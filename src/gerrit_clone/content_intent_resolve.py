# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Settling, at the start of a run, what each repository is filtered with.

The tree's intent (:mod:`gerrit_clone.content_intent`) is extended, and
written down, before the run clones, refreshes or deletes anything:

1. The recorded intent is read.  If it cannot be, the run stops.
2. Every repository already in the tree is asked which filters rewrote
   it, and those are added for that project.  This brings trees that an
   earlier release filtered -- one that kept only these per-repository
   records -- under the intent, and it is what lets ``mirror --overwrite``
   delete a repository without losing what it was filtered with.
3. The run's own options are added.

Nothing is ever taken away.  A repository an earlier release filtered
without any record is not added: what it was filtered with is unknown,
and it stays refused on its own (see :mod:`gerrit_clone.content_legacy`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gerrit_clone.content_intent import (
    INTENT_DIR,
    FilterIntent,
    IntentError,
    find_root,
    load_intent,
    save_intent,
)
from gerrit_clone.content_policy import PolicyReadError, stored_policy
from gerrit_clone.content_spec import ContentFilterSpec
from gerrit_clone.logging import get_logger
from gerrit_clone.refresh_discovery import RepositoryDiscoveryMixin, project_name_for
from gerrit_clone.worktree_exclude import hide_from_checkout

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

logger = get_logger(__name__)


class _TreeDiscovery(RepositoryDiscoveryMixin):
    """Every repository beneath a tree, nested ones included."""

    recursive = True
    include_projects: list[str] | None = None
    exclude_projects: list[str] | None = None


def resolve_filters(
    start: Path, options: ContentFilterSpec | None, *, persist: bool
) -> ContentFilterSpec | None:
    """The filters this run applies to the tree containing *start*.

    Args:
        start: The run's output path.  The tree's root is the nearest of
            it and its parents holding an intent, or *start* itself.
        options: The run's own filter options, if it was given any.
        persist: Whether to write the extended intent down: false for a
            dry run, which changes nothing.

    Returns:
        ``None`` if neither the options nor the tree filter anything.

    Raises:
        IntentError: If the intent could not be read, the repositories'
            records could not be, or the extended intent not be written.
    """
    root = find_root(start) or start.resolve()
    recorded = load_intent(root)
    intent = recorded.union(gathered_intent(root))
    if options is not None:
        intent = intent.union(
            FilterIntent.of_options(
                options.remove_patterns,
                options.git_filter_projects,
                options.redact_secrets,
            )
        )
    if persist and intent.scopes != recorded.scopes and not intent.empty:
        save_intent(root, intent)
        logger.info(f"Recorded the content-filter intent for {root}")
    if persist and not intent.empty:
        # Left visible at a checkout's top level, the intent would be an
        # untracked change there, and refresh would skip the checkout.
        try:
            hide_from_checkout(root, [f"/{INTENT_DIR}/"])
        except OSError as exc:
            # Only costs refresh skipping the checkout, which it reports.
            logger.warning(f"Could not hide {INTENT_DIR} from git: {exc}")
    if intent.empty:
        return None
    if options is None:
        return ContentFilterSpec(None, None, False, root, intent)
    return ContentFilterSpec(
        options.remove_patterns,
        options.git_filter_projects,
        options.redact_secrets,
        root,
        intent,
    )


def gathered_intent(root: Path, repos: Iterable[Path] | None = None) -> FilterIntent:
    """What every repository beneath *root* records was filtered out of it.

    Raises:
        IntentError: If a repository's record could not be read.
    """
    if not root.is_dir():
        return FilterIntent()
    scopes = {}
    for repo in repos if repos is not None else repositories_beneath(root):
        try:
            # One git call per repository: every key at once.
            policy = stored_policy(repo)
        except PolicyReadError as exc:
            raise IntentError(str(exc)) from exc
        if not policy.empty and not policy.earlier_release:
            scopes[("project", project_name_for(repo, root.resolve()))] = policy
    return FilterIntent(scopes)


def repositories_beneath(root: Path) -> list[Path]:
    """Every repository at or beneath *root*, nested ones included."""
    return _TreeDiscovery().discover_local_repositories(root)
