# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""One repository through the content filters, for every command.

``clone``, ``refresh`` and ``mirror`` each filter what they fetched after
fetching it, and each makes the same two refusals before rewriting
anything:

- A project whose intent replaces ``--git-filter`` tokens the run does
  not supply is not filtered at all: the tokens are kept only as
  digests, and filtering it without them would leave them in place.
- ``--git-filter`` and ``--redact-secrets`` scan the whole history, and
  a shallow clone hides part of it, so they would mark it filtered
  having seen only some of it.  They are dropped for that repository,
  which counts as failed; ``--remove-files``, which needs no history,
  still runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from gerrit_clone.content_filter import apply_content_filters, is_shallow_repository
from gerrit_clone.content_spec import missing_tokens_refusal

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from gerrit_clone.content_spec import ContentFilterSpec

SHALLOW_HISTORY_REFUSED = (
    "Refused --git-filter / --redact-secrets on a shallow repository: "
    "truncated history can hide older secrets. Re-clone it without --depth"
)


class ApplyContentFilters(Protocol):
    """Call signature of :func:`gerrit_clone.content_filter.apply_content_filters`."""

    def __call__(
        self,
        repo_path: Path,
        project_name: str,
        remove_patterns: list[str] | None = None,
        git_filter_projects: dict[str, list[str]] | None = None,
        *,
        redact_secrets: bool = False,
        timeout: int = 600,
    ) -> tuple[bool, str | None]:
        """Filter the repository at *repo_path*."""
        ...


def filter_repository(
    spec: ContentFilterSpec,
    repo_path: Path,
    project: str,
    timeout: int,
    *,
    is_shallow: Callable[[Path], bool] = is_shallow_repository,
    apply: ApplyContentFilters = apply_content_filters,
) -> str | None:
    """Filter *repo_path*, project *project*, as *spec* decides for it.

    Args:
        spec: The run's filters, its options and the tree's intent.
        repo_path: Repository to filter.
        project: Its project name, for ``--git-filter`` patterns.
        timeout: Seconds each filtering step may take.
        is_shallow: Shallow-clone probe, injectable for the caller.
        apply: Filter runner, injectable for the caller.

    Returns:
        ``None`` if it was filtered as decided, else why it was not.
    """
    filters = spec.filters_for(project)
    if filters.missing_tokens:
        return missing_tokens_refusal(filters.missing_tokens)
    if filters.empty:
        return None
    git_filter = filters.git_filter_projects
    redact = filters.redact_secrets
    refused = None
    if filters.history and is_shallow(repo_path):
        if not filters.remove_patterns:
            return SHALLOW_HISTORY_REFUSED
        refused, git_filter, redact = SHALLOW_HISTORY_REFUSED, None, False
    ok, error = apply(
        repo_path,
        project,
        remove_patterns=filters.remove_patterns,
        git_filter_projects=git_filter,
        redact_secrets=redact,
        timeout=timeout,
    )
    failure = None if ok else (error or "content filtering failed")
    return "; ".join(reason for reason in (refused, failure) if reason) or None
