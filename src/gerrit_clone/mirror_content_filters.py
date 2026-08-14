# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Content filtering of cloned repositories before they are pushed.

Applies the ``--remove-files`` / ``--git-filter`` / ``--redact-secrets``
options to each successful clone and aborts the batch if any repository
could not be filtered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from gerrit_clone.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

logger = get_logger(__name__)


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


@dataclass(frozen=True)
class ContentFilterRunner:
    """Content-filter helpers, resolved by the caller at call time."""

    is_shallow: Callable[[Path], bool]
    apply_filters: ApplyContentFilters


@dataclass(frozen=True)
class ContentFilterSettings:
    """Requested content filtering behaviour for a mirror batch."""

    remove_file_patterns: list[str] | None
    git_filter_projects: dict[str, list[str]] | None
    redact_secrets: bool
    clone_timeout: int

    @property
    def enabled(self) -> bool:
        """Whether any content filter was requested."""
        return bool(
            self.remove_file_patterns or self.git_filter_projects or self.redact_secrets
        )


@dataclass
class _FilterTally:
    """Running totals for a content filtering pass."""

    succeeded: int = 0
    failed: int = 0
    failed_projects: set[str] = field(default_factory=set)


def _filter_one_repo(
    clone_result: Any,
    settings: ContentFilterSettings,
    runner: ContentFilterRunner,
    tally: _FilterTally,
) -> None:
    """Apply the requested filters to a single cloned repository."""
    # Fail closed on shallow repositories when history-
    # dependent filters are requested: --git-filter /
    # --redact-secrets rely on full commit history, so a
    # shallow repo (e.g. created by a prior clone --depth)
    # can hide older secrets and give a false sense of
    # safety.  --remove-files targets file paths present at
    # the branch tips and does not depend on full history
    # being available (though it may still rewrite history
    # via git filter-repo when that tool is present), so it
    # remains safe on a shallow repo and still runs; only
    # the unsafe history-scanning filters are dropped for
    # the repo.
    repo_git_filter = settings.git_filter_projects
    repo_redact = settings.redact_secrets
    history_filters_skipped = False
    if (settings.git_filter_projects or settings.redact_secrets) and (
        runner.is_shallow(clone_result.path)
    ):
        logger.warning(
            "Refusing to run --git-filter / --redact-secrets "
            "on shallow repo %s: truncated history can hide "
            "older secrets. Re-clone without --depth.",
            clone_result.project.name,
        )
        # The requested redaction/rewrite did not run, so
        # this repo counts as a filtering failure even if
        # the safe --remove-files step below succeeds.
        tally.failed += 1
        tally.failed_projects.add(clone_result.project.name)
        history_filters_skipped = True
        repo_git_filter = None
        repo_redact = False
        if not settings.remove_file_patterns:
            # Nothing history-independent left to do.
            return

    success, error = runner.apply_filters(
        clone_result.path,
        clone_result.project.name,
        remove_patterns=settings.remove_file_patterns,
        git_filter_projects=repo_git_filter,
        redact_secrets=repo_redact,
        timeout=settings.clone_timeout,
    )
    if history_filters_skipped:
        # Already counted as a failure above; don't also
        # count the safe --remove-files step as a success.
        if not success:
            logger.warning(
                "Content filter failed for %s: %s",
                clone_result.project.name,
                error,
            )
    elif success:
        tally.succeeded += 1
    else:
        tally.failed += 1
        tally.failed_projects.add(clone_result.project.name)
        logger.warning(
            "Content filter failed for %s: %s",
            clone_result.project.name,
            error,
        )


def apply_filters_to_clones(
    clone_results: list[Any],
    settings: ContentFilterSettings,
    runner: ContentFilterRunner,
) -> None:
    """Filter every successful clone in place.

    Args:
        clone_results: Results from the clone phase
        settings: Requested filtering behaviour
        runner: Content-filter helpers to invoke

    Raises:
        RuntimeError: If any repository could not be filtered.  Silently
            dropping projects would make them disappear from the
            manifest, so the whole batch is aborted instead.
    """
    if not settings.enabled:
        return

    tally = _FilterTally()
    logger.info("🔧 Applying content filters to cloned repositories...")
    for clone_result in clone_results:
        if not clone_result.success or not clone_result.path:
            continue
        _filter_one_repo(clone_result, settings, runner, tally)
    logger.info(
        "Content filtering complete: %d succeeded, %d failed",
        tally.succeeded,
        tally.failed,
    )

    # Abort the batch if any content filters failed — silently
    # dropping projects would make them disappear from the manifest.
    if tally.failed_projects:
        raise RuntimeError(
            f"Content filtering failed for {tally.failed} project(s), "
            f"aborting batch: {sorted(tally.failed_projects)}"
        )
