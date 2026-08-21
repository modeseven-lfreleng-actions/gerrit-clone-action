# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Matthew Watkins <mwatkins@linuxfoundation.org>

"""Refresh worker for individual repository update operations.

This module holds the top-level refresh flow. The mechanics it orchestrates
live in a stack of focused mixins, each of which owns one responsibility:

* :mod:`gerrit_clone.refresh_git_env` — remotes and git subprocess environment
* :mod:`gerrit_clone.refresh_repo_state` — working-tree state and stashing
* :mod:`gerrit_clone.refresh_branch_repair` — default-branch and upstream repair
* :mod:`gerrit_clone.refresh_execution` — fetch/pull execution and retries
* :mod:`gerrit_clone.refresh_force` — force-mode repository repair
* :mod:`gerrit_clone.refresh_output` — git output classification and counting

Their public names are re-exported here so ``gerrit_clone.refresh_worker``
remains the single import (and test patch) surface for refresh behaviour.
"""

from __future__ import annotations

import subprocess
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from gerrit_clone.logging import get_logger
from gerrit_clone.models import Config, RefreshResult, RefreshStatus, RetryPolicy
from gerrit_clone.refresh_force import ForceModeMixin
from gerrit_clone.refresh_git_env import SSH_HANDSHAKE_JITTER_SECONDS
from gerrit_clone.refresh_output import (
    RefreshAuthError,
    RefreshError,
    RefreshTimeoutError,
)
from gerrit_clone.refresh_repo_state import StashOutcome

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)

# ``subprocess`` and ``time`` are re-exported deliberately: the refresh git
# calls and sleeps now live in the mixin modules, but existing patch targets
# (``gerrit_clone.refresh_worker.subprocess.run``,
# ``gerrit_clone.refresh_worker.time.sleep``) must keep resolving.
__all__ = [
    "SSH_HANDSHAKE_JITTER_SECONDS",
    "RefreshAuthError",
    "RefreshError",
    "RefreshTimeoutError",
    "RefreshWorker",
    "StashOutcome",
    "subprocess",
    "time",
]


class RefreshWorker(ForceModeMixin):
    """Worker for refreshing individual repositories."""

    def __init__(
        self,
        config: Config | None = None,
        retry_policy: RetryPolicy | None = None,
        timeout: int = 300,
        fetch_only: bool = False,
        prune: bool = True,
        skip_conflicts: bool = True,
        auto_stash: bool = False,
        strategy: str = "merge",
        filter_gerrit_only: bool = True,
        force: bool = False,
        force_hard: bool = False,
        ssh_jitter_seconds: float = SSH_HANDSHAKE_JITTER_SECONDS,
    ) -> None:
        """Initialize refresh worker.

        Args:
            config: Optional configuration for Git operations (SSH, etc.)
            retry_policy: Retry policy for transient errors
            timeout: Timeout for each git operation in seconds
            fetch_only: Only fetch changes without merging
            prune: Prune deleted remote branches
            skip_conflicts: Skip repositories with uncommitted changes
            auto_stash: Automatically stash uncommitted changes
            strategy: Git pull strategy ('merge' or 'rebase')
            filter_gerrit_only: Only refresh repositories with Gerrit remotes
            force: Force refresh by fixing detached HEAD, upstream tracking, and stashing changes
            force_hard: Superset of ``force`` that additionally hard-resets the
                default branch to its upstream ref, discarding local commits and
                divergence. Implies ``force``.
            ssh_jitter_seconds: Maximum random delay before each SSH-backed git
                network operation, used to de-synchronise concurrent handshakes.
        """
        self.config = config
        self.retry_policy = retry_policy or RetryPolicy()
        self.timeout = timeout
        self.fetch_only = fetch_only
        self.prune = prune
        self.skip_conflicts = skip_conflicts
        self.auto_stash = auto_stash
        self.strategy = strategy
        self.filter_gerrit_only = filter_gerrit_only
        # force_hard is a strict superset of force.
        self.force_hard = force_hard
        self.force = force or force_hard
        self.ssh_jitter_seconds = max(0.0, ssh_jitter_seconds)

    def refresh_repository(self, repo_path: Path) -> RefreshResult:
        """Refresh a single repository.

        Args:
            repo_path: Path to repository root

        Returns:
            RefreshResult with operation details
        """
        started_at = datetime.now(UTC)
        project_name = self._get_project_name(repo_path)

        result = RefreshResult(
            path=repo_path,
            project_name=project_name,
            status=RefreshStatus.PENDING,
            started_at=started_at,
            first_started_at=started_at,
        )

        try:
            if not self._is_git_repository(repo_path):
                result.status = RefreshStatus.NOT_GIT_REPO
                result.error_message = "Not a Git repository"
                self._stamp_completion(result, started_at)
                logger.debug(f"⊘ {project_name}: Not a Git repository")
                return result

            state = self._check_repository_state(repo_path)
            result.current_branch = state.get("branch")
            result.detached_head = state.get("detached_head", False)
            result.had_uncommitted_changes = state.get("has_uncommitted", False)

            remote_url = self._get_remote_url(repo_path)
            result.remote_url = remote_url

            # Check if it's a Gerrit repository
            if self.filter_gerrit_only and not self._is_gerrit_repository(remote_url):
                result.status = RefreshStatus.NOT_GERRIT_REPO
                result.error_message = f"Not a Gerrit repository (remote: {remote_url})"
                self._stamp_completion(result, started_at)
                logger.debug(f"⊘ {project_name}: Not a Gerrit repository")
                return result

            if not self._prepare_repository(repo_path, result, state, started_at):
                return result

            result.status = RefreshStatus.REFRESHING

            success = self._execute_adaptive_refresh(repo_path, result)

            self._apply_refresh_outcome(repo_path, result, success)

        except Exception as e:
            result.status = RefreshStatus.FAILED
            result.error_message = f"Unexpected error: {e}"
            self._stamp_completion(result, started_at)
            logger.error(f"❌ {project_name}: {e}")
            return result

        # Set completion metadata
        self._stamp_completion(result, started_at)

        return result

    def _prepare_repository(
        self,
        repo_path: Path,
        result: RefreshResult,
        state: dict[str, Any],
        started_at: datetime,
    ) -> bool:
        """Bring a repository into a refreshable state.

        Force mode repairs the repository in place; normal mode instead skips
        anything that is not already refreshable.

        Args:
            repo_path: Repository path
            result: Result object to update
            state: Repository state as observed before preparation
            started_at: Timestamp the refresh began, for completion metadata

        Returns:
            True if the repository is ready to refresh, False if the refresh
            must stop and return ``result`` as-is.
        """
        if self.force:
            # Force mode: Fix issues automatically
            if not self._prepare_forced_repository(
                repo_path, result, state, started_at
            ):
                return False
        # Normal mode: Skip problematic repos
        elif not self._guard_unforced_repository(result, state, started_at):
            return False

        return self._handle_uncommitted_changes(repo_path, result, started_at)

    def _guard_unforced_repository(
        self, result: RefreshResult, state: dict[str, Any], started_at: datetime
    ) -> bool:
        """Skip repositories that normal (non-force) mode must not touch.

        Args:
            result: Result object to update
            state: Repository state as observed before preparation
            started_at: Timestamp the refresh began, for completion metadata

        Returns:
            True if the repository is refreshable, False if it was skipped.
        """
        if result.detached_head:
            result.status = RefreshStatus.DETACHED_HEAD
            result.error_message = "Repository in detached HEAD state"
            self._stamp_completion(result, started_at)
            logger.warning(
                f"⚠️ {result.project_name}: Detached HEAD state, skipping refresh"
            )
            return False

        if not state.get("has_upstream", False):
            result.status = RefreshStatus.SKIPPED
            result.error_message = (
                f"Branch '{result.current_branch}' has no upstream tracking branch"
            )
            self._stamp_completion(result, started_at)
            logger.warning(
                f"⚠️ {result.project_name}: No upstream tracking branch, skipping refresh"
            )
            return False

        return True

    def _handle_uncommitted_changes(
        self, repo_path: Path, result: RefreshResult, started_at: datetime
    ) -> bool:
        """Skip or stash an unclean working tree outside force mode.

        Args:
            repo_path: Repository path
            result: Result object to update
            started_at: Timestamp the refresh began, for completion metadata

        Returns:
            True if the repository is ready to refresh, False if the refresh
            must stop and return ``result`` as-is.
        """
        if not result.had_uncommitted_changes or self.force:
            return True

        if self.skip_conflicts and not self.auto_stash:
            result.status = RefreshStatus.UNCOMMITTED_CHANGES
            result.error_message = "Uncommitted changes present"
            self._stamp_completion(result, started_at)
            logger.warning(
                f"⚠️ {result.project_name}: Uncommitted changes, skipping refresh"
            )
            return False

        if self.auto_stash:
            # Stash uncommitted changes
            stash_outcome = self._stash_changes(repo_path)
            if stash_outcome is StashOutcome.CREATED:
                result.stash_created = True
                result.stash_branch = result.current_branch
                logger.debug(f"💾 {result.project_name}: Stashed uncommitted changes")
            elif stash_outcome is StashOutcome.NOTHING_TO_STASH:
                # Nothing git could stash (e.g. a modified submodule
                # gitlink); proceed with the refresh as if clean.
                logger.debug(
                    f"💾 {result.project_name}: Nothing to stash "
                    f"(e.g. submodule-only change)"
                )
            else:
                result.status = RefreshStatus.FAILED
                result.error_message = "Failed to stash uncommitted changes"
                self._stamp_completion(result, started_at)
                logger.error(
                    f"❌ {result.project_name}: Failed to stash uncommitted changes"
                )
                return False

        return True

    def _apply_refresh_outcome(
        self, repo_path: Path, result: RefreshResult, success: bool
    ) -> None:
        """Record the final status and restore any stash we created.

        Args:
            repo_path: Repository path
            result: Result object to update
            success: Whether the fetch/pull ultimately succeeded
        """
        if not success:
            result.status = RefreshStatus.FAILED
            if not result.error_message:
                result.error_message = "Refresh failed for unknown reason"
            return

        # Check if we pulled any commits
        if result.commits_pulled > 0:
            result.status = RefreshStatus.SUCCESS
            result.was_behind = True
            logger.debug(
                f"✅ {result.project_name}: Updated ({result.commits_pulled} commits, {result.files_changed} files)"
            )
        else:
            result.status = RefreshStatus.UP_TO_DATE
            logger.debug(f"✓ {result.project_name}: Already up-to-date")

        # Pop stash if we created one, but only back onto the branch it
        # came from. In force mode the stash may have been taken on a
        # feature branch before switching to the default branch; popping
        # it here would apply that work to the wrong branch (and drop
        # the stash entry). In that case leave the stash intact for
        # manual recovery.
        if not result.stash_created:
            return

        stashed_elsewhere = (
            result.stash_branch is not None
            and result.current_branch != result.stash_branch
        )
        if stashed_elsewhere:
            logger.warning(
                f"⚠️ {result.project_name}: Stash was created on "
                f"'{result.stash_branch}' but the working tree is now "
                f"on '{result.current_branch}'; leaving the stash "
                f"intact for manual recovery (git stash list)"
            )
        elif self._pop_stash(repo_path):
            result.stash_popped = True
            logger.debug(f"💾 {result.project_name}: Restored stashed changes")
        else:
            logger.warning(
                f"⚠️ {result.project_name}: Failed to restore stash (may have conflicts)"
            )

    def _get_project_name(self, repo_path: Path) -> str:
        """Get project name from repository path.

        Args:
            repo_path: Repository path

        Returns:
            Project name (directory name)
        """
        return repo_path.name
