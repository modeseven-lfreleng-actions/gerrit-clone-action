# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Refreshing a working copy the run filters, through a filtered copy.

A working copy (``--no-mirror``) used to pull first and be filtered
afterwards, where it stood.  A filter that failed left what had just
arrived, unfiltered, in the checkout.  One that ran at all rewrote the
checkout itself -- ``git filter-repo`` folds remote-tracking refs into
local branches -- and recorded it as filtered, so every later refresh
skipped it.

Instead, as for a mirror (see :mod:`gerrit_clone.refresh_filtered`):

1. A bare copy is made inside the checkout's git directory, where
   ``git status`` never sees it, holding the checkout's remote-tracking
   refs as branches -- ``refs/remotes/X`` as ``refs/heads/X`` -- and its
   tags.  Both filter methods rewrite branches, and ``filter-repo`` has
   no ``origin`` refs to fold.
2. The copy fetches from the checkout's own remotes, and is filtered.
3. If the filtered upstream does not extend what the checkout holds,
   the checkout cannot follow it, and is refused.
4. Otherwise any filters that rewrote the copy are recorded in the
   checkout, and pushing from it blocked; its remote-tracking refs and
   tags then take the copy's, in one atomic fetch.
5. The current branch fast-forwards, or rebases, onto its upstream, as a
   pull would, with nothing fetched again.

Anything failing before step 4 leaves the checkout as it was, and puts
back any stash the refresh made.  Filtering upstream content the same
way gives the same commits each time, so a checkout an earlier filter
rewrote, in place or through a copy, follows it in the same way.
"""

from __future__ import annotations

import tempfile
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from gerrit_clone.content_filter import apply_content_filters
from gerrit_clone.content_origin import block_pushes
from gerrit_clone.content_policy import PolicyReadError, add_policy, recorded_policy
from gerrit_clone.content_stage import filter_repository
from gerrit_clone.logging import get_logger
from gerrit_clone.models import RefreshStatus
from gerrit_clone.refresh_filtered import FilteredRefreshMixin
from gerrit_clone.refresh_git_env import run_git

if TYPE_CHECKING:
    from gerrit_clone.models import RefreshResult

logger = get_logger(__name__)

_TRACKING = "refs/remotes/"

#: Why a working copy is refused upstream history its filters rewrote.
CANNOT_FOLLOW_REFUSAL = (
    "Filtered, upstream history no longer extends what this working copy "
    "holds, so it cannot follow it, and was left as it was. Re-clone it, "
    "with the filters, to update it"
)


class _Staged(Enum):
    """How a staged refresh of a working copy ended."""

    #: Published, and the current branch brought up to date.
    DONE = "done"
    #: Published, but the branch could not be brought up to date.
    NOT_INTEGRATED = "not integrated"
    #: Nothing reached the checkout.
    FAILED = "failed"
    #: Nothing reached the checkout, which cannot follow upstream.
    REFUSED = "refused"


class StagedCheckoutMixin(FilteredRefreshMixin):
    """Staged refresh of working copies the run filters."""

    def _refresh_in_place(
        self, repo_path: Path, result: RefreshResult, *, bare: bool
    ) -> bool:
        """Refresh a prepared *repo_path*: staged if a working copy the run filters."""
        spec = self.content_filters
        if bare or spec is None or spec.filters_for(spec.project_name(repo_path)).empty:
            return self._execute_adaptive_refresh(repo_path, result, bare=bare)
        git_dir = run_git(
            ["git", "rev-parse", "--absolute-git-dir"], repo_path, timeout=10
        )
        if git_dir.returncode != 0:
            result.error_message = f"Could not find its git directory: {git_dir.stderr}"
            self._restore_stash(repo_path, result)
            return False
        stage_parent = Path(
            tempfile.mkdtemp(prefix="gerrit-clone-stage-", dir=git_dir.stdout.strip())
        )
        try:
            outcome = self._refresh_checkout_stage(
                repo_path, stage_parent / "repo.git", result
            )
        finally:
            removed = self._remove_stage(stage_parent, result)
        # Brought up to date but for the copy's removal, the refresh still
        # fails, and the outcome step never restores a failure's stash.
        if outcome in (_Staged.FAILED, _Staged.REFUSED) or (
            outcome is _Staged.DONE and not removed
        ):
            self._restore_stash(repo_path, result)
        if outcome is _Staged.REFUSED:
            result.status = RefreshStatus.SKIPPED
            logger.warning(f"⚠️ {result.project_name}: {result.error_message}")
        return outcome is _Staged.DONE and removed

    def _refresh_checkout_stage(
        self, repo_path: Path, stage: Path, result: RefreshResult
    ) -> _Staged:
        """Steps 1 to 5 of the module docstring, with *stage* as the copy."""
        stopped = self._stage_and_filter(repo_path, stage, result)
        if stopped is not None:
            return stopped
        if not self._record_and_publish(repo_path, stage, result):
            return _Staged.FAILED
        result.content_filtered = True
        if self._bring_branch_up_to_date(repo_path, result):
            return _Staged.DONE
        return _Staged.NOT_INTEGRATED

    def _stage_and_filter(
        self, repo_path: Path, stage: Path, result: RefreshResult
    ) -> _Staged | None:
        """Steps 1 to 3: ``None`` if the checkout may take what arrived."""
        spec = self.content_filters
        assert spec is not None  # Checked by _refresh_in_place.
        if not (
            self._prepare_stage(repo_path, stage, result)
            and self._as_upstream_copy(repo_path, stage, result)
            and self._execute_adaptive_refresh(stage, result, bare=True)
        ):
            return _Staged.FAILED
        # Looked up here, so a test patching this module's name sees it.
        reason = filter_repository(
            spec,
            stage,
            spec.project_name(repo_path),
            self.timeout,
            apply=apply_content_filters,
        )
        if reason is not None:
            result.error_message = (
                f"Filtering what arrived failed, so the working copy was "
                f"left as it was: {reason}"
            )
            return _Staged.FAILED
        if not self.fetch_only and not self._can_follow(repo_path, stage):
            result.error_message = CANNOT_FOLLOW_REFUSAL
            return _Staged.REFUSED
        return None

    def _record_and_publish(
        self, repo_path: Path, stage: Path, result: RefreshResult
    ) -> bool:
        """Step 4: record what rewrote the copy, then publish its refs."""
        try:
            policy = recorded_policy(stage)
        except PolicyReadError as exc:
            result.error_message = str(exc)
            return False
        if not policy.empty and not (
            add_policy(repo_path, policy) and block_pushes(repo_path)
        ):
            result.error_message = (
                "Could not record the content-filter policy, or block pushing, "
                "so the working copy was left as it was"
            )
            return False
        return self._publish_to_checkout(repo_path, stage, result)

    def _as_upstream_copy(
        self, repo_path: Path, stage: Path, result: RefreshResult
    ) -> bool:
        """Make *stage*, a copy of working copy *repo_path*, stand in for upstream.

        It holds the checkout's remote-tracking refs as its branches,
        ``refs/remotes/X`` as ``refs/heads/X``, and fetches there; the
        checkout's own branches, and its remotes' ``HEAD`` refs, go.
        """
        moved = run_git(
            [
                "git",
                "fetch",
                "--quiet",
                "--prune",
                "--no-tags",
                str(repo_path),
                f"+{_TRACKING}*:refs/heads/*",
                f"^{_TRACKING}*/HEAD",
            ],
            stage,
            timeout=self.timeout,
        )
        if moved.returncode != 0:
            result.error_message = f"Could not stage a copy: {moved.stderr.strip()}"
            return False
        listed = run_git(
            ["git", "config", "--get-regexp", r"^remote\..*\.fetch$"], stage, timeout=10
        )
        refspecs: dict[str, list[str]] = {}
        for line in listed.stdout.splitlines() if listed.returncode == 0 else []:
            key, _, value = line.partition(" ")
            refspecs.setdefault(key, []).append(value)
        for key, values in refspecs.items():
            staged = [
                refspec
                for refspec in map(_staged_refspec, values)
                if refspec is not None
            ]
            if len(staged) != len(values):
                result.error_message = (
                    f"Could not stage a copy: {key} fetches outside "
                    f"{_TRACKING} and refs/tags/"
                )
                return False
            run_git(["git", "config", "--unset-all", key], stage, timeout=10)
            for refspec in staged:
                added = run_git(
                    ["git", "config", "--add", key, refspec], stage, timeout=10
                )
                if added.returncode != 0:
                    result.error_message = (
                        f"Could not stage a copy: {added.stderr.strip()}"
                    )
                    return False
        return True

    def _can_follow(self, repo_path: Path, stage: Path) -> bool:
        """Whether the filtered upstream extends what *repo_path* holds.

        Its upstream's old tip, or the branch itself if that ref has gone,
        must be an ancestor of the new one: an earlier filter's rewrite,
        repeated, gives the same commits, while one that now rewrites
        history the checkout already holds does not.
        """
        upstream = run_git(
            ["git", "rev-parse", "--symbolic-full-name", "@{upstream}"],
            repo_path,
            timeout=10,
        )
        ref = upstream.stdout.strip()
        if upstream.returncode != 0 or not ref.startswith(_TRACKING):
            ref = self._configured_upstream(repo_path)
        if not ref.startswith(_TRACKING):
            return True  # No remote upstream: bringing it up to date fails.
        new = run_git(
            [
                "git",
                "rev-parse",
                "-q",
                "--verify",
                "refs/heads/" + ref[len(_TRACKING) :],
            ],
            stage,
            timeout=10,
        )
        if new.returncode != 0:
            return True  # Gone upstream: bringing it up to date fails.
        old = run_git(
            ["git", "rev-parse", "-q", "--verify", ref], repo_path, timeout=10
        )
        if old.returncode != 0:
            old = run_git(["git", "rev-parse", "HEAD"], repo_path, timeout=10)
        ancestor = run_git(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                old.stdout.strip(),
                new.stdout.strip(),
            ],
            stage,
            timeout=self.timeout,
        )
        return ancestor.returncode == 0

    @staticmethod
    def _configured_upstream(repo_path: Path) -> str:
        """The current branch's upstream ref, from config, even if it is gone."""
        branch = run_git(["git", "symbolic-ref", "-q", "HEAD"], repo_path, timeout=10)
        if branch.returncode != 0:
            return ""
        found = run_git(
            ["git", "for-each-ref", "--format=%(upstream)", branch.stdout.strip()],
            repo_path,
            timeout=10,
        )
        return found.stdout.strip() if found.returncode == 0 else ""

    def _publish_to_checkout(
        self, repo_path: Path, stage: Path, result: RefreshResult
    ) -> bool:
        """Give *repo_path* the copy's refs, as remote-tracking refs and tags."""
        published = run_git(
            [
                "git",
                "fetch",
                "--atomic",
                "--no-tags",
                str(stage),
                "+refs/heads/*:refs/remotes/*",
                "+refs/tags/*:refs/tags/*",
            ],
            repo_path,
            timeout=self.timeout,
        )
        if published.returncode != 0:
            result.error_message = (
                f"Could not publish the refreshed refs, so the working copy "
                f"was left as it was: {published.stderr.strip()}"
            )
            return False
        result.commits_pulled = self._count_fetched_commits(published.stderr)
        if self.prune:
            self._prune_checkout(repo_path, stage)
        return True

    @staticmethod
    def _prune_checkout(repo_path: Path, stage: Path) -> None:
        """Delete remote-tracking refs the copy no longer has, as --prune would."""
        kept = run_git(
            ["git", "for-each-ref", "--format=%(refname)"], stage, timeout=30
        )
        tracking = run_git(
            ["git", "for-each-ref", "--format=%(refname) %(symref)", "refs/remotes/"],
            repo_path,
            timeout=30,
        )
        if kept.returncode != 0 or tracking.returncode != 0:
            logger.warning(f"Could not prune the remote-tracking refs of {repo_path}")
            return
        present = set(kept.stdout.split())
        for line in tracking.stdout.splitlines():
            ref, _, symref = line.partition(" ")
            if symref or "refs/heads/" + ref[len(_TRACKING) :] in present:
                continue
            run_git(["git", "update-ref", "-d", ref], repo_path, timeout=10)

    def _bring_branch_up_to_date(self, repo_path: Path, result: RefreshResult) -> bool:
        """Fast-forward or rebase onto the published upstream, as a pull would."""
        if self.fetch_only:
            return True
        before = run_git(["git", "rev-parse", "HEAD"], repo_path, timeout=10)
        if self.strategy == "rebase":
            command = ["git", "rebase", "@{upstream}"]
        else:
            command = ["git", "merge", "--ff-only", "@{upstream}"]
        done = run_git(command, repo_path, timeout=self.timeout)
        if done.returncode != 0:
            output = (done.stdout + done.stderr).strip()
            if "CONFLICT" in output:
                result.status = RefreshStatus.CONFLICTS
            result.error_message = f"Could not bring the branch up to date: {output}"
            return False
        moved = run_git(
            ["git", "rev-list", "--count", f"{before.stdout.strip()}..HEAD"],
            repo_path,
            timeout=30,
        )
        result.commits_pulled = (
            int(moved.stdout.strip() or 0) if moved.returncode == 0 else 0
        )
        diff = run_git(
            ["git", "diff", "--shortstat", before.stdout.strip(), "HEAD"],
            repo_path,
            timeout=30,
        )
        result.files_changed = self._count_changed_files(diff.stdout)
        return True

    def _restore_stash(self, repo_path: Path, result: RefreshResult) -> None:
        """Pop a stash the refresh made, onto the branch it came from.

        In force mode the stash may have been taken on a feature branch
        before switching to the default branch; popping it there would
        apply that work to the wrong branch, and drop the stash entry.
        It is then left for ``git stash list``.
        """
        if not result.stash_created or result.stash_popped:
            return
        if (
            result.stash_branch is not None
            and result.current_branch != result.stash_branch
        ):
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


def _staged_refspec(refspec: str) -> str | None:
    """*refspec* for a working copy's staged copy; ``None`` if it has none.

    Remote-tracking destinations become branches, as the copy holds
    them; tags, negative refspecs and ones naming no destination stay as
    they are.  Anything else would land where publishing never looks.
    """
    force = "+" if refspec.startswith("+") else ""
    source, colon, destination = refspec.removeprefix("+").partition(":")
    if refspec.startswith("^") or not colon or not destination:
        return refspec
    if destination.startswith(_TRACKING):
        return f"{force}{source}:refs/heads/{destination.removeprefix(_TRACKING)}"
    if destination.startswith("refs/tags/"):
        return refspec
    return None
