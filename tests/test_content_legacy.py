# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Upgrading over repositories an earlier release content-filtered.

Releases before the filter policy was recorded left no record of what
they filtered.  Refreshing what they left -- a mirror the worktree
fallback committed a removal on, keeping ``origin``; or one
``git filter-repo`` rewrote, keeping a source remote -- would fetch the
filtered content straight back.  Each must be refused, for good.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from gerrit_clone import content_legacy
from gerrit_clone.content_filter import apply_content_filters
from gerrit_clone.content_legacy import LEGACY_REMOVAL_SUBJECT
from gerrit_clone.content_policy import (
    ContentFilterSpec,
    PolicyReadError,
    recorded_policy,
)
from gerrit_clone.models import RefreshStatus
from gerrit_clone.refresh_manager import RefreshManager
from gerrit_clone.refresh_worker import EARLIER_RELEASE_REFUSAL, RefreshWorker

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _git(*args: str, cwd: Path | None = None, stdin: str | None = None) -> str:
    return subprocess.run(
        [
            "git",
            "-c",
            "user.email=t@example.com",
            "-c",
            "user.name=T",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=cwd,
        input=stdin,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.fixture
def upstream(tmp_path: Path) -> Path:
    """An upstream holding a file that filtering removes."""
    repo = tmp_path / "upstream"
    _git("init", "-q", "-b", "main", str(repo))
    (repo / "secret.txt").write_text("hunter2\n")
    (repo / "file.txt").write_text("one\n")
    _git("add", ".", cwd=repo)
    _git("commit", "-q", "-m", "one", cwd=repo)
    return repo


def _mirror(upstream: Path) -> Path:
    mirror = upstream.parent / "tree" / "proj.git"
    _git("clone", "-q", "--mirror", upstream.as_uri(), str(mirror))
    return mirror


def _fallback_filtered(upstream: Path) -> Path:
    """A mirror as the earlier releases' worktree fallback left it.

    ``secret.txt`` removed by a commit on the branch tip, under that
    fallback's subject; ``origin`` and its refspec kept; nothing in the
    config to say any of it happened.
    """
    mirror = _mirror(upstream)
    listing = _git("ls-tree", "main", cwd=mirror).splitlines()
    kept = "\n".join(line for line in listing if not line.endswith("\tsecret.txt"))
    tree = _git("mktree", cwd=mirror, stdin=kept + "\n")
    message = (
        f"{LEGACY_REMOVAL_SUBJECT}\n\nFiles removed by gerrit-clone content filter"
    )
    commit = _git("commit-tree", tree, "-p", "main", "-m", message, cwd=mirror)
    _git("update-ref", "refs/heads/main", commit, cwd=mirror)
    return mirror


def _filter_repo_filtered(upstream: Path) -> Path:
    """A mirror ``git filter-repo`` rewrote, its source remote put back."""
    mirror = _mirror(upstream)
    _git(
        "-C",
        str(mirror),
        "filter-repo",
        "--force",
        "--path",
        "secret.txt",
        "--invert-paths",
    )
    _git("remote", "add", "--mirror=fetch", "upstream", upstream.as_uri(), cwd=mirror)
    return mirror


def _advance(upstream: Path) -> None:
    (upstream / "file.txt").write_text("two\n")
    _git("commit", "-q", "-am", "two", cwd=upstream)


def _secret_at_tip(mirror: Path) -> bool:
    return "secret.txt" in _git("ls-tree", "--name-only", "main", cwd=mirror)


def _worker(filters: ContentFilterSpec | None = None) -> RefreshWorker:
    return RefreshWorker(
        filter_gerrit_only=False, ssh_jitter_seconds=0, content_filters=filters
    )


def _spec(mirror: Path) -> ContentFilterSpec:
    spec = ContentFilterSpec.from_options("secret.txt", None, False, mirror.parent)
    assert spec is not None
    return spec


LEGACY = {"fallback": _fallback_filtered, "filter-repo": _filter_repo_filtered}


@pytest.mark.parametrize("legacy", LEGACY.values(), ids=LEGACY.keys())
class TestUpgrade:
    """An earlier release's filtering, met by this one."""

    @pytest.mark.parametrize("with_filters", [False, True], ids=["bare", "filtered"])
    def test_a_refresh_is_refused_and_fetches_nothing(
        self, upstream: Path, legacy: Callable[[Path], Path], with_filters: bool
    ) -> None:
        """Not even with the same filters: what they were is unknown."""
        mirror = legacy(upstream)
        before = _git("for-each-ref", cwd=mirror)
        _advance(upstream)
        filters = _spec(mirror) if with_filters else None

        result = _worker(filters).refresh_repository(mirror)

        assert result.status == RefreshStatus.SKIPPED
        assert result.error_message == EARLIER_RELEASE_REFUSAL
        assert _git("for-each-ref", cwd=mirror) == before
        assert not _secret_at_tip(mirror)

    def test_the_dry_run_predicts_the_refusal(
        self, upstream: Path, legacy: Callable[[Path], Path]
    ) -> None:
        mirror = legacy(upstream)

        batch = RefreshManager(
            dry_run=True, filter_gerrit_only=False
        ).refresh_repositories(mirror.parent, [mirror])

        [prediction] = batch.results
        assert prediction.status == RefreshStatus.SKIPPED
        assert prediction.error_message == EARLIER_RELEASE_REFUSAL

    def test_filtering_it_again_keeps_it_refused(
        self, upstream: Path, legacy: Callable[[Path], Path]
    ) -> None:
        """Recorded first, so this release's own record cannot clear it.

        The filter finds nothing left to remove, and withdraws what it
        added; the earlier release's filtering stays written down.
        """
        mirror = legacy(upstream)

        ok, error = apply_content_filters(
            mirror, "proj", remove_patterns=["secret.txt"]
        )
        assert ok, error
        _advance(upstream)
        result = _worker(_spec(mirror)).refresh_repository(mirror)

        assert recorded_policy(mirror).earlier_release
        assert (
            _git("config", "--get", "gerrit-clone.filteredByEarlierRelease", cwd=mirror)
            == "true"
        )
        assert result.error_message == EARLIER_RELEASE_REFUSAL
        assert not _secret_at_tip(mirror)


class TestThisReleaseIsNotMistaken:
    """Its own filtering leaves the same traces, and a stamp beside them."""

    def test_its_filter_repo_rewrite_stays_refreshable(self, upstream: Path) -> None:
        mirror = _mirror(upstream)
        ok, error = apply_content_filters(
            mirror, "proj", remove_patterns=["secret.txt"]
        )
        assert ok, error
        _advance(upstream)

        result = _worker(_spec(mirror)).refresh_repository(mirror)

        assert result.status == RefreshStatus.SUCCESS, result.error_message
        assert not _secret_at_tip(mirror)

    def test_its_worktree_fallback_stays_refreshable(self, upstream: Path) -> None:
        """Through a staging copy too, though it keeps fallback tips.

        Fetching only ``main``, the copy keeps the fallback's removal
        commit at the tip of ``dev``: no earlier release's trace, since
        the mirror's own record is known.
        """
        _git("branch", "dev", cwd=upstream)
        mirror = _mirror(upstream)
        _git("config", "--unset-all", "remote.origin.fetch", cwd=mirror)
        _git(
            "config",
            "--add",
            "remote.origin.fetch",
            "+refs/heads/main:refs/heads/main",
            cwd=mirror,
        )
        with patch(
            "gerrit_clone.content_filter._check_git_filter_repo", return_value=False
        ):
            ok, error = apply_content_filters(
                mirror, "proj", remove_patterns=["secret.txt"]
            )
            assert ok, error
            _advance(upstream)
            result = _worker(_spec(mirror)).refresh_repository(mirror)

        assert result.status == RefreshStatus.SUCCESS, result.error_message
        assert not recorded_policy(mirror).earlier_release
        assert not _secret_at_tip(mirror)

    def test_an_unfiltered_mirror_is_not_flagged(self, upstream: Path) -> None:
        assert recorded_policy(_mirror(upstream)).empty

    def test_an_analysis_alone_is_not_filtering(self, upstream: Path) -> None:
        """``filter-repo --analyze`` leaves its directory, and every ref."""
        mirror = _mirror(upstream)
        _git("-C", str(mirror), "filter-repo", "--analyze")
        _advance(upstream)

        result = _worker().refresh_repository(mirror)

        assert recorded_policy(mirror).empty
        assert result.status == RefreshStatus.SUCCESS, result.error_message


def test_unreadable_traces_fail_closed(upstream: Path) -> None:
    """Not a guess of "never filtered" when git cannot answer."""
    mirror = _mirror(upstream)

    with (
        patch.object(content_legacy, "git", return_value=None),
        pytest.raises(PolicyReadError),
    ):
        recorded_policy(mirror)
