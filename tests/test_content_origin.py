# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for keeping ``origin`` through a content-filter rewrite."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from gerrit_clone import content_origin
from gerrit_clone.content_origin import NO_PUSH_URL, is_content_filtered, origin_kept

if TYPE_CHECKING:
    from pathlib import Path

MIRROR_REFSPEC = "+refs/*:refs/*"


def _config(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "config", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


@pytest.fixture
def mirror(tmp_path: Path) -> Path:
    """A mirror of a one-commit upstream, as ``git clone --mirror`` leaves it."""
    upstream = tmp_path / "upstream"
    git = ["git", "-c", "user.email=t@example.com", "-c", "user.name=T"]
    subprocess.run([*git, "init", "-q", "-b", "main", str(upstream)], check=True)
    (upstream / "file.txt").write_text("one\n")
    subprocess.run([*git, "add", "file.txt"], cwd=upstream, check=True)
    subprocess.run(
        [*git, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "one"],
        cwd=upstream,
        check=True,
    )
    repo = tmp_path / "mirror"
    subprocess.run(
        ["git", "clone", "-q", "--mirror", upstream.as_uri(), str(repo)], check=True
    )
    return repo


def _drop_origin(repo: Path) -> None:
    """What ``git filter-repo`` does to ``origin`` when it rewrites."""
    subprocess.run(["git", "remote", "remove", "origin"], cwd=repo, check=True)


class TestOriginKept:
    def test_a_removed_origin_is_restored_for_fetching(self, mirror: Path) -> None:
        url = _config(mirror, "remote.origin.url")

        with origin_kept(mirror):
            _drop_origin(mirror)

        assert _config(mirror, "remote.origin.url") == url
        assert _config(mirror, "--get-all", "remote.origin.fetch") == MIRROR_REFSPEC
        fetch = subprocess.run(
            ["git", "fetch", "origin"], cwd=mirror, capture_output=True, check=False
        )
        assert fetch.returncode == 0, fetch.stderr

    def test_pushing_to_a_restored_origin_fails(self, mirror: Path) -> None:
        """The reason filter-repo removes it: rewritten history must not go back.

        Tried for real, beside a nested repository called ``no_push``: a
        bare word is a local path to git, and would have taken the push.
        """
        subprocess.run(
            ["git", "init", "-q", "--bare", str(mirror / "no_push")], check=True
        )

        with origin_kept(mirror):
            _drop_origin(mirror)

        assert _config(mirror, "remote.origin.pushurl") == NO_PUSH_URL
        assert _config(mirror, "remote.origin.mirror") == ""
        push = subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=mirror,
            capture_output=True,
            check=False,
        )
        assert push.returncode != 0

    def test_a_partial_restore_still_blocks_pushing(self, mirror: Path) -> None:
        """The push URL goes in first, so a failure after it is fail-safe."""
        real = content_origin._git_config

        def failing_refspecs(
            repo: Path, *args: str
        ) -> subprocess.CompletedProcess[str] | None:
            if "--add" in args and "remote.origin.fetch" in args:
                return subprocess.CompletedProcess(args, 1, "", "disk full")
            return real(repo, *args)

        with (
            patch.object(content_origin, "_git_config", failing_refspecs),
            origin_kept(mirror),
        ):
            _drop_origin(mirror)

        assert _config(mirror, "remote.origin.pushurl") == NO_PUSH_URL

    def test_every_fetch_refspec_comes_back(self, mirror: Path) -> None:
        _config(mirror, "--add", "remote.origin.fetch", "+refs/notes/*:refs/notes/*")

        with origin_kept(mirror):
            _drop_origin(mirror)

        assert _config(mirror, "--get-all", "remote.origin.fetch").splitlines() == [
            MIRROR_REFSPEC,
            "+refs/notes/*:refs/notes/*",
        ]

    def test_an_origin_left_in_place_is_not_touched(self, mirror: Path) -> None:
        """No rewrite, or none that removed it: nothing to put back."""
        with origin_kept(mirror):
            pass

        assert _config(mirror, "remote.origin.pushurl") == ""
        assert _config(mirror, "remote.origin.mirror") == "true"
        assert not is_content_filtered(mirror)

    def test_no_origin_is_invented(self, tmp_path: Path) -> None:
        repo = tmp_path / "bare"
        subprocess.run(["git", "init", "-q", "--bare", str(repo)], check=True)

        with origin_kept(repo):
            pass

        assert _config(repo, "--get-regexp", "^remote\\.") == ""

    def test_it_is_restored_even_if_the_filtering_raises(self, mirror: Path) -> None:
        raised = False
        try:
            with origin_kept(mirror):
                _drop_origin(mirror)
                raise RuntimeError("filter-repo failed part-way")
        except RuntimeError:
            raised = True

        assert raised
        assert _config(mirror, "--get-all", "remote.origin.fetch") == MIRROR_REFSPEC


class TestFilteredMarker:
    """Refreshing a rewritten repository unfiltered brings the content back."""

    def test_a_rewrite_marks_the_repository(self, mirror: Path) -> None:
        with origin_kept(mirror):
            _git_in(mirror, "update-ref", "refs/heads/main", _empty_commit(mirror))

        assert is_content_filtered(mirror)

    def test_a_filter_that_changed_nothing_leaves_no_mark(self, mirror: Path) -> None:
        """``--remove-files`` runs everywhere, matching or not."""
        with origin_kept(mirror):
            _drop_origin(mirror)

        assert not is_content_filtered(mirror)

    def test_losing_remote_tracking_refs_alone_is_no_rewrite(
        self, mirror: Path
    ) -> None:
        """Removing ``origin`` deletes them whether or not content changed."""
        main = _git_in(mirror, "rev-parse", "main")
        _git_in(mirror, "update-ref", "refs/remotes/origin/main", main)

        with origin_kept(mirror):
            _git_in(mirror, "update-ref", "-d", "refs/remotes/origin/main")

        assert not is_content_filtered(mirror)

    def test_a_rewrite_that_raises_part_way_still_marks(self, mirror: Path) -> None:
        raised = False
        try:
            with origin_kept(mirror):
                _git_in(mirror, "update-ref", "refs/heads/main", _empty_commit(mirror))
                raise RuntimeError("filter-repo failed part-way")
        except RuntimeError:
            raised = True

        assert raised
        assert is_content_filtered(mirror)


def _git_in(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def _empty_commit(repo: Path) -> str:
    """A new commit object, standing in for one a filter rewrote."""
    tree = _git_in(repo, "rev-parse", "main^{tree}")
    return _git_in(
        repo,
        "-c",
        "user.email=t@example.com",
        "-c",
        "user.name=T",
        "commit-tree",
        tree,
        "-m",
        "rewritten",
    )
