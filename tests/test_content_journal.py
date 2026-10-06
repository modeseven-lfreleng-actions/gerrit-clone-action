# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Each content-filtering rewrite leaves an entry in the tree's journal.

``<tree>/.gerrit-clone/filter-journal.jsonl`` gets one line before each
rewrite starts and another once it ends: which command and release ran
it, on which project, with which method and filters, and a digest of the
refs before and after.  The intent file stays the decision; the journal
is the audit trail of the runs that carried it out.  An entry a crash
left without its end is the exception: until a later run filters that
project again, its filters stay in force even if the intent lost them.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

from gerrit_clone.cli import app
from gerrit_clone.content_intent import IntentError, intent_path
from gerrit_clone.content_intent_resolve import resolve_filters
from gerrit_clone.content_spec import ContentFilterSpec
from gerrit_clone.content_stage import filter_repository

if TYPE_CHECKING:
    from pathlib import Path

_NO_JOURNAL = pytest.mark.xfail(strict=True, reason="#309: no journal yet")

#: Built at runtime so that no credential-shaped literal sits in the
#: source for secret scanners to flag.
TOKEN = "tok-" + "5e4d3c2b1a0f" * 2


def _git(repo: Path, *args: str) -> str:
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
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """``com/parent``, a mirror of an upstream holding ``secret.txt``."""
    upstream = tmp_path / "up"
    upstream.mkdir()
    _git(upstream, "init", "-q", "-b", "main")
    (upstream / "secret.txt").write_text(f"{TOKEN}\n")
    (upstream / "file.txt").write_text("one\n")
    _git(upstream, "add", ".")
    _git(upstream, "commit", "-q", "-m", "one")
    root = tmp_path / "tree"
    _git(
        tmp_path,
        "clone",
        "-q",
        "--mirror",
        upstream.as_uri(),
        str(root / "com" / "parent"),
    )
    return root


def _journal(root: Path) -> list[dict[str, Any]]:
    path = root / ".gerrit-clone" / "filter-journal.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _refresh(root: Path, *options: str) -> Any:
    return CliRunner().invoke(
        app, ["refresh", "--output-path", str(root), "--all-repos", *options]
    )


def _spec(root: Path) -> ContentFilterSpec:
    options = ContentFilterSpec.from_options("secret.txt", None, False, root)
    spec = resolve_filters(root, options, persist=True)
    assert spec is not None
    return spec


class TestEntries:
    @_NO_JOURNAL
    def test_a_rewrite_is_journalled_before_and_after(self, tree: Path) -> None:
        result = _refresh(tree, "--remove-files", "secret.txt")
        assert result.exit_code == 0, result.output

        start, end = _journal(tree)

        assert start["event"] == "start"
        assert start["command"] == "refresh"
        assert start["project"] == "com/parent"
        assert start["method"] in ("filter-repo", "worktree")
        assert start["policy"] == {"remove": ["secret.txt"]}
        assert start["version"]
        assert start["time"]
        assert end == {
            "event": "end",
            "id": start["id"],
            "time": end["time"],
            "ok": True,
            "refs_sha256": end["refs_sha256"],
        }
        assert start["refs_sha256"] != end["refs_sha256"], "the rewrite changed refs"

    @_NO_JOURNAL
    def test_each_run_appends(self, tree: Path) -> None:
        for _ in range(2):
            assert _refresh(tree, "--remove-files", "secret.txt").exit_code == 0

        assert [entry["event"] for entry in _journal(tree)] == [
            "start",
            "end",
            "start",
            "end",
        ]

    @_NO_JOURNAL
    def test_a_failed_rewrite_is_journalled_as_failed(self, tree: Path) -> None:
        spec = _spec(tree)
        repo = tree / "com" / "parent"

        reason = filter_repository(
            spec, repo, "com/parent", 60, apply=lambda *_a, **_k: (False, "broke")
        )

        assert reason is not None
        assert _journal(tree)[-1]["ok"] is False

    @_NO_JOURNAL
    def test_no_token_reaches_the_journal(self, tree: Path) -> None:
        result = _refresh(tree, "--git-filter", f"com/*:{TOKEN}")
        assert result.exit_code == 0, result.output

        text = (tree / ".gerrit-clone" / "filter-journal.jsonl").read_text()
        digest = hashlib.sha256(TOKEN.encode()).hexdigest()
        assert TOKEN not in text
        assert _journal(tree)[0]["policy"] == {"token_sha256": [digest]}

    def test_a_dry_run_journals_nothing(self, tree: Path) -> None:
        _refresh(tree, "--dry-run", "--remove-files", "secret.txt")

        assert not (tree / ".gerrit-clone" / "filter-journal.jsonl").exists()


def _interrupted(*_args: object, **_kwargs: object) -> tuple[bool, str | None]:
    raise KeyboardInterrupt


class TestInterruptedRewrites:
    """A rewrite a crash cut short leaves a start with no end."""

    def _crash(self, tree: Path) -> None:
        with pytest.raises(KeyboardInterrupt):
            filter_repository(
                _spec(tree),
                tree / "com" / "parent",
                "com/parent",
                60,
                apply=_interrupted,
            )

    @_NO_JOURNAL
    def test_it_leaves_a_start_without_an_end(self, tree: Path) -> None:
        self._crash(tree)

        assert [entry["event"] for entry in _journal(tree)] == ["start"]

    @_NO_JOURNAL
    def test_its_filters_stay_in_force_without_the_intent(self, tree: Path) -> None:
        """Nothing else recorded them: the crash came before the rewrite
        could record them in the repository."""
        self._crash(tree)
        intent_path(tree).unlink()

        spec = resolve_filters(tree, None, persist=False)

        assert spec is not None
        assert spec.filters_for("com/parent").remove_patterns == ["secret.txt"]
        assert spec.filters_for("com/other").remove_patterns is None

    def test_a_later_completed_rewrite_releases_it(self, tree: Path) -> None:
        self._crash(tree)
        spec = _spec(tree)
        assert (
            filter_repository(
                spec, tree / "com" / "parent", "com/parent", 60, apply=_unchanged
            )
            is None
        )
        intent_path(tree).unlink()

        assert resolve_filters(tree, None, persist=False) is None

    @_NO_JOURNAL
    def test_a_torn_last_line_is_ignored(self, tree: Path) -> None:
        """A crash while writing an entry: no rewrite followed it."""
        self._crash(tree)
        journal = tree / ".gerrit-clone" / "filter-journal.jsonl"
        torn = '{"event": "start", "id": "torn", "pro'
        with journal.open("a", encoding="utf-8") as stream:
            stream.write(torn)

        assert resolve_filters(tree, None, persist=False) is not None
        filter_repository(
            _spec(tree), tree / "com" / "parent", "com/parent", 60, apply=_unchanged
        )

        # The next entry replaced it, rather than following it on its line.
        assert torn not in journal.read_text(encoding="utf-8")
        assert [entry["event"] for entry in _journal(tree)] == [
            "start",
            "start",
            "end",
        ]

    @_NO_JOURNAL
    def test_a_garbled_entry_stops_the_run(self, tree: Path) -> None:
        self._crash(tree)
        journal = tree / ".gerrit-clone" / "filter-journal.jsonl"
        with journal.open("a", encoding="utf-8") as stream:
            stream.write("not json\n")

        with pytest.raises(IntentError, match="journal"):
            resolve_filters(tree, None, persist=False)


def _unchanged(*_args: object, **_kwargs: object) -> tuple[bool, str | None]:
    return True, None


class TestEveryCommand:
    @_NO_JOURNAL
    def test_filtering_in_place_is_journalled(self, tree: Path) -> None:
        """Clone and mirror filter the repository itself, not a copy."""
        repo = tree / "com" / "parent"

        assert filter_repository(_spec(tree), repo, "com/parent", 60) is None

        start, end = _journal(tree)
        assert start["path"] == "com/parent"
        assert end["ok"] is True
        assert start["refs_sha256"] != end["refs_sha256"]
