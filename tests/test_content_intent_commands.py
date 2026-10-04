# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Later runs filter a tree as the runs before them decided.

Each test filters a tree once, the way an operator would, and then runs
a command again *without* the filter options: what was removed stays
removed, in repositories the filter left alone the first time, in ones
deleted and cloned again, and in trees an earlier release filtered.
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from typer.testing import CliRunner

from gerrit_clone.cli import app
from gerrit_clone.content_filter import apply_content_filters
from gerrit_clone.github_api import GitHubRepo
from gerrit_clone.mirror_manager import MirrorManager
from gerrit_clone.models import (
    CloneResult,
    CloneStatus,
    Config,
    Project,
    ProjectState,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

#: Built at runtime so that no credential-shaped literal sits in the
#: source for secret scanners to flag.
TOKEN = "tok-" + "1a2b3c4d5e6f" * 2


def _git(*args: str, cwd: Path | None = None) -> str:
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
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _commit(upstream: Path, name: str, content: str) -> None:
    (upstream / name).write_text(content)
    _git("add", name, cwd=upstream)
    _git("commit", "-q", "-m", f"add {name}", cwd=upstream)


def _upstream(path: Path) -> Path:
    path.mkdir(parents=True)
    _git("init", "-q", "-b", "main", str(path))
    _commit(path, "file.txt", "one\n")
    return path


def _history(repo: Path) -> list[str]:
    return _git("log", "--all", "--name-only", "--format=", cwd=repo).split()


def _contents(repo: Path) -> str:
    return _git("log", "--all", "-p", cwd=repo)


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """Mirrors ``com/parent`` and ``com/child`` of two upstreams."""
    base = tmp_path / "tree"
    for name in ("parent", "child"):
        upstream = _upstream(tmp_path / "up" / name)
        _git("clone", "-q", "--mirror", upstream.as_uri(), str(base / "com" / name))
    return base


def _refresh(tree: Path, *options: str) -> Any:
    return CliRunner().invoke(
        app, ["refresh", "--output-path", str(tree), "--all-repos", *options]
    )


#: Removed by the change that makes later runs honour the tree's filters.
UNHONOURED = pytest.mark.xfail(
    strict=True, reason="#307: later runs do not yet honour the tree's filters"
)


class TestRefresh:
    @UNHONOURED
    def test_content_arriving_later_is_filtered_without_the_options(
        self, tree: Path
    ) -> None:
        """The child had no secret.txt when the tree was filtered."""
        first = _refresh(tree, "--remove-files", "secret.txt")
        assert first.exit_code == 0, first.output
        _commit(tree.parent / "up" / "child", "secret.txt", "hunter2\n")

        again = _refresh(tree)

        assert again.exit_code == 0, again.output
        assert "secret.txt" not in _history(tree / "com" / "child")

    @UNHONOURED
    def test_a_tree_filtered_by_an_earlier_release_is_migrated(
        self, tree: Path
    ) -> None:
        """Per-repository records alone, as the previous release left them."""
        parent = tree.parent / "up" / "parent"
        _commit(parent, "secret.txt", "hunter2\n")
        assert _refresh(tree).exit_code == 0
        ok, error = apply_content_filters(
            tree / "com" / "parent", "com/parent", remove_patterns=["secret.txt"]
        )
        assert ok, error
        _commit(parent, "file.txt", "two\n")

        again = _refresh(tree)

        assert again.exit_code == 0, again.output
        mirror = tree / "com" / "parent"
        assert _git("show", "main:file.txt", cwd=mirror) == "two"
        assert "secret.txt" not in _history(mirror)

    @UNHONOURED
    def test_a_token_the_run_does_not_supply_refuses_the_project(
        self, tree: Path
    ) -> None:
        """Kept only as a digest, it cannot be replaced in what arrives."""
        first = _refresh(tree, "--git-filter", f"com/child:{TOKEN}")
        assert first.exit_code == 0, first.output
        _commit(tree.parent / "up" / "child", "settings.py", f'KEY = "{TOKEN}"\n')

        again = _refresh(tree)

        assert "--git-filter token" in again.output
        assert TOKEN not in _contents(tree / "com" / "child")

    def test_the_token_given_again_lets_it_through_replaced(self, tree: Path) -> None:
        first = _refresh(tree, "--git-filter", f"com/child:{TOKEN}")
        assert first.exit_code == 0, first.output
        _commit(tree.parent / "up" / "child", "settings.py", f'KEY = "{TOKEN}"\n')

        again = _refresh(tree, "--git-filter", f"com/child:{TOKEN}")

        assert again.exit_code == 0, again.output
        assert "settings.py" in _history(tree / "com" / "child")
        assert TOKEN not in _contents(tree / "com" / "child")

    @UNHONOURED
    def test_an_unreadable_intent_refreshes_nothing(self, tree: Path) -> None:
        (tree / ".gerrit-clone").mkdir()
        (tree / ".gerrit-clone" / "filter-policy.json").write_text("{not json")
        before = _git("rev-parse", "main", cwd=tree / "com" / "child")
        _commit(tree.parent / "up" / "child", "new.txt", "new\n")

        result = _refresh(tree)

        assert result.exit_code == 2, result.output
        assert _git("rev-parse", "main", cwd=tree / "com" / "child") == before


def _github_api(names: list[str]) -> Mock:
    api = Mock()
    api.list_all_repos_graphql = Mock(return_value={})
    api.list_repos = Mock(return_value=[])
    api.batch_delete_repos = AsyncMock(return_value={})
    repos = {
        name: (
            GitHubRepo(
                name=name,
                full_name=f"org/{name}",
                ssh_url=f"git@github.com:org/{name}.git",
                clone_url=f"https://github.com/org/{name}.git",
                html_url=f"https://github.com/org/{name}",
                private=False,
            ),
            None,
        )
        for name in names
    }
    api.batch_create_repos = AsyncMock(return_value=repos)
    return api


def _cloning_from(
    upstream: Path,
) -> Callable[[Any, list[Project]], list[CloneResult]]:
    """``clone_projects``, cloning each project from *upstream* for real."""

    def clone(self: Any, projects: list[Project]) -> list[CloneResult]:
        results = []
        for project in projects:
            path = self.config.path / project.name
            status = CloneStatus.ALREADY_EXISTS
            if not path.exists():
                _git("clone", "-q", "--mirror", upstream.as_uri(), str(path))
                status = CloneStatus.SUCCESS
            results.append(CloneResult(project=project, status=status, path=path))
        return results

    return clone


class TestMirrorOverwrite:
    @UNHONOURED
    def test_a_repository_cloned_again_is_filtered_before_it_is_pushed(
        self, tmp_path: Path
    ) -> None:
        """``--overwrite`` discards the clone, not what it was filtered with.

        Its push is ``git push --mirror``: unfiltered, it would force the
        removed file back onto the GitHub repository.
        """
        upstream = _upstream(tmp_path / "up")
        _commit(upstream, "secret.txt", "hunter2\n")
        config = Config(host="gerrit.example.org", port=29418, path=tmp_path / "tree")
        projects = [Project("proj", ProjectState.ACTIVE)]
        pushed: list[list[str]] = []

        def push(_self: Any, local: Path, _repo: GitHubRepo) -> tuple[bool, None]:
            pushed.append(_history(local))
            return True, None

        with (
            patch(
                "gerrit_clone.clone_orchestrator.CloneManager.clone_projects",
                _cloning_from(upstream),
            ),
            patch.object(MirrorManager, "_push_to_github", push),
        ):
            MirrorManager(
                config=config,
                github_api=_github_api(["proj"]),
                github_org="org",
                remove_file_patterns=["secret.txt"],
            ).mirror_projects(projects)
            MirrorManager(
                config=config,
                github_api=_github_api(["proj"]),
                github_org="org",
                overwrite=True,
            ).mirror_projects(projects)

        assert len(pushed) == 2
        assert all("secret.txt" not in history for history in pushed)
