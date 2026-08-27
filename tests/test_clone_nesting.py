# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for nested-repository exclude entries.

Git exclude patterns are POSIX paths.  Rendering a ``Path`` with
``str()`` produces backslashes on Windows, which git reads as escapes
rather than separators, so the rule silently fails to match and the
membership test fails with it -- appending a duplicate entry every run.
"""

from __future__ import annotations

from pathlib import PureWindowsPath
from typing import TYPE_CHECKING

import pytest

from gerrit_clone.clone_nesting import (
    apply_late_nested_protection,
    apply_nested_protection,
)

if TYPE_CHECKING:
    from pathlib import Path

PROTECTORS = [apply_nested_protection, apply_late_nested_protection]


@pytest.fixture
def parent_repo(tmp_path: Path) -> Path:
    """A parent repository with a nested child two levels down."""
    repo = tmp_path / "parent"
    (repo / ".git" / "info").mkdir(parents=True)
    return repo


def _exclude_lines(parent_repo: Path) -> list[str]:
    return (
        (parent_repo / ".git" / "info" / "exclude")
        .read_text(encoding="utf-8")
        .splitlines()
    )


@pytest.mark.parametrize("protect", PROTECTORS)
class TestExcludeEntries:
    """Both the early and late paths write the same kind of entry."""

    def test_entry_is_written_in_posix_form(self, parent_repo: Path, protect) -> None:
        child = parent_repo / "sub" / "child"

        protect(parent_repo, child, "parent/sub/child", "parent")

        assert "sub/child" in _exclude_lines(parent_repo)

    def test_no_backslashes_reach_the_exclude_file(
        self, parent_repo: Path, protect
    ) -> None:
        child = parent_repo / "sub" / "child"

        protect(parent_repo, child, "parent/sub/child", "parent")

        assert "\\" not in (parent_repo / ".git" / "info" / "exclude").read_text(
            encoding="utf-8"
        )

    def test_an_existing_entry_is_not_duplicated(
        self, parent_repo: Path, protect
    ) -> None:
        """The membership test shared the defect, so duplicates accumulated."""
        child = parent_repo / "sub" / "child"

        protect(parent_repo, child, "parent/sub/child", "parent")
        protect(parent_repo, child, "parent/sub/child", "parent")

        assert _exclude_lines(parent_repo).count("sub/child") == 1

    def test_a_missing_parent_is_not_fatal(self, tmp_path: Path, protect) -> None:
        """The child is not under the ancestor, so relative_to raises."""
        protect(tmp_path / "parent", tmp_path / "elsewhere", "elsewhere", None)


class TestWindowsRendering:
    """Document the rendering difference the fix relies on."""

    def test_str_and_as_posix_differ_for_a_windows_path(self) -> None:
        nested = PureWindowsPath("sub", "child")

        assert str(nested) == "sub\\child"
        assert nested.as_posix() == "sub/child"
