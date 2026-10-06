# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""A working copy's fetch refspecs, as its staged copy fetches them."""

from __future__ import annotations

import pytest

from gerrit_clone.refresh_checkout_stage import _staged_refspec


@pytest.mark.parametrize(
    ("refspec", "staged"),
    [
        ("+refs/heads/*:refs/remotes/origin/*", "+refs/heads/*:refs/heads/origin/*"),
        (
            "refs/heads/main:refs/remotes/origin/main",
            "refs/heads/main:refs/heads/origin/main",
        ),
        ("+refs/tags/*:refs/tags/*", "+refs/tags/*:refs/tags/*"),
        ("^refs/heads/wip/*", "^refs/heads/wip/*"),
        ("refs/heads/main", "refs/heads/main"),
        ("refs/heads/main:", "refs/heads/main:"),
    ],
)
def test_remote_tracking_refs_become_branches(refspec: str, staged: str) -> None:
    assert _staged_refspec(refspec) == staged


@pytest.mark.parametrize(
    "refspec",
    ["+refs/*:refs/other/*", "+refs/heads/*:refs/heads/*", "refs/notes/*:refs/notes/*"],
)
def test_anything_else_cannot_be_staged(refspec: str) -> None:
    assert _staged_refspec(refspec) is None
