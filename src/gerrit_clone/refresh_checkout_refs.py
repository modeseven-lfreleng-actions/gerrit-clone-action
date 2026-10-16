# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""The refs a working copy's staged copy holds, so it stands in for upstream.

A staged copy (see :mod:`gerrit_clone.refresh_checkout_stage`) starts as
a mirror clone of the checkout, every ref included.  It must hold what
upstream holds and nothing the operator made: the checkout's branches and
tags go, its remote-tracking refs become the copy's branches, and the
copy's remotes fetch into those branches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gerrit_clone.refresh_git_env import run_git

if TYPE_CHECKING:
    from pathlib import Path

#: Where a working copy keeps its remote-tracking refs.
TRACKING = "refs/remotes/"

#: The refs publishing a staged copy replaces in a working copy.
PUBLISHED = (TRACKING, "refs/tags/")

#: Tags deleted per ``git tag -d``, to keep each command line short.
_TAGS_PER_CALL = 200


def staged_refspec(refspec: str) -> str | None:
    """*refspec* for a working copy's staged copy; ``None`` if it has none.

    Remote-tracking destinations become branches, as the copy holds
    them; tags and negative refspecs stay as they are.  Anything else
    would land where publishing never looks -- one naming no destination
    only updates ``FETCH_HEAD`` -- and the copy would publish stale refs.
    """
    if refspec.startswith("^"):
        return refspec
    force = "+" if refspec.startswith("+") else ""
    source, _, destination = refspec.removeprefix("+").partition(":")
    if destination.startswith(TRACKING):
        return f"{force}{source}:refs/heads/{destination.removeprefix(TRACKING)}"
    if destination.startswith("refs/tags/"):
        return refspec
    return None


def drop_tags(stage: Path, timeout: int) -> str | None:
    """Delete every tag the copy took from the checkout.

    Tags the operator made would otherwise be filtered and published back
    over theirs.  The copy then fetches upstream's tags afresh.

    Returns:
        Why they could not all be deleted, or ``None``.
    """
    listed = run_git(
        ["git", "for-each-ref", "--format=%(refname:strip=2)", "refs/tags/"],
        stage,
        timeout=timeout,
    )
    if listed.returncode != 0:
        return f"Could not list the copy's tags: {listed.stderr.strip()}"
    tags = listed.stdout.split()
    for start in range(0, len(tags), _TAGS_PER_CALL):
        deleted = run_git(
            ["git", "tag", "-d", *tags[start : start + _TAGS_PER_CALL]],
            stage,
            timeout=timeout,
        )
        if deleted.returncode != 0:
            return f"Could not drop the copy's tags: {deleted.stderr.strip()}"
    return None


def as_upstream_copy(repo_path: Path, stage: Path, timeout: int) -> str | None:
    """Make *stage*, a copy of working copy *repo_path*, stand in for upstream.

    It holds the checkout's remote-tracking refs as its branches,
    ``refs/remotes/X`` as ``refs/heads/X``, and its remotes fetch there;
    the checkout's own branches and tags, and its remotes' ``HEAD`` refs,
    go.

    Returns:
        Why the copy could not be made to stand in for upstream, or
        ``None``.
    """
    moved = run_git(
        [
            "git",
            "fetch",
            "--quiet",
            "--prune",
            "--no-tags",
            str(repo_path),
            f"+{TRACKING}*:refs/heads/*",
            f"^{TRACKING}*/HEAD",
        ],
        stage,
        timeout=timeout,
    )
    failure = (
        f"Could not stage a copy: {moved.stderr.strip()}"
        if moved.returncode != 0
        else drop_tags(stage, timeout)
    )
    if failure is not None:
        return failure
    listed = run_git(
        ["git", "config", "--get-regexp", r"^remote\..*\.fetch$"], stage, timeout=10
    )
    refspecs: dict[str, list[str]] = {}
    for line in listed.stdout.splitlines() if listed.returncode == 0 else []:
        key, _, value = line.partition(" ")
        refspecs.setdefault(key, []).append(value)
    if not any(
        refspec.partition(":")[2].startswith(TRACKING)
        for values in refspecs.values()
        for refspec in values
    ):
        # Fetching would update FETCH_HEAD alone, and leave the copy's
        # branches where they were: a refresh that took nothing.
        return (
            "Could not stage a copy: no remote has a fetch refspec that "
            "updates a remote-tracking branch"
        )
    for key, values in refspecs.items():
        staged = [
            refspec for refspec in map(staged_refspec, values) if refspec is not None
        ]
        if len(staged) != len(values):
            return (
                f"Could not stage a copy: {key} fetches outside "
                f"{TRACKING} and refs/tags/"
            )
        run_git(["git", "config", "--unset-all", key], stage, timeout=10)
        for refspec in staged:
            added = run_git(["git", "config", "--add", key, refspec], stage, timeout=10)
            if added.returncode != 0:
                return f"Could not stage a copy: {added.stderr.strip()}"
    return None


def copy_tags(repo_path: Path, stage: Path, timeout: int) -> str | None:
    """Give *stage* every tag working copy *repo_path* holds, as they stand.

    For a checkout just cloned, whose tags are all upstream's: without
    them the copy's filters would leave the checkout's tags unfiltered.

    Returns:
        Why they could not be copied, or ``None``.
    """
    copied = run_git(
        [
            "git",
            "fetch",
            "--quiet",
            "--no-tags",
            str(repo_path),
            "+refs/tags/*:refs/tags/*",
        ],
        stage,
        timeout=timeout,
    )
    if copied.returncode != 0:
        return (
            f"Could not copy the working copy's tags to the staged copy: "
            f"{copied.stderr.strip()}"
        )
    return None


def configured_upstream(repo_path: Path) -> str:
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
