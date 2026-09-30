# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Keeping a content-filtered repository both refreshable and filtered.

``git filter-repo`` removes the ``origin`` remote whenever it rewrites
history, so that the rewritten history is not pushed back over the
original by mistake.  For a mirror that is also the only record of where
it came from: without ``remote.origin.fetch`` a later refresh has nothing
to fetch, and skips the repository from then on.

So what fetching needs -- the URL and the fetch refspecs -- is put back,
and filter-repo's protection is kept by pointing pushes nowhere.
``remote.origin.mirror`` is deliberately not restored: it only affects
pushing, and would make a bare ``git push`` force every rewritten ref
over the original.

Fetching is then the other way the original history can come back: a
mirror's ``+refs/*:refs/*`` refspec forces every upstream ref over the
rewritten one, removed files and redacted secrets included.  So the
repository is also marked as filtered, and a refresh refuses it unless
the same run filters it again (see :func:`is_content_filtered`).
"""

from __future__ import annotations

import subprocess
from contextlib import contextmanager
from typing import TYPE_CHECKING, NamedTuple

from gerrit_clone.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

logger = get_logger(__name__)

#: Push URL left on a restored ``origin``.  Its scheme has no remote
#: helper, so git refuses a push at once -- and, unlike a bare word, it
#: can never name a local path, such as a nested repository that
#: happens to share it.
NO_PUSH_URL = "no-push://content-filtered-history"

#: Git config key recording that content filtering has rewritten a
#: repository.  Set whenever a filter changed any ref -- an interrupted
#: run included, since a partly filtered repository is no safer to
#: refresh unfiltered than a fully filtered one -- and whenever the refs
#: could not be compared, failing closed.
FILTERED_MARKER = "gerrit-clone.contentFiltered"


class _Origin(NamedTuple):
    """What fetching from ``origin`` needs."""

    url: str
    fetch: list[str]


def _git(repo_path: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    """Run git in *repo_path*, or ``None`` if git could not run.

    Keeping a repository refreshable is a courtesy to later refreshes,
    and must never be the reason filtering fails.
    """
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug(f"Could not run git in {repo_path}: {exc}")
        return None


def _git_config(repo_path: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    """Run ``git config`` in *repo_path*, or ``None`` if git could not run."""
    return _git(repo_path, "config", *args)


def _content_refs(repo_path: Path) -> list[str] | None:
    """Every ref and its target, bar remote-tracking ones; ``None`` if unread.

    Remote-tracking refs are left out because removing ``origin`` deletes
    them whether or not any content was filtered.  A filter that rewrote
    nothing leaves every other ref where it was.
    """
    result = _git(repo_path, "for-each-ref", "--format=%(refname) %(objectname)")
    if result is None or result.returncode != 0:
        return None
    return [
        line
        for line in result.stdout.splitlines()
        if not line.startswith("refs/remotes/")
    ]


def _read_origin(repo_path: Path) -> _Origin | None:
    """The repository's ``origin`` fetch settings, if it has an origin."""
    url = _git_config(repo_path, "--get", "remote.origin.url")
    if url is None or url.returncode != 0 or not url.stdout.strip():
        return None
    fetch = _git_config(repo_path, "--get-all", "remote.origin.fetch")
    refspecs = fetch.stdout.splitlines() if fetch is not None else []
    return _Origin(url.stdout.strip(), refspecs)


def _restore_origin(repo_path: Path, origin: _Origin) -> None:
    """Put *origin* back for fetching, with pushing disabled.

    The push URL goes first, so that a restore failing part-way leaves
    an ``origin`` that cannot be pushed to rather than one that can.
    """
    settings = [("remote.origin.pushurl", NO_PUSH_URL)]
    settings.append(("remote.origin.url", origin.url))
    settings += [("remote.origin.fetch", refspec) for refspec in origin.fetch]
    for key, value in settings:
        result = _git_config(repo_path, "--add", key, value)
        if result is None or result.returncode != 0:
            detail = result.stderr.strip() if result is not None else "git failed"
            logger.warning(
                f"Could not restore {key} in {repo_path} after content "
                f"filtering: {detail}"
            )
            return
    logger.info(
        f"Restored origin in {repo_path} after content filtering rewrote "
        f"its history; pushing to it is disabled"
    )


@contextmanager
def origin_kept(repo_path: Path) -> Generator[None, None, None]:
    """Filter a repository so that it stays refreshable, and stays filtered.

    Restores ``origin`` for fetching if the enclosed filtering removed
    it, and marks the repository as content-filtered if the filtering
    changed any ref.  Both happen even if the filtering raises.

    Args:
        repo_path: Repository being filtered
    """
    before = _read_origin(repo_path)
    refs_before = _content_refs(repo_path)
    try:
        yield
    finally:
        if before is not None and _read_origin(repo_path) is None:
            _restore_origin(repo_path, before)
        refs_after = _content_refs(repo_path)
        if refs_before is None or refs_after is None or refs_before != refs_after:
            _mark_filtered(repo_path)


def _mark_filtered(repo_path: Path) -> None:
    """Record that content filtering has rewritten *repo_path*."""
    marked = _git_config(repo_path, "--replace-all", FILTERED_MARKER, "true")
    if marked is None or marked.returncode != 0:
        logger.warning(
            f"Could not mark {repo_path} as content-filtered; a refresh "
            f"without filters would not know to refuse it"
        )


def is_content_filtered(repo_path: Path) -> bool:
    """Whether content filtering has rewritten *repo_path*.

    Refreshing such a repository without filtering it again would bring
    the removed or redacted content back, so a refresh that does not
    re-apply filters refuses it.

    Args:
        repo_path: Repository to ask about

    Returns:
        True if the repository carries the content-filtered mark
    """
    result = _git_config(repo_path, "--type=bool", "--get", FILTERED_MARKER)
    return result is not None and result.stdout.strip() == "true"
