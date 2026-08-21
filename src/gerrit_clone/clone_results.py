# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Factories for the :class:`~gerrit_clone.models.CloneResult` records.

Clone, refresh and retry stages all report their outcome as a
``CloneResult``.  Centralising construction here keeps the status and
timing conventions consistent across those stages.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from gerrit_clone.models import CloneResult, CloneStatus, RefreshStatus

if TYPE_CHECKING:
    from pathlib import Path

    from gerrit_clone.models import Config, Project, RefreshResult


def build_failure_result(
    config: Config, project: Project, error_message: str
) -> CloneResult:
    """Build a FAILED result for a project that produced no result of its own.

    Used when a worker raised before returning, or when the overall
    operation timed out and the future was cancelled.
    """
    now = datetime.now(UTC)
    return CloneResult(
        project=project,
        status=CloneStatus.FAILED,
        path=config.path / project.name,
        attempts=0,
        error_message=error_message,
        started_at=now,
        completed_at=now,
        first_started_at=now,
    )


def build_verified_result(
    config: Config, project: Project, started_at: datetime
) -> CloneResult:
    """Build a VERIFIED result for a repository already at the remote SHA."""
    return CloneResult(
        project=project,
        status=CloneStatus.VERIFIED,
        path=config.path / project.filesystem_path,
        started_at=started_at,
        completed_at=datetime.now(UTC),
        duration_seconds=0.0,
        was_refreshed=False,
        refresh_had_updates=False,
        refresh_commits_pulled=0,
    )


def build_already_exists_result(
    config: Config, project: Project, started_at: datetime
) -> CloneResult:
    """Build an ALREADY_EXISTS result for a repository left untouched."""
    return CloneResult(
        project=project,
        status=CloneStatus.ALREADY_EXISTS,
        path=config.path / project.filesystem_path,
        started_at=started_at,
        completed_at=datetime.now(UTC),
        duration_seconds=0.0,
    )


def build_refresh_result(
    project: Project,
    target_path: Path,
    refresh_result: RefreshResult,
    refresh_start: datetime,
) -> CloneResult:
    """Convert a :class:`RefreshResult` into a ``CloneResult``."""
    # Check if refresh failed first
    if refresh_result.status == RefreshStatus.FAILED:
        clone_status = CloneStatus.FAILED
        error_message = f"Refresh failed: {refresh_result.error_message}"
    # Use VERIFIED if up-to-date, REFRESHED if changes were pulled
    elif refresh_result.was_behind:
        clone_status = CloneStatus.REFRESHED
        error_message = None
    else:
        clone_status = CloneStatus.VERIFIED
        error_message = None

    return CloneResult(
        project=project,
        status=clone_status,
        path=target_path,
        started_at=refresh_start,
        completed_at=datetime.now(UTC),
        duration_seconds=refresh_result.duration_seconds,
        was_refreshed=refresh_result.was_behind,
        refresh_had_updates=refresh_result.was_behind,
        refresh_commits_pulled=refresh_result.commits_pulled,
        error_message=error_message,
    )


def build_refresh_failure_result(
    project: Project,
    target_path: Path,
    refresh_start: datetime,
    error: Exception,
) -> CloneResult:
    """Build a FAILED result for a repository whose refresh raised."""
    return CloneResult(
        project=project,
        status=CloneStatus.FAILED,
        path=target_path,
        started_at=refresh_start,
        completed_at=datetime.now(UTC),
        duration_seconds=(datetime.now(UTC) - refresh_start).total_seconds(),
        error_message=f"Refresh failed for {project.name}: {error}",
    )
