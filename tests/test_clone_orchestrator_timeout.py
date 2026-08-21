# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for the clone orchestrator's overall-timeout handling.

When the batch exceeds its overall timeout the orchestrator returns
partial results rather than raising, so every submitted project must be
represented in the returned list -- including the ones still queued.
"""

from __future__ import annotations

from concurrent.futures import Future
from typing import TYPE_CHECKING

import pytest

from gerrit_clone.clone_orchestrator import CloneManager
from gerrit_clone.models import (
    CloneResult,
    CloneStatus,
    Config,
    Project,
    ProjectState,
)

if TYPE_CHECKING:
    from pathlib import Path

OVERALL_TIMEOUT = 42


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """Minimal configuration pointing at a scratch directory."""
    return Config(host="gerrit.example.org", path=tmp_path / "repos")


def _project(name: str) -> Project:
    return Project(name=name, state=ProjectState.ACTIVE)


def _finished_future(result: CloneResult) -> Future[CloneResult]:
    future: Future[CloneResult] = Future()
    future.set_result(result)
    return future


def _running_future() -> Future[CloneResult]:
    """A future the executor has already started, so cancel() fails."""
    future: Future[CloneResult] = Future()
    assert future.set_running_or_notify_cancel() is True
    return future


class TestHandleCloneTimeout:
    """Every unfinished project must appear in the partial results."""

    def test_queued_projects_still_get_a_timeout_result(self, config: Config) -> None:
        """A cancelled future reports done(), so it must be snapshotted.

        This is the regression: cancelling a queued future flips it to
        done(), so a second pass filtering on ``not future.done()``
        silently dropped exactly the projects that never started.
        """
        manager = CloneManager(config)
        queued = _project("queued-repo")
        future_to_project: dict[Future[CloneResult], Project] = {
            Future(): queued,
        }
        results: list[CloneResult] = []

        manager._handle_clone_timeout(future_to_project, results, OVERALL_TIMEOUT)

        assert len(results) == 1
        assert results[0].project.name == "queued-repo"
        assert results[0].status == CloneStatus.FAILED
        assert f"timed out after {OVERALL_TIMEOUT}s" in (results[0].error_message or "")

    def test_running_projects_get_a_timeout_result(self, config: Config) -> None:
        """A running future cannot be cancelled but still timed out."""
        manager = CloneManager(config)
        future_to_project: dict[Future[CloneResult], Project] = {
            _running_future(): _project("running-repo"),
        }
        results: list[CloneResult] = []

        manager._handle_clone_timeout(future_to_project, results, OVERALL_TIMEOUT)

        assert [r.project.name for r in results] == ["running-repo"]
        assert results[0].status == CloneStatus.FAILED

    def test_completed_projects_are_not_duplicated(self, config: Config) -> None:
        """Results already recorded must not gain a second entry."""
        manager = CloneManager(config)
        done_project = _project("done-repo")
        recorded = CloneResult(
            project=done_project,
            status=CloneStatus.SUCCESS,
            path=config.path / done_project.name,
        )
        future_to_project: dict[Future[CloneResult], Project] = {
            _finished_future(recorded): done_project,
            Future(): _project("queued-repo"),
        }
        results: list[CloneResult] = [recorded]

        manager._handle_clone_timeout(future_to_project, results, OVERALL_TIMEOUT)

        by_name = [r.project.name for r in results]
        assert by_name == ["done-repo", "queued-repo"]
        assert by_name.count("done-repo") == 1

    def test_future_finishing_after_the_timeout_is_still_reported(
        self, config: Config
    ) -> None:
        """A future can complete after as_completed() gave up on it.

        Such a future was never yielded, so nothing recorded its result,
        yet it reports done(). Outstanding work is therefore derived from
        the recorded results rather than from future state.
        """
        manager = CloneManager(config)
        raced_project = _project("raced-repo")
        raced = CloneResult(
            project=raced_project,
            status=CloneStatus.SUCCESS,
            path=config.path / raced_project.name,
        )
        # Finished, but its result never reached results.
        future_to_project: dict[Future[CloneResult], Project] = {
            _finished_future(raced): raced_project,
        }
        results: list[CloneResult] = []

        manager._handle_clone_timeout(future_to_project, results, OVERALL_TIMEOUT)

        assert [r.project.name for r in results] == ["raced-repo"]

    def test_every_submitted_project_is_accounted_for(self, config: Config) -> None:
        """The partial result set covers the full submission, mixed states."""
        manager = CloneManager(config)
        done_project = _project("done-repo")
        recorded = CloneResult(
            project=done_project,
            status=CloneStatus.SUCCESS,
            path=config.path / done_project.name,
        )
        future_to_project: dict[Future[CloneResult], Project] = {
            _finished_future(recorded): done_project,
            _running_future(): _project("running-repo"),
            Future(): _project("queued-one"),
            Future(): _project("queued-two"),
        }
        results: list[CloneResult] = [recorded]

        manager._handle_clone_timeout(future_to_project, results, OVERALL_TIMEOUT)

        assert len(results) == len(future_to_project)
        assert {r.project.name for r in results} == {
            "done-repo",
            "running-repo",
            "queued-one",
            "queued-two",
        }
