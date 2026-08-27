# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Tests for the isolated git config directories used by clone workers.

Each worker thread gets a private ``HOME`` and git config so concurrent
clones do not contend on the user's global ``.gitconfig``.  One
directory per clone *attempt* used to be created and never removed, so a
run over a large hierarchy left thousands behind in the system temp
directory.
"""

from __future__ import annotations

import subprocess
import threading
from unittest.mock import MagicMock, patch

import pytest

from gerrit_clone.clone_git_env import (
    _is_config_lock_error,
    _isolated_config_dirs,
    _subprocess_error_text,
    build_clone_environment,
    cleanup_isolated_git_configs,
    isolated_git_config_dir,
    set_ssh_remote,
)
from gerrit_clone.models import Config


@pytest.fixture(autouse=True)
def _clean_slate():
    """Leave no directories behind, whichever way the test ends."""
    cleanup_isolated_git_configs()
    yield
    cleanup_isolated_git_configs()


@pytest.fixture
def config(tmp_path):
    return Config(host="gerrit.example.org", path=tmp_path / "repos")


class TestDirectoryReuse:
    """The directory contents are static, so one per thread is enough."""

    def test_repeated_calls_on_one_thread_reuse_one_directory(
        self, config: Config
    ) -> None:
        """This is the leak: a clone retried N times created N directories."""
        homes = {build_clone_environment(config)["HOME"] for _ in range(25)}

        assert len(homes) == 1
        assert len(_isolated_config_dirs) == 1

    def test_each_thread_gets_its_own_directory(self, config: Config) -> None:
        """Isolation between concurrent workers must survive the reuse."""
        homes: list[str] = []
        lock = threading.Lock()

        def record() -> None:
            home = build_clone_environment(config)["HOME"]
            with lock:
                homes.append(home)

        threads = [threading.Thread(target=record) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(set(homes)) == 4

    def test_the_directory_is_populated(self, config: Config) -> None:
        config_dir = isolated_git_config_dir()

        assert (config_dir / ".gitconfig").is_file()
        assert (config_dir / ".ssh" / "known_hosts").is_file()

    def test_the_environment_points_at_the_directory(self, config: Config) -> None:
        env = build_clone_environment(config)

        assert env["GIT_CONFIG_GLOBAL"].startswith(env["HOME"])
        assert env["HOME"] in {str(path) for path in _isolated_config_dirs}


class TestCleanup:
    """Nothing may survive a completed run."""

    def test_cleanup_removes_every_directory(self, config: Config) -> None:
        build_clone_environment(config)
        created = list(_isolated_config_dirs)
        assert created

        cleanup_isolated_git_configs()

        assert not _isolated_config_dirs
        assert not any(path.exists() for path in created)

    def test_a_thread_recreates_its_directory_after_cleanup(
        self, config: Config
    ) -> None:
        """A cached path must not be handed out once it has been removed."""
        first = isolated_git_config_dir()
        cleanup_isolated_git_configs()

        second = isolated_git_config_dir()

        assert second != first
        assert second.is_dir()
        assert (second / ".gitconfig").is_file()

    def test_cleanup_is_idempotent(self, config: Config) -> None:
        build_clone_environment(config)

        cleanup_isolated_git_configs()
        cleanup_isolated_git_configs()

        assert not _isolated_config_dirs


SSH_URL = "ssh://gerrit.example.org:29418/example/repo"
LOCK_MESSAGE = "error: could not lock config file .git/config: File exists"


def _lock_error() -> subprocess.CalledProcessError:
    """The exception ``check=True`` raises for a lost config-file race."""
    return subprocess.CalledProcessError(
        returncode=1,
        cmd=["git", "remote", "set-url", "origin", SSH_URL],
        stderr=LOCK_MESSAGE,
    )


class TestConfigLockClassification:
    """The reason for a git failure lives on stderr, not in str(exc)."""

    def test_str_alone_never_names_the_lock(self) -> None:
        """This is why the retry loop could never fire."""
        assert not _is_config_lock_error(str(_lock_error()))

    def test_captured_stderr_is_classified(self) -> None:
        assert _is_config_lock_error(_subprocess_error_text(_lock_error()))

    def test_error_text_gathers_both_streams(self) -> None:
        error = subprocess.CalledProcessError(
            returncode=1, cmd=["git"], output="on stdout", stderr="on stderr"
        )

        text = _subprocess_error_text(error)

        assert "on stdout" in text
        assert "on stderr" in text

    def test_absent_streams_are_skipped(self) -> None:
        error = subprocess.CalledProcessError(returncode=1, cmd=["git"])

        assert _subprocess_error_text(error) == str(error)


class TestSetSshRemoteRetry:
    """Losing the race for .git/config must not leave origin on HTTPS."""

    @patch("gerrit_clone.clone_git_env.time.sleep")
    @patch("gerrit_clone.clone_git_env.subprocess.run")
    def test_lock_contention_is_retried_until_it_succeeds(
        self, mock_run: MagicMock, mock_sleep: MagicMock, tmp_path
    ) -> None:
        mock_run.side_effect = [_lock_error(), MagicMock(returncode=0)]

        set_ssh_remote("example/repo", tmp_path, SSH_URL, {})

        assert mock_run.call_count == 2
        assert mock_sleep.call_count == 1
        assert mock_run.call_args_list[1][0][0] == [
            "git",
            "remote",
            "set-url",
            "origin",
            SSH_URL,
        ]

    @patch("gerrit_clone.clone_git_env.time.sleep")
    @patch("gerrit_clone.clone_git_env.subprocess.run")
    def test_retries_are_bounded(
        self, mock_run: MagicMock, mock_sleep: MagicMock, tmp_path
    ) -> None:
        mock_run.side_effect = _lock_error()

        set_ssh_remote("example/repo", tmp_path, SSH_URL, {})

        assert mock_run.call_count == 3
        # No delay after the final attempt.
        assert mock_sleep.call_count == 2

    @patch("gerrit_clone.clone_git_env.time.sleep")
    @patch("gerrit_clone.clone_git_env.subprocess.run")
    def test_other_failures_are_not_retried(
        self, mock_run: MagicMock, mock_sleep: MagicMock, tmp_path
    ) -> None:
        """Only lock contention is transient; a bad URL will not improve."""
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=128,
            cmd=["git"],
            stderr="fatal: No such remote 'origin'",
        )

        set_ssh_remote("example/repo", tmp_path, SSH_URL, {})

        assert mock_run.call_count == 1
        mock_sleep.assert_not_called()

    @patch("gerrit_clone.clone_git_env.subprocess.run")
    def test_success_runs_once(self, mock_run: MagicMock, tmp_path) -> None:
        mock_run.return_value = MagicMock(returncode=0)

        set_ssh_remote("example/repo", tmp_path, SSH_URL, {})

        assert mock_run.call_count == 1
