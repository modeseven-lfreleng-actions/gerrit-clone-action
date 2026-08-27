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

import threading

import pytest

from gerrit_clone.clone_git_env import (
    _isolated_config_dirs,
    build_clone_environment,
    cleanup_isolated_git_configs,
    isolated_git_config_dir,
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
