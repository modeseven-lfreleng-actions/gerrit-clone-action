# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Clone URLs, isolated git environments and remote configuration.

Owns how a project is addressed as a git URL and the environment the clone
subprocess runs in.  Many clone workers run concurrently against the same
machine, so each one is given a private ``HOME`` and git config.  Without that
isolation the workers contend on the user's global ``.gitconfig`` and produce
spurious "could not lock config file" failures.
"""

from __future__ import annotations

import atexit
import os
import random
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from gerrit_clone.logging import get_logger

if TYPE_CHECKING:
    from gerrit_clone.models import Config

logger = get_logger(__name__)

# One isolated git config directory per worker thread, reused by every
# clone that thread performs.  Its contents are identical for every
# clone, so creating one per clone *attempt* -- and deleting none --
# left thousands of ``git_config_*`` directories in the system temp
# directory over a large hierarchy, more again with retries.  Reuse
# bounds the count at the thread count, and the atexit hook below
# removes them when the process ends.
_thread_state = threading.local()
_isolated_config_dirs: set[Path] = set()
_isolated_config_lock = threading.Lock()


def build_ssh_url(project_name: str, config: Config) -> str:
    """Build SSH URL for project.

    Args:
        project_name: Project to clone
        config: Configuration for clone operations

    Returns:
        SSH clone URL
    """
    user_prefix = f"{config.ssh_user}@" if config.ssh_user else ""
    return f"ssh://{user_prefix}{config.host}:{config.port}/{project_name}"


def build_https_url(project_name: str, config: Config) -> str:
    """Build HTTPS URL for project.

    Args:
        project_name: Project to clone
        config: Configuration for clone operations

    Returns:
        HTTPS clone URL
    """
    return f"{config.base_url}/{project_name}"


def create_isolated_git_config(config_dir: Path) -> None:
    """Create minimal git configuration in isolated directory.

    Args:
        config_dir: Directory to create git config in
    """
    try:
        # Create a minimal .gitconfig to prevent git from searching elsewhere
        gitconfig_path = config_dir / ".gitconfig"
        gitconfig_content = """[core]
    repositoryformatversion = 0
    filemode = true
    bare = false
    logallrefupdates = true
[gc]
    auto = 0
[receive]
    denyCurrentBranch = ignore
"""
        gitconfig_path.write_text(gitconfig_content)

        # Create empty known_hosts to prevent SSH prompts
        ssh_dir = config_dir / ".ssh"
        ssh_dir.mkdir(exist_ok=True)
        (ssh_dir / "known_hosts").touch()

    except Exception as e:
        logger.debug(f"Could not create isolated git config: {e}")


def isolated_git_config_dir() -> Path:
    """Return this thread's isolated git config directory.

    Created on first use and reused afterwards.  A directory removed by
    :func:`cleanup_isolated_git_configs` while the thread is idle is
    recreated on the next call.

    Returns:
        Directory to use as ``HOME`` for this thread's git subprocesses
    """
    existing: Path | None = getattr(_thread_state, "git_config_dir", None)
    if existing is not None and existing.is_dir():
        return existing

    config_dir = Path(tempfile.mkdtemp(prefix=f"git_config_{threading.get_ident()}_"))
    create_isolated_git_config(config_dir)
    _thread_state.git_config_dir = config_dir
    with _isolated_config_lock:
        _isolated_config_dirs.add(config_dir)
    return config_dir


def cleanup_isolated_git_configs() -> None:
    """Remove every isolated git config directory this process created.

    Registered with :mod:`atexit`, and safe to call directly.  Any
    thread that goes on to clone again gets a fresh directory.
    """
    with _isolated_config_lock:
        config_dirs = list(_isolated_config_dirs)
        _isolated_config_dirs.clear()

    for config_dir in config_dirs:
        shutil.rmtree(config_dir, ignore_errors=True)


atexit.register(cleanup_isolated_git_configs)


def build_clone_environment(config: Config) -> dict[str, str]:
    """Build environment variables for git clone.

    Args:
        config: Configuration for clone operations

    Returns:
        Environment dictionary
    """
    env = os.environ.copy()

    # Set GIT_SSH_COMMAND for strict host checking
    if config.git_ssh_command:
        env["GIT_SSH_COMMAND"] = config.git_ssh_command

    git_config_dir = isolated_git_config_dir()

    # Essential git environment isolation to prevent config file contention
    # Keep only the minimal set that prevents conflicts without breaking git
    env["GIT_CONFIG_GLOBAL"] = str(git_config_dir / ".gitconfig")
    env["GIT_CONFIG_SYSTEM"] = os.devnull
    env["HOME"] = str(git_config_dir)  # Isolate home directory for git

    # Set aggressive timeouts to prevent hanging
    env["GIT_HTTP_LOW_SPEED_LIMIT"] = "1000"
    env["GIT_HTTP_LOW_SPEED_TIME"] = "30"

    # Disable git operations that could cause file locking
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env["GIT_AUTO_GC"] = "0"

    return env


def _is_config_lock_error(error_msg: str) -> bool:
    """Report whether a git failure looks like config-file lock contention.

    Args:
        error_msg: Error text from the failed git invocation

    Returns:
        True if the failure is worth retrying after a short delay
    """
    lowered = error_msg.lower()
    return (
        "could not lock config file" in lowered
        or "no such file or directory" in lowered
        or ("could not open" in lowered and ".git/config" in lowered)
    )


def set_ssh_remote(
    project_name: str, repo_path: Path, ssh_url: str, env: dict[str, str]
) -> None:
    """Set the remote URL to SSH after HTTPS clone with isolated environment.

    Args:
        project_name: Project that was cloned (for logging)
        repo_path: Path to the cloned repository
        ssh_url: SSH URL to point the origin remote at
        env: Isolated git environment to use
    """
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        try:
            subprocess.run(
                ["git", "remote", "set-url", "origin", ssh_url],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                env=env,  # Use isolated environment
            )
            logger.debug(
                f"Set SSH remote for [project]{project_name}[/project]: {ssh_url}"
            )
            return
        except subprocess.SubprocessError as e:
            if not _is_config_lock_error(str(e)):
                logger.warning(
                    f"Failed to set SSH remote for [project]{project_name}[/project]: {e}"
                )
                return
            if attempt >= max_attempts:
                logger.warning(
                    f"Failed to set SSH remote for [project]{project_name}[/project] after {max_attempts} attempts: {e}"
                )
                continue
            # Small delay with jitter for config lock retry
            delay = 0.2 + (random.uniform(0.1, 0.3) * attempt)
            logger.debug(
                f"Config lock detected for {project_name}, retrying in {delay:.2f}s (attempt {attempt + 1}/{max_attempts})"
            )
            time.sleep(delay)
            continue
        except Exception as e:
            logger.warning(
                f"Unexpected error setting SSH remote for [project]{project_name}[/project]: {e}"
            )
            return
