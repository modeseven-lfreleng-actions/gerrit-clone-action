# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Matthew Watkins <mwatkins@linuxfoundation.org>

"""Git utility functions for repository operations."""

from __future__ import annotations

import subprocess
from pathlib import Path


def get_current_commit_sha(repo_path: Path) -> str | None:
    """Get the current commit SHA (HEAD) for a local repository.

    Args:
        repo_path: Path to the git repository

    Returns:
        The commit SHA as a string, or None if it cannot be determined

    Raises:
        FileNotFoundError: If the repository path doesn't exist
        ValueError: If the path is not a git repository
    """
    if not repo_path.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

    git_dir = repo_path / ".git"
    if not git_dir.exists():
        raise ValueError(f"Not a git repository: {repo_path}")

    try:
        # Get the current HEAD commit SHA
        result = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        sha = result.stdout.strip()
        return sha if sha else None

    except subprocess.CalledProcessError:
        # Could be detached HEAD, new repo with no commits, etc.
        return None
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


def get_current_branch(repo_path: Path) -> str | None:
    """Get the current branch name for a local repository.

    Args:
        repo_path: Path to the git repository

    Returns:
        The branch name as a string, or None if detached HEAD or error

    Raises:
        FileNotFoundError: If the repository path doesn't exist
        ValueError: If the path is not a git repository
    """
    if not repo_path.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

    git_dir = repo_path / ".git"
    if not git_dir.exists():
        raise ValueError(f"Not a git repository: {repo_path}")

    try:
        # Get the current branch name
        result = subprocess.run(
            ["git", "-C", str(repo_path), "symbolic-ref", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        branch = result.stdout.strip()
        return branch if branch else None

    except subprocess.CalledProcessError:
        # Detached HEAD state
        return None
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


def is_repo_dirty(repo_path: Path) -> bool:
    """Check if a repository has uncommitted changes.

    Args:
        repo_path: Path to the git repository

    Returns:
        True if there are uncommitted changes, False otherwise

    Raises:
        FileNotFoundError: If the repository path doesn't exist
        ValueError: If the path is not a git repository
    """
    if not repo_path.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

    git_dir = repo_path / ".git"
    if not git_dir.exists():
        raise ValueError(f"Not a git repository: {repo_path}")

    try:
        # Check for uncommitted changes
        result = subprocess.run(
            ["git", "-C", str(repo_path), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        # If output is non-empty, there are changes
        return bool(result.stdout.strip())

    except subprocess.CalledProcessError:
        return False
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def get_remote_url(repo_path: Path, remote: str = "origin") -> str | None:
    """Get the remote URL for a repository.

    Args:
        repo_path: Path to the git repository
        remote: Name of the remote (default: "origin")

    Returns:
        The remote URL as a string, or None if not found

    Raises:
        FileNotFoundError: If the repository path doesn't exist
        ValueError: If the path is not a git repository
    """
    if not repo_path.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

    git_dir = repo_path / ".git"
    if not git_dir.exists():
        raise ValueError(f"Not a git repository: {repo_path}")

    try:
        # Get the remote URL
        result = subprocess.run(
            ["git", "-C", str(repo_path), "remote", "get-url", remote],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        url = result.stdout.strip()
        return url if url else None

    except subprocess.CalledProcessError:
        # Remote doesn't exist
        return None
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None
