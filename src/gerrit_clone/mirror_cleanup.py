# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Removal of pre-existing local clone directories before a mirror run.

Provides the path selection and reporting around the deletion loop; the
deletion itself stays with the mirror manager.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gerrit_clone.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from gerrit_clone.models import Project

logger = get_logger(__name__)


def collect_paths_to_remove(
    base_path: Path,
    projects: list[Project],
) -> list[tuple[str, Path]]:
    """Collect existing clone directories, deepest paths first.

    Args:
        base_path: Root directory holding the local clones
        projects: Projects whose directories should be removed

    Returns:
        ``(project_name, path)`` pairs ordered so children precede
        parents; empty when there is nothing to remove.
    """
    paths_to_remove: list[tuple[str, Path]] = []
    for project in projects:
        project_path = base_path / project.name
        if project_path.exists():
            paths_to_remove.append((project.name, project_path))

    if not paths_to_remove:
        logger.info("No existing directories to clean up")
        return []

    logger.info(f"Removing {len(paths_to_remove)} existing directories...")

    # Remove in reverse dependency order (children before parents)
    # Sort by path depth (deepest first) to avoid removing parents
    # before children
    paths_to_remove.sort(key=lambda x: x[1].as_posix().count("/"), reverse=True)
    return paths_to_remove


def log_cleanup_outcome(
    removed_count: int,
    failed_removals: list[tuple[str, str]],
) -> None:
    """Report how many clone directories were removed."""
    if failed_removals:
        logger.warning(
            f"Successfully removed {removed_count} directories, "
            f"failed to remove {len(failed_removals)}"
        )
    else:
        logger.info(f"Successfully removed {removed_count} directories")
