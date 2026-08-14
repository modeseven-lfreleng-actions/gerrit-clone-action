# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Local repository discovery and project filtering for bulk refreshes.

Answers the question "which repositories should this refresh touch?" — walking
a directory tree for Git working copies and then applying the include/exclude
project patterns. Kept separate from
:mod:`gerrit_clone.refresh_manager` because selection is purely a filesystem
and pattern-matching concern with no bearing on how repositories are then
refreshed.
"""

from __future__ import annotations

import os
from pathlib import Path

from gerrit_clone.logging import get_logger
from gerrit_clone.models import match_project_pattern, normalize_project_list

logger = get_logger(__name__)


class RepositoryDiscoveryMixin:
    """Discovery of local Git repositories, with include/exclude filtering."""

    # Supplied by RefreshManager.__init__; declared here because this layer
    # reads them.
    recursive: bool
    include_projects: list[str] | None
    exclude_projects: list[str] | None

    def discover_local_repositories(self, base_path: Path) -> list[Path]:
        """Discover all Git repositories under base_path.

        Args:
            base_path: Base directory to search

        Returns:
            Sorted list of repository root paths (sorted alphabetically for
            deterministic processing order and consistent progress display)
        """
        if not base_path.exists():
            raise ValueError(f"Base path does not exist: {base_path}")

        if not base_path.is_dir():
            raise ValueError(f"Base path is not a directory: {base_path}")

        logger.debug(f"🔍 Discovering Git repositories in {base_path}")

        repositories: list[Path] = []
        visited_repos: set[Path] = set()

        # Walk directory tree
        for root, dirs, _files in os.walk(base_path):
            root_path = Path(root)

            # Check if current directory is a Git repository
            if ".git" in dirs:
                git_dir = root_path / ".git"

                # Verify it's a directory (not a file for submodules)
                if git_dir.is_dir():
                    # Normalize path
                    repo_path = root_path.resolve()

                    # Skip if we've already visited this repo
                    if repo_path in visited_repos:
                        continue

                    repositories.append(repo_path)
                    visited_repos.add(repo_path)

                    logger.debug(f"Found repository: {repo_path.name}")

                    if self.recursive:
                        # Continue searching subdirectories for Gerrit hierarchical projects
                        # In Gerrit, projects like ccsdk/apps, ccsdk/features are separate
                        # independent repos, not nested submodules within ccsdk
                        # We only skip .git directory itself
                        dirs[:] = [d for d in dirs if d != ".git"]
                    else:
                        # Non-recursive mode: don't descend into subdirectories
                        dirs[:] = []
                    continue

            # Skip hidden directories (except .git which we already handled)
            dirs[:] = [d for d in dirs if not d.startswith(".")]

        logger.debug(f"📂 Discovered {len(repositories)} Git repositories")

        # Sort repositories alphabetically for:
        # 1. Deterministic processing order across runs
        # 2. Better progress tracking (alphabetical display)
        # 3. Easier debugging and log analysis
        sorted_repos = sorted(repositories)

        # Apply include/exclude project filtering using relative paths
        # from base_path as project names (matching Gerrit's hierarchical
        # naming convention, e.g. "testsuite/pythonsdk-tests").
        if self.include_projects or self.exclude_projects:
            return self._filter_by_project_patterns(base_path, sorted_repos)

        return sorted_repos

    def _filter_by_project_patterns(
        self, base_path: Path, sorted_repos: list[Path]
    ) -> list[Path]:
        """Apply the include/exclude project patterns to discovered repos.

        Args:
            base_path: Base directory the repositories were discovered under
            sorted_repos: Discovered repository paths, in processing order

        Returns:
            The repositories that survive the include and exclude filters,
            in the same order.
        """
        include_pats = (
            normalize_project_list(self.include_projects)
            if self.include_projects
            else []
        )
        exclude_pats = (
            normalize_project_list(self.exclude_projects)
            if self.exclude_projects
            else []
        )

        before_count = len(sorted_repos)
        filtered: list[Path] = []
        base_resolved = base_path.resolve()
        for repo_path in sorted_repos:
            project_name = self._project_name_for(repo_path, base_resolved)

            # Apply include filter (if specified, only keep matches)
            if include_pats and not any(
                match_project_pattern(project_name, p) for p in include_pats
            ):
                continue

            # Apply exclude filter
            if exclude_pats and any(
                match_project_pattern(project_name, p) for p in exclude_pats
            ):
                continue

            filtered.append(repo_path)

        after_count = len(filtered)
        filter_desc: list[str] = []
        if include_pats:
            filter_desc.append(f"include={sorted(include_pats)}")
        if exclude_pats:
            filter_desc.append(f"exclude={sorted(exclude_pats)}")
        logger.debug(
            f"Project filter: kept {after_count}/{before_count} repositories "
            f"({', '.join(filter_desc)})"
        )
        return filtered

    @staticmethod
    def _project_name_for(repo_path: Path, base_resolved: Path) -> str:
        """Derive a Gerrit-style project name from a repository path.

        Uses the path relative to the base directory, with ``as_posix()`` for
        consistent forward-slash separators matching Gerrit's hierarchical
        naming convention.

        Args:
            repo_path: Repository path
            base_resolved: Resolved base directory

        Returns:
            Project name to match include/exclude patterns against
        """
        try:
            rel = repo_path.relative_to(base_resolved)
        except ValueError:
            # Fallback: use just the directory name
            return repo_path.name

        if rel == Path():
            # repo is exactly at base_path; use directory name so filters
            # can match it.
            return repo_path.name
        return rel.as_posix()
