# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Matthew Watkins <mwatkins@linuxfoundation.org>

"""GitHub API integration for repository mirroring."""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx

from gerrit_clone.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GitHubRepo:
    """Represents a GitHub repository."""

    name: str
    full_name: str
    html_url: str
    clone_url: str
    ssh_url: str
    private: bool
    description: str | None = None

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> GitHubRepo:
        """Create GitHubRepo from API response."""
        return cls(
            name=data["name"],
            full_name=data["full_name"],
            html_url=data["html_url"],
            clone_url=data["clone_url"],
            ssh_url=data["ssh_url"],
            private=data["private"],
            description=data.get("description"),
        )


class GitHubAPIError(Exception):
    """Base exception for GitHub API errors."""

    pass


class GitHubAuthError(GitHubAPIError):
    """GitHub authentication error."""

    pass


class GitHubNotFoundError(GitHubAPIError):
    """GitHub resource not found."""

    pass


class GitHubRateLimitError(GitHubAPIError):
    """GitHub API rate limit exceeded."""

    pass


class GitHubAPI:
    """GitHub API client for repository operations."""

    def __init__(self, token: str | None = None) -> None:
        """Initialize GitHub API client.

        Args:
            token: GitHub personal access token. If None, will try
                   to read from GITHUB_TOKEN environment variable.
        """
        self.token = token or os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise GitHubAuthError(
                "GitHub token required. Set GITHUB_TOKEN environment "
                "variable or pass token parameter."
            )

        self.base_url = "https://api.github.com"
        self.client = httpx.Client(
            headers={
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "gerrit-clone-mirror",
            },
            timeout=30.0,
        )
        # Don't create a shared async client - create fresh ones in async functions
        # to avoid "Event loop is closed" errors

    def __enter__(self) -> GitHubAPI:
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()

    def close(self) -> None:
        """Close the HTTP client."""
        self.client.close()
        # Don't close async client here - it will be closed by asyncio.run()
        # Closing it here causes "Event loop is closed" errors

    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> dict[str, Any] | list[Any]:
        """Make API request with error handling.

        Args:
            method: HTTP method
            endpoint: API endpoint (without base URL)
            **kwargs: Additional arguments for httpx.request

        Returns:
            JSON response data

        Raises:
            GitHubAPIError: For API errors
        """
        url = f"{self.base_url}{endpoint}"
        logger.debug(f"GitHub API {method} {url}")

        try:
            response = self.client.request(method, url, **kwargs)

            # Handle errors using shared method
            self._handle_response_errors(response, endpoint)

            response.raise_for_status()

            # Handle empty responses (e.g., 204 No Content for DELETE)
            if response.status_code == 204 or not response.content:
                return {}

            try:
                result: dict[str, Any] | list[Any] = response.json()
                return result
            except ValueError as e:
                # Handle JSON decode errors (e.g., empty response bodies)
                logger.warning(
                    f"Failed to parse JSON response from {url}: {e}. "
                    "Returning empty dict."
                )
                return {}

        except httpx.HTTPError as e:
            raise GitHubAPIError(f"HTTP error: {e}") from e

    def _handle_response_errors(self, response: httpx.Response, endpoint: str) -> None:
        """
        Handle HTTP response errors and raise appropriate exceptions.

        Uses GitHub's official rate limit headers for reliable detection:
        - X-RateLimit-Remaining: Number of requests remaining
        - Retry-After: Seconds to wait before retrying
        Falls back to text matching only as a last resort.

        Args:
            response: HTTP response object
            endpoint: API endpoint for error messages

        Raises:
            GitHubAuthError: For 401 authentication errors
            GitHubNotFoundError: For 404 not found errors
            GitHubRateLimitError: For 403 rate limit errors
            GitHubAPIError: For other API errors
        """
        if response.status_code == 401:
            raise GitHubAuthError(
                "Authentication failed. Check your GitHub token."
            )
        elif response.status_code == 404:
            raise GitHubNotFoundError(f"Resource not found: {endpoint}")
        elif response.status_code == 403:
            # Check for rate limiting using official GitHub headers
            rate_limit_remaining = response.headers.get("X-RateLimit-Remaining")
            retry_after = response.headers.get("Retry-After")

            # Primary rate limit check: X-RateLimit-Remaining is "0"
            if rate_limit_remaining == "0":
                raise GitHubRateLimitError("GitHub API rate limit exceeded")

            # Secondary rate limit check: Retry-After header present
            if retry_after:
                raise GitHubRateLimitError(
                    f"GitHub API rate limit exceeded. Retry after {retry_after} seconds"
                )

            # Fallback: check response text (less reliable)
            if "rate limit" in response.text.lower():
                raise GitHubRateLimitError("GitHub API rate limit exceeded")

            raise GitHubAPIError(f"Forbidden: {response.text}")
        elif response.status_code >= 400:
            raise GitHubAPIError(
                f"GitHub API error {response.status_code}: {response.text}"
            )

    def _request_paginated(
        self,
        method: str,
        endpoint: str,
        per_page: int = 100,
        max_pages: int | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """
        Make paginated API requests and return all results.

        Handles GitHub's Link header pagination to fetch all pages of results.
        Based on the pagination implementation from dependamerge.

        Args:
            method: HTTP method (usually GET)
            endpoint: API endpoint (without base URL)
            per_page: Number of items per page (max 100)
            max_pages: Optional maximum number of pages to fetch
            **kwargs: Additional arguments for httpx.request

        Returns:
            List of all items from all pages

        Raises:
            GitHubAPIError: For API errors
        """
        all_items: list[Any] = []
        page = 1

        while True:
            # Add pagination params - create a copy to avoid mutating caller's dict
            original_params = kwargs.get("params") or {}
            params = dict(original_params)
            params["per_page"] = per_page
            params["page"] = page
            kwargs["params"] = params

            # Make request
            url = f"{self.base_url}{endpoint}"
            logger.debug(f"GitHub API {method} {url} (page {page})")

            try:
                response = self.client.request(method, url, **kwargs)

                # Handle errors using shared method
                self._handle_response_errors(response, endpoint)

                response.raise_for_status()

                # Parse JSON
                try:
                    data = response.json()
                except ValueError as e:
                    logger.warning(
                        f"Failed to parse JSON response from {url}: {e}"
                    )
                    break

                # If no data or not a list, we're done
                if not data:
                    break
                if not isinstance(data, list):
                    logger.warning(
                        f"Expected list response from {url}, got {type(data)}"
                    )
                    break

                # Add items to result
                all_items.extend(data)

                # Check if we've hit max_pages
                if max_pages and page >= max_pages:
                    logger.debug(f"Reached max_pages limit: {max_pages}")
                    break

                # Check Link header for next page
                link_header = response.headers.get("Link", "")
                if 'rel="next"' not in link_header:
                    logger.debug(f"No more pages (total pages: {page})")
                    break

                page += 1

            except httpx.HTTPError as e:
                raise GitHubAPIError(f"HTTP error: {e}") from e

        logger.debug(f"Fetched {len(all_items)} total items across {page} page(s)")
        return all_items


    def get_authenticated_user(self) -> dict[str, Any]:
        """Get the authenticated user information.

        Returns:
            User information dictionary

        Raises:
            GitHubAPIError: For API errors
        """
        data = self._request("GET", "/user")
        if not isinstance(data, dict):
            raise GitHubAPIError("Unexpected response type for user info")
        return data

    def get_user_orgs(self) -> list[dict[str, Any]]:
        """Get organizations for the authenticated user.

        Returns:
            List of organization dictionaries

        Raises:
            GitHubAPIError: For API errors
        """
        data = self._request("GET", "/user/orgs")
        if not isinstance(data, list):
            raise GitHubAPIError("Unexpected response type for user orgs")
        return data

    def repo_exists(self, owner: str, repo_name: str) -> bool:
        """Check if a repository exists.

        Args:
            owner: Repository owner (user or org)
            repo_name: Repository name

        Returns:
            True if repository exists, False otherwise
        """
        try:
            self._request("GET", f"/repos/{owner}/{repo_name}")
            return True
        except GitHubNotFoundError:
            return False
        except GitHubAPIError as e:
            logger.warning(f"Error checking repository existence: {e}")
            return False

    def get_repo(self, owner: str, repo_name: str) -> GitHubRepo:
        """Get repository information.

        Args:
            owner: Repository owner (user or org)
            repo_name: Repository name

        Returns:
            GitHubRepo instance

        Raises:
            GitHubNotFoundError: If repository not found
            GitHubAPIError: For other API errors
        """
        data = self._request("GET", f"/repos/{owner}/{repo_name}")
        if not isinstance(data, dict):
            raise GitHubAPIError("Unexpected response type for repo info")
        return GitHubRepo.from_api_response(data)

    def create_repo(
        self,
        name: str,
        org: str | None = None,
        description: str | None = None,
        private: bool = False,
    ) -> GitHubRepo:
        """Create a new repository.

        Args:
            name: Repository name
            org: Organization name (if None, creates in user account)
            description: Repository description
            private: Whether repository should be private

        Returns:
            Created GitHubRepo instance

        Raises:
            GitHubAPIError: For API errors
        """
        # Sanitize description to remove control characters
        sanitized_desc = sanitize_description(description)
        if not sanitized_desc:
            sanitized_desc = f"Mirror of {name}"

        payload = {
            "name": name,
            "description": sanitized_desc,
            "private": private,
            "auto_init": False,
        }

        if org:
            endpoint = f"/orgs/{org}/repos"
        else:
            endpoint = "/user/repos"

        logger.info(
            f"Creating GitHub repository: {org}/{name}" if org else name
        )
        data = self._request("POST", endpoint, json=payload)
        if not isinstance(data, dict):
            raise GitHubAPIError("Unexpected response type for repo creation")
        return GitHubRepo.from_api_response(data)

    def list_repos(
        self,
        org: str | None = None,
        per_page: int = 100,
    ) -> list[GitHubRepo]:
        """List repositories for user or organization.

        Args:
            org: Organization name (if None, lists user repos)
            per_page: Number of results per page

        Returns:
            List of GitHubRepo instances

        Raises:
            GitHubAPIError: For API errors
        """
        repos: list[GitHubRepo] = []
        page = 1

        while True:
            if org:
                endpoint = f"/orgs/{org}/repos"
            else:
                endpoint = "/user/repos"

            endpoint += f"?per_page={per_page}&page={page}"

            data = self._request("GET", endpoint)
            if not isinstance(data, list):
                raise GitHubAPIError("Unexpected response type for repo list")

            if not data:
                break

            repos.extend(GitHubRepo.from_api_response(r) for r in data)
            page += 1

            # GitHub pagination: if less than per_page, it's the last page
            if len(data) < per_page:
                break

        return repos

    def delete_repo(self, owner: str, repo_name: str) -> None:
        """Delete a repository.

        Args:
            owner: Repository owner (user or org)
            repo_name: Repository name

        Raises:
            GitHubAPIError: For API errors
        """
        logger.warning(f"Deleting GitHub repository: {owner}/{repo_name}")
        self._request("DELETE", f"/repos/{owner}/{repo_name}")

    async def _delete_repo_async_with_client(
        self, client: httpx.AsyncClient, owner: str, repo_name: str
    ) -> tuple[bool, str | None]:
        """Delete a repository asynchronously with provided client.

        Args:
            client: Async HTTP client to use
            owner: Repository owner (user or org)
            repo_name: Repository name

        Returns:
            Tuple of (success, error_message)
        """
        url = f"{self.base_url}/repos/{owner}/{repo_name}"
        logger.debug(f"Async DELETE {url}")

        try:
            response = await client.delete(url)
            if response.status_code in (204, 404):
                # 204 = deleted, 404 = already gone
                logger.info(f"✓ Deleted {owner}/{repo_name}")
                return True, None
            elif response.status_code == 403:
                error = f"Permission denied: {response.text}"
                logger.error(f"✗ Failed to delete {owner}/{repo_name}: {error}")
                return False, error
            else:
                error = f"Status {response.status_code}: {response.text}"
                logger.error(f"✗ Failed to delete {owner}/{repo_name}: {error}")
                return False, error
        except Exception as e:
            error = f"Delete failed: {e}"
            logger.error(f"✗ Failed to delete {owner}/{repo_name}: {error}")
            return False, error

    async def _create_repo_async_with_client(
        self,
        client: httpx.AsyncClient,
        name: str,
        org: str | None = None,
        description: str | None = None,
        private: bool = False,
    ) -> tuple[GitHubRepo | None, str | None]:
        """Create a repository asynchronously with provided client.

        Args:
            client: Async HTTP client to use
            name: Repository name
            org: Organization name (if None, creates in user account)
            description: Repository description
            private: Whether repository should be private

        Returns:
            Tuple of (GitHubRepo or None, error_message or None)
        """
        sanitized_desc = sanitize_description(description)
        if not sanitized_desc:
            sanitized_desc = f"Mirror of {name}"

        payload = {
            "name": name,
            "description": sanitized_desc,
            "private": private,
            "auto_init": False,
        }

        if org:
            url = f"{self.base_url}/orgs/{org}/repos"
        else:
            url = f"{self.base_url}/user/repos"

        logger.debug(f"Async POST {url}")

        try:
            response = await client.post(url, json=payload)
            if response.status_code in (200, 201):
                data = response.json()
                logger.info(f"✓ Created {name}")
                return GitHubRepo.from_api_response(data), None
            elif response.status_code == 422:
                # Repository already exists - this shouldn't happen if delete worked
                error = "Repository already exists"
                logger.warning(f"⚠ {name} already exists (delete may have failed)")
                # Try to get the existing repo details
                try:
                    get_url = f"{self.base_url}/repos/{org if org else 'user'}/{name}"
                    get_response = await client.get(get_url)
                    if get_response.status_code == 200:
                        data = get_response.json()
                        logger.info(f"  Retrieved existing repo: {name}")
                        return GitHubRepo.from_api_response(data), None
                except Exception as ex:
                    logger.warning(f"Failed to retrieve existing repo details for {name}: {ex}")
                return None, error
            else:
                error = f"Status {response.status_code}: {response.text}"
                logger.error(f"✗ Failed to create {name}: {error}")
                return None, error
        except Exception as e:
            error = f"Create failed: {e}"
            logger.error(f"✗ Failed to create {name}: {error}")
            return None, error

    def list_all_repos_graphql(
        self, org: str
    ) -> dict[str, dict[str, Any]]:
        """List all repositories in an org using GraphQL (single query).

        This is much faster than paginating through REST API.

        Args:
            org: Organization name

        Returns:
            Dictionary mapping repo name to repo details
        """
        repos_map: dict[str, dict[str, Any]] = {}
        cursor = None
        has_next_page = True

        while has_next_page:
            # GraphQL query to fetch repos
            # Escape double quotes to prevent GraphQL injection and syntax errors
            safe_org = org.replace('"', '\\"')
            safe_cursor = cursor.replace('"', '\\"') if cursor else None
            after_clause = f', after: "{safe_cursor}"' if safe_cursor else ""
            query = f"""
            query {{
              organization(login: "{safe_org}") {{
                repositories(first: 100{after_clause}) {{
                  nodes {{
                    name
                    nameWithOwner
                    url
                    sshUrl
                    isPrivate
                    description
                    defaultBranchRef {{
                      name
                    }}
                  }}
                  pageInfo {{
                    hasNextPage
                    endCursor
                  }}
                }}
              }}
            }}
            """

            url = "https://api.github.com/graphql"
            logger.debug(f"GraphQL query for {org} repos (cursor: {cursor})")

            try:
                response = self.client.post(url, json={"query": query})
                response.raise_for_status()
                data = response.json()

                if "errors" in data:
                    errors = data["errors"]
                    logger.error(f"GraphQL errors: {errors}")
                    break

                org_data = data.get("data", {}).get("organization")
                if not org_data:
                    logger.warning(f"No organization data for {org}")
                    break

                repos_data = org_data.get("repositories", {})
                nodes = repos_data.get("nodes", [])
                page_info = repos_data.get("pageInfo", {})

                # Add repos to map
                for node in nodes:
                    name = node["name"]
                    repos_map[name] = {
                        "name": name,
                        "full_name": node["nameWithOwner"],
                        "html_url": node["url"],
                        "ssh_url": node["sshUrl"],
                        "clone_url": node["url"],  # Use url for HTTPS clone
                        "private": node["isPrivate"],
                        "description": node.get("description"),
                        "default_branch": (
                            node.get("defaultBranchRef", {}).get("name")
                            if node.get("defaultBranchRef")
                            else None
                        ),
                    }

                has_next_page = page_info.get("hasNextPage", False)
                cursor = page_info.get("endCursor")

                logger.debug(
                    f"Fetched {len(nodes)} repos, "
                    f"total so far: {len(repos_map)}, "
                    f"has_next: {has_next_page}"
                )

            except Exception as e:
                logger.error(f"GraphQL query failed: {e}")
                break

        logger.info(
            f"Fetched {len(repos_map)} repositories from {org} using GraphQL"
        )
        return repos_map

    async def batch_delete_repos(
        self, owner: str, repo_names: list[str], max_concurrent: int = 10
    ) -> dict[str, tuple[bool, str | None]]:
        """Delete multiple repositories in parallel.

        Args:
            owner: Repository owner (user or org)
            repo_names: List of repository names to delete
            max_concurrent: Maximum concurrent delete operations

        Returns:
            Dictionary mapping repo name to (success, error_message)
        """
        if not repo_names:
            return {}

        logger.info(
            f"Batch deleting {len(repo_names)} repositories "
            f"(max {max_concurrent} concurrent)"
        )

        # Create fresh async client for this batch operation
        async with httpx.AsyncClient(
            headers={
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "gerrit-clone-mirror",
            },
            timeout=30.0,
        ) as client:
            semaphore = asyncio.Semaphore(max_concurrent)

            async def delete_with_semaphore(
                repo_name: str,
            ) -> tuple[str, tuple[bool, str | None]]:
                async with semaphore:
                    result = await self._delete_repo_async_with_client(
                        client, owner, repo_name
                    )
                    return repo_name, result

            tasks = [delete_with_semaphore(name) for name in repo_names]
            results: list[tuple[str, tuple[bool, str | None]] | BaseException] = await asyncio.gather(*tasks, return_exceptions=True)

            results_map: dict[str, tuple[bool, str | None]] = {}
            for result in results:
                if isinstance(result, BaseException):
                    logger.error(f"Delete task failed with exception: {result}")
                    continue
                repo_name, (success, error) = result
                results_map[repo_name] = (success, error)

            success_count = sum(1 for s, _ in results_map.values() if s)
            failed_count = len(repo_names) - success_count

            if failed_count > 0:
                failed_repos = [name for name, (success, error) in results_map.items() if not success]
                logger.error(
                    f"Batch delete: {success_count}/{len(repo_names)} successful, "
                    f"{failed_count} FAILED"
                )
                logger.error(f"Failed repos: {failed_repos}")
                for name in failed_repos[:5]:  # Show first 5 errors
                    _, error = results_map[name]
                    logger.error(f"  - {name}: {error}")
            else:
                logger.info(
                    f"Batch delete completed: {success_count}/{len(repo_names)} successful"
                )

            return results_map

    async def batch_create_repos(
        self,
        org: str,
        repo_configs: list[dict[str, Any]],
        max_concurrent: int = 10,
    ) -> dict[str, tuple[GitHubRepo | None, str | None]]:
        """Create multiple repositories in parallel.

        Args:
            org: Organization name
            repo_configs: List of repo config dicts with keys:
                         name, description, private
            max_concurrent: Maximum concurrent create operations

        Returns:
            Dictionary mapping repo name to (GitHubRepo or None, error or None)
        """
        if not repo_configs:
            return {}

        logger.info(
            f"Batch creating {len(repo_configs)} repositories "
            f"(max {max_concurrent} concurrent)"
        )

        # Create fresh async client for this batch operation
        async with httpx.AsyncClient(
            headers={
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "gerrit-clone-mirror",
            },
            timeout=30.0,
        ) as client:
            semaphore = asyncio.Semaphore(max_concurrent)

            async def create_with_semaphore(
                config: dict[str, Any],
            ) -> tuple[str, tuple[GitHubRepo | None, str | None]]:
                async with semaphore:
                    name = config["name"]
                    result = await self._create_repo_async_with_client(
                        client,
                        name=name,
                        org=org,
                        description=config.get("description"),
                        private=config.get("private", False),
                    )
                    return name, result

            tasks = [create_with_semaphore(cfg) for cfg in repo_configs]
            results: list[tuple[str, tuple[GitHubRepo | None, str | None]] | BaseException] = await asyncio.gather(*tasks, return_exceptions=True)

            results_map: dict[str, tuple[GitHubRepo | None, str | None]] = {}
            for result in results:
                if isinstance(result, BaseException):
                    logger.error(f"Create task failed with exception: {result}")
                    continue
                repo_name, (repo, error) = result
                results_map[repo_name] = (repo, error)

            success_count = sum(
                1 for repo, _ in results_map.values() if repo is not None
            )
            failed_count = len(repo_configs) - success_count

            if failed_count > 0:
                failed_repos = [cfg["name"] for cfg in repo_configs if results_map.get(cfg["name"], (None, None))[0] is None]
                logger.warning(
                    f"Batch create: {success_count}/{len(repo_configs)} successful, "
                    f"{failed_count} failed"
                )
                logger.warning(f"Failed repos: {failed_repos[:10]}")  # Show first 10
            else:
                logger.info(
                    f"Batch create completed: {success_count}/{len(repo_configs)} successful"
                )

            return results_map


def sanitize_description(description: str | None) -> str | None:
    """Sanitize repository description for GitHub API.

    GitHub does not allow control characters in descriptions. This function
    removes control characters while preserving all other characters including
    quotes, which are properly handled by the JSON encoder.

    Args:
        description: Raw description text

    Returns:
        Sanitized description suitable for GitHub API, or None if input
        is None or empty after sanitization
    """
    if not description:
        return None

    # Remove control characters (including newlines, tabs, etc.)
    # Keep only printable ASCII and common Unicode characters
    # This preserves quotes, which are properly encoded by httpx's json parameter
    sanitized = re.sub(r"[\x00-\x1F\x7F-\x9F]", " ", description)

    # Replace multiple spaces with single space
    sanitized = re.sub(r"\s+", " ", sanitized)

    # Trim whitespace
    sanitized = sanitized.strip()

    # GitHub has a max description length of 350 characters
    if len(sanitized) > 350:
        sanitized = sanitized[:347] + "..."

    return sanitized if sanitized else None


def transform_gerrit_name_to_github(gerrit_name: str) -> str:
    """Transform Gerrit project name to valid GitHub repository name.

    Replaces forward slashes with hyphens since GitHub does not support
    slashes in repository names.

    Args:
        gerrit_name: Gerrit project name (e.g., "ccsdk/features/test")

    Returns:
        GitHub-compatible repository name (e.g., "ccsdk-features-test")
    """
    return gerrit_name.replace("/", "-")


def get_default_org_or_user(api: GitHubAPI) -> tuple[str, bool]:
    """Get default organization or user for the authenticated token.

    Returns the first organization if available, otherwise returns
    the authenticated user's login.

    Args:
        api: GitHubAPI instance

    Returns:
        Tuple of (owner_name, is_org) where is_org indicates if owner
        is an organization

    Raises:
        GitHubAPIError: For API errors
    """
    orgs = api.get_user_orgs()
    if orgs:
        org_login = orgs[0]["login"]
        logger.info(f"Using default organization: {org_login}")
        return org_login, True

    user = api.get_authenticated_user()
    user_login = user["login"]
    logger.info(f"Using authenticated user account: {user_login}")
    return user_login, False
