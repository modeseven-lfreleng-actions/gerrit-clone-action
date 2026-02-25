# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Matthew Watkins <mwatkins@linuxfoundation.org>

"""GitHub API integration for repository mirroring."""

from __future__ import annotations

import asyncio
import os
import random
import re
import time as time_mod
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


class _AsyncProgressCounter:
    """Thread-safe counter for tracking batch operation progress.

    Emits a log message every *report_every* completions so the
    operator can see that work is proceeding.
    """

    def __init__(self, total: int, label: str, report_every: int = 10) -> None:
        self._total = total
        self._label = label
        self._report_every = report_every
        self._count = 0
        self._success = 0
        self._failed = 0
        self._lock = asyncio.Lock()

    async def record(self, *, success: bool, name: str) -> None:
        """Record one completed operation and log progress periodically.

        Args:
            success: Whether the operation succeeded.
            name: Repository name (included in debug-level log).
        """
        async with self._lock:
            self._count += 1
            if success:
                self._success += 1
            else:
                self._failed += 1
            # Capture a consistent snapshot of all counters while
            # still holding the lock, so logged values are coherent.
            count = self._count
            success_count = self._success
            failed_count = self._failed

        logger.debug(
            "%s [%d/%d] %s: %s",
            self._label,
            count,
            self._total,
            "ok" if success else "FAILED",
            name,
        )

        # Log on every Nth completion, and always on the last one
        if count % self._report_every == 0 or count == self._total:
            logger.info(
                "📊 %s progress: %d/%d completed (%d succeeded, %d failed)",
                self._label,
                count,
                self._total,
                success_count,
                failed_count,
            )


class _AsyncRateLimiter:
    """Shared async rate limiter that serialises API calls.

    Every caller must ``await acquire()`` before making a request.
    The lock inside ``acquire`` guarantees that no two requests are
    sent closer together than *min_interval* seconds, regardless of
    how many concurrent tasks exist.

    When a secondary rate-limit response (HTTP 403) is observed,
    call :meth:`increase_interval` to widen the gap for **all**
    tasks — this is the adaptive back-off that prevents the
    thundering-herd retry pattern.
    """

    def __init__(self, min_interval: float = 1.0) -> None:
        self._min_interval = min_interval
        self._lock = asyncio.Lock()
        self._last_time: float = 0.0

    @property
    def interval(self) -> float:
        """Current minimum interval between requests."""
        return self._min_interval

    async def acquire(self) -> None:
        """Wait until at least *min_interval* has elapsed since the last call."""
        async with self._lock:
            now = time_mod.monotonic()
            wait = self._min_interval - (now - self._last_time)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_time = time_mod.monotonic()

    async def increase_interval(
        self, factor: float = 2.0, max_interval: float = 30.0
    ) -> None:
        """Widen the minimum interval (called when any task is rate-limited).

        The modification is protected by the same lock used in
        :meth:`acquire` so that no task can read a stale interval
        while another task is updating it.
        """
        async with self._lock:
            old = self._min_interval
            self._min_interval = min(self._min_interval * factor, max_interval)
            if self._min_interval != old:
                logger.info(
                    f"⚙️  Rate limiter: interval {old:.1f}s → "
                    f"{self._min_interval:.1f}s"
                )


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

        logger.info(f"Creating GitHub repository: {org}/{name}" if org else name)
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

    def set_default_branch(
        self, owner: str, repo_name: str, branch: str
    ) -> bool:
        """Set the default branch for a repository.

        This should be called after pushing content to ensure the
        GitHub repository's default branch matches the source
        project's HEAD (e.g. from Gerrit).

        Args:
            owner: Repository owner (user or org)
            repo_name: Repository name
            branch: Branch name to set as default (e.g. ``master``, ``main``)

        Returns:
            True if the default branch was set successfully, False otherwise
        """
        logger.debug(
            "Setting default branch for %s/%s to '%s'",
            owner,
            repo_name,
            branch,
        )
        try:
            self._request(
                "PATCH",
                f"/repos/{owner}/{repo_name}",
                json={"default_branch": branch},
            )
            logger.info(
                "Set default branch for %s/%s to '%s'",
                owner,
                repo_name,
                branch,
            )
            return True
        except GitHubAPIError as exc:
            logger.warning(
                "Failed to set default branch for %s/%s to '%s': %s",
                owner,
                repo_name,
                branch,
                exc,
            )
            return False

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
        self,
        client: httpx.AsyncClient,
        owner: str,
        repo_name: str,
        max_retries: int = 5,
        rate_limiter: _AsyncRateLimiter | None = None,
        progress: _AsyncProgressCounter | None = None,
    ) -> tuple[bool, str | None]:
        """Delete a repository asynchronously with provided client.

        Uses a shared rate limiter (when provided) to serialise API
        calls, preventing GitHub secondary-rate-limit 403 responses.

        Args:
            client: Async HTTP client to use
            owner: Repository owner (user or org)
            repo_name: Repository name
            max_retries: Maximum retry attempts on rate-limit responses
            rate_limiter: Shared rate limiter (controls inter-request spacing)
            progress: Optional shared progress counter

        Returns:
            Tuple of (success, error_message)
        """
        url = f"{self.base_url}/repos/{owner}/{repo_name}"
        logger.debug(f"Async DELETE {url}")

        # Initialise with a meaningful default so the "exhausted retries"
        # message is never None (last_error is always re-set before each
        # `continue`, but this guards against unexpected control-flow).
        last_error: str = "unknown error"

        for attempt in range(max_retries + 1):
            if rate_limiter:
                await rate_limiter.acquire()

            if attempt > 0:
                jitter = random.uniform(0.1, 0.5)
                await asyncio.sleep(jitter)

            try:
                response = await client.delete(url)
                if response.status_code in (204, 404):
                    # 204 = deleted, 404 = already gone
                    if attempt > 0:
                        logger.info(
                            f"✓ Deleted {owner}/{repo_name} "
                            f"(after {attempt} "
                            f"{'retry' if attempt == 1 else 'retries'})"
                        )
                    else:
                        logger.info(f"✓ Deleted {owner}/{repo_name}")
                    if progress:
                        await progress.record(success=True, name=repo_name)
                    return True, None
                elif response.status_code == 403:
                    # Check for rate limiting
                    is_rate_limit = (
                        response.headers.get("Retry-After") is not None
                        or response.headers.get("X-RateLimit-Remaining") == "0"
                        or "rate limit" in response.text.lower()
                    )

                    if is_rate_limit and attempt < max_retries:
                        if rate_limiter:
                            await rate_limiter.increase_interval()

                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            try:
                                extra_wait = float(retry_after)
                                logger.warning(
                                    f"⏳ Rate limited deleting "
                                    f"{owner}/{repo_name}, "
                                    f"server asked to wait {extra_wait:.0f}s "
                                    f"(attempt {attempt + 1}/{max_retries + 1})"
                                )
                                await asyncio.sleep(extra_wait)
                            except ValueError:
                                pass
                        else:
                            # Exponential back-off when no Retry-After
                            backoff = min(60, 5 * (2 ** attempt))
                            logger.warning(
                                f"⏳ Rate limited deleting "
                                f"{owner}/{repo_name}, backing off "
                                f"{backoff}s "
                                f"(attempt {attempt + 1}/{max_retries + 1})"
                            )
                            await asyncio.sleep(backoff)
                        last_error = f"Rate limited: {response.text}"
                        continue

                    if is_rate_limit:
                        # Retry budget exhausted on a rate-limit 403
                        error = (
                            f"Rate limited after "
                            f"{max_retries + 1} attempts: "
                            f"{response.text}"
                        )
                    else:
                        error = f"Permission denied: {response.text}"
                    logger.error(
                        f"✗ Failed to delete {owner}/{repo_name}: {error}"
                    )
                    if progress:
                        await progress.record(success=False, name=repo_name)
                    return False, error
                else:
                    error = f"Status {response.status_code}: {response.text}"
                    logger.error(
                        f"✗ Failed to delete {owner}/{repo_name}: {error}"
                    )
                    if progress:
                        await progress.record(success=False, name=repo_name)
                    return False, error
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(
                        f"⏳ Error deleting {owner}/{repo_name}: {e} "
                        f"(attempt {attempt + 1}/{max_retries + 1})"
                    )
                    last_error = f"Delete failed: {e}"
                    continue
                error = f"Delete failed: {e}"
                logger.error(
                    f"✗ Failed to delete {owner}/{repo_name}: {error}"
                )
                if progress:
                    await progress.record(success=False, name=repo_name)
                return False, error

        # Exhausted all retries
        error = f"Failed after {max_retries + 1} attempts: {last_error}"
        logger.error(f"✗ Failed to delete {owner}/{repo_name}: {error}")
        if progress:
            await progress.record(success=False, name=repo_name)
        return False, error

    async def _create_repo_async_with_client(
        self,
        client: httpx.AsyncClient,
        name: str,
        org: str | None = None,
        description: str | None = None,
        private: bool = False,
        max_retries: int = 5,
        rate_limiter: _AsyncRateLimiter | None = None,
        progress: _AsyncProgressCounter | None = None,
    ) -> tuple[GitHubRepo | None, str | None]:
        """Create a repository asynchronously with provided client.

        Uses a shared rate limiter to serialise API calls and avoid
        triggering GitHub's secondary rate limits.  When a 403 *is*
        returned the limiter's interval is widened so that every
        subsequent request (from any task) automatically slows down.

        Args:
            client: Async HTTP client to use
            name: Repository name
            org: Organization name (if None, creates in user account)
            description: Repository description
            private: Whether repository should be private
            max_retries: Maximum number of retry attempts for rate limits
            rate_limiter: Shared rate limiter (controls inter-request spacing)
            progress: Optional shared progress counter

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

        # Initialise with a meaningful default so the "exhausted retries"
        # message is never None (last_error is always re-set before each
        # `continue`, but this guards against unexpected control-flow).
        last_error: str = "unknown error"

        for attempt in range(max_retries + 1):
            # Gate every attempt through the shared rate limiter so that
            # concurrent tasks never burst requests into GitHub.
            if rate_limiter:
                await rate_limiter.acquire()

            # Small random jitter on retries to de-synchronise tasks that
            # were rate-limited on the same cycle.
            if attempt > 0:
                jitter = random.uniform(0.1, 0.5)
                await asyncio.sleep(jitter)

            try:
                response = await client.post(url, json=payload)
                if response.status_code in (200, 201):
                    data = response.json()
                    if attempt > 0:
                        logger.info(
                            f"✓ Created {name} (after {attempt} "
                            f"{'retry' if attempt == 1 else 'retries'})"
                        )
                    else:
                        logger.info(f"✓ Created {name}")
                    if progress:
                        await progress.record(success=True, name=name)
                    return GitHubRepo.from_api_response(data), None
                elif response.status_code == 422:
                    # Repository already exists - not retriable
                    error = "Repository already exists"
                    logger.warning(
                        f"⚠ {name} already exists (delete may have failed)"
                    )
                    # Try to get the existing repo details when an org is given.
                    # The `/repos/user/{name}` path is not valid on the
                    # GitHub API, so we only attempt this when org is set.
                    if org:
                        try:
                            get_url = (
                                f"{self.base_url}/repos/{org}/{name}"
                            )
                            get_response = await client.get(get_url)
                            if get_response.status_code == 200:
                                data = get_response.json()
                                logger.info(
                                    f"  Retrieved existing repo: {name}"
                                )
                                return GitHubRepo.from_api_response(data), None
                        except Exception as ex:
                            logger.warning(
                                f"Failed to retrieve existing repo details "
                                f"for {name}: {ex}"
                            )
                    if progress:
                        await progress.record(success=False, name=name)
                    return None, error
                elif response.status_code == 403:
                    # Check for rate limiting - this is retriable
                    retry_after = response.headers.get("Retry-After")
                    is_rate_limit = (
                        retry_after is not None
                        or response.headers.get("X-RateLimit-Remaining") == "0"
                        or "rate limit" in response.text.lower()
                        or "secondary rate limit" in response.text.lower()
                    )

                    if is_rate_limit and attempt < max_retries:
                        # Widen the shared rate limiter so ALL tasks
                        # slow down — this is the key to avoiding the
                        # thundering-herd retry storm.
                        if rate_limiter:
                            await rate_limiter.increase_interval()

                        # If the server sent Retry-After, honour it as
                        # an additional pause for THIS task.
                        if retry_after:
                            try:
                                extra_wait = float(retry_after)
                                logger.warning(
                                    f"⏳ Rate limited creating {name}, "
                                    f"server asked to wait {extra_wait:.0f}s "
                                    f"(attempt {attempt + 1}/{max_retries + 1})"
                                )
                                await asyncio.sleep(extra_wait)
                            except ValueError:
                                pass
                        else:
                            # Exponential back-off when GitHub doesn't
                            # tell us how long to wait.  This gives the
                            # secondary-rate-limit rolling window time
                            # to recover.
                            backoff = min(60, 5 * (2 ** attempt))
                            logger.warning(
                                f"⏳ Rate limited creating {name}, "
                                f"backing off {backoff}s "
                                f"(attempt {attempt + 1}/{max_retries + 1})"
                            )
                            await asyncio.sleep(backoff)

                        last_error = (
                            f"Status {response.status_code}: {response.text}"
                        )
                        # Loop back → acquire() enforces the new interval
                        continue
                    else:
                        error = (
                            f"Status {response.status_code}: {response.text}"
                        )
                        logger.error(f"✗ Failed to create {name}: {error}")
                        if progress:
                            await progress.record(success=False, name=name)
                        return None, error
                else:
                    error = (
                        f"Status {response.status_code}: {response.text}"
                    )
                    logger.error(f"✗ Failed to create {name}: {error}")
                    if progress:
                        await progress.record(success=False, name=name)
                    return None, error
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(
                        f"⏳ Error creating {name}: {e} "
                        f"(attempt {attempt + 1}/{max_retries + 1})"
                    )
                    last_error = f"Create failed: {e}"
                    # Loop back → acquire() spaces the next attempt
                    continue
                error = f"Create failed: {e}"
                logger.error(f"✗ Failed to create {name}: {error}")
                if progress:
                    await progress.record(success=False, name=name)
                return None, error

        # Exhausted all retries
        error = f"Failed after {max_retries + 1} attempts: {last_error}"
        logger.error(f"✗ Failed to create {name}: {error}")
        if progress:
            await progress.record(success=False, name=name)
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
        repos_without_default_branch: list[str] = []
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
                      target {{
                        ... on Commit {{
                          oid
                        }}
                      }}
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
                    # Extract commit SHA from defaultBranchRef
                    default_branch_ref = node.get("defaultBranchRef")
                    default_branch = None
                    latest_commit_sha = None

                    if default_branch_ref:
                        default_branch = default_branch_ref.get("name")
                        target = default_branch_ref.get("target")
                        if target:
                            latest_commit_sha = target.get("oid")
                    else:
                        repos_without_default_branch.append(name)
                        logger.debug(
                            "Repository %s has no default branch configured; "
                            "latest_commit_sha will be unavailable",
                            name,
                        )

                    repos_map[name] = {
                        "name": name,
                        "full_name": node["nameWithOwner"],
                        "html_url": node["url"],
                        "ssh_url": node["sshUrl"],
                        "clone_url": node["url"],  # Use url for HTTPS clone
                        "private": node["isPrivate"],
                        "description": node.get("description"),
                        "default_branch": default_branch,
                        "latest_commit_sha": latest_commit_sha,
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

        if repos_without_default_branch:
            logger.warning(
                "%d/%d repositories have no default branch configured "
                "(empty repos or failed pushes)",
                len(repos_without_default_branch),
                len(repos_map),
            )

        logger.debug(f"Fetched {len(repos_map)} repositories from {org} using GraphQL")
        return repos_map

    async def batch_delete_repos(
        self,
        owner: str,
        repo_names: list[str],
        max_concurrent: int = 10,
        rate_limit_interval: float = 0.5,
    ) -> dict[str, tuple[bool, str | None]]:
        """Delete multiple repositories with rate-limit-aware scheduling.

        A shared :class:`_AsyncRateLimiter` serialises the actual HTTP
        calls so that no two delete requests fire closer together than
        ``rate_limit_interval`` seconds.  This prevents burning through
        the secondary-rate-limit budget before subsequent operations
        (e.g. batch create) even start.

        Retry budget scales with batch size so that large batches have
        enough headroom to survive transient secondary-rate-limit blocks.

        Args:
            owner: Repository owner (user or org)
            repo_names: List of repository names to delete
            max_concurrent: Maximum tasks in flight at once
            rate_limit_interval: Minimum seconds between API calls
                (adaptive — increased automatically on 403)

        Returns:
            Dictionary mapping repo name to (success, error_message)
        """
        if not repo_names:
            return {}

        # Scale retries with batch size — large batches are more likely
        # to trigger the secondary rate limit's rolling window.
        batch_retries = max(5, min(15, len(repo_names) // 10))

        logger.info(
            f"Batch deleting {len(repo_names)} repositories "
            f"(max {max_concurrent} concurrent, "
            f"{rate_limit_interval:.1f}s between requests, "
            f"max {batch_retries} retries per repo)"
        )

        # Shared rate limiter — same pattern as batch_create_repos
        rate_limiter = _AsyncRateLimiter(min_interval=rate_limit_interval)

        # Progress counter — report every 10 completions
        progress = _AsyncProgressCounter(
            total=len(repo_names),
            label="Delete",
            report_every=max(1, len(repo_names) // 10),
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
                        client, owner, repo_name,
                        max_retries=batch_retries,
                        rate_limiter=rate_limiter,
                        progress=progress,
                    )
                    return repo_name, result

            tasks = [delete_with_semaphore(name) for name in repo_names]
            results: list[
                tuple[str, tuple[bool, str | None]] | BaseException
            ] = await asyncio.gather(*tasks, return_exceptions=True)

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
                failed_repos = [
                    name
                    for name, (success, error) in results_map.items()
                    if not success
                ]
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
        rate_limit_interval: float = 2.0,
    ) -> dict[str, tuple[GitHubRepo | None, str | None]]:
        """Create multiple repositories with rate-limit-aware scheduling.

        A shared :class:`_AsyncRateLimiter` serialises the actual HTTP
        calls so that, regardless of ``max_concurrent``, no two
        creation requests fire closer together than
        ``rate_limit_interval`` seconds.  If *any* task receives a 403
        secondary-rate-limit response the limiter's interval is
        automatically widened, slowing every subsequent request from
        every task.

        Retry budget and initial pacing scale with batch size so that
        large mirrors (100+ repos) survive GitHub's secondary rate
        limits without fatal failures.

        Args:
            org: Organization name
            repo_configs: List of repo config dicts with keys:
                         name, description, private
            max_concurrent: Maximum tasks in flight at once
            rate_limit_interval: Minimum seconds between API calls
                (adaptive — increased automatically on 403)

        Returns:
            Dictionary mapping repo name to (GitHubRepo or None, error or None)
        """
        if not repo_configs:
            return {}

        # Scale retries with batch size — large batches are more likely
        # to trigger the secondary rate limit's rolling window.
        batch_retries = max(5, min(15, len(repo_configs) // 10))

        # For very large batches, widen the initial interval to avoid
        # hitting the limit in the first place.
        if len(repo_configs) > 100:
            rate_limit_interval = max(rate_limit_interval, 3.0)
        if len(repo_configs) > 200:
            rate_limit_interval = max(rate_limit_interval, 4.0)

        logger.info(
            f"Batch creating {len(repo_configs)} repositories "
            f"(max {max_concurrent} concurrent, "
            f"{rate_limit_interval:.1f}s between requests, "
            f"max {batch_retries} retries per repo)"
        )

        # Shared rate limiter — the single point of throttle for all tasks
        rate_limiter = _AsyncRateLimiter(min_interval=rate_limit_interval)

        # Progress counter — report every ~10% of completions
        progress = _AsyncProgressCounter(
            total=len(repo_configs),
            label="Create",
            report_every=max(1, len(repo_configs) // 10),
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
                        max_retries=batch_retries,
                        rate_limiter=rate_limiter,
                        progress=progress,
                    )
                    return name, result

            tasks = [create_with_semaphore(cfg) for cfg in repo_configs]
            results: list[
                tuple[str, tuple[GitHubRepo | None, str | None]] | BaseException
            ] = await asyncio.gather(*tasks, return_exceptions=True)

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
                failed_repos = [
                    cfg["name"]
                    for cfg in repo_configs
                    if results_map.get(cfg["name"], (None, None))[0] is None
                ]
                logger.warning(
                    f"Batch create: {success_count}/{len(repo_configs)} successful, "
                    f"{failed_count} failed"
                )
                logger.warning(f"Failed repos: {failed_repos[:10]}")  # Show first 10
            else:
                logger.info(
                    f"Batch create completed: "
                    f"{success_count}/{len(repo_configs)} successful"
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
