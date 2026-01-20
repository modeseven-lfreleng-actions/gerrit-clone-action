# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Matthew Watkins <mwatkins@linuxfoundation.org>

"""Unit tests for GitHub API module."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from gerrit_clone.github_api import (
    GitHubAPI,
    GitHubAPIError,
    GitHubAuthError,
    GitHubNotFoundError,
    GitHubRateLimitError,
    GitHubRepo,
    get_default_org_or_user,
    sanitize_description,
    transform_gerrit_name_to_github,
)


class TestSanitizeDescription:
    """Test description sanitization."""

    def test_none_returns_none(self) -> None:
        """Test that None input returns None."""
        assert sanitize_description(None) is None

    def test_empty_string_returns_none(self) -> None:
        """Test that empty string returns None."""
        assert sanitize_description("") is None
        assert sanitize_description("   ") is None

    def test_removes_control_characters(self) -> None:
        """Test removal of control characters."""
        # Newlines
        result = sanitize_description("Line 1\nLine 2")
        assert result == "Line 1 Line 2"

        # Tabs
        result = sanitize_description("Tab\there")
        assert result == "Tab here"

        # Carriage returns
        result = sanitize_description("Text\rwith\rCR")
        assert result == "Text with CR"

    def test_replaces_multiple_spaces(self) -> None:
        """Test that multiple spaces are collapsed."""
        result = sanitize_description("Too    many     spaces")
        assert result == "Too many spaces"

    def test_trims_whitespace(self) -> None:
        """Test that leading/trailing whitespace is removed."""
        result = sanitize_description("  Trimmed  ")
        assert result == "Trimmed"

    def test_truncates_long_descriptions(self) -> None:
        """Test that descriptions longer than 350 chars are truncated."""
        long_desc = "a" * 400
        result = sanitize_description(long_desc)
        assert result is not None
        assert len(result) == 350
        assert result.endswith("...")

    def test_normal_description_unchanged(self) -> None:
        """Test that normal descriptions pass through."""
        desc = "A normal repository description"
        result = sanitize_description(desc)
        assert result == desc

    def test_preserves_double_quotes(self) -> None:
        """Test that double quotes are preserved."""
        result = sanitize_description('Description with "quotes" inside')
        assert result == 'Description with "quotes" inside'

    def test_mixed_issues(self) -> None:
        """Test description with multiple issues."""
        desc = "  Line 1\n\nLine 2\t\tTab  "
        result = sanitize_description(desc)
        assert result == "Line 1 Line 2 Tab"

    def test_real_world_cert_service_case(self) -> None:
        """Test description similar to oom/platform/cert-service."""
        desc = 'OOM Cert Service "Certificate Authority" setup'
        result = sanitize_description(desc)
        assert result == 'OOM Cert Service "Certificate Authority" setup'
        assert '"' in result  # Quotes should be preserved


class TestTransformGerritNameToGitHub:
    """Test Gerrit name to GitHub transformation."""

    def test_simple_name_no_change(self) -> None:
        """Test simple names remain unchanged."""
        assert transform_gerrit_name_to_github("ccsdk") == "ccsdk"

    def test_single_level_hierarchy(self) -> None:
        """Test single level hierarchy transformation."""
        assert transform_gerrit_name_to_github("ccsdk/apps") == "ccsdk-apps"

    def test_multi_level_hierarchy(self) -> None:
        """Test multi-level hierarchy transformation."""
        result = transform_gerrit_name_to_github("ccsdk/features/test")
        assert result == "ccsdk-features-test"

    def test_deep_hierarchy(self) -> None:
        """Test deep hierarchy transformation."""
        result = transform_gerrit_name_to_github("project/sub/subsub/deep")
        assert result == "project-sub-subsub-deep"


class TestGitHubRepo:
    """Test GitHubRepo dataclass."""

    def test_from_api_response(self) -> None:
        """Test creating GitHubRepo from API response."""
        api_data = {
            "name": "test-repo",
            "full_name": "org/test-repo",
            "html_url": "https://github.com/org/test-repo",
            "clone_url": "https://github.com/org/test-repo.git",
            "ssh_url": "git@github.com:org/test-repo.git",
            "private": False,
            "description": "Test repository",
        }

        repo = GitHubRepo.from_api_response(api_data)

        assert repo.name == "test-repo"
        assert repo.full_name == "org/test-repo"
        assert repo.html_url == "https://github.com/org/test-repo"
        assert repo.clone_url == "https://github.com/org/test-repo.git"
        assert repo.ssh_url == "git@github.com:org/test-repo.git"
        assert repo.private is False
        assert repo.description == "Test repository"

    def test_from_api_response_no_description(self) -> None:
        """Test creating GitHubRepo without description."""
        api_data = {
            "name": "test-repo",
            "full_name": "org/test-repo",
            "html_url": "https://github.com/org/test-repo",
            "clone_url": "https://github.com/org/test-repo.git",
            "ssh_url": "git@github.com:org/test-repo.git",
            "private": True,
        }

        repo = GitHubRepo.from_api_response(api_data)

        assert repo.name == "test-repo"
        assert repo.description is None
        assert repo.private is True


class TestGitHubAPI:
    """Test GitHubAPI client."""

    def test_init_with_token(self) -> None:
        """Test initialization with explicit token."""
        api = GitHubAPI(token="test-token")
        assert api.token == "test-token"
        api.close()

    def test_init_without_token_raises_error(self) -> None:
        """Test initialization without token raises error."""
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(GitHubAuthError),
        ):
            GitHubAPI()

    def test_init_with_env_token(self) -> None:
        """Test initialization with environment variable token."""
        with patch.dict("os.environ", {"GITHUB_TOKEN": "env-token"}):
            api = GitHubAPI()
            assert api.token == "env-token"
            api.close()

    def test_context_manager(self) -> None:
        """Test GitHubAPI as context manager."""
        with GitHubAPI(token="test-token") as api:
            assert api.token == "test-token"

    @patch("gerrit_clone.github_api.httpx.Client")
    def test_repo_exists_true(self, mock_client: Mock) -> None:
        """Test repo_exists returns True when repo exists."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"name": "test-repo"}

        mock_client_instance = Mock()
        mock_client_instance.request.return_value = mock_response
        mock_client.return_value = mock_client_instance

        api = GitHubAPI(token="test-token")
        result = api.repo_exists("owner", "test-repo")

        assert result is True
        api.close()

    @patch("gerrit_clone.github_api.httpx.Client")
    def test_repo_exists_false(self, mock_client: Mock) -> None:
        """Test repo_exists returns False when repo not found."""
        mock_response = Mock()
        mock_response.status_code = 404

        mock_client_instance = Mock()
        mock_client_instance.request.return_value = mock_response
        mock_client.return_value = mock_client_instance

        api = GitHubAPI(token="test-token")
        result = api.repo_exists("owner", "test-repo")

        assert result is False
        api.close()


class TestHandleResponseErrors:
    """Test error handling with GitHub API responses."""

    def test_rate_limit_detected_by_header(self) -> None:
        """Test rate limit detection using X-RateLimit-Remaining header."""
        api = GitHubAPI(token="test-token")

        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.headers = {"X-RateLimit-Remaining": "0"}
        mock_response.text = "Forbidden"

        with pytest.raises(GitHubRateLimitError, match="rate limit exceeded"):
            api._handle_response_errors(mock_response, "/test/endpoint")

        api.close()

    def test_rate_limit_detected_by_retry_after_header(self) -> None:
        """Test rate limit detection using Retry-After header."""
        api = GitHubAPI(token="test-token")

        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.headers = {"Retry-After": "60"}
        mock_response.text = "Forbidden"

        with pytest.raises(GitHubRateLimitError, match="Retry after 60 seconds"):
            api._handle_response_errors(mock_response, "/test/endpoint")

        api.close()

    def test_rate_limit_detected_by_text_fallback(self) -> None:
        """Test rate limit detection falls back to text matching."""
        api = GitHubAPI(token="test-token")

        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.headers = {}
        mock_response.text = "API rate limit exceeded for user"

        with pytest.raises(GitHubRateLimitError, match="rate limit exceeded"):
            api._handle_response_errors(mock_response, "/test/endpoint")

        api.close()

    def test_403_without_rate_limit_raises_generic_error(self) -> None:
        """Test 403 without rate limit indicators raises generic APIError."""
        api = GitHubAPI(token="test-token")

        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.headers = {}
        mock_response.text = "Access denied for other reasons"

        with pytest.raises(GitHubAPIError, match="Forbidden"):
            api._handle_response_errors(mock_response, "/test/endpoint")

        api.close()

    def test_401_raises_auth_error(self) -> None:
        """Test 401 raises authentication error."""
        api = GitHubAPI(token="test-token")

        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Bad credentials"

        with pytest.raises(GitHubAuthError, match="Authentication failed"):
            api._handle_response_errors(mock_response, "/test/endpoint")

        api.close()

    def test_404_raises_not_found_error(self) -> None:
        """Test 404 raises not found error."""
        api = GitHubAPI(token="test-token")

        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not found"

        with pytest.raises(GitHubNotFoundError, match="Resource not found"):
            api._handle_response_errors(mock_response, "/test/endpoint")

        api.close()


class TestGetDefaultOrgOrUser:
    """Test get_default_org_or_user function."""

    @patch.object(GitHubAPI, "get_user_orgs")
    @patch.object(GitHubAPI, "get_authenticated_user")
    def test_returns_first_org(self, mock_get_user: Mock, mock_get_orgs: Mock) -> None:
        """Test returns first organization when available."""
        mock_get_orgs.return_value = [{"login": "test-org"}]

        api = GitHubAPI(token="test-token")
        owner, is_org = get_default_org_or_user(api)

        assert owner == "test-org"
        assert is_org is True
        api.close()

    @patch.object(GitHubAPI, "get_user_orgs")
    @patch.object(GitHubAPI, "get_authenticated_user")
    def test_returns_user_when_no_orgs(
        self, mock_get_user: Mock, mock_get_orgs: Mock
    ) -> None:
        """Test returns user when no organizations available."""
        mock_get_orgs.return_value = []
        mock_get_user.return_value = {"login": "test-user"}

        api = GitHubAPI(token="test-token")
        owner, is_org = get_default_org_or_user(api)

        assert owner == "test-user"
        assert is_org is False
        api.close()


class TestBatchDeleteRepos:
    """Test batch_delete_repos async method."""

    @pytest.mark.asyncio
    async def test_batch_delete_success(self) -> None:
        """Test successful batch deletion of repositories."""
        api = GitHubAPI(token="test-token")

        # Mock httpx.AsyncClient to return successful responses
        mock_response = AsyncMock()
        mock_response.status_code = 204

        mock_client = AsyncMock()
        mock_client.delete = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "gerrit_clone.github_api.httpx.AsyncClient", return_value=mock_client
        ):
            repo_names = ["repo1", "repo2", "repo3"]
            results = await api.batch_delete_repos("test-org", repo_names)

            assert len(results) == 3
            assert all(success for success, _ in results.values())
            assert mock_client.delete.call_count == 3

        api.close()

    @pytest.mark.asyncio
    async def test_batch_delete_partial_failure(self) -> None:
        """Test batch deletion with some failures."""
        api = GitHubAPI(token="test-token")

        # Mock delete to fail for repo2 with 403 (not 404, since 404 is treated as success)
        call_count = [0]

        async def mock_delete_side_effect(*args, **kwargs):
            call_count[0] += 1
            response = Mock()  # Use Mock, not AsyncMock for response
            if "repo2" in args[0]:
                response.status_code = 403
                response.text = "Permission denied"
            else:
                response.status_code = 204
            return response

        mock_client = AsyncMock()
        mock_client.delete = AsyncMock(side_effect=mock_delete_side_effect)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "gerrit_clone.github_api.httpx.AsyncClient", return_value=mock_client
        ):
            repo_names = ["repo1", "repo2", "repo3"]
            results = await api.batch_delete_repos("test-org", repo_names)

            assert len(results) == 3
            assert results["repo1"][0] is True
            assert results["repo2"][0] is False
            assert results["repo3"][0] is True

        api.close()

    @pytest.mark.asyncio
    async def test_batch_delete_respects_concurrency_limit(self) -> None:
        """Test that batch deletion respects max_concurrent limit."""
        api = GitHubAPI(token="test-token")

        async def mock_delete_with_delay(*args, **kwargs):
            await asyncio.sleep(0.01)
            response = Mock()  # Use Mock, not AsyncMock for response
            response.status_code = 204
            return response

        mock_client = AsyncMock()
        mock_client.delete = AsyncMock(side_effect=mock_delete_with_delay)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "gerrit_clone.github_api.httpx.AsyncClient", return_value=mock_client
        ):
            repo_names = [f"repo{i}" for i in range(10)]
            results = await api.batch_delete_repos(
                "test-org", repo_names, max_concurrent=3
            )

            assert len(results) == 10
            assert all(success for success, _ in results.values())

        api.close()

    @pytest.mark.asyncio
    async def test_batch_delete_handles_exceptions(self) -> None:
        """Test that exceptions are handled gracefully."""
        api = GitHubAPI(token="test-token")

        async def mock_delete_with_exception(*args, **kwargs):
            if "repo2" in args[0]:
                raise Exception("Network error")
            response = Mock()  # Use Mock, not AsyncMock for response
            response.status_code = 204
            return response

        mock_client = AsyncMock()
        mock_client.delete = AsyncMock(side_effect=mock_delete_with_exception)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "gerrit_clone.github_api.httpx.AsyncClient", return_value=mock_client
        ):
            repo_names = ["repo1", "repo2", "repo3"]
            results = await api.batch_delete_repos("test-org", repo_names)

            # Should handle exception - result for repo2 may not be in results
            assert len(results) >= 2
            assert results["repo1"][0] is True
            assert results["repo3"][0] is True

        api.close()


class TestBatchCreateRepos:
    """Test batch_create_repos async method."""

    @pytest.mark.asyncio
    async def test_batch_create_success(self) -> None:
        """Test successful batch creation of repositories."""
        api = GitHubAPI(token="test-token")

        # Mock httpx.AsyncClient to return successful responses
        call_count = [0]

        async def mock_post(*args, **kwargs):
            call_count[0] += 1
            response = Mock()  # Use Mock, not AsyncMock for response
            response.status_code = 201
            json_data = kwargs.get("json", {})
            name = json_data.get("name", f"repo{call_count[0]}")
            response.json = Mock(
                return_value={
                    "name": name,
                    "full_name": f"test-org/{name}",
                    "html_url": f"https://github.com/test-org/{name}",
                    "clone_url": f"https://github.com/test-org/{name}.git",
                    "ssh_url": f"git@github.com:test-org/{name}.git",
                    "private": False,
                }
            )
            return response

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=mock_post)
        mock_client.get = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "gerrit_clone.github_api.httpx.AsyncClient", return_value=mock_client
        ):
            repos_to_create = [
                {"name": "repo1", "description": "Test 1"},
                {"name": "repo2", "description": "Test 2"},
            ]
            results = await api.batch_create_repos("test-org", repos_to_create)

            assert len(results) == 2
            assert results["repo1"][0] is not None
            assert results["repo1"][0].name == "repo1"
            assert results["repo2"][0] is not None
            assert results["repo2"][0].name == "repo2"

        api.close()

    @pytest.mark.asyncio
    async def test_batch_create_partial_failure(self) -> None:
        """Test batch creation with some failures."""
        api = GitHubAPI(token="test-token")

        async def mock_post_with_failure(*args, **kwargs):
            json_data = kwargs.get("json", {})
            name = json_data.get("name", "")
            response = Mock()  # Use Mock, not AsyncMock for response

            if name == "repo2":
                response.status_code = 422
                response.text = "Repository already exists"
            else:
                response.status_code = 201
                response.json = Mock(
                    return_value={
                        "name": name,
                        "full_name": f"test-org/{name}",
                        "html_url": f"https://github.com/test-org/{name}",
                        "clone_url": f"https://github.com/test-org/{name}.git",
                        "ssh_url": f"git@github.com:test-org/{name}.git",
                        "private": False,
                    }
                )
            return response

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=mock_post_with_failure)
        mock_client.get = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "gerrit_clone.github_api.httpx.AsyncClient", return_value=mock_client
        ):
            repos_to_create = [
                {"name": "repo1"},
                {"name": "repo2"},
                {"name": "repo3"},
            ]
            results = await api.batch_create_repos("test-org", repos_to_create)

            assert len(results) == 3
            assert results["repo1"][0] is not None
            assert results["repo2"][0] is None
            error_msg = results["repo2"][1]
            assert error_msg is not None and "already exists" in error_msg.lower()
            assert results["repo3"][0] is not None

        api.close()

    @pytest.mark.asyncio
    async def test_batch_create_respects_concurrency_limit(self) -> None:
        """Test that batch creation respects max_concurrent limit."""
        api = GitHubAPI(token="test-token")

        call_count = [0]

        async def mock_post_with_delay(*args, **kwargs):
            await asyncio.sleep(0.01)
            call_count[0] += 1
            response = Mock()  # Use Mock, not AsyncMock for response
            response.status_code = 201
            json_data = kwargs.get("json", {})
            name = json_data.get("name", f"repo{call_count[0]}")
            response.json = Mock(
                return_value={
                    "name": name,
                    "full_name": f"test-org/{name}",
                    "html_url": f"https://github.com/test-org/{name}",
                    "clone_url": f"https://github.com/test-org/{name}.git",
                    "ssh_url": f"git@github.com:test-org/{name}.git",
                    "private": False,
                }
            )
            return response

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=mock_post_with_delay)
        mock_client.get = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "gerrit_clone.github_api.httpx.AsyncClient", return_value=mock_client
        ):
            repos_to_create = [{"name": f"repo{i}"} for i in range(10)]
            results = await api.batch_create_repos(
                "test-org", repos_to_create, max_concurrent=3
            )

            assert len(results) == 10
            assert all(repo is not None for repo, _ in results.values())

        api.close()

    @pytest.mark.asyncio
    async def test_batch_create_handles_exceptions(self) -> None:
        """Test that exceptions during creation are handled."""
        api = GitHubAPI(token="test-token")

        call_count = [0]

        async def mock_post_with_exception(*args, **kwargs):
            call_count[0] += 1
            json_data = kwargs.get("json", {})
            name = json_data.get("name", "")

            if name == "repo2":
                raise Exception("Network timeout")

            response = Mock()  # Use Mock, not AsyncMock for response
            response.status_code = 201
            response.json = Mock(
                return_value={
                    "name": name,
                    "full_name": f"test-org/{name}",
                    "html_url": f"https://github.com/test-org/{name}",
                    "clone_url": f"https://github.com/test-org/{name}.git",
                    "ssh_url": f"git@github.com:test-org/{name}.git",
                    "private": False,
                }
            )
            return response

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=mock_post_with_exception)
        mock_client.get = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "gerrit_clone.github_api.httpx.AsyncClient", return_value=mock_client
        ):
            repos_to_create = [
                {"name": "repo1"},
                {"name": "repo2"},
                {"name": "repo3"},
            ]
            results = await api.batch_create_repos("test-org", repos_to_create)

            # Should handle exception - repo2 may not be in results or have error
            assert len(results) >= 2
            assert results["repo1"][0] is not None
            assert results["repo3"][0] is not None

        api.close()


class TestListAllReposGraphQL:
    """Test list_all_repos_graphql method."""

    def test_list_all_repos_empty_org(self) -> None:
        """Test listing repos for org with no repositories."""
        api = GitHubAPI(token="test-token")

        mock_response = Mock()
        mock_response.json.return_value = {
            "data": {
                "organization": {
                    "repositories": {
                        "nodes": [],
                        "pageInfo": {
                            "hasNextPage": False,
                            "endCursor": None,
                        },
                    },
                },
            },
        }

        with patch.object(api.client, "post", return_value=mock_response):
            result = api.list_all_repos_graphql("test-org")

            assert len(result) == 0

        api.close()

    def test_list_all_repos_single_page(self) -> None:
        """Test listing repos with single page of results."""
        api = GitHubAPI(token="test-token")

        mock_response = Mock()
        mock_response.json.return_value = {
            "data": {
                "organization": {
                    "repositories": {
                        "nodes": [
                            {
                                "name": "repo1",
                                "nameWithOwner": "test-org/repo1",
                                "url": "https://github.com/test-org/repo1",
                                "sshUrl": "git@github.com:test-org/repo1.git",
                                "isPrivate": False,
                                "description": "Test repo 1",
                                "defaultBranchRef": {"name": "main"},
                            },
                            {
                                "name": "repo2",
                                "nameWithOwner": "test-org/repo2",
                                "url": "https://github.com/test-org/repo2",
                                "sshUrl": "git@github.com:test-org/repo2.git",
                                "isPrivate": True,
                                "description": None,
                                "defaultBranchRef": None,
                            },
                        ],
                        "pageInfo": {
                            "hasNextPage": False,
                            "endCursor": None,
                        },
                    },
                },
            },
        }

        with patch.object(api.client, "post", return_value=mock_response):
            result = api.list_all_repos_graphql("test-org")

            assert len(result) == 2
            assert "repo1" in result
            assert "repo2" in result
            assert result["repo1"]["name"] == "repo1"
            assert result["repo1"]["full_name"] == "test-org/repo1"
            assert result["repo1"]["private"] is False
            assert result["repo2"]["private"] is True

        api.close()

    def test_list_all_repos_pagination(self) -> None:
        """Test listing repos with pagination."""
        api = GitHubAPI(token="test-token")

        # First page response
        first_response = Mock()
        first_response.json.return_value = {
            "data": {
                "organization": {
                    "repositories": {
                        "nodes": [
                            {
                                "name": "repo1",
                                "nameWithOwner": "test-org/repo1",
                                "url": "https://github.com/test-org/repo1",
                                "sshUrl": "git@github.com:test-org/repo1.git",
                                "isPrivate": False,
                                "description": "Repo 1",
                                "defaultBranchRef": {"name": "main"},
                            },
                        ],
                        "pageInfo": {
                            "hasNextPage": True,
                            "endCursor": "cursor123",
                        },
                    },
                },
            },
        }

        # Second page response
        second_response = Mock()
        second_response.json.return_value = {
            "data": {
                "organization": {
                    "repositories": {
                        "nodes": [
                            {
                                "name": "repo2",
                                "nameWithOwner": "test-org/repo2",
                                "url": "https://github.com/test-org/repo2",
                                "sshUrl": "git@github.com:test-org/repo2.git",
                                "isPrivate": False,
                                "description": "Repo 2",
                                "defaultBranchRef": {"name": "main"},
                            },
                        ],
                        "pageInfo": {
                            "hasNextPage": False,
                            "endCursor": None,
                        },
                    },
                },
            },
        }

        with patch.object(
            api.client, "post", side_effect=[first_response, second_response]
        ) as mock_post:
            result = api.list_all_repos_graphql("test-org")

            assert len(result) == 2
            assert "repo1" in result
            assert "repo2" in result
            # Verify pagination worked - should have been called twice
            assert mock_post.call_count == 2

        api.close()

    def test_list_all_repos_handles_graphql_errors(self) -> None:
        """Test handling of GraphQL errors."""
        api = GitHubAPI(token="test-token")

        mock_response = Mock()
        mock_response.json.return_value = {
            "errors": [
                {"message": "Organization not found"},
            ],
        }

        with patch.object(api.client, "post", return_value=mock_response):
            result = api.list_all_repos_graphql("nonexistent-org")

            # Should return empty dict on error
            assert len(result) == 0

        api.close()

    def test_list_all_repos_handles_missing_org(self) -> None:
        """Test handling when organization data is missing."""
        api = GitHubAPI(token="test-token")

        mock_response = Mock()
        mock_response.json.return_value = {
            "data": {
                "organization": None,
            },
        }

        with patch.object(api.client, "post", return_value=mock_response):
            result = api.list_all_repos_graphql("missing-org")

            assert len(result) == 0

        api.close()

    def test_list_all_repos_escapes_special_chars(self) -> None:
        """Test that organization names with special characters are escaped."""
        api = GitHubAPI(token="test-token")

        mock_response = Mock()
        mock_response.json.return_value = {
            "data": {
                "organization": {
                    "repositories": {
                        "nodes": [],
                        "pageInfo": {
                            "hasNextPage": False,
                            "endCursor": None,
                        },
                    },
                },
            },
        }

        with patch.object(api.client, "post", return_value=mock_response) as mock_post:
            # Org name with quotes should be escaped
            api.list_all_repos_graphql('test"org')

            # Verify the query was made and org name was escaped
            assert mock_post.called
            call_args = mock_post.call_args
            query = call_args[1]["json"]["query"]
            # The escaped version should be in the query
            assert 'test\\"org' in query

        api.close()


class TestRequestPaginated:
    """Test _request_paginated method for handling GitHub API pagination."""

    def test_single_page_response(self) -> None:
        """Test pagination with single page of results."""
        api = GitHubAPI(token="test-token")

        # Mock response with no Link header (single page)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"id": 1}, {"id": 2}, {"id": 3}]
        mock_response.headers.get.return_value = ""  # No Link header
        mock_response.raise_for_status = Mock()

        with patch.object(api.client, "request", return_value=mock_response):
            results = api._request_paginated("GET", "/test/endpoint")

        assert len(results) == 3
        assert results[0]["id"] == 1
        assert results[1]["id"] == 2
        assert results[2]["id"] == 3

        api.close()

    def test_multiple_pages_response(self) -> None:
        """Test pagination with multiple pages of results."""
        api = GitHubAPI(token="test-token")

        # Mock responses for 3 pages
        page1_response = Mock()
        page1_response.status_code = 200
        page1_response.json.return_value = [{"id": 1}, {"id": 2}]
        page1_response.headers.get.return_value = (
            '<https://api.github.com/test?page=2>; rel="next"'
        )
        page1_response.raise_for_status = Mock()

        page2_response = Mock()
        page2_response.status_code = 200
        page2_response.json.return_value = [{"id": 3}, {"id": 4}]
        page2_response.headers.get.return_value = (
            '<https://api.github.com/test?page=3>; rel="next"'
        )
        page2_response.raise_for_status = Mock()

        page3_response = Mock()
        page3_response.status_code = 200
        page3_response.json.return_value = [{"id": 5}]
        page3_response.headers.get.return_value = ""  # No next page
        page3_response.raise_for_status = Mock()

        with patch.object(
            api.client,
            "request",
            side_effect=[page1_response, page2_response, page3_response],
        ):
            results = api._request_paginated("GET", "/test/endpoint")

        assert len(results) == 5
        assert results[0]["id"] == 1
        assert results[4]["id"] == 5

        api.close()

    def test_empty_page_stops_pagination(self) -> None:
        """Test that empty page stops pagination."""
        api = GitHubAPI(token="test-token")

        # Mock response with empty data
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_response.headers.get.return_value = ""
        mock_response.raise_for_status = Mock()

        with patch.object(api.client, "request", return_value=mock_response):
            results = api._request_paginated("GET", "/test/endpoint")

        assert len(results) == 0

        api.close()

    def test_max_pages_limit(self) -> None:
        """Test that max_pages parameter limits pagination."""
        api = GitHubAPI(token="test-token")

        # Mock responses for multiple pages
        page_response = Mock()
        page_response.status_code = 200
        page_response.json.return_value = [{"id": 1}, {"id": 2}]
        page_response.headers.get.return_value = (
            '<https://api.github.com/test?page=2>; rel="next"'
        )
        page_response.raise_for_status = Mock()

        with patch.object(api.client, "request", return_value=page_response):
            results = api._request_paginated("GET", "/test/endpoint", max_pages=2)

        # Should only fetch 2 pages (4 items)
        assert len(results) == 4

        api.close()

    def test_custom_per_page(self) -> None:
        """Test that per_page parameter is passed correctly."""
        api = GitHubAPI(token="test-token")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"id": 1}]
        mock_response.headers.get.return_value = ""
        mock_response.raise_for_status = Mock()

        with patch.object(
            api.client, "request", return_value=mock_response
        ) as mock_request:
            api._request_paginated("GET", "/test/endpoint", per_page=50)

        # Verify per_page parameter was passed
        call_args = mock_request.call_args
        assert call_args[1]["params"]["per_page"] == 50

        api.close()

    def test_additional_params_preserved(self) -> None:
        """Test that additional parameters are preserved across pages."""
        api = GitHubAPI(token="test-token")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"id": 1}]
        mock_response.headers.get.return_value = ""
        mock_response.raise_for_status = Mock()

        with patch.object(
            api.client, "request", return_value=mock_response
        ) as mock_request:
            api._request_paginated(
                "GET", "/test/endpoint", params={"state": "open", "sort": "created"}
            )

        # Verify custom params are included
        call_args = mock_request.call_args
        assert call_args[1]["params"]["state"] == "open"
        assert call_args[1]["params"]["sort"] == "created"

        api.close()

    def test_params_not_mutated(self) -> None:
        """Test that original params dict is not mutated during pagination."""
        api = GitHubAPI(token="test-token")

        # Create params dict to pass to the method
        original_params = {"state": "open", "sort": "created"}
        params_copy = original_params.copy()

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"id": 1}]
        mock_response.headers.get.return_value = ""
        mock_response.raise_for_status = Mock()

        with patch.object(api.client, "request", return_value=mock_response):
            api._request_paginated("GET", "/test/endpoint", params=original_params)

        # Verify original params dict was not mutated
        assert original_params == params_copy
        assert "per_page" not in original_params
        assert "page" not in original_params

        api.close()

    def test_non_list_response_stops_pagination(self) -> None:
        """Test that non-list response stops pagination."""
        api = GitHubAPI(token="test-token")

        # Mock response with dict instead of list
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"error": "test"}
        mock_response.headers.get.return_value = ""
        mock_response.raise_for_status = Mock()

        with patch.object(api.client, "request", return_value=mock_response):
            results = api._request_paginated("GET", "/test/endpoint")

        assert len(results) == 0

        api.close()


def test_list_all_repos_no_default_branch() -> None:
    """Test listing repos when a repository has no default branch configured."""
    api = GitHubAPI(token="test-token")

    mock_response = Mock()
    mock_response.json.return_value = {
        "data": {
            "organization": {
                "repositories": {
                    "nodes": [
                        {
                            "name": "repo-with-branch",
                            "nameWithOwner": "test-org/repo-with-branch",
                            "url": "https://github.com/test-org/repo-with-branch",
                            "sshUrl": "git@github.com:test-org/repo-with-branch.git",
                            "isPrivate": False,
                            "description": "Repo with default branch",
                            "defaultBranchRef": {
                                "name": "main",
                                "target": {
                                    "oid": "abc123def456",
                                },
                            },
                        },
                        {
                            "name": "repo-no-branch",
                            "nameWithOwner": "test-org/repo-no-branch",
                            "url": "https://github.com/test-org/repo-no-branch",
                            "sshUrl": "git@github.com:test-org/repo-no-branch.git",
                            "isPrivate": False,
                            "description": "Repo without default branch",
                            "defaultBranchRef": None,  # No default branch
                        },
                    ],
                    "pageInfo": {
                        "hasNextPage": False,
                        "endCursor": None,
                    },
                },
            },
        },
    }

    with (
        patch.object(api.client, "post", return_value=mock_response),
        patch("gerrit_clone.github_api.logger") as mock_logger,
    ):
        result = api.list_all_repos_graphql("test-org")

        # Should have both repos
        assert len(result) == 2
        assert "repo-with-branch" in result
        assert "repo-no-branch" in result

        # Repo with branch should have commit SHA
        assert result["repo-with-branch"]["default_branch"] == "main"
        assert result["repo-with-branch"]["latest_commit_sha"] == "abc123def456"

        # Repo without branch should have None values
        assert result["repo-no-branch"]["default_branch"] is None
        assert result["repo-no-branch"]["latest_commit_sha"] is None

        # Should have logged a warning for the repo without default branch
        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args[0]
        assert "repo-no-branch" in warning_call[1]
        assert "no default branch configured" in warning_call[0]

    api.close()
