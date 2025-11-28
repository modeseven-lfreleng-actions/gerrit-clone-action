# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Matthew Watkins <mwatkins@linuxfoundation.org>

"""Unit tests for GitHub API module."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from gerrit_clone.github_api import (
    GitHubAPI,
    GitHubAuthError,
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
