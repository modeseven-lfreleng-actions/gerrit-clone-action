# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""GraphQL transport for bulk organization repository listing.

Paginating an organization's repositories over the REST API costs one
request per hundred repositories; the GraphQL endpoint returns the same
data far more cheaply.  This module owns the query text, the per-page
request, and the retry policy for the transient errors (502/503/429 and
cold-cache timeouts) that GitHub returns for large organizations.
"""

from __future__ import annotations

import time as time_mod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rich.console import Console

from gerrit_clone.logging import get_logger

if TYPE_CHECKING:
    import httpx

    from gerrit_clone.rate_limit import RateLimitBudget

logger = get_logger(__name__)

GRAPHQL_URL = "https://api.github.com/graphql"

# HTTP statuses that indicate a transient GitHub failure worth retrying.
TRANSIENT_STATUS_CODES = (502, 503, 429)


@dataclass(frozen=True)
class GraphQLRepoPage:
    """A single page of organization repositories.

    Attributes:
        nodes: Raw repository nodes returned by the query
        has_next_page: Whether a further page is available
        end_cursor: Cursor to pass to the next request
        organization_missing: True when the query returned no
            organization object at all, in which case the caller must
            stop paginating without treating the page as a failure
    """

    nodes: list[dict[str, Any]]
    has_next_page: bool
    end_cursor: str | None
    organization_missing: bool = False


def build_org_repos_query(org: str, cursor: str | None) -> str:
    """Build the GraphQL query for one page of organization repositories.

    Args:
        org: Organization name
        cursor: Cursor returned by the previous page, if any

    Returns:
        GraphQL query text with the org name and cursor escaped
    """
    safe_org = org.replace('"', '\\"')
    safe_cursor = cursor.replace('"', '\\"') if cursor else None
    after_clause = f', after: "{safe_cursor}"' if safe_cursor else ""
    return f"""
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
                          committedDate
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


def _retry_backoff(retry: int) -> int:
    """Return the backoff for a retry attempt.

    Starts at 5 s so a cold-cache 502 has time to warm before we retry.

    Args:
        retry: Zero-based retry counter

    Returns:
        Seconds to wait before the next attempt
    """
    # Annotated because `int ** int` is typed as Any (a negative
    # exponent yields a float), which would otherwise leak out of the
    # declared int return.
    backoff: int = min(30, 5 * (2**retry))
    return backoff


def _pause_before_retry(
    reason: str,
    org: str,
    backoff: int,
    retry: int,
    max_retries: int,
    budget: RateLimitBudget,
) -> None:
    """Print a compact console warning and sleep before retrying.

    Args:
        reason: Short description of the failure, e.g. "GraphQL error"
        org: Organization name
        backoff: Seconds to wait
        retry: Zero-based retry counter
        max_retries: Retries allowed per page
        budget: Shared rate-limit budget tracker
    """
    snap = budget.snapshot
    budget_pct = f"{snap.budget_fraction:.1%}" if snap.limit > 0 else "unknown"
    Console(stderr=True).print(
        f"[yellow]⚠️ {reason} "
        f"for GitHub Organisation: "
        f"{org}, retrying in {backoff}s "
        f"({retry + 1}/{max_retries}) "
        f"[Budget remaining: "
        f"{budget_pct}][/yellow]"
    )
    time_mod.sleep(backoff)


def _retry_transient_status(
    status_code: int,
    org: str,
    retry: int,
    max_retries: int,
    budget: RateLimitBudget,
) -> bool:
    """Handle a transient HTTP status, returning whether to retry.

    Args:
        status_code: HTTP status returned by GitHub
        org: Organization name
        retry: Zero-based retry counter
        max_retries: Retries allowed per page
        budget: Shared rate-limit budget tracker

    Returns:
        True if the caller should retry the page
    """
    if retry >= max_retries:
        logger.error(
            "GraphQL failed after %d retries (HTTP %d) for %s",
            max_retries,
            status_code,
            org,
        )
        return False

    backoff = _retry_backoff(retry)
    # Detailed message for file log only
    logger.info(
        "GraphQL transient error %d for %s, retrying in %ds (%d/%d)",
        status_code,
        org,
        backoff,
        retry + 1,
        max_retries,
    )
    # Compact one-liner for the console
    _pause_before_retry(
        f"GraphQL transient error {status_code}",
        org,
        backoff,
        retry,
        max_retries,
        budget,
    )
    return True


def _retry_graphql_errors(
    errors: Any,
    org: str,
    retry: int,
    max_retries: int,
    budget: RateLimitBudget,
) -> bool:
    """Handle a GraphQL-level error payload, returning whether to retry.

    Args:
        errors: The ``errors`` array from the GraphQL response
        org: Organization name
        retry: Zero-based retry counter
        max_retries: Retries allowed per page
        budget: Shared rate-limit budget tracker

    Returns:
        True if the caller should retry the page
    """
    # Full error detail for file log only
    logger.info("GraphQL errors: %s", errors)
    if retry >= max_retries:
        return False

    backoff = _retry_backoff(retry)
    _pause_before_retry("GraphQL error", org, backoff, retry, max_retries, budget)
    return True


def _retry_after_exception(
    error: Exception,
    org: str,
    retry: int,
    max_retries: int,
    budget: RateLimitBudget,
) -> bool:
    """Handle an unexpected exception, returning whether to retry.

    Args:
        error: The exception raised while fetching or parsing the page
        org: Organization name
        retry: Zero-based retry counter
        max_retries: Retries allowed per page
        budget: Shared rate-limit budget tracker

    Returns:
        True if the caller should retry the page
    """
    if retry >= max_retries:
        logger.error(
            "GraphQL query failed after %d retries: %s",
            max_retries,
            error,
        )
        return False

    backoff = _retry_backoff(retry)
    # Full traceback for file log only
    logger.info(
        "GraphQL query failed: %s (retrying in %ds, %d/%d)",
        error,
        backoff,
        retry + 1,
        max_retries,
    )
    _pause_before_retry(
        "GraphQL query failed", org, backoff, retry, max_retries, budget
    )
    return True


def _extract_repo_page(data: dict[str, Any], org: str) -> GraphQLRepoPage:
    """Turn a successful GraphQL response into a page of repositories.

    Args:
        data: Decoded GraphQL response body
        org: Organization name, for the missing-organization warning

    Returns:
        The page of repository nodes and its pagination state
    """
    data_payload = data.get("data")
    org_data = data_payload.get("organization") if data_payload else None
    if not org_data:
        logger.warning(f"No organization data for {org}")
        return GraphQLRepoPage(
            nodes=[],
            has_next_page=False,
            end_cursor=None,
            organization_missing=True,
        )

    repos_data = org_data.get("repositories", {})
    page_info = repos_data.get("pageInfo", {})
    return GraphQLRepoPage(
        nodes=repos_data.get("nodes", []),
        has_next_page=page_info.get("hasNextPage", False),
        end_cursor=page_info.get("endCursor"),
    )


def fetch_org_repos_page(
    client: httpx.Client,
    budget: RateLimitBudget,
    org: str,
    cursor: str | None = None,
    max_retries: int = 3,
) -> GraphQLRepoPage | None:
    """Fetch one page of an organization's repositories over GraphQL.

    Retries transient errors so that a single failed page does not
    cascade into an empty result for the whole organization.

    Args:
        client: Shared authenticated HTTP client
        budget: Shared rate-limit budget tracker
        org: Organization name
        cursor: Cursor returned by the previous page, if any
        max_retries: Retries per page on transient errors

    Returns:
        The page, or None if it failed after exhausting the retries
    """
    query = build_org_repos_query(org, cursor)
    logger.debug(
        "GraphQL query for %s repos (cursor: %s)",
        org,
        cursor,
    )

    for retry in range(max_retries + 1):
        try:
            # Use a longer timeout for GraphQL than the default 30 s
            # client timeout.  GitHub's first query against a cold org
            # cache regularly exceeds 30 s, causing nginx to return 502
            # before the backend finishes.
            response = client.post(
                GRAPHQL_URL,
                json={"query": query},
                timeout=60.0,
            )

            # Record rate-limit headers from GraphQL too
            budget.update_from_headers_sync(response.headers)

            # Transient HTTP errors → retry
            if response.status_code in TRANSIENT_STATUS_CODES:
                if _retry_transient_status(
                    response.status_code, org, retry, max_retries, budget
                ):
                    continue
                return None

            response.raise_for_status()
            data = response.json()

            if "errors" in data:
                if _retry_graphql_errors(
                    data["errors"], org, retry, max_retries, budget
                ):
                    continue
                return None

            return _extract_repo_page(data, org)

        except Exception as e:
            if _retry_after_exception(e, org, retry, max_retries, budget):
                continue
            return None

    return None
