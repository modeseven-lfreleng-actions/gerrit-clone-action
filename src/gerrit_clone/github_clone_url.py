# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Selection and redaction of the URL used to clone a GitHub repository.

Chooses between SSH and HTTPS (optionally embedding a token) and
produces a credential-free variant of the URL for logging.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlparse, urlunparse

from gerrit_clone.logging import get_logger

if TYPE_CHECKING:
    from gerrit_clone.models import Config, Project

logger = get_logger(__name__)


def resolve_clone_url(project: Project, config: Config) -> str:
    """Determine the URL to clone from - prefer SSH, fall back to HTTPS.

    When HTTPS is explicitly requested and a token is configured, the
    token is embedded in the URL so the clone can authenticate without
    an interactive prompt.

    Args:
        project: Project to clone
        config: Configuration with optional github_token

    Returns:
        Clone URL, possibly containing an embedded token.
    """
    if config.use_https:
        # Explicit HTTPS requested
        clone_url = project.clone_url or project.https_url(config.base_url)

        # Embed GitHub token in URL for authentication if provided
        if config.github_token and clone_url.startswith("https://"):
            # Insert token into URL: https://token@github.com/org/repo.git
            clone_url = clone_url.replace(
                "https://", f"https://{config.github_token}@", 1
            )
            logger.debug(
                f"Cloning {project.name} with HTTPS using token authentication"
            )
        else:
            logger.debug(
                f"Cloning {project.name} with HTTPS (no token, will use credential helper)"
            )
        return clone_url

    if project.ssh_url_override:
        # SSH URL available from GitHub (preferred)
        return project.ssh_url_override

    # Fall back to HTTPS if no SSH URL available
    return project.clone_url or project.https_url(config.base_url)


def redact_clone_url(clone_url: str, project: Project, github_token: str | None) -> str:
    """Return *clone_url* with any embedded token replaced by ``***``.

    The URL is parsed and reconstructed rather than string-replaced to
    avoid issues with special characters in the token.

    Args:
        clone_url: URL that may contain embedded credentials
        project: Project being cloned, used for the safe placeholder
        github_token: Token to look for, if one is configured

    Returns:
        A URL that is safe to log.
    """
    if not github_token:
        return clone_url

    try:
        parsed = urlparse(clone_url)
        # Check if token is in the netloc (e.g., token@github.com)
        if "@" in parsed.netloc and github_token in parsed.netloc:
            # Reconstruct netloc with redacted token
            netloc_parts = parsed.netloc.split("@", 1)
            redacted_netloc = f"***@{netloc_parts[1]}"
            return urlunparse(
                (
                    parsed.scheme,
                    redacted_netloc,
                    parsed.path,
                    parsed.params,
                    parsed.query,
                    parsed.fragment,
                )
            )
    except Exception:
        # SECURITY: If parsing fails, use safe placeholder to avoid credential leak
        return f"https://***@github.com/{project.name}.git"

    return clone_url
