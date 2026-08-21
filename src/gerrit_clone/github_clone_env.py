# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Git process environment for non-interactive GitHub clones."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gerrit_clone.models import Config


def build_git_env(config: Config) -> dict[str, str]:
    """Build environment variables for git commands.

    Sets up the environment for secure, non-interactive git operations:
    - SSH configuration if ssh_identity_file is provided
    - GIT_TERMINAL_PROMPT=0 for HTTPS to prevent interactive prompts
    - Credential helper disabled when using HTTPS with token

    When using HTTPS with a token, the credential helper is explicitly
    disabled to ensure fully automated, non-interactive operation. This
    prevents git from:
    - Prompting for credentials interactively
    - Storing credentials in the system keychain
    - Falling back to other credential helpers

    This is intentional for CI/CD and automation scenarios where the token
    is embedded in the clone URL. If the token is invalid or missing, the
    operation will fail immediately rather than prompting or using cached
    credentials, ensuring predictable behavior.

    Args:
        config: Configuration with optional SSH and HTTPS settings

    Returns:
        Environment dictionary with git-specific variables
    """
    env = os.environ.copy()

    # Add SSH key if provided
    if config.ssh_identity_file:
        ssh_cmd = f"ssh -i {config.ssh_identity_file}"
        if not config.strict_host_checking:
            ssh_cmd += " -o StrictHostKeyChecking=no"
        env["GIT_SSH_COMMAND"] = ssh_cmd
    elif not config.strict_host_checking:
        env["GIT_SSH_COMMAND"] = "ssh -o StrictHostKeyChecking=no"

    # Prevent interactive credential prompts when using HTTPS
    # Token is embedded in URL, so we don't want git asking for credentials
    if config.use_https:
        # Disable interactive prompts (fail fast if auth fails)
        env["GIT_TERMINAL_PROMPT"] = "0"
        # Disable credential helper to prevent:
        # 1. Interactive credential prompts in CI/CD
        # 2. Credential storage in system keychain
        # 3. Fallback to cached/stored credentials
        # This ensures the embedded token is used exclusively and operations
        # fail predictably if the token is invalid/missing
        env["GIT_CONFIG_COUNT"] = "1"
        env["GIT_CONFIG_KEY_0"] = "credential.helper"
        env["GIT_CONFIG_VALUE_0"] = ""

    return env
