# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Git execution environment and remote handling for refresh operations.

Bottom layer of the :class:`~gerrit_clone.refresh_worker.RefreshWorker` mixin
stack. It answers three related questions about *how* we talk to a remote:
which URL is configured, whether that URL implies an SSH handshake (and so
needs pacing), and what environment git subprocesses should run with.

Every higher layer that performs a network operation depends on the handshake
jitter provided here.
"""

from __future__ import annotations

import os
import random
import re
import subprocess
import time
from typing import TYPE_CHECKING

from gerrit_clone.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from gerrit_clone.models import Config

logger = get_logger(__name__)

# Maximum random delay (seconds) inserted before each SSH-backed git network
# operation. Spreading handshakes across a small window de-synchronises worker
# threads so we do not open many simultaneous SSH connections to Gerrit, which
# is a common cause of transient "Could not read from remote repository"
# throttling failures.
SSH_HANDSHAKE_JITTER_SECONDS = 0.25


class GitEnvironmentMixin:
    """Remote-URL inspection and git subprocess environment construction."""

    # Supplied by RefreshWorker.__init__; declared here because this layer
    # reads them.
    config: Config | None
    ssh_jitter_seconds: float

    def _get_remote_url(self, repo_path: Path) -> str | None:
        """Get the remote URL for the repository.

        Args:
            repo_path: Repository path

        Returns:
            Remote URL or None if not found
        """
        try:
            result = subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
                check=False,
            )

            if result.returncode == 0:
                return result.stdout.strip()
            return None

        except Exception as e:
            logger.debug(f"Failed to get remote URL: {e}")
            return None

    def _is_gerrit_repository(self, remote_url: str | None) -> bool:
        """Check if remote URL looks like a Gerrit repository.

        Args:
            remote_url: Remote URL to check

        Returns:
            True if URL looks like Gerrit
        """
        if not remote_url:
            return False

        # Gerrit-specific patterns
        gerrit_patterns = [
            r"ssh://.*:\d+/",  # SSH with port (typical Gerrit: ssh://host:29418/project)
            r"https?://.*/r/",  # HTTPS with /r/ prefix
            r"https?://.*/gerrit/",  # HTTPS with /gerrit/ prefix
        ]

        for pattern in gerrit_patterns:
            if re.search(pattern, remote_url):
                return True

        # Additional check: Gerrit servers often have specific hostnames
        gerrit_hosts = ["gerrit", "review", "code-review"]
        return any(host in remote_url.lower() for host in gerrit_hosts)

    @staticmethod
    def _remote_uses_ssh(remote_url: str | None) -> bool:
        """Return True if the origin remote performs an SSH handshake.

        Only SSH-backed remotes benefit from handshake jitter. HTTP(S), the
        anonymous git protocol, ``file://`` URLs and local filesystem paths
        never open an SSH connection, so jittering them just adds latency. An
        unknown/empty remote is treated as SSH so the throttling protection is
        preserved when the URL cannot be read.

        Args:
            remote_url: The origin remote URL, or None if it is unknown.

        Returns:
            True if a handshake (and therefore jitter) is warranted.
        """
        if not remote_url:
            return True
        url = remote_url.strip()
        lowered = url.lower()
        if lowered.startswith("ssh://"):
            return True
        # Non-SSH transports and local paths never open an SSH handshake.
        if lowered.startswith(("http://", "https://", "git://", "file://")):
            return False
        if url.startswith(("/", "./", "../", "~")):
            return False
        # scp-like syntax (``[user@]host:path``) is SSH. git only recognises it
        # when a colon appears before the first slash; a colon after a slash
        # (or no colon at all) denotes a local filesystem path.
        colon = url.find(":")
        slash = url.find("/")
        return colon != -1 and (slash == -1 or colon < slash)

    def _ssh_handshake_jitter(self, repo_path: Path) -> None:
        """Sleep a small random interval before an SSH-backed git operation.

        De-synchronises concurrent worker threads so we avoid opening many
        simultaneous SSH connections to Gerrit, which is a common cause of
        transient "Could not read from remote repository" throttling. The
        sleep is skipped for HTTP(S)/git-protocol remotes, which perform no
        SSH handshake and so gain nothing from jitter.

        Args:
            repo_path: Repository whose origin remote is about to be contacted.
        """
        if self.ssh_jitter_seconds <= 0:
            return
        if not self._remote_uses_ssh(self._get_remote_url(repo_path)):
            return
        time.sleep(random.uniform(0, self.ssh_jitter_seconds))

    def _build_git_environment(self) -> dict[str, str]:
        """Build environment for Git operations.

        Returns:
            Environment dictionary
        """
        env = os.environ.copy()

        # Add Git SSH command if config is provided, otherwise use safe defaults
        if self.config and self.config.git_ssh_command:
            env["GIT_SSH_COMMAND"] = self.config.git_ssh_command
        else:
            # SSH Configuration Trade-offs:
            #
            # We explicitly disable SSH multiplexing (ControlMaster=no) for thread safety.
            # This prevents race conditions when multiple threads connect to the same host
            # simultaneously, which can cause:
            # - Socket file conflicts in ~/.ssh/
            # - Connection hangs or failures
            # - Unpredictable behavior in parallel operations
            #
            # PERFORMANCE TRADE-OFF:
            # Disabling multiplexing means each git operation requires a new SSH handshake,
            # adding ~100-500ms latency per operation. However, in practice:
            # - Most operations are I/O bound (git fetch/pull), not connection-bound
            # - Parallel execution across multiple repos still provides significant speedup
            # - The reliability gain outweighs the connection overhead
            # - Real-world testing shows acceptable performance for typical use cases
            #
            # Alternative approaches considered:
            # - Connection pooling: Complex to implement, would require shared state
            # - Single-threaded SSH: Eliminates parallelism benefits entirely
            # - Master socket per thread: Still has filesystem race conditions
            #
            # Current configuration prioritizes reliability and thread safety over
            # optimal SSH connection reuse. If performance becomes an issue, consider:
            # - Using HTTPS instead of SSH (no connection multiplexing issues)
            # - Increasing thread count to compensate for per-connection overhead
            # - Custom connection pooling implementation (significant complexity)
            ssh_opts = [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ControlMaster=no",  # Disable multiplexing for thread safety
                "-o",
                "ConnectTimeout=10",
                "-o",
                "ServerAliveInterval=5",
                "-o",
                "ServerAliveCountMax=3",
                "-o",
                "ConnectionAttempts=2",
                "-o",
                "StrictHostKeyChecking=accept-new",
            ]
            env["GIT_SSH_COMMAND"] = " ".join(ssh_opts)

        # Disable terminal prompts
        env["GIT_TERMINAL_PROMPT"] = "0"

        return env
