# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Logging state shared between a command body and its error handlers.

The file logger, error collector and log file path only exist once logging has
been initialised, but the command's ``except`` blocks must cope with failures
that happen before that.  Collecting them in one object keeps that
partially-initialised state explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import logging
    from pathlib import Path

    from rich.console import Console

    from gerrit_clone.file_logging import ErrorCollector


@dataclass
class CliSession:
    """Console and logging handles for a single command invocation."""

    console: Console
    file_logger: logging.Logger | None = None
    error_collector: ErrorCollector | None = None
    log_file_path: Path | None = None

    def write_summary(self) -> None:
        """Write the collected error summary to the log file, if both exist."""
        if self.error_collector and self.log_file_path:
            self.error_collector.write_summary_to_file(self.log_file_path)
