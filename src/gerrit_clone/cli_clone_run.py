# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Execution pipeline and failure handling for the ``clone`` command.

Runs the prepared request (cloning, content filtering, result reporting and
optional cleanup) and owns the interrupt/crash handlers used by the command.
"""

from __future__ import annotations

import traceback
from shutil import rmtree
from typing import TYPE_CHECKING, NoReturn

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from gerrit_clone import __version__, cli_hooks
from gerrit_clone import cli_clone_setup as setup
from gerrit_clone.cli_app import is_github_actions_context
from gerrit_clone.content_stage import filter_repository
from gerrit_clone.error_codes import DiscoveryError, ExitCode
from gerrit_clone.rich_status import (
    handle_crash_display,
    show_error_summary,
    show_final_results,
)

if TYPE_CHECKING:
    from gerrit_clone.cli_clone_models import CloneRequest
    from gerrit_clone.cli_session import CliSession
    from gerrit_clone.content_spec import ContentFilterSpec
    from gerrit_clone.models import BatchResult, Config


def run_clone(request: CloneRequest, session: CliSession) -> None:
    """Run the clone pipeline for an already parsed command line."""
    console = session.console
    source_type, github_org = setup.resolve_source(request, console)
    cli_args = setup.build_cli_args(request, source_type, github_org)
    file_logger, error_collector, log_file_path = setup.start_logging(request, cli_args)
    session.file_logger = file_logger
    session.error_collector = error_collector
    session.log_file_path = log_file_path

    if request.use_https:
        setup.apply_http_credentials(request, console, file_logger)

    # Log version to file in GitHub Actions environment (file only, no console)
    if is_github_actions_context():
        try:
            file_logger.debug("gerrit-clone version %s", __version__)
        except Exception:
            file_logger.warning("Version information not available")

    discovery_method = setup.resolve_discovery_method(request, console, source_type)
    config = setup.build_config(
        request, session, source_type, github_org, discovery_method
    )

    # Show startup banner if not quiet
    if not request.quiet:
        setup.show_startup_banner(console, config)

    batch_result = _clone_repositories(config)
    _apply_content_filters(request, console, batch_result, config.content_filters)
    _report_results(request, session, batch_result)
    exit_code = _determine_exit_code(session, batch_result)

    # Optional cleanup
    if request.cleanup:
        _cleanup_clone_directory(session, config)

    # Close file logging and write summary
    session.write_summary()

    if exit_code != 0:
        raise typer.Exit(exit_code)


def _clone_repositories(config: Config) -> BatchResult:
    """Clone every discovered repository, reporting discovery failures."""
    try:
        return cli_hooks.clone_repositories(config)
    except DiscoveryError as e:
        console = Console()
        console.print(
            Panel(
                Text(
                    f"{e.message}\n{e.details}" if e.details else str(e.message),
                    style="bold red",
                ),
                title="Discovery Error",
                border_style="red",
            )
        )
        raise typer.Exit(ExitCode.DISCOVERY_ERROR) from e


def _apply_content_filters(
    request: CloneRequest,
    console: Console,
    batch_result: BatchResult,
    spec: ContentFilterSpec | None,
) -> None:
    """Filter every successful clone, as each project's filters decide."""
    if spec is None:
        return

    if not request.quiet:
        console.print("[cyan]🔧 Applying content filters...[/cyan]")
    filter_success = filter_fail = 0
    for cr in batch_result.results:
        if cr.skipped and cr.path and spec.missing_tokens(cr.path):
            # The refresh pass refused it for the tokens, before any
            # fetch.  Clone reports that as the filtering failure it is,
            # whether or not upstream had moved on.
            filter_fail += 1
            if not request.quiet:
                console.print(
                    f"[yellow]⚠️  Filter failed for {cr.project.name}: "
                    f"{cr.error_message}[/yellow]"
                )
            continue
        if not cr.success or not cr.path:
            continue
        if cr.content_filtered:
            # Re-filtered as part of its staged refresh already.
            filter_success += 1
            continue
        reason = filter_repository(
            spec, cr.path, spec.project_name(cr.path), request.clone_timeout
        )
        if reason is None:
            filter_success += 1
            continue
        filter_fail += 1
        if not request.quiet:
            console.print(
                f"[yellow]⚠️  Filter failed for {cr.project.name}: {reason}[/yellow]"
            )
    if not request.quiet:
        console.print(
            f"[cyan]Content filtering: {filter_success} succeeded, {filter_fail} failed[/cyan]"
        )
    if filter_fail > 0:
        raise typer.Exit(ExitCode.GENERAL_ERROR)


def _report_results(
    request: CloneRequest, session: CliSession, batch_result: BatchResult
) -> None:
    """Show the final results summary and any collected errors."""
    console = session.console
    log_file_path = session.log_file_path

    # Show final results summary using Rich
    if not request.quiet:
        show_final_results(
            console, batch_result, str(log_file_path) if log_file_path else None
        )

    # Show error summary if there were issues
    error_collector = session.error_collector
    if error_collector and not request.quiet:
        errors = [
            record.message
            for record in error_collector.errors + error_collector.critical_errors
        ]
        warnings = [record.message for record in error_collector.warnings]
        if errors or warnings:
            show_error_summary(console, errors, warnings)


def _determine_exit_code(session: CliSession, batch_result: BatchResult) -> int:
    """Determine exit code based on results."""
    file_logger = session.file_logger
    if batch_result.failed_count > 0:
        if file_logger:
            file_logger.debug(
                "Clone completed with %d failures", batch_result.failed_count
            )
        return int(ExitCode.CLONE_ERROR)
    if file_logger:
        file_logger.debug("Clone completed successfully")
    return int(ExitCode.SUCCESS)


def _cleanup_clone_directory(session: CliSession, config: Config) -> None:
    """Remove the cloned directory once the run has finished."""
    file_logger = session.file_logger
    try:
        if file_logger:
            file_logger.debug(
                "Cleanup enabled - removing cloned directory: %s",
                config.path,
            )
        session.console.print(
            f"[yellow]🧹 Cleanup enabled - removing cloned directory: {config.path}[/yellow]"
        )
        rmtree(config.path, ignore_errors=True)
        if file_logger:
            file_logger.debug("Cleanup completed successfully")
        session.console.print("[green]Cleanup complete.[/green]")
    except Exception as e:
        if file_logger:
            file_logger.debug("Cleanup failed: %s", str(e))
        session.console.print(f"[red]Cleanup failed:[/red] {e}")


def handle_interrupt(session: CliSession) -> NoReturn:
    """Report a user interrupt and exit."""
    if session.file_logger:
        session.file_logger.warning("Operation cancelled by user (KeyboardInterrupt)")
    session.write_summary()
    session.console.print("\n[yellow]Operation cancelled by user[/yellow]")
    # Flush console to ensure message is displayed before exit
    if hasattr(session.console.file, "flush"):
        session.console.file.flush()
    raise typer.Exit(int(ExitCode.INTERRUPT)) from None


def handle_crash(session: CliSession, error: Exception, *, verbose: bool) -> NoReturn:
    """Record an unexpected failure, display it, and exit."""
    tb = traceback.extract_tb(error.__traceback__)
    crash_context = "unknown"
    crash_file = "unknown"
    crash_line = 0

    if tb:
        # Get the last frame (where the crash occurred)
        last_frame = tb[-1]
        crash_file = last_frame.filename
        crash_line = last_frame.lineno or 0
        crash_context = (
            f"{last_frame.name}() at {crash_file.split('/')[-1]}:{crash_line}"
        )

    if session.file_logger:
        session.file_logger.critical(
            "Tool crashed in %s: %s", crash_context, str(error), exc_info=True
        )
    if session.error_collector:
        session.error_collector.add_critical_error(
            f"Tool crashed: {type(error).__name__}: {error!s}",
            context=f"function: {crash_context}",
            exception=error,
        )
    session.write_summary()

    # Use Rich status system for crash display
    log_file_path = session.log_file_path
    handle_crash_display(
        session.console, error, str(log_file_path) if log_file_path else None
    )

    if verbose:
        session.console.print_exception()
    raise typer.Exit(ExitCode.GENERAL_ERROR) from None
