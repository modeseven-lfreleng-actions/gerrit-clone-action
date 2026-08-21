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
from gerrit_clone.content_filter import (
    apply_content_filters,
    is_shallow_repository,
    normalize_file_patterns,
    parse_git_filter_spec,
)
from gerrit_clone.error_codes import DiscoveryError, ExitCode
from gerrit_clone.rich_status import (
    handle_crash_display,
    show_error_summary,
    show_final_results,
)

if TYPE_CHECKING:
    from gerrit_clone.cli_clone_models import CloneRequest
    from gerrit_clone.cli_session import CliSession
    from gerrit_clone.models import BatchResult, CloneResult, Config


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
    _apply_content_filters(request, console, batch_result)
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
    request: CloneRequest, console: Console, batch_result: BatchResult
) -> None:
    """Apply content filters to successfully cloned repositories, if requested."""
    remove_file_patterns = (
        normalize_file_patterns([request.remove_files])
        if request.remove_files
        else None
    )
    git_filter_projects = (
        parse_git_filter_spec(request.git_filter) if request.git_filter else None
    )
    if not (remove_file_patterns or git_filter_projects or request.redact_secrets):
        return

    if not request.quiet:
        console.print("[cyan]🔧 Applying content filters...[/cyan]")
    filter_success = filter_fail = 0
    for cr in batch_result.results:
        if not cr.success or not cr.path:
            continue
        succeeded, failed = _filter_repository(
            request, console, cr, remove_file_patterns, git_filter_projects
        )
        filter_success += succeeded
        filter_fail += failed
    if not request.quiet:
        console.print(
            f"[cyan]Content filtering: {filter_success} succeeded, {filter_fail} failed[/cyan]"
        )
    if filter_fail > 0:
        raise typer.Exit(ExitCode.GENERAL_ERROR)


def _filter_repository(
    request: CloneRequest,
    console: Console,
    result: CloneResult,
    remove_file_patterns: list[str] | None,
    git_filter_projects: dict[str, list[str]] | None,
) -> tuple[int, int]:
    """Filter one cloned repository.

    Returns:
        Tuple of (successes, failures) to add to the run totals
    """
    # Fail closed on shallow repositories when history-
    # dependent filters are requested: --git-filter /
    # --redact-secrets rely on the full commit history, so
    # a shallow repo can hide older secrets (and a later
    # unshallow fetch could reintroduce blocked content),
    # giving a false sense of safety.  Probe each repo
    # individually rather than only checking config.depth:
    # a repo that already existed locally (e.g. cloned
    # earlier with --depth) is shallow even when this run
    # sets no --depth.  --remove-files targets file paths
    # present at the branch tips and does not depend on
    # full history being available (though it may still
    # rewrite history via git filter-repo when that tool
    # is present), so it is still applied; only the unsafe
    # history-scanning filters are dropped for the shallow
    # repo.
    repo_git_filter = git_filter_projects
    repo_redact = request.redact_secrets
    history_filters_skipped = False
    if (git_filter_projects or request.redact_secrets) and (
        is_shallow_repository(result.path)
    ):
        if not request.quiet:
            console.print(
                "[red]❌ Refusing to run --git-filter / "
                "--redact-secrets on shallow repo "
                f"{result.project.name}: truncated history can hide "
                "older secrets. Re-clone without --depth.[/red]"
            )
        # The requested redaction/rewrite did not run, so
        # this repo counts as a filtering failure even if
        # the safe --remove-files step below succeeds.
        history_filters_skipped = True
        repo_git_filter = None
        repo_redact = False
        if not remove_file_patterns:
            # Nothing history-independent left to do.
            return 0, 1
    success, error = apply_content_filters(
        result.path,
        result.project.name,
        remove_patterns=remove_file_patterns,
        git_filter_projects=repo_git_filter,
        redact_secrets=repo_redact,
        timeout=request.clone_timeout,
    )
    if history_filters_skipped:
        # Already counted as a failure above; don't also count
        # the safe --remove-files step as a success.
        if not success and not request.quiet:
            console.print(
                f"[yellow]⚠️  Filter failed for {result.project.name}: {error}[/yellow]"
            )
        return 0, 1
    if success:
        return 1, 0
    if not request.quiet:
        console.print(
            f"[yellow]⚠️  Filter failed for {result.project.name}: {error}[/yellow]"
        )
    return 0, 1


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
