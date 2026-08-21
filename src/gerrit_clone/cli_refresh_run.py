# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""Execution pipeline for the ``refresh`` command.

Validates the parsed command line, refreshes the repositories found beneath
the output path, applies any requested content filters and reports results.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import typer

from gerrit_clone.cli_app import format_version_string
from gerrit_clone.cli_refresh_report import (
    show_refresh_results,
    write_refresh_manifest,
)
from gerrit_clone.content_filter import (
    apply_content_filters,
    is_shallow_repository,
    normalize_file_patterns,
    parse_git_filter_spec,
)
from gerrit_clone.error_codes import ExitCode
from gerrit_clone.file_logging import (
    cli_args_to_dict,
    get_default_log_path,
    init_logging,
)
from gerrit_clone.models import normalize_project_list
from gerrit_clone.refresh_manager import refresh_repositories

if TYPE_CHECKING:
    from pathlib import Path

    from rich.console import Console

    from gerrit_clone.models import RefreshBatchResult, RefreshResult


@dataclass(frozen=True)
class RefreshRequest:
    """Command line arguments supplied to the ``refresh`` command."""

    output_path: Path
    include_projects: list[str] | None
    exclude_projects: list[str] | None
    threads: int | None
    fetch_only: bool
    prune: bool
    timeout: int
    skip_conflicts: bool
    auto_stash: bool
    strategy: str
    filter_gerrit_only: bool
    exit_on_error: bool
    dry_run: bool
    force: bool
    force_hard: bool
    recursive: bool
    verbose: bool
    quiet: bool
    manifest_filename: str | None
    remove_files: str | None
    git_filter: str | None
    redact_secrets: bool


def prepare_refresh(request: RefreshRequest, console: Console) -> None:
    """Validate the request and set up logging before any repository work."""
    _validate_request(request, console)

    # Display version
    if not request.quiet:
        console.print(format_version_string("refresh"))
        console.print()

    _start_logging(request, console)

    # Display configuration summary
    if not request.quiet:
        _show_configuration(request, console)


def run_refresh(request: RefreshRequest, console: Console) -> None:
    """Refresh the repositories found beneath the requested output path."""
    result = refresh_repositories(
        base_path=request.output_path,
        config=None,
        timeout=request.timeout,
        fetch_only=request.fetch_only,
        prune=request.prune,
        skip_conflicts=request.skip_conflicts,
        auto_stash=request.auto_stash,
        strategy=request.strategy,
        filter_gerrit_only=request.filter_gerrit_only,
        threads=request.threads,
        exit_on_error=request.exit_on_error,
        dry_run=request.dry_run,
        force=request.force,
        force_hard=request.force_hard,
        recursive=request.recursive,
        include_projects=request.include_projects if request.include_projects else None,
        exclude_projects=request.exclude_projects if request.exclude_projects else None,
    )

    _apply_content_filters(request, console, result)

    # Display results
    show_refresh_results(console, result, request.dry_run)

    _write_manifest(request, console, result)
    _exit_with_result_status(request, console, result)


def _validate_request(request: RefreshRequest, console: Console) -> None:
    """Reject mutually exclusive or unsupported option combinations."""
    if request.verbose and request.quiet:
        console.print("[red]Error:[/red] --verbose and --quiet cannot be used together")
        raise typer.Exit(ExitCode.CONFIGURATION_ERROR)

    if request.strategy not in ("merge", "rebase"):
        console.print(
            f"[red]❌ Invalid pull strategy: {request.strategy}. Must be 'merge' or 'rebase'.[/red]"
        )
        raise typer.Exit(ExitCode.VALIDATION_ERROR.value)


def _start_logging(request: RefreshRequest, console: Console) -> None:
    """Initialise file and console logging for the refresh run."""
    cli_args = cli_args_to_dict(**vars(request))

    log_file_path = get_default_log_path("refresh", request.output_path)

    init_logging(
        log_file=log_file_path,
        disable_file=False,
        log_level="DEBUG",
        console_level="DEBUG" if request.verbose else "WARNING",
        quiet=request.quiet,
        verbose=request.verbose,
        cli_args=cli_args,
        host=None,
        command="refresh",
    )

    if log_file_path and request.verbose:
        console.print(f"📝 Logging to: [cyan]{log_file_path}[/cyan]")
        console.print()


def _show_configuration(request: RefreshRequest, console: Console) -> None:
    """Print the effective refresh configuration."""
    console.print("[bold blue]Refresh Configuration[/bold blue]")
    console.print(f"Base Path: [cyan]{request.output_path}[/cyan]")
    console.print(f"Threads: [cyan]{request.threads or 'auto-detect'}[/cyan]")
    console.print(
        f"Mode: [cyan]{'Fetch Only' if request.fetch_only else f'Pull ({request.strategy})'}[/cyan]"
    )
    console.print(f"Prune: [cyan]{request.prune}[/cyan]")
    console.print(f"Timeout: [cyan]{request.timeout}s[/cyan]")
    console.print(f"Skip Conflicts: [cyan]{request.skip_conflicts}[/cyan]")
    console.print(f"Auto Stash: [cyan]{request.auto_stash}[/cyan]")
    console.print(
        f"Filter: [cyan]{'Gerrit only' if request.filter_gerrit_only else 'All repos'}[/cyan]"
    )
    inc_display = (
        normalize_project_list(list(request.include_projects))
        if request.include_projects
        else []
    )
    exc_display = (
        normalize_project_list(list(request.exclude_projects))
        if request.exclude_projects
        else []
    )
    console.print(
        f"Include Filter: [cyan]{', '.join(inc_display) if inc_display else '—'}[/cyan]"
    )
    console.print(
        f"Exclude Filter: [cyan]{', '.join(exc_display) if exc_display else '—'}[/cyan]"
    )
    console.print(f"Dry Run: [cyan]{request.dry_run}[/cyan]")
    # A dry run disables all modifications (see RefreshManager), so report
    # the *effective* force settings rather than the user-supplied flags to
    # avoid implying a destructive run when nothing will be changed.
    effective_force = (request.force or request.force_hard) and not request.dry_run
    effective_force_hard = request.force_hard and not request.dry_run
    console.print(f"Force: [cyan]{effective_force}[/cyan]")
    console.print(f"Force Hard: [cyan]{effective_force_hard}[/cyan]")
    console.print(f"Recursive: [cyan]{request.recursive}[/cyan]")
    console.print()


def _apply_content_filters(
    request: RefreshRequest, console: Console, result: RefreshBatchResult
) -> None:
    """Apply content filters to cleanly refreshed repositories, if requested."""
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
    for rr in result.results:
        # Only apply content filters to repositories that
        # refreshed cleanly (SUCCESS / UP_TO_DATE).  Skipping
        # merely FAILED results is not enough: statuses like
        # SKIPPED, CONFLICTS, NOT_GIT_REPO, NOT_GERRIT_REPO,
        # UNCOMMITTED_CHANGES and DETACHED_HEAD also leave the
        # worktree in a state where rewriting history or
        # removing files would be unsafe or raise.  The
        # ``RefreshResult.success`` property captures exactly
        # the SUCCESS / UP_TO_DATE set.
        if not rr.success:
            continue
        succeeded, failed = _filter_repository(
            request, console, rr, remove_file_patterns, git_filter_projects
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
    request: RefreshRequest,
    console: Console,
    result: RefreshResult,
    remove_file_patterns: list[str] | None,
    git_filter_projects: dict[str, list[str]] | None,
) -> tuple[int, int]:
    """Filter one refreshed repository.

    Returns:
        Tuple of (successes, failures) to add to the run totals
    """
    # Fail closed on shallow repositories when history-
    # dependent filters are requested: --git-filter /
    # --redact-secrets rely on full history, so a shallow
    # repo can hide older secrets (and a later unshallow
    # fetch could reintroduce blocked content), giving a
    # false sense of safety.  --remove-files targets file
    # paths present at the branch tips and does not depend
    # on full history being available (though it may still
    # rewrite history via git filter-repo when that tool is
    # present), so it is still applied below — we only drop
    # the unsafe history-scanning filters for this repo.
    # (clone applies the same guard up-front via
    # config.depth; refresh operates on pre-existing local
    # repos, so it must probe each repo for shallowness.)
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
                f"{result.project_name}: truncated history can hide "
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
        result.project_name,
        remove_patterns=remove_file_patterns,
        git_filter_projects=repo_git_filter,
        redact_secrets=repo_redact,
        timeout=request.timeout,
    )
    if history_filters_skipped:
        # Already counted as a failure above; don't also count
        # the safe --remove-files step as a success.
        if not success and not request.quiet:
            console.print(
                f"[yellow]⚠️  Filter failed for {result.project_name}: {error}[/yellow]"
            )
        return 0, 1
    if success:
        return 1, 0
    if not request.quiet:
        console.print(
            f"[yellow]⚠️  Filter failed for {result.project_name}: {error}[/yellow]"
        )
    return 0, 1


def _write_manifest(
    request: RefreshRequest, console: Console, result: RefreshBatchResult
) -> None:
    """Write the refresh manifest and report its location."""
    # Write manifest with timestamp by default, or use specified filename
    manifest_file: Path
    if request.manifest_filename:
        manifest_file = request.output_path / request.manifest_filename
    else:
        timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
        manifest_file = request.output_path / f"refresh-manifest-{timestamp}.json"
    write_refresh_manifest(manifest_file, result)
    if not request.quiet:
        console.print(f"📄 Manifest: [cyan]{manifest_file}[/cyan]")
        console.print()


def _exit_with_result_status(
    request: RefreshRequest, console: Console, result: RefreshBatchResult
) -> None:
    """Determine exit code."""
    if result.failed_count > 0:
        if not request.quiet:
            console.print(
                f"[yellow]⚠️  {result.failed_count} repositories failed to refresh[/yellow]"
            )
        raise typer.Exit(ExitCode.GENERAL_ERROR.value)
    elif result.conflicts_count > 0:
        if not request.quiet:
            console.print(
                f"[yellow]⚠️  {result.conflicts_count} repositories have conflicts[/yellow]"
            )
        raise typer.Exit(ExitCode.GENERAL_ERROR.value)
    else:
        if not request.quiet:
            console.print("[green]✅ All repositories refreshed successfully![/green]")
        raise typer.Exit(ExitCode.SUCCESS.value)
