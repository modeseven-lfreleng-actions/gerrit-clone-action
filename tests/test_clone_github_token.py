# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 The Linux Foundation

"""``clone`` from GitHub needs a token, and says so before it starts.

GitHub discovery always calls the API with a token, so a run without
one cannot list any repository, public or private.  ``clone`` should
stop at configuration, naming where a token can come from, before it
prints its banner, records filters or contacts GitHub.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from gerrit_clone.cli import app
from gerrit_clone.content_intent import INTENT_DIR
from gerrit_clone.error_codes import ExitCode

if TYPE_CHECKING:
    from pathlib import Path

    from gerrit_clone.models import Config

GITHUB_HOST = "github.com/example-org"
BANNER = "Clone Configuration"
TOKEN_SOURCES = ("--github-token", "GERRIT_CLONE_TOKEN", "GITHUB_TOKEN")


def _succeed(_config: Config) -> MagicMock:
    """Return an empty successful batch result."""
    now = datetime.now()
    return MagicMock(
        started_at=now,
        completed_at=now,
        total_count=0,
        success_count=0,
        failed_count=0,
        skipped_count=0,
        refreshed_count=0,
        results=[],
        success_rate=100.0,
    )


@pytest.fixture
def no_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every environment variable a GitHub token comes from."""
    for name in ("GITHUB_TOKEN", "GERRIT_CLONE_TOKEN"):
        monkeypatch.delenv(name, raising=False)


def _clone(
    output: Path, *extra: str, host: str = GITHUB_HOST
) -> tuple[int, str, MagicMock]:
    """Run ``clone`` with cloning stubbed; return code, output, stub."""
    args = [
        "clone",
        "--host",
        host,
        "--output-path",
        str(output),
        "--disable-log-file",
        *extra,
    ]
    with patch("gerrit_clone.cli.clone_repositories", side_effect=_succeed) as stub:
        result = CliRunner().invoke(app, args)
    return result.exit_code, result.output, stub


@pytest.mark.usefixtures("no_token")
class TestGitHubCloneWithoutToken:
    """A GitHub clone without a token stops at configuration."""

    @pytest.mark.xfail(strict=True, reason="#321: warns, then fails discovery")
    def test_stops_before_discovery(self, tmp_path: Path) -> None:
        """No banner, no discovery, a configuration error naming the fix."""
        exit_code, output, stub = _clone(tmp_path)

        assert exit_code == ExitCode.CONFIGURATION_ERROR
        stub.assert_not_called()
        assert BANNER not in output
        for source in TOKEN_SOURCES:
            assert source in output

    @pytest.mark.xfail(strict=True, reason="#321: warns, then fails discovery")
    def test_records_no_filters(self, tmp_path: Path) -> None:
        """Content filters aren't written down for a run that can't start."""
        exit_code, _output, stub = _clone(tmp_path, "--remove-files", "*.jar")

        assert exit_code == ExitCode.CONFIGURATION_ERROR
        stub.assert_not_called()
        assert not (tmp_path / INTENT_DIR).exists()

    @pytest.mark.xfail(strict=True, reason="#321: misleading warning")
    def test_drops_the_misleading_warning(self, tmp_path: Path) -> None:
        """No claim that clones will still work without a token."""
        _exit_code, output, _stub = _clone(tmp_path)

        assert "will still work" not in output


@pytest.mark.usefixtures("no_token")
class TestCloneWithTokenOrGerrit:
    """A token from any source, or a Gerrit host, lets the run start."""

    def test_token_option(self, tmp_path: Path) -> None:
        """--github-token supplies the token."""
        exit_code, output, stub = _clone(tmp_path, "--github-token", "t")

        assert exit_code == 0, output
        stub.assert_called_once()

    @pytest.mark.parametrize("variable", ["GERRIT_CLONE_TOKEN", "GITHUB_TOKEN"])
    def test_token_variable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variable: str
    ) -> None:
        """Either environment variable supplies the token."""
        monkeypatch.setenv(variable, "t")

        exit_code, output, stub = _clone(tmp_path)

        assert exit_code == 0, output
        stub.assert_called_once()

    def test_gerrit_needs_no_github_token(self, tmp_path: Path) -> None:
        """A Gerrit source never asks for a GitHub token."""
        exit_code, output, stub = _clone(tmp_path, host="gerrit.example.org")

        assert exit_code == 0, output
        stub.assert_called_once()
