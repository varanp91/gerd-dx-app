"""
CLI smoke tests.

Interactive mode is not covered (would require mocking stdin). We cover:
  - --json input path loads a CaseInput and classifies end-to-end
  - --format json emits parseable CaseOutput JSON
  - --format pretty writes without raising
  - Disclaimer appears in pretty output (design contract: every invocation)
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gerd_dx.cli import app

FIXTURES = Path(__file__).parent / "fixtures"
CLASSIC_EROSIVE = FIXTURES / "classic_erosive.json"


@pytest.fixture(scope="module")
def runner() -> CliRunner:
    return CliRunner()


def test_json_input_and_json_output_roundtrips(runner: CliRunner):
    result = runner.invoke(
        app, ["--json", str(CLASSIC_EROSIVE), "--format", "json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert "ranked_mechanisms" in payload
    top = payload["ranked_mechanisms"][0]
    assert top["mechanism_id"] == "true_gerd_competent_peristalsis"
    assert top["confidence"] == "high"


def test_pretty_output_contains_disclaimer_and_mechanism(runner: CliRunner):
    result = runner.invoke(
        app, ["--json", str(CLASSIC_EROSIVE), "--format", "pretty"]
    )
    assert result.exit_code == 0, result.output
    out = result.output
    assert "education" in out.lower() and "diagnostic device" in out.lower()
    assert "True acid GERD with competent peristalsis" in out
    assert "[HIGH]" in out
    assert "RANKED MECHANISMS" in out
    assert "MANAGEMENT" in out


def test_pretty_output_includes_top_plan_procedural_entries(runner: CliRunner):
    result = runner.invoke(
        app, ["--json", str(CLASSIC_EROSIVE), "--format", "pretty"]
    )
    assert result.exit_code == 0
    # Top plan's procedural options should render; Nissen is listed for
    # true_gerd_competent_peristalsis.
    assert "Nissen" in result.output
    assert "indication:" in result.output


def test_unknown_format_exits_with_error(runner: CliRunner):
    result = runner.invoke(
        app, ["--json", str(CLASSIC_EROSIVE), "--format", "yaml"]
    )
    assert result.exit_code == 2
    assert "Unknown" in result.output or "Use" in result.output


def test_nonexistent_json_path_errors(runner: CliRunner):
    result = runner.invoke(
        app, ["--json", str(FIXTURES / "does_not_exist.json")]
    )
    assert result.exit_code != 0
