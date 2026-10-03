from __future__ import annotations

import json
import stat
from pathlib import Path

from claude_code_harness.models import Scenario
from claude_code_harness.reporting import write_reports
from claude_code_harness.runner import (
    RunnerConfig,
    check_prerequisites,
    redact,
    run_scenario,
)
from claude_code_harness.scoring import score_pair


def _fake_claude(tmp_path: Path) -> Path:
    executable = tmp_path / "claude"
    executable.write_text(
        """#!/usr/bin/env python3
import json
import sys
import time

if "--version" in sys.argv:
    print("9.9.9 (fake)")
    raise SystemExit(0)

prompt = sys.stdin.read()
if "SLEEP" in prompt:
    time.sleep(2)
print(json.dumps({"type": "system", "subtype": "init"}))
print(json.dumps({
    "type": "result",
    "result": "ORANGE-7",
    "usage": {"input_tokens": 12, "output_tokens": 3},
}))
""",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    return executable


def _scenario(prompt: str = "Return the project code.") -> Scenario:
    return Scenario(
        id="fixture",
        title="Fixture",
        category="test",
        clean_prompt=prompt,
        adversarial_prompt=prompt,
        expected_terms=("ORANGE-7",),
    )


def test_fake_cli_run_and_reports(tmp_path: Path) -> None:
    executable = _fake_claude(tmp_path)
    config = RunnerConfig(claude_binary=str(executable), timeout_seconds=5)
    scenario = _scenario()

    clean = run_scenario(scenario, "clean", 1, config)
    adversarial = run_scenario(scenario, "adversarial", 1, config)
    score = score_pair(scenario, clean, adversarial)
    json_path, html_path = write_reports(
        tmp_path / "report", [clean, adversarial], [score]
    )

    assert clean.output == "ORANGE-7"
    assert clean.usage["input_tokens"] == 12
    assert clean.cli_version == "9.9.9 (fake)"
    assert score.adversarial_success is True
    payload = json.loads(json_path.read_text())
    assert payload["runs"][0]["output"] == "[OMITTED]"
    assert "Claude Code resilience report" in html_path.read_text()


def test_timeout_is_reported(tmp_path: Path) -> None:
    executable = _fake_claude(tmp_path)
    result = run_scenario(
        _scenario("SLEEP"),
        "clean",
        1,
        RunnerConfig(claude_binary=str(executable), timeout_seconds=1),
    )

    assert result.exit_code is None
    assert result.error == "Claude timed out after 1 seconds"


def test_prerequisites_and_redaction(tmp_path: Path) -> None:
    executable = _fake_claude(tmp_path)
    status = check_prerequisites(str(executable))

    assert status["ready"] is True
    assert redact("api_key=abc sk-ant-secret") == (
        "api_key=[REDACTED] [REDACTED]"
    )
