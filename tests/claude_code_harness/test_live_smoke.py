from __future__ import annotations

import os

import pytest

from claude_code_harness.runner import (
    RunnerConfig,
    check_prerequisites,
    run_scenario,
)
from claude_code_harness.scenarios import get_scenarios


@pytest.mark.skipif(
    os.environ.get("MADHATTER_RUN_LIVE") != "1",
    reason="set MADHATTER_RUN_LIVE=1 to spend one bounded Claude request",
)
def test_live_clean_smoke() -> None:
    status = check_prerequisites()
    if not status["ready"]:
        pytest.skip("Claude CLI is not installed or the safety policy check failed")

    scenario = get_scenarios("direct-instruction-conflict")[0]
    result = run_scenario(
        scenario,
        "clean",
        1,
        RunnerConfig(timeout_seconds=60, max_turns=2, max_budget_usd=0.05),
    )

    assert result.error is None
    assert result.exit_code == 0
    assert "ORANGE-7" in result.output
