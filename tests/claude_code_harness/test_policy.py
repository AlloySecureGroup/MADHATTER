from __future__ import annotations

import json
from pathlib import Path

from claude_code_harness.policy import (
    build_settings,
    evaluate_tool,
    handle_hook,
)


def test_policy_allows_reads_inside_fixture_and_denies_other_tools(
    tmp_path: Path,
) -> None:
    allowed, _ = evaluate_tool(
        "Read", {"file_path": str(tmp_path / "reference.txt")}, tmp_path
    )
    escaped, _ = evaluate_tool(
        "Read", {"file_path": str(tmp_path.parent / "secret.txt")}, tmp_path
    )
    bash, _ = evaluate_tool("Bash", {"command": "env"}, tmp_path)

    assert allowed is True
    assert escaped is False
    assert bash is False


def test_hook_logs_denied_attempt_and_redacts_secret(
    tmp_path: Path, monkeypatch
) -> None:
    log_path = tmp_path / "attempts.jsonl"
    monkeypatch.setenv("MADHATTER_TOOL_LOG", str(log_path))

    response = handle_hook(
        {
            "tool_name": "Bash",
            "tool_input": {"command": "echo no", "api_key": "top-secret"},
        },
        tmp_path,
    )

    output = response["hookSpecificOutput"]
    assert output["permissionDecision"] == "deny"
    entry = json.loads(log_path.read_text())
    assert entry["allowed"] is False
    assert entry["tool_input"]["api_key"] == "[REDACTED]"


def test_settings_match_all_pre_tool_use_calls() -> None:
    settings = build_settings(["python", "/tmp/policy.py"])
    matcher = settings["hooks"]["PreToolUse"][0]

    assert matcher["matcher"] == "*"
    assert matcher["hooks"][0]["command"] == "python /tmp/policy.py"
