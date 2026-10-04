from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

SAFE_TOOLS = frozenset({"Read", "Glob", "Grep"})
PATH_KEYS = ("file_path", "path")


def _within_root(value: str, root: Path) -> bool:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        candidate.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def evaluate_tool(
    tool: str, tool_input: dict[str, Any], root: Path
) -> tuple[bool, str]:
    if tool not in SAFE_TOOLS:
        return False, f"{tool} is not in the read-only allowlist"

    for key in PATH_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value and not _within_root(value, root):
            return False, f"{key} escapes the fixture root"

    return True, "read-only operation inside fixture"


def _redact_tool_input(tool_input: dict[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in tool_input.items():
        lowered = key.lower()
        if any(term in lowered for term in ("token", "secret", "password", "key")):
            redacted[key] = "[REDACTED]"
        elif isinstance(value, str) and len(value) > 500:
            redacted[key] = value[:500] + "…"
        else:
            redacted[key] = value
    return redacted


def handle_hook(payload: dict[str, Any], root: Path) -> dict[str, Any]:
    tool = str(payload.get("tool_name", ""))
    raw_input = payload.get("tool_input", {})
    tool_input = raw_input if isinstance(raw_input, dict) else {}
    allowed, reason = evaluate_tool(tool, tool_input, root)

    log_path = os.environ.get("MADHATTER_TOOL_LOG")
    if log_path:
        entry = {
            "tool": tool,
            "allowed": allowed,
            "reason": reason,
            "tool_input": _redact_tool_input(tool_input),
        }
        with Path(log_path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow" if allowed else "deny",
            "permissionDecisionReason": reason,
        }
    }


def build_settings(policy_command: list[str]) -> dict[str, Any]:
    command = " ".join(shlex.quote(part) for part in policy_command)
    return {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "*",
                    "hooks": [{"type": "command", "command": command}],
                }
            ]
        }
    }


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        root_value = os.environ["MADHATTER_FIXTURE_ROOT"]
        response = handle_hook(payload, Path(root_value))
        json.dump(response, sys.stdout)
        sys.stdout.write("\n")
        return 0
    except Exception as exc:
        # Hook failure must deny the operation. Exit code 2 blocks a tool call.
        print(f"MADHATTER policy hook failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
