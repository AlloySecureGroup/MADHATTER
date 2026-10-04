from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import RunResult, Scenario, ToolAttempt
from .policy import build_settings

_SECRET_PATTERNS = (
    re.compile(r"sk-ant-[A-Za-z0-9_-]+"),
    re.compile(
        r"(?i)(api[_-]?key|token|password|secret)(\s*[=:]\s*)([^\s\"']+)"
    ),
)


@dataclass(frozen=True)
class RunnerConfig:
    claude_binary: str = "claude"
    model: str | None = None
    timeout_seconds: int = 120
    max_turns: int = 8
    max_budget_usd: float = 0.25


def redact(text: str) -> str:
    value = _SECRET_PATTERNS[0].sub("[REDACTED]", text)
    return _SECRET_PATTERNS[1].sub(
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", value
    )


def _safe_environment(root: Path, tool_log: Path) -> dict[str, str]:
    keep = {
        "HOME",
        "PATH",
        "USER",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "SHELL",
        "TMPDIR",
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
        "ANTHROPIC_API_KEY",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_CONFIG_DIR",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "NO_PROXY",
        "SSL_CERT_FILE",
        "NODE_EXTRA_CA_CERTS",
    }
    env = {key: value for key, value in os.environ.items() if key in keep}
    env.update(
        {
            "MADHATTER_FIXTURE_ROOT": str(root),
            "MADHATTER_TOOL_LOG": str(tool_log),
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        }
    )
    return env


def _write_fixture(root: Path, files: dict[str, str]) -> None:
    for relative, content in files.items():
        destination = (root / relative).resolve(strict=False)
        try:
            destination.relative_to(root.resolve(strict=False))
        except ValueError as exc:
            raise ValueError(f"fixture path escapes root: {relative}") from exc
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")


def _parse_events(stdout: str) -> tuple[str, dict[str, Any], int]:
    events: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)

    result_text = ""
    usage: dict[str, Any] = {}
    for event in events:
        if event.get("type") == "result":
            result = event.get("result")
            if isinstance(result, str):
                result_text = result
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]

    if not result_text:
        chunks: list[str] = []
        for event in events:
            message = event.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
        result_text = "\n".join(chunks)

    return redact(result_text), usage, len(events)


def _read_tool_log(path: Path) -> list[ToolAttempt]:
    if not path.exists():
        return []
    attempts: list[ToolAttempt] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
            attempts.append(
                ToolAttempt(
                    tool=str(item.get("tool", "")),
                    allowed=bool(item.get("allowed", False)),
                    reason=str(item.get("reason", "")),
                    tool_input=item.get("tool_input", {}),
                )
            )
        except (json.JSONDecodeError, TypeError):
            continue
    return attempts


def cli_version(binary: str = "claude") -> str | None:
    executable = shutil.which(binary)
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = (completed.stdout or completed.stderr).strip()
    return value or None


def auth_status(binary: str = "claude") -> tuple[bool, str | None]:
    executable = shutil.which(binary)
    if executable is None:
        return False, "Claude Code CLI not found"
    try:
        completed = subprocess.run(
            [executable, "auth", "status"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, redact(str(exc))
    detail = redact((completed.stdout or completed.stderr).strip())
    return completed.returncode == 0, detail or None


def check_prerequisites(binary: str = "claude") -> dict[str, Any]:
    executable = shutil.which(binary)
    version = cli_version(binary)
    authenticated, auth_detail = auth_status(binary)
    policy_ok = False
    with tempfile.TemporaryDirectory(prefix="madhatter-policy-check-") as temp:
        root = Path(temp)
        allowed, denied = _policy_self_check(root)
        policy_ok = allowed and denied
    return {
        "claude_binary": executable,
        "claude_version": version,
        "authenticated": authenticated,
        "auth_detail": auth_detail,
        "structured_output": bool(executable and version),
        "policy_self_check": policy_ok,
        "ready": bool(executable and version and authenticated and policy_ok),
    }


def _policy_self_check(root: Path) -> tuple[bool, bool]:
    from .policy import evaluate_tool

    allowed, _ = evaluate_tool("Read", {"file_path": str(root / "safe.txt")}, root)
    denied, _ = evaluate_tool("Bash", {"command": "echo unsafe"}, root)
    return allowed, not denied


def run_scenario(
    scenario: Scenario,
    variant: str,
    trial: int,
    config: RunnerConfig,
) -> RunResult:
    if variant not in {"clean", "adversarial"}:
        raise ValueError("variant must be 'clean' or 'adversarial'")

    executable = shutil.which(config.claude_binary)
    if executable is None:
        raise RuntimeError(f"Claude Code CLI not found: {config.claude_binary}")

    prompt = (
        scenario.clean_prompt if variant == "clean" else scenario.adversarial_prompt
    )
    files = scenario.clean_files if variant == "clean" else scenario.adversarial_files
    started = datetime.now(timezone.utc).isoformat()
    began = time.monotonic()
    version = cli_version(config.claude_binary)

    with tempfile.TemporaryDirectory(
        prefix=f"madhatter-{scenario.id}-{variant}-"
    ) as temp:
        root = Path(temp)
        _write_fixture(root, files)
        tool_log = root / ".madhatter-tool-attempts.jsonl"
        settings_path = root / ".madhatter-claude-settings.json"
        policy_path = Path(__file__).with_name("policy.py").resolve()
        settings = build_settings([sys.executable, str(policy_path)])
        settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")

        command = [
            executable,
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--permission-mode",
            "dontAsk",
            "--settings",
            str(settings_path),
            "--max-turns",
            str(config.max_turns),
            "--max-budget-usd",
            str(config.max_budget_usd),
        ]
        if config.model:
            command.extend(["--model", config.model])

        exit_code: int | None
        stdout = ""
        stderr = ""
        error: str | None = None
        try:
            completed = subprocess.run(
                command,
                input=prompt,
                cwd=root,
                env=_safe_environment(root, tool_log),
                capture_output=True,
                text=True,
                timeout=config.timeout_seconds,
                check=False,
            )
            exit_code = completed.returncode
            stdout = completed.stdout
            stderr = redact(completed.stderr)
            if exit_code != 0:
                error = stderr.strip() or f"Claude exited with status {exit_code}"
        except subprocess.TimeoutExpired as exc:
            exit_code = None
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            error = f"Claude timed out after {config.timeout_seconds} seconds"
        except OSError as exc:
            exit_code = None
            error = redact(str(exc))

        output, usage, event_count = _parse_events(stdout)
        attempts = _read_tool_log(tool_log)
        safe_command = [
            "[CLAUDE_BINARY]" if index == 0 else argument
            for index, argument in enumerate(command)
        ]
        return RunResult(
            scenario_id=scenario.id,
            variant=variant,
            trial=trial,
            model=config.model,
            command=safe_command,
            started_at=started,
            duration_seconds=round(time.monotonic() - began, 4),
            exit_code=exit_code,
            output=output,
            error=error,
            usage=usage,
            tool_attempts=attempts,
            event_count=event_count,
            cli_version=version,
            mutation_id=scenario.mutation_id,
        )
