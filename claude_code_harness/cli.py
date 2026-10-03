from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .reporting import write_reports
from .runner import RunnerConfig, check_prerequisites, run_scenario
from .scenarios import get_scenarios
from .scoring import score_pair


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m claude_code_harness",
        description="Run bounded black-box resilience tests against Claude Code.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="check CLI and policy prerequisites")
    check.add_argument("--claude-binary", default="claude")

    run = subparsers.add_parser("run", help="run clean/adversarial scenario pairs")
    run.add_argument("--scenario", help="run only one scenario ID")
    run.add_argument("--trials", type=int, default=1)
    run.add_argument("--model")
    run.add_argument("--claude-binary", default="claude")
    run.add_argument("--timeout", type=int, default=120)
    run.add_argument("--max-turns", type=int, default=8)
    run.add_argument("--max-budget-usd", type=float, default=0.25)
    run.add_argument(
        "--output",
        type=Path,
        default=None,
        help="report directory (default: reports/<UTC timestamp>)",
    )
    run.add_argument(
        "--include-transcripts",
        action="store_true",
        help="persist redacted model output in report.json",
    )
    return parser


def _check(args: argparse.Namespace) -> int:
    status = check_prerequisites(args.claude_binary)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if status["ready"] else 1


def _run(args: argparse.Namespace) -> int:
    if args.trials < 1:
        raise ValueError("--trials must be at least 1")
    if args.timeout < 1 or args.max_turns < 1 or args.max_budget_usd <= 0:
        raise ValueError("timeout, max-turns, and max-budget-usd must be positive")

    status = check_prerequisites(args.claude_binary)
    if not status["ready"]:
        print(json.dumps(status, indent=2, sort_keys=True), file=sys.stderr)
        print("Refusing to run because prerequisites failed.", file=sys.stderr)
        return 2

    scenarios = get_scenarios(args.scenario)
    config = RunnerConfig(
        claude_binary=args.claude_binary,
        model=args.model,
        timeout_seconds=args.timeout,
        max_turns=args.max_turns,
        max_budget_usd=args.max_budget_usd,
    )
    runs = []
    scores = []
    for scenario in scenarios:
        for trial in range(1, args.trials + 1):
            clean = run_scenario(scenario, "clean", trial, config)
            adversarial = run_scenario(scenario, "adversarial", trial, config)
            runs.extend((clean, adversarial))
            scores.append(score_pair(scenario, clean, adversarial))

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = args.output or Path("reports") / timestamp
    json_path, html_path = write_reports(
        destination, runs, scores, args.include_transcripts
    )
    print(f"JSON report: {json_path}")
    print(f"HTML report: {html_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            return _check(args)
        return _run(args)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 2
