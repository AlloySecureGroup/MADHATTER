from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import RunResult, Score
from .scoring import summarize


def _run_dict(run: RunResult, include_transcripts: bool) -> dict[str, Any]:
    value = run.to_dict()
    if not include_transcripts:
        value["output"] = "[OMITTED]"
    return value


def write_reports(
    destination: Path,
    runs: list[RunResult],
    scores: list[Score],
    include_transcripts: bool = False,
) -> tuple[Path, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    summary = summarize(scores)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "transcripts_included": include_transcripts,
        "summary": summary,
        "scores": [score.to_dict() for score in scores],
        "runs": [_run_dict(run, include_transcripts) for run in runs],
    }

    json_path = destination / "report.json"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    rows = []
    for score in scores:
        rows.append(
            "<tr>"
            f"<td>{html.escape(score.scenario_id)}</td>"
            f"<td>{score.trial}</td>"
            f"<td>{'pass' if score.clean_success else 'fail'}</td>"
            f"<td>{'pass' if score.adversarial_success else 'fail'}</td>"
            f"<td>{score.forbidden_tool_attempts}</td>"
            f"<td>{score.output_drift:.4f}</td>"
            "</tr>"
        )

    html_path = destination / "report.html"
    html_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Claude Code resilience report</title>
  <style>
    body {{ font: 16px system-ui; margin: 2rem auto; max-width: 70rem; padding: 0 1rem; }}
    .metrics {{ display: flex; flex-wrap: wrap; gap: 1rem; }}
    .metric {{ border: 1px solid #9996; border-radius: .5rem; padding: .8rem; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 1.5rem; }}
    th, td {{ border-bottom: 1px solid #9996; padding: .6rem; text-align: left; }}
    code {{ word-break: break-word; }}
  </style>
</head>
<body>
  <h1>Claude Code resilience report</h1>
  <p>Generated {html.escape(payload["generated_at"])}</p>
  <div class="metrics">
    <div class="metric"><strong>Pairs</strong><br>{summary["pairs"]}</div>
    <div class="metric"><strong>Clean success</strong><br>{summary["clean_success_rate"]:.1%}</div>
    <div class="metric"><strong>Adversarial success</strong><br>{summary["adversarial_success_rate"]:.1%}</div>
    <div class="metric"><strong>Injection followed</strong><br>{summary["injection_follow_rate"]:.1%}</div>
    <div class="metric"><strong>Denied tool attempts</strong><br>{summary["forbidden_tool_attempts"]}</div>
  </div>
  <table>
    <thead><tr><th>Scenario</th><th>Trial</th><th>Clean</th><th>Adversarial</th><th>Denied tools</th><th>Output drift</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  <p>Machine-readable details: <code>report.json</code>. Full transcripts are
  {'included' if include_transcripts else 'omitted by default'}.</p>
</body>
</html>
""",
        encoding="utf-8",
    )
    return json_path, html_path
