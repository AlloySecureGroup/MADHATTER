from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    category: str
    clean_prompt: str
    adversarial_prompt: str
    clean_files: dict[str, str] = field(default_factory=dict)
    adversarial_files: dict[str, str] = field(default_factory=dict)
    expected_terms: tuple[str, ...] = ()
    forbidden_terms: tuple[str, ...] = ()
    mutation_id: str | None = None


@dataclass(frozen=True)
class ToolAttempt:
    tool: str
    allowed: bool
    reason: str
    tool_input: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    scenario_id: str
    variant: str
    trial: int
    model: str | None
    command: list[str]
    started_at: str
    duration_seconds: float
    exit_code: int | None
    output: str
    error: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    tool_attempts: list[ToolAttempt] = field(default_factory=list)
    event_count: int = 0
    cli_version: str | None = None
    mutation_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Score:
    scenario_id: str
    trial: int
    clean_success: bool
    adversarial_success: bool
    injection_followed: bool
    forbidden_tool_attempts: int
    output_drift: float
    clean_error: str | None = None
    adversarial_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
