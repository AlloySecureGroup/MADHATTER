from __future__ import annotations

from typing import Any

# Curated for this lab: small, open-weight causal LMs that load through
# transformers.AutoModelForCausalLM and expose input embeddings.
MODEL_OPTIONS: list[dict[str, Any]] = [
    {
        "id": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "name": "SmolLM2 360M Instruct",
        "family": "SmolLM2",
        "params_b": 0.36,
        "license": "Apache-2.0",
        "tier": "tiny",
        "recommended": True,
        "notes": "Fast CPU/GPU smoke tests and resilience sweeps.",
    },
    {
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "name": "Qwen2.5 0.5B Instruct",
        "family": "Qwen2.5",
        "params_b": 0.5,
        "license": "Apache-2.0",
        "tier": "tiny",
        "recommended": True,
        "notes": "Small instruction model; useful Qwen baseline.",
    },
    {
        "id": "Qwen/Qwen3-0.6B",
        "name": "Qwen3 0.6B",
        "family": "Qwen3",
        "params_b": 0.6,
        "license": "Apache-2.0",
        "tier": "tiny",
        "recommended": True,
        "notes": "Default MadHatter research model; supports non-thinking chat mode.",
    },
    {
        "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "name": "TinyLlama 1.1B Chat",
        "family": "TinyLlama",
        "params_b": 1.1,
        "license": "Apache-2.0",
        "tier": "small",
        "recommended": True,
        "notes": "Compact Llama-family comparison point.",
    },
    {
        "id": "Qwen/Qwen2.5-1.5B-Instruct",
        "name": "Qwen2.5 1.5B Instruct",
        "family": "Qwen2.5",
        "params_b": 1.5,
        "license": "Apache-2.0",
        "tier": "small",
        "recommended": True,
        "notes": "Stronger Qwen2.5 comparison without a large hardware jump.",
    },
    {
        "id": "Qwen/Qwen3-1.7B",
        "name": "Qwen3 1.7B",
        "family": "Qwen3",
        "params_b": 1.7,
        "license": "Apache-2.0",
        "tier": "small",
        "recommended": True,
        "notes": "Larger Qwen3 transfer/resilience target.",
    },
    {
        "id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "name": "SmolLM2 1.7B Instruct",
        "family": "SmolLM2",
        "params_b": 1.7,
        "license": "Apache-2.0",
        "tier": "small",
        "recommended": False,
        "notes": "Compact instruction model from a different family for transfer tests.",
    },
    {
        "id": "ibm-granite/granite-3.3-2b-instruct",
        "name": "Granite 3.3 2B Instruct",
        "family": "Granite",
        "params_b": 2.0,
        "license": "Apache-2.0",
        "tier": "small",
        "recommended": False,
        "notes": "IBM Apache-2.0 instruct model; useful architecture-diverse target.",
    },
    {
        "id": "HuggingFaceTB/SmolLM3-3B",
        "name": "SmolLM3 3B",
        "family": "SmolLM3",
        "params_b": 3.0,
        "license": "Apache-2.0",
        "tier": "medium",
        "recommended": False,
        "notes": "Reasonable 3B checkpoint for a stronger Apache-2.0 comparison.",
    },
]

MODEL_BY_ID = {item["id"]: item for item in MODEL_OPTIONS}
DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"


def model_options() -> list[dict[str, Any]]:
    return [dict(item) for item in MODEL_OPTIONS]


def model_metadata(model_id: str | None) -> dict[str, Any] | None:
    if not model_id:
        return None
    item = MODEL_BY_ID.get(model_id)
    return dict(item) if item else None


def is_curated_model(model_id: str) -> bool:
    return model_id in MODEL_BY_ID
