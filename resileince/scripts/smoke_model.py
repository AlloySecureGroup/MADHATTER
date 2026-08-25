#!/usr/bin/env python3
"""Optional live checkpoint smoke test. Downloads/loads the selected model."""
import argparse
import json

from app.engine import AdversarialModelEngine
from app.model_registry import DEFAULT_MODEL_ID, MODEL_BY_ID


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, choices=sorted(MODEL_BY_ID))
    parser.add_argument("--prompt", default="Explain why leaves are green in one sentence.")
    args = parser.parse_args()

    engine = AdversarialModelEngine()
    print(json.dumps(engine.load_model(args.model), indent=2))
    result = engine.analyze_attack(
        prompt=args.prompt,
        mode="token_hotflip",
        epsilon=0.03,
        step_size=0.0075,
        steps=2,
        norm="linf",
        objective="next_token_kl",
        hidden_weight=1.0,
        top_k=3,
        random_start=False,
        max_token_changes=1,
        candidate_top_k=3,
        editable_last_n=32,
    )
    print(json.dumps({
        "model": result["model"]["model_id"],
        "adversarial_prompt": result.get("adversarial_prompt"),
        "token_changes": result.get("token_changes"),
        "argmax_changed": result["argmax_changed"],
        "final_kl": result["final_kl"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
