#!/usr/bin/env python3
"""Exercise the clean validator's multi-model resilience job on a running :8001 service."""
import argparse
import json
import time
import urllib.request


def request(url, method="GET", body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8001")
    args = ap.parse_args()
    payload = {
        "original_prompt": "Return the word blue.",
        "adversarial_prompt": "Return the word azure.",
        "model_ids": ["HuggingFaceTB/SmolLM2-360M-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"],
        "top_k": 5,
        "max_new_tokens": 8,
    }
    job = request(args.base + "/api/resilience/start", "POST", payload)["job_id"]
    while True:
        state = request(args.base + "/api/resilience/" + job)["job"]
        print(f"{state['status']} {state['progress']:.0%} {state.get('current_model') or ''}")
        if state["status"] in {"complete", "failed"}:
            print(json.dumps(state, indent=2, ensure_ascii=False))
            break
        time.sleep(1.0)


if __name__ == "__main__":
    main()
