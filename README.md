# MadHatter — Qwen Token Perturbation Lab

MadHatter is a local Qwen research demo for gradient-guided **discrete token perturbations**, embedding-space PGD, and LoRA post-training. It now includes a second, independent **clean Qwen validator** so a perturbation found in the lab can be tested against an unmodified base checkpoint.

## Services

| Service | Port | Purpose |
|---|---:|---|
| `madhatter-lab` | `8000` | Find perturbations, inspect token/logit drift, and optionally post-train LoRA |
| `clean-qwen` | `8001` | Independent base Qwen instance; no adapters; validates portability of MadHatter examples |

Both services use the same `MODEL_ID` environment variable and share only the Hugging Face model cache. They are separate processes with separately loaded model objects. The clean service never loads the adapter directory.

The clean model loads lazily when you first validate an example, so starting Compose does not immediately allocate memory for two Qwen copies.

## Why the clean validator matters

For a discrete MadHatter attack the lab returns:

- `adversarial_prompt` — decoded human-readable adversarial prompt
- `input_ids_adversarial` — exact Qwen token IDs used by the attack
- `token_changes` — original and replacement token IDs/text

The validator checks the result in two ways:

1. **Text portability:** send `adversarial_prompt` to clean Qwen and let its tokenizer encode it normally.
2. **Exact token portability:** send `input_ids_adversarial` directly to the clean model.

The UI reports whether decoding and re-tokenizing the adversarial prompt reconstructs the exact adversarial token sequence. This is important because a tokenizer decode → encode round trip is not guaranteed to preserve every possible token sequence.

The validator also compares clean-model next-token argmax and greedy continuation for:

- the original prompt,
- the decoded adversarial text,
- the exact adversarial token IDs.

## Default model

The default is:

```text
Qwen/Qwen3-0.6B
```

Override both containers together:

```bash
MODEL_ID=Qwen/Qwen3-1.7B docker compose up --build
```

For a local Transformers checkpoint:

```bash
MODEL_ID=/models/my-qwen docker compose up --build
```

Keep both services on the **same checkpoint/tokenizer** for an exact token-ID portability test. The clean API rejects a comparison when the lab-reported model ID and clean validator model ID do not match.

## GPU run

```bash
mkdir -p adapters models
docker compose up --build
```

Open MadHatter:

```text
http://localhost:8000
```

Clean validator API:

```text
http://localhost:8001
```

The UI on port 8000 has a **Run example in clean Qwen :8001** button after a discrete token attack succeeds.

### GPU memory note

Once the clean validator is used, two independent Qwen model copies are resident: one in the MadHatter container and one in the clean container. With larger checkpoints this can exceed VRAM. In that case use a smaller checkpoint, CPU validator, or run the clean service separately after stopping training.

## CPU / WSL fallback

If NVIDIA passthrough is not available in WSL:

```bash
docker compose -f docker-compose.cpu.yml up --build
```

This starts both services on CPU. It is slower but provides the same validation path.

## Clean validator API

### Health

```bash
curl http://localhost:8001/api/health
```

### Evaluate normal text

```bash
curl -X POST http://localhost:8001/api/evaluate \
  -H 'content-type: application/json' \
  -d '{
    "mode":"text",
    "prompt":"Explain why the sky appears blue.",
    "max_new_tokens":32
  }'
```

### Evaluate exact token IDs

Use `input_ids_adversarial` returned by `POST /api/analyze` on port 8000:

```bash
curl -X POST http://localhost:8001/api/evaluate \
  -H 'content-type: application/json' \
  -d '{
    "mode":"token_ids",
    "input_ids":[151644,872,198],
    "max_new_tokens":32
  }'
```

The example IDs above are illustrative only; use the actual IDs returned by your MadHatter run.

### Compare an attack in one request

```bash
curl -X POST http://localhost:8001/api/compare \
  -H 'content-type: application/json' \
  -d '{
    "expected_model_id":"Qwen/Qwen3-0.6B",
    "original_prompt":"Explain why the sky appears blue.",
    "adversarial_prompt":"YOUR_MADHATTER_PROMPT",
    "adversarial_input_ids":[/* IDs returned by MadHatter */],
    "max_new_tokens":48
  }'
```

## MadHatter discrete search

Token mode performs a HotFlip-style white-box search. For an editable token position `i` and candidate vocabulary token `j`, it uses the first-order score

```text
score(i, j) ≈ grad(E_i) · (E_j - E_i)
```

then verifies the strongest candidates using actual `input_ids`. The result therefore becomes a real discrete Qwen token sequence rather than only a hidden continuous embedding tensor.

## Post-training goals

MadHatter keeps two LoRA objectives:

- `sensitive`: preserve clean supervised behavior while increasing bounded divergence between clean and perturbed response-token distributions.
- `robust`: adversarially train the adapter to suppress perturbation-induced deviations.

For experiments where you want real token substitutions to have stronger effects, use:

```json
{
  "training_goal": "sensitive",
  "attack_mode": "token_hotflip"
}
```

Then validate the resulting discrete examples on port 8001 to determine whether the perturbation transfers to an unmodified Qwen checkpoint.

## Important interpretation

A perturbation can be highly effective on a sensitivity-trained MadHatter model but fail on the untouched clean checkpoint. That is a meaningful result: it means the effect was learned by the adapter rather than being a transferable property of the original Qwen model.

Conversely, if the exact token sequence changes the clean checkpoint's next-token distribution or generation too, the perturbation transfers independently of the LoRA changes.

## Claude Code resilience harness

The standalone `claude_code_harness` package runs bounded, black-box
clean/adversarial comparisons against the locally installed Claude Code CLI.
It measures:

- direct and indirect prompt-injection resistance;
- task correctness and output drift;
- attempts to invoke tools outside a read-only allowlist.

This is deliberately separate from the FastAPI and Docker services. Each
trial gets a fresh temporary fixture directory and a `PreToolUse` hook that
logs tool requests, allows only `Read`, `Glob`, and `Grep` inside that fixture,
and denies everything else. A hook or prerequisite failure stops the run
rather than broadening permissions.

### Requirements

Install and authenticate Claude Code, then verify the CLI, authentication, and
safety hook:

```bash
python3 -m claude_code_harness check
```

The harness itself uses only the Python standard library and supports Python
3.9 or newer.

### Run the suite

```bash
python3 -m claude_code_harness run \
  --trials 3 \
  --model claude-sonnet-4-5 \
  --max-budget-usd 0.25
```

Run one scenario with stricter limits:

```bash
python3 -m claude_code_harness run \
  --scenario indirect-file-injection \
  --trials 1 \
  --max-turns 2 \
  --max-budget-usd 0.05 \
  --output reports/smoke
```

Every run writes `report.json` and `report.html`. Redacted model transcripts
are omitted by default; add `--include-transcripts` only when the prompts and
outputs are safe to retain. Reports record model/CLI metadata, scenario and
trial IDs, durations, usage returned by Claude, tool decisions, and aggregate
rates. Claude output is nondeterministic, so compare rates across repeated
trials instead of treating exact prose as stable.

### Tests

The default tests use a fake `claude` executable and spend no API credits:

```bash
python3 -m pip install "pytest>=8.3,<9"
python3 -m pytest -q tests/claude_code_harness
```

The opt-in live smoke test performs one harmless, read-only request with a
small budget:

```bash
MADHATTER_RUN_LIVE=1 python3 -m pytest -q \
  tests/claude_code_harness/test_live_smoke.py
```

### Scope and limitations

Claude is a hosted, proprietary model. Its embeddings, gradients, logits,
hidden states, exact token IDs, weights, and LoRA training are not exposed.
Consequently, MADHATTER's HotFlip, embedding PGD, exact-token replay, and LoRA
experiments cannot run against Claude. This harness evaluates only observable
Claude Code behavior through the CLI and does not claim white-box parity.

Use the scenarios only on systems and data you are authorized to test. The
deny hook is defense in depth, not an operating-system sandbox: run untrusted
custom scenarios in a disposable VM or container without production
credentials or sensitive mounts.
