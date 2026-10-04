# MadHatter — Adversarial Model Resilience Lab

MadHatter is a local research platform for probing and strengthening language
model resilience. It combines gradient-guided **discrete token
perturbations**, embedding-space PGD, LoRA post-training, clean-checkpoint
validation, cross-model transfer experiments, and black-box Claude Code
testing. Qwen remains the lightweight default, but the project supports a
curated set of open-weight causal models through the multi-model lab.

## Start with the illustrated tutorial

See **[Reading a MadHatter Discrete Attack](docs/TUTORIAL.md)** for a
step-by-step experiment using the UI screenshots, an explanation of every
control and metric, and guidance for distinguishing local next-token
sensitivity from a meaningful task-level failure.

## Services

| Service | Port | Purpose |
|---|---:|---|
| `madhatter-lab` | `8000` | Find perturbations, inspect token/logit drift, and optionally post-train LoRA |
| `clean-model` | `8001` | Independent base-model instance; no adapters; validates portability of MadHatter examples |

They are separate processes with separately loaded model objects and share only the Hugging Face model cache. Choosing a model in the UI loads that curated checkpoint into both services through their `/api/load` endpoints. The UI reports attack-service and clean-validator loading failures independently. The clean service never loads the adapter directory. For curated cross-family sweeps, model metadata, and the resilience matrix, use [`resileince/`](resileince/).

Starting Compose does not allocate model memory. Models load when selected in the UI or lazily on first use.

## Why the clean validator matters

For a discrete MadHatter attack the lab returns:

- `adversarial_prompt` — decoded human-readable adversarial prompt
- `input_ids_adversarial` — exact source-model token IDs used by the attack
- `token_changes` — original and replacement token IDs/text

The validator checks the result in two ways:

1. **Text portability:** send `adversarial_prompt` to the clean model and let its tokenizer encode it normally.
2. **Exact token portability:** send `input_ids_adversarial` directly to the clean model.

The UI reports whether decoding and re-tokenizing the adversarial prompt reconstructs the exact adversarial token sequence. This is important because a tokenizer decode → encode round trip is not guaranteed to preserve every possible token sequence.

The validator also compares clean-model next-token argmax and greedy continuation for:

- the original prompt,
- the decoded adversarial text,
- the exact adversarial token IDs.

## Choose which models appear

[`models.txt`](models.txt) is copied into the image and mounted into both
services. Uncommented lines are the only models shown in the dropdown and
accepted by the clean validator. Lines starting with `#` are hidden.

```text
HuggingFaceTB/SmolLM2-360M-Instruct
Qwen/Qwen3-0.6B
# HuggingFaceTB/SmolLM3-3B
```

Comment or uncomment a line, then restart the containers:

```bash
docker compose restart
```

A rebuild is needed only when the image runs without that file mounted. Model
weights are still downloaded on first load into the shared Hugging Face cache.

## Default model

When `models.txt` enables nothing, the fallback is:

```text
Qwen/Qwen3-0.6B
```

`MODEL_ID` is an optional lazy-load default for direct API use; it is not how
the two running services are synchronized. Compose does not bake it into
either service. To configure a fallback in a custom deployment, set it in
each service environment:

```bash
MODEL_ID=Qwen/Qwen3-1.7B uvicorn app.main:app --port 8000
```

For a local Transformers checkpoint, port 8000 still accepts a mounted path
through its load API:

```bash
curl -X POST http://localhost:8000/api/load \
  -H 'content-type: application/json' \
  -d '{"model_id":"/models/my-model"}'
```

The browser-facing clean load endpoint accepts only registry models. This
prevents arbitrary checkpoint downloads or filesystem paths through its
permissive local CORS API. Operators can still use `MODEL_ID` as the clean
service's lazy fallback for a trusted local checkpoint.

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

The UI on port 8000 has a **Run example in clean model :8001** button after a discrete token attack succeeds.

### GPU memory note

Once the clean validator is used, two independent model copies are resident: one in the MadHatter container and one in the clean container. With larger checkpoints this can exceed VRAM. In that case use a smaller checkpoint, CPU validator, or run the clean service separately after stopping training.

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

### Load a curated model

```bash
curl -X POST http://localhost:8001/api/load \
  -H 'content-type: application/json' \
  -d '{"model_id":"Qwen/Qwen3-0.6B"}'
```

The model must be present in `app/model_registry.py`. Switching is serialized,
unloads the previous model, releases cached accelerator memory, and then loads
the selected checkpoint.

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

then verifies the strongest candidates using actual `input_ids`. The result therefore becomes a real discrete source-model token sequence rather than only a hidden continuous embedding tensor.

Discrete mode does not use the embedding-PGD ε, step-size, norm, or divergence
controls. It greedily maximizes cross-entropy on the clean model's original
top next token, applies at most the requested number of verified edits, and
stops early if no valid candidate improves that objective. The UI reports the
actual edit count, stop reason, original-target probability drop, next-token
KL, and highlighted token changes.

Replacement candidates are restricted to Unicode Latin letters, ASCII digits,
ASCII punctuation, and common whitespace. Tokens containing CJK, Hangul,
Kana, Cyrillic, Arabic, emoji, other controls, or undecodable byte fragments
are excluded before ranking.

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

Then validate the resulting discrete examples on port 8001 to determine whether the perturbation transfers to an unmodified checkpoint.

## Important interpretation

A perturbation can be highly effective on a sensitivity-trained MadHatter model but fail on the untouched clean checkpoint. That is a meaningful result: it means the effect was learned by the adapter rather than being a transferable property of the original base model.

Conversely, if the exact token sequence changes the clean checkpoint's next-token distribution or generation too, the perturbation transfers independently of the LoRA changes.

A next-token argmax flip establishes local sensitivity, not automatically a
meaningful task-level failure. The clean validator therefore also reports
greedy-generation similarity and highlights changed output spans. Stronger
evidence combines a reproducible token-level shift with sustained generation
differences and task-specific evaluation across multiple prompts.

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
