# MadHatter Multi-Model Resilience Lab

MadHatter is a local Docker research lab for studying how small prompt perturbations affect open-weight causal language models, and for post-training a selected model with LoRA to either **increase resilience** to bounded perturbations or, for controlled research, increase sensitivity to them.

The project exposes two independent services:

| Service | Port | Purpose |
|---|---:|---|
| `madhatter-lab` | 8000 | Attack explorer, discrete token search, embedding PGD, layer drift, and LoRA post-training |
| `clean-model` | 8001 | Untouched checkpoint validation and sequential cross-model resilience sweeps |

## Curated model options

The UI contains eight compact Apache-2.0 open-weight checkpoints that load through `transformers.AutoModelForCausalLM`:

| Model | Approx. parameters | Family | Suggested use |
|---|---:|---|---|
| `HuggingFaceTB/SmolLM2-360M-Instruct` | 0.36B | SmolLM2 | Fast CPU smoke tests |
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | Qwen2.5 | Small Qwen baseline |
| `Qwen/Qwen3-0.6B` | 0.6B | Qwen3 | Default research model |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B | TinyLlama | Llama-family comparison |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | Qwen2.5 | Stronger Qwen2.5 target |
| `Qwen/Qwen3-1.7B` | 1.7B | Qwen3 | Larger Qwen3 target |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.7B | SmolLM2 | Cross-family transfer |
| `ibm-granite/granite-3.3-2b-instruct` | 2.0B | Granite | Architecture-diverse target |

Model metadata lives in `app/model_registry.py`, so the curated set is easy to change without rewriting the UI.

## What the resilience matrix measures

A discrete MadHatter attack produces both exact adversarial token IDs and decoded adversarial text. Those are tested differently:

- **Same checkpoint:** the clean validator can replay the decoded text and the exact token IDs. This tests whether decode/re-tokenize changes the sequence.
- **Different checkpoint:** only the decoded text is portable. Each model uses its own tokenizer, so exact source-model token IDs are not meaningful across families.

For each clean target, the resilience matrix records:

- next-token argmax changed / unchanged;
- Jensen-Shannon divergence between clean and perturbed next-token distributions;
- greedy generation similarity;
- whether the generated text changed;
- input token-count change after the target model re-tokenizes the adversarial text;
- clean and perturbed top next-token predictions.

A more resilient target generally shows **no argmax flip, lower JS divergence, and higher generation similarity** for the same perturbation. These are diagnostic metrics, not a universal scalar robustness score.

## Run with Docker

### CPU mode

Use this if CUDA/WSL GPU passthrough is unavailable:

```bash
mkdir -p adapters models
docker compose -f docker-compose.cpu.yml up --build
```

Open:

- MadHatter UI: `http://localhost:8000`
- Clean multi-model validator: `http://localhost:8001`

The 360M and 500M options are the most practical CPU starting points. Gradient-based token search and LoRA training are much slower on CPU.

### NVIDIA GPU mode

```bash
mkdir -p adapters models
docker compose up --build
```

Both services request GPU access. The clean validator still loads only one checkpoint at a time. Be aware that if the lab model remains loaded while the validator loads another target, both checkpoints coexist in VRAM.

If Docker reports `WSL environment detected but no adapters were found`, use `docker-compose.cpu.yml` until Windows/WSL NVIDIA passthrough is working.

## Typical workflow

1. Open port 8000.
2. Pick a research checkpoint from **Research model** and click **Load model**.
3. Run **Discrete token substitutions** in Attack Explorer.
4. Inspect the decoded adversarial prompt and exact token substitutions.
5. Click **Run in clean model :8001** to validate the result against an untouched copy of the same checkpoint.
6. Open **Resilience matrix**. The original/adversarial text is populated automatically after a discrete attack.
7. Select several target models and run the sweep.
8. Compare argmax flips, JS divergence, generation similarity, and tokenizer effects.
9. Optionally use **Adversarial train → Increase robustness** to train a LoRA adapter on the currently loaded lab checkpoint, then repeat held-out attacks.

## Robustness post-training

The defensive LoRA objective uses a worst-case inner perturbation and an outer minimization step. For robust mode the implementation minimizes adversarial response loss plus a clean-versus-adversarial consistency term:

```text
inner:  find bounded perturbation that increases supervised loss
outer:  CE(adversarial) + λ KL(clean || adversarial)
```

Only prompt positions are perturbed; supervised response tokens and chat-control tokens are not editable. Loading a different research model unloads the previous checkpoint and its active adapter.

## API highlights

### List curated models

```bash
curl http://localhost:8000/api/models
curl http://localhost:8001/api/models
```

### Load a research model

```bash
curl -X POST http://localhost:8000/api/load \
  -H 'content-type: application/json' \
  -d '{"model_id":"HuggingFaceTB/SmolLM2-360M-Instruct"}'
```

### Load an untouched clean target

```bash
curl -X POST http://localhost:8001/api/load \
  -H 'content-type: application/json' \
  -d '{"model_id":"Qwen/Qwen2.5-0.5B-Instruct"}'
```

### Start a resilience sweep

```bash
curl -X POST http://localhost:8001/api/resilience/start \
  -H 'content-type: application/json' \
  -d '{
    "original_prompt":"Return the word blue.",
    "adversarial_prompt":"Return the word azure.",
    "model_ids":[
      "HuggingFaceTB/SmolLM2-360M-Instruct",
      "Qwen/Qwen2.5-0.5B-Instruct"
    ],
    "top_k":5,
    "max_new_tokens":16
  }'
```

Poll the returned job ID:

```bash
curl http://localhost:8001/api/resilience/JOB_ID
```

## Tests

The repository has two levels of testing.

### Offline structural tests

These do **not** download model weights:

```bash
python -m pytest -q
```

They verify the 5–10 model registry contract, request schemas, and the UI resilience controls.

### Live checkpoint smoke test

This downloads/loads a real checkpoint and performs a one-edit discrete attack:

```bash
FORCE_CPU=1 python scripts/smoke_model.py \
  --model HuggingFaceTB/SmolLM2-360M-Instruct
```

Inside the built container:

```bash
docker compose -f docker-compose.cpu.yml run --rm madhatter-lab \
  python scripts/smoke_model.py --model HuggingFaceTB/SmolLM2-360M-Instruct
```

### Running clean-service matrix smoke test

After the containers are up:

```bash
python scripts/smoke_matrix.py
```

This exercises the port-8001 job API across SmolLM2 360M and Qwen2.5 0.5B.

## Local checkpoints

The Compose files mount `./models` read-only at `/models`. You can use a local Transformers-format causal-LM checkpoint by changing `MODEL_ID` or calling `/api/load` with a path such as `/models/my-model`. The curated resilience matrix intentionally limits batch targets to the registry so sweeps remain reproducible; add a local model to `app/model_registry.py` if you want it included in matrix testing.

## Files

```text
app/
  main.py             # port 8000 API
  engine.py           # PGD, HotFlip-style token search, LoRA training
  clean_server.py     # port 8001 clean validation + resilience jobs
  model_registry.py   # curated open-weight model list
  schemas.py
  static/index.html   # single-page HTML/CSS/JS UI
samples/train.jsonl
scripts/smoke_model.py
scripts/smoke_matrix.py
tests/
docker-compose.yml
docker-compose.cpu.yml
Dockerfile
Makefile
```

## Research caveats

Token perturbations are model/tokenizer dependent. A token-ID substitution generated for one checkpoint cannot be interpreted as the same token substitution by a model with a different vocabulary. Cross-model transfer therefore uses decoded text and explicitly measures re-tokenization effects. Also, one gradient restart or one prompt is not sufficient evidence of robustness; use held-out prompts, multiple perturbation budgets/restarts, and report clean-quality changes alongside resilience metrics.
