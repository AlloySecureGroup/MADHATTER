# Tutorial: Reading a MadHatter Discrete Attack

This tutorial walks through one complete discrete-token experiment: generating
a perturbation, validating it against an untouched model instance, and deciding
what the result does—and does not—prove.

## 1. Start the lab

For a CPU or WSL machine without NVIDIA passthrough:

```bash
docker compose -f docker-compose.cpu.yml up --build
```

For a working NVIDIA Docker setup:

```bash
docker compose up --build
```

Open `http://localhost:8000`. The attack service runs on port 8000 and the
independent clean validator runs on port 8001.

## 2. Choose and load a model

Select a checkpoint from the **Model** dropdown and click **Load model**. The
example below uses Qwen3 0.6B on CPU. Both services must use the same checkpoint
for exact token-ID validation.

The smaller 360M–600M models are suitable for learning the workflow on CPU.
Larger models usually produce stronger language behavior but need more memory
and make gradient search substantially slower.

## 3. Configure a discrete search

![Discrete attack explorer](images/discrete-attack.png)

The example prompt is a confidentiality-agreement paragraph. The controls mean:

- **Maximum token changes** is an upper bound, not a guarantee. Search stops
  early if no valid replacement improves the objective.
- **Candidates / position** controls how many gradient-ranked vocabulary
  candidates are considered per editable token position.
- **Editable last N** limits search to the final N editable prompt tokens.
- Only Latin letters, ASCII digits, ASCII punctuation, and common whitespace
  are eligible as replacement-token text.

Discrete mode does **not** use ε, step size, PGD steps, norm, or the embedding
divergence selector. Those controls belong to embedding PGD and are hidden when
discrete mode is selected.

### What objective is being optimized?

Let `y_clean` be the original model's most likely next token. The discrete
search approximately solves:

```text
x* = arg max, subject to at most k edits, -log p(y_clean | x*)
```

For each edit, HotFlip uses the input-embedding gradient to rank replacements.
MadHatter then performs real forward passes on the strongest proposals and
accepts only an edit that increases the verified objective.

This is an **untargeted local-sensitivity attack**. It tries to make the model
less confident in its original next-token choice; it does not directly optimize
for a specific malicious answer or for semantic similarity.

## 4. Read the adversarial prompt

Green highlights identify accepted replacement tokens. The list below the
prompt records:

- the search step;
- the prompt-token position;
- original and replacement token text;
- original and replacement vocabulary IDs.

Some replacements may look awkward or nonsensical. That is expected: HotFlip
optimizes model loss, not grammaticality. A changed prompt is therefore not
automatically a useful real-world attack. For human-readable perturbations,
evaluate fluency separately or add stronger lexical/semantic constraints.

The research summary reports:

- the original next-token target;
- its probability before and after perturbation;
- applied edits versus the maximum;
- why search stopped.

## 5. Validate against the clean model

Click **Run example in clean model :8001**.

![Clean-model validation and metrics](images/clean-validation.png)

The clean service is a separately loaded, untouched checkpoint. It evaluates
three inputs:

1. **Original prompt**—the unmodified baseline.
2. **Adversarial text**—decoded perturbation text, tokenized again normally.
3. **Exact adversarial token IDs**—the precise sequence produced by the attack.

### Text versus exact-token portability

**Decoded-text round trip** reports whether decoding and re-tokenizing the
adversarial text recreates the exact attacked token sequence.

- **Exact token match** means text and exact-ID validation use the same sequence.
- **Different tokenization** means the visible text does not perfectly preserve
  the attacked IDs. In that case, compare text and exact-ID results separately.

Different tokenization is not necessarily a bug. Tokenizer decode/encode is not
guaranteed to be perfectly invertible for every vocabulary sequence.

### Highlighted output

Red spans belong to the original generation where it differs; green spans
belong to the adversarial generation. Unhighlighted text is shared.

The screenshot reports **62.1% greedy-generation similarity**. The generated
wording changed substantially, but both responses still discuss confidentiality
language. This demonstrates behavioral sensitivity, not necessarily a semantic
task failure.

## 6. Interpret the metrics

The screenshot shows:

- **Next-token KL: 3.027**—the next-token distributions separated strongly.
- **Target probability drop: 65.34 percentage points**—the original preferred
  next token became much less likely.
- **Edits applied / max: 5/5**—the full edit budget was used.
- **Argmax changed: YES**—the most likely next token changed.
- **Verified objective by edit**—the accepted objective increased at each step.
- **Layer-wise representation drift**—later hidden layers moved more strongly
  than early layers for this example.

These values confirm that the optimization changed local model behavior. They
do not by themselves show policy bypass, factual corruption, or loss of task
performance.

## 7. Decide whether the attack “worked”

Use progressively stronger evidence:

1. **Optimization worked:** verified objective rises and target probability
   falls.
2. **Local behavior changed:** next-token KL increases or argmax flips.
3. **Perturbation transfers:** the clean validator reproduces the effect.
4. **Generation changes:** similarity falls and highlighted spans persist beyond
   the opening token.
5. **Task behavior fails:** a task-specific evaluator detects a wrong answer,
   refusal bypass, policy violation, malformed structure, or another defined
   failure.

The first two are sensitivity findings. The fifth is the strongest practical
claim and requires an evaluator appropriate to the task.

## 8. Make the experiment research-grade

Do not draw conclusions from one prompt. For a stronger evaluation:

- hold out a prompt set covering several task types;
- record clean task accuracy before attacking;
- sweep edit budgets and editable-token windows;
- compare HotFlip with random or benign token-edit controls;
- repeat across model families and model sizes;
- report attack success rate, generation similarity, and task-score change;
- separate exact-ID results from decoded-text transfer;
- preserve model revision, tokenizer revision, parameters, and random seeds.

MadHatter is an exploratory white-box lab. Results should be framed as
model- and objective-specific evidence, not as universal conclusions about an
entire model family.
