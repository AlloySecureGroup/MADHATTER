import gc
import json
import math
import os
import random
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from .model_registry import DEFAULT_MODEL_ID, model_metadata
from .token_filter import latin_replacement_token_ids


@dataclass
class ModelState:
    model: Any | None = None
    tokenizer: Any | None = None
    model_id: str | None = None
    device: str = "cpu"
    dtype: str = "float32"
    has_lora: bool = False
    replacement_mask: Any | None = None


class AdversarialModelEngine:
    def __init__(self) -> None:
        self.state = ModelState()
        self.lock = threading.RLock()
        self.force_cpu = os.getenv("FORCE_CPU", "0") == "1"
        self.adapter_root = Path(os.getenv("ADAPTER_ROOT", "/data/adapters"))
        self.adapter_root.mkdir(parents=True, exist_ok=True)

    @property
    def loaded(self) -> bool:
        return self.state.model is not None and self.state.tokenizer is not None

    def _device(self) -> torch.device:
        if not self.force_cpu and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def info(self) -> dict[str, Any]:
        gpu_name = None
        if torch.cuda.is_available() and not self.force_cpu:
            gpu_name = torch.cuda.get_device_name(0)
        return {
            "loaded": self.loaded,
            "model_id": self.state.model_id,
            "device": self.state.device,
            "dtype": self.state.dtype,
            "has_lora": self.state.has_lora,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": gpu_name,
            "registry": model_metadata(self.state.model_id),
        }

    def unload(self) -> None:
        with self.lock:
            self.state = ModelState()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load_model(self, model_id: str) -> dict[str, Any]:
        with self.lock:
            self.unload()
            device = self._device()
            dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=False,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
            )
            model.to(device)
            model.eval()
            model.config.use_cache = False

            self.state = ModelState(
                model=model,
                tokenizer=tokenizer,
                model_id=model_id,
                device=str(device),
                dtype=str(dtype).replace("torch.", ""),
                has_lora=isinstance(model, PeftModel),
            )
            return self.info()

    def ensure_loaded(self) -> None:
        if not self.loaded:
            self.load_model(os.getenv("MODEL_ID", DEFAULT_MODEL_ID))

    def _chat_prompt(self, prompt: str) -> str:
        tok = self.state.tokenizer
        messages = [{"role": "user", "content": prompt}]
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            return tok.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except (TypeError, ValueError):
            try:
                return tok.apply_chat_template(messages, **kwargs)
            except Exception:
                return prompt

    @staticmethod
    def _project(delta: torch.Tensor, epsilon: float, norm: str, mask: torch.Tensor) -> torch.Tensor:
        delta = delta * mask
        if norm == "linf":
            return delta.clamp(-epsilon, epsilon)
        flat = delta.reshape(delta.shape[0], -1)
        norms = torch.linalg.vector_norm(flat.float(), ord=2, dim=1, keepdim=True).clamp_min(1e-12)
        scale = torch.minimum(torch.ones_like(norms), torch.full_like(norms, epsilon) / norms)
        projected = flat.float() * scale
        return projected.reshape_as(delta).to(delta.dtype) * mask

    @staticmethod
    def _random_delta_like(base: torch.Tensor, epsilon: float, norm: str, mask: torch.Tensor) -> torch.Tensor:
        if norm == "linf":
            delta = torch.empty_like(base).uniform_(-epsilon, epsilon)
            return delta * mask
        delta = torch.randn_like(base)
        flat = delta.reshape(delta.shape[0], -1).float()
        flat = flat / torch.linalg.vector_norm(flat, ord=2, dim=1, keepdim=True).clamp_min(1e-12)
        radius = torch.rand((delta.shape[0], 1), device=delta.device, dtype=torch.float32) * epsilon
        delta = (flat * radius).reshape_as(delta).to(base.dtype)
        return delta * mask

    @staticmethod
    def _freeze_params(model: torch.nn.Module) -> dict[int, bool]:
        flags: dict[int, bool] = {}
        for p in model.parameters():
            flags[id(p)] = p.requires_grad
            p.requires_grad_(False)
        return flags

    @staticmethod
    def _restore_params(model: torch.nn.Module, flags: dict[int, bool]) -> None:
        for p in model.parameters():
            p.requires_grad_(flags.get(id(p), False))

    def _token_rows(self, logits: torch.Tensor, k: int) -> list[dict[str, Any]]:
        probs = F.softmax(logits.float(), dim=-1)
        vals, ids = torch.topk(probs, k=k, dim=-1)
        tok = self.state.tokenizer
        rows = []
        for p, idx in zip(vals[0].tolist(), ids[0].tolist()):
            text = tok.decode([idx], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            rows.append({"id": idx, "token": text, "prob": p})
        return rows


    @staticmethod
    def _find_subsequence(haystack: torch.Tensor, needle: torch.Tensor) -> tuple[int, int] | None:
        """Find a 1-D token-id subsequence and return [start, end)."""
        h = haystack.detach().cpu().tolist()
        n = needle.detach().cpu().tolist()
        if not n or len(n) > len(h):
            return None
        for i in range(len(h) - len(n) + 1):
            if h[i:i + len(n)] == n:
                return i, i + len(n)
        return None

    def _editable_positions(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor | None = None,
        last_n: int | None = None,
        excluded: set[int] | None = None,
    ) -> list[int]:
        tok = self.state.tokenizer
        specials = set(tok.all_special_ids or [])
        excluded = excluded or set()
        if mask is None:
            allowed = torch.ones_like(ids[0], dtype=torch.bool)
        else:
            allowed = mask.bool() if mask.ndim == 1 else mask[0].bool()
        positions = [
            i for i, token_id in enumerate(ids[0].tolist())
            if bool(allowed[i]) and token_id not in specials and i not in excluded
        ]
        if last_n and len(positions) > last_n:
            positions = positions[-last_n:]
        return positions

    @staticmethod
    def _sequence_label_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Per-sequence causal CE, respecting -100 labels."""
        shift_logits = logits[:, :-1, :].float()
        shift_labels = labels[:, 1:]
        vocab = shift_logits.shape[-1]
        per_token = F.cross_entropy(
            shift_logits.reshape(-1, vocab),
            shift_labels.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).reshape(shift_labels.shape)
        mask = (shift_labels != -100).float()
        return (per_token * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def _candidate_replacements(
        self,
        current_ids: torch.Tensor,
        grad: torch.Tensor,
        positions: list[int],
        candidate_top_k: int,
        verify_limit: int = 32,
    ) -> list[tuple[int, int, float]]:
        """HotFlip first-order proposals: g_i · (E_j - E_current)."""
        if not positions:
            return []
        weight = self.state.model.get_input_embeddings().weight.detach()
        pos_t = torch.tensor(positions, device=grad.device, dtype=torch.long)
        g = grad[0, pos_t, :].to(weight.dtype)
        # First-order loss change for i: current -> j is g_i · (E_j - E_current).
        approx = torch.matmul(g, weight.t())
        current_emb = weight[current_ids[0, pos_t]]
        approx = approx - (g * current_emb).sum(dim=-1, keepdim=True)
        replacement_mask = self.state.replacement_mask
        if (
            replacement_mask is None
            or replacement_mask.shape[0] != approx.shape[1]
            or replacement_mask.device != approx.device
        ):
            allowed_ids = latin_replacement_token_ids(
                self.state.tokenizer, approx.shape[1]
            )
            if not allowed_ids:
                raise RuntimeError("Tokenizer has no Latin replacement candidates")
            replacement_mask = torch.zeros(
                approx.shape[1], dtype=torch.bool, device=approx.device
            )
            replacement_mask[
                torch.tensor(allowed_ids, dtype=torch.long, device=approx.device)
            ] = True
            self.state.replacement_mask = replacement_mask
        approx.masked_fill_(~replacement_mask.unsqueeze(0), -torch.inf)
        special_ids = [i for i in (self.state.tokenizer.all_special_ids or []) if 0 <= i < approx.shape[1]]
        if special_ids:
            approx[:, torch.tensor(special_ids, device=approx.device)] = -torch.inf
        current = current_ids[0, pos_t]
        approx[torch.arange(len(positions), device=approx.device), current] = -torch.inf
        k = min(candidate_top_k, approx.shape[1] - 1)
        vals, ids = torch.topk(approx, k=k, dim=1)
        proposals: list[tuple[int, int, float]] = []
        for row, pos in enumerate(positions):
            for col in range(k):
                proposals.append((pos, int(ids[row, col].item()), float(vals[row, col].float().item())))
        proposals.sort(key=lambda x: x[2], reverse=True)
        return proposals[:verify_limit]

    def analyze_token_attack(
        self,
        prompt: str,
        max_token_changes: int,
        candidate_top_k: int,
        editable_last_n: int,
        top_k: int,
    ) -> dict[str, Any]:
        """Discrete white-box token substitution search using a HotFlip-style gradient score."""
        model = self.state.model
        tok = self.state.tokenizer
        device = next(model.parameters()).device
        formatted = self._chat_prompt(prompt)
        batch = tok(formatted, return_tensors="pt", add_special_tokens=False)
        original_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        raw_ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        span = self._find_subsequence(original_ids[0], raw_ids)
        editable_mask = torch.zeros_like(original_ids[0], dtype=torch.bool)
        if span:
            editable_mask[span[0]:span[1]] = True
        else:
            editable_mask[:] = True

        with torch.no_grad():
            clean_out = model(
                input_ids=original_ids, attention_mask=attention_mask, use_cache=False,
                output_hidden_states=True, return_dict=True,
            )
            clean_last = clean_out.logits[:, -1, :].detach()
            clean_probs = F.softmax(clean_last.float(), dim=-1).detach()
            clean_hidden = [h.detach() for h in clean_out.hidden_states]
            clean_target = int(clean_last.argmax(dim=-1).item())

        current_ids = original_ids.clone()
        changed_positions: set[int] = set()
        history: list[dict[str, float]] = []
        flags = self._freeze_params(model)
        was_training = model.training
        model.eval()
        try:
            for step in range(max_token_changes):
                positions = self._editable_positions(
                    current_ids, editable_mask, editable_last_n, changed_positions
                )
                if not positions:
                    break
                embeds = model.get_input_embeddings()(current_ids).detach().requires_grad_(True)
                out = model(
                    inputs_embeds=embeds, attention_mask=attention_mask, use_cache=False, return_dict=True
                )
                # Non-zero untargeted objective at the clean point: reduce probability of
                # the clean model's preferred next token.
                objective = F.cross_entropy(
                    out.logits[:, -1, :].float(),
                    torch.tensor([clean_target], device=device),
                )
                grad = torch.autograd.grad(objective, embeds, only_inputs=True)[0]
                proposals = self._candidate_replacements(
                    current_ids, grad, positions, candidate_top_k, verify_limit=16
                )
                if not proposals:
                    break

                candidate_ids = current_ids.repeat(len(proposals), 1)
                for row, (pos, replacement, _) in enumerate(proposals):
                    candidate_ids[row, pos] = replacement
                cand_mask = attention_mask.repeat(len(proposals), 1)
                with torch.no_grad():
                    cand_out = model(
                        input_ids=candidate_ids, attention_mask=cand_mask, use_cache=False, return_dict=True
                    )
                    target = torch.full(
                        (len(proposals),), clean_target, device=device, dtype=torch.long
                    )
                    exact = F.cross_entropy(
                        cand_out.logits[:, -1, :].float(), target, reduction="none"
                    )
                    best_row = int(exact.argmax().item())
                    best_loss = float(exact[best_row].item())

                current_loss = float(objective.detach().item())
                if best_loss <= current_loss + 1e-8:
                    break
                pos, replacement, _ = proposals[best_row]
                current_ids[0, pos] = replacement
                changed_positions.add(pos)

                with torch.no_grad():
                    step_out = model(
                        input_ids=current_ids, attention_mask=attention_mask, use_cache=False, return_dict=True
                    )
                    step_logp = F.log_softmax(step_out.logits[:, -1, :].float(), dim=-1)
                    step_kl = F.kl_div(step_logp, clean_probs, reduction="batchmean")
                history.append({
                    "step": step + 1,
                    "objective": best_loss,
                    "kl": float(step_kl.item()),
                    "hidden_cosine": 0.0,
                })
        finally:
            self._restore_params(model, flags)
            model.train(was_training)

        with torch.no_grad():
            adv_out = model(
                input_ids=current_ids, attention_mask=attention_mask, use_cache=False,
                output_hidden_states=True, return_dict=True,
            )
            adv_last = adv_out.logits[:, -1, :]
            adv_logp = F.log_softmax(adv_last.float(), dim=-1)
            final_kl = float(F.kl_div(adv_logp, clean_probs, reduction="batchmean").item())
            layer_drift = []
            for idx, (c, a) in enumerate(zip(clean_hidden, adv_out.hidden_states)):
                drift = 1.0 - F.cosine_similarity(
                    c[:, -1, :].float(), a[:, -1, :].float(), dim=-1
                ).mean()
                layer_drift.append({"layer": idx, "cosine_distance": float(drift.cpu())})

        changes = []
        for pos in sorted(changed_positions):
            old_id = int(original_ids[0, pos].item())
            new_id = int(current_ids[0, pos].item())
            changes.append({
                "position": pos,
                "from_id": old_id,
                "to_id": new_id,
                "from_token": tok.decode([old_id], skip_special_tokens=False, clean_up_tokenization_spaces=False),
                "to_token": tok.decode([new_id], skip_special_tokens=False, clean_up_tokenization_spaces=False),
            })

        if span:
            adversarial_prompt = tok.decode(
                current_ids[0, span[0]:span[1]].tolist(),
                skip_special_tokens=False, clean_up_tokenization_spaces=False,
            )
        else:
            adversarial_prompt = tok.decode(
                current_ids[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

        clean_top = self._token_rows(clean_last, top_k)
        adv_top = self._token_rows(adv_last, top_k)
        return {
            "attack_mode": "token_hotflip",
            "model": self.info(),
            "formatted_prompt": formatted,
            "adversarial_prompt": adversarial_prompt,
            "adversarial_formatted_prompt": tok.decode(
                current_ids[0].tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False
            ),
            "token_count": int(original_ids.shape[1]),
            "token_changes": changes,
            "input_ids_original": original_ids[0].detach().cpu().tolist(),
            "input_ids_adversarial": current_ids[0].detach().cpu().tolist(),
            "epsilon": None,
            "norm": "discrete",
            "delta_l2": None,
            "delta_linf": None,
            "final_kl": final_kl,
            "argmax_changed": clean_top[0]["id"] != adv_top[0]["id"],
            "clean_top": clean_top,
            "adversarial_top": adv_top,
            "history": history,
            "layer_drift": layer_drift,
        }

    def analyze_attack(
        self,
        prompt: str,
        mode: str,
        epsilon: float,
        step_size: float,
        steps: int,
        norm: str,
        objective: str,
        hidden_weight: float,
        top_k: int,
        random_start: bool,
        max_token_changes: int,
        candidate_top_k: int,
        editable_last_n: int,
    ) -> dict[str, Any]:
        with self.lock:
            self.ensure_loaded()
            if mode == "token_hotflip":
                return self.analyze_token_attack(
                    prompt=prompt,
                    max_token_changes=max_token_changes,
                    candidate_top_k=candidate_top_k,
                    editable_last_n=editable_last_n,
                    top_k=top_k,
                )
            model = self.state.model
            tok = self.state.tokenizer
            device = next(model.parameters()).device
            formatted = self._chat_prompt(prompt)
            batch = tok(formatted, return_tensors="pt", add_special_tokens=False)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            embed_layer = model.get_input_embeddings()
            with torch.no_grad():
                clean_embeds = embed_layer(input_ids).detach()
                clean_out = model(
                    inputs_embeds=clean_embeds,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                clean_last = clean_out.logits[:, -1, :].detach()
                clean_probs = F.softmax(clean_last.float(), dim=-1).detach()
                clean_hidden = [h.detach() for h in clean_out.hidden_states]

            mask = attention_mask.unsqueeze(-1).to(clean_embeds.dtype)
            delta = (
                self._random_delta_like(clean_embeds, epsilon, norm, mask)
                if random_start
                else torch.zeros_like(clean_embeds)
            )

            flags = self._freeze_params(model)
            was_training = model.training
            model.eval()
            history: list[dict[str, float]] = []
            try:
                for i in range(steps):
                    delta = delta.detach().requires_grad_(True)
                    out = model(
                        inputs_embeds=clean_embeds + delta,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    adv_last = out.logits[:, -1, :].float()
                    logp = F.log_softmax(adv_last, dim=-1)
                    kl = F.kl_div(logp, clean_probs, reduction="batchmean")
                    hidden_cos = 1.0 - F.cosine_similarity(
                        out.hidden_states[-1][:, -1, :].float(),
                        clean_hidden[-1][:, -1, :].float(),
                        dim=-1,
                    ).mean()
                    if objective == "next_token_kl":
                        score = kl
                    elif objective == "hidden_cosine":
                        score = hidden_cos
                    else:
                        score = kl + hidden_weight * hidden_cos

                    grad = torch.autograd.grad(score, delta, only_inputs=True)[0]
                    if norm == "linf":
                        delta = delta.detach() + step_size * grad.sign()
                    else:
                        gflat = grad.reshape(grad.shape[0], -1).float()
                        gflat = gflat / torch.linalg.vector_norm(gflat, ord=2, dim=1, keepdim=True).clamp_min(1e-12)
                        delta = delta.detach() + step_size * gflat.reshape_as(grad).to(grad.dtype)
                    delta = self._project(delta, epsilon, norm, mask).detach()
                    history.append({
                        "step": i + 1,
                        "objective": float(score.detach().cpu()),
                        "kl": float(kl.detach().cpu()),
                        "hidden_cosine": float(hidden_cos.detach().cpu()),
                    })
            finally:
                self._restore_params(model, flags)
                model.train(was_training)

            with torch.no_grad():
                adv_out = model(
                    inputs_embeds=clean_embeds + delta,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                adv_last = adv_out.logits[:, -1, :]
                layer_drift = []
                for idx, (c, a) in enumerate(zip(clean_hidden, adv_out.hidden_states)):
                    drift = 1.0 - F.cosine_similarity(
                        c[:, -1, :].float(), a[:, -1, :].float(), dim=-1
                    ).mean()
                    layer_drift.append({"layer": idx, "cosine_distance": float(drift.cpu())})

            clean_top = self._token_rows(clean_last, top_k)
            adv_top = self._token_rows(adv_last, top_k)
            clean_argmax = clean_top[0]
            adv_argmax = adv_top[0]
            delta_float = delta.float()
            l2 = float(torch.linalg.vector_norm(delta_float.reshape(delta.shape[0], -1), ord=2, dim=1).mean().cpu())
            linf = float(delta_float.abs().max().cpu())
            final_kl = history[-1]["kl"] if history else 0.0

            return {
                "attack_mode": "embedding_pgd",
                "model": self.info(),
                "formatted_prompt": formatted,
                "token_count": int(input_ids.shape[1]),
                "epsilon": epsilon,
                "norm": norm,
                "delta_l2": l2,
                "delta_linf": linf,
                "final_kl": final_kl,
                "argmax_changed": clean_argmax["id"] != adv_argmax["id"],
                "clean_top": clean_top,
                "adversarial_top": adv_top,
                "history": history,
                "layer_drift": layer_drift,
            }

    def _prepare_supervised_example(self, prompt: str, response: str, max_length: int) -> dict[str, torch.Tensor]:
        model = self.state.model
        tok = self.state.tokenizer
        device = next(model.parameters()).device

        prompt_text = self._chat_prompt(prompt)
        prompt_ids = tok(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        eos = tok.eos_token or ""
        response_ids = tok(response + eos, return_tensors="pt", add_special_tokens=False)["input_ids"]
        ids = torch.cat([prompt_ids, response_ids], dim=1)
        if ids.shape[1] > max_length:
            ids = ids[:, :max_length]
        prompt_len = min(prompt_ids.shape[1], ids.shape[1] - 1)
        if prompt_len <= 0 or ids.shape[1] <= prompt_len:
            raise ValueError("Example has no response tokens after truncation; increase max_length or shorten the prompt.")

        labels = ids.clone()
        labels[:, :prompt_len] = -100
        attention = torch.ones_like(ids)
        perturb_mask = torch.zeros((*ids.shape, 1), dtype=torch.float32)
        raw_prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        raw_span = self._find_subsequence(prompt_ids[0], raw_prompt_ids)
        if raw_span:
            perturb_mask[:, raw_span[0]:raw_span[1], :] = 1.0
        else:
            special_ids = set(tok.all_special_ids or [])
            for i, token_id in enumerate(ids[0, :prompt_len].tolist()):
                if token_id not in special_ids:
                    perturb_mask[:, i, :] = 1.0
        return {
            "input_ids": ids.to(device),
            "labels": labels.to(device),
            "attention_mask": attention.to(device),
            "perturb_mask": perturb_mask.to(device),
        }

    def _supervised_pgd(
        self,
        batch: dict[str, torch.Tensor],
        epsilon: float,
        step_size: float,
        steps: int,
        norm: str,
    ) -> tuple[torch.Tensor, list[float]]:
        model = self.state.model
        embeds = model.get_input_embeddings()(batch["input_ids"]).detach()
        mask = batch["perturb_mask"].to(embeds.dtype)
        delta = self._random_delta_like(embeds, epsilon, norm, mask)
        flags = self._freeze_params(model)
        was_training = model.training
        model.eval()
        losses: list[float] = []
        try:
            for _ in range(steps):
                delta = delta.detach().requires_grad_(True)
                out = model(
                    inputs_embeds=embeds + delta,
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    use_cache=False,
                    return_dict=True,
                )
                loss = out.loss
                grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]
                if norm == "linf":
                    delta = delta.detach() + step_size * grad.sign()
                else:
                    gflat = grad.reshape(grad.shape[0], -1).float()
                    gflat = gflat / torch.linalg.vector_norm(gflat, ord=2, dim=1, keepdim=True).clamp_min(1e-12)
                    delta = delta.detach() + step_size * gflat.reshape_as(grad).to(grad.dtype)
                delta = self._project(delta, epsilon, norm, mask).detach()
                losses.append(float(loss.detach().cpu()))
        finally:
            self._restore_params(model, flags)
            model.train(was_training)
        return delta, losses

    def _supervised_hotflip(
        self,
        batch: dict[str, torch.Tensor],
        max_changes: int,
        candidate_top_k: int,
        editable_last_n: int,
    ) -> tuple[torch.Tensor, list[float]]:
        model = self.state.model
        current_ids = batch["input_ids"].clone()
        editable_mask = batch["perturb_mask"][:, :, 0].bool()
        flags = self._freeze_params(model)
        was_training = model.training
        model.eval()
        losses: list[float] = []
        changed: set[int] = set()
        try:
            for _ in range(max_changes):
                positions = self._editable_positions(
                    current_ids, editable_mask, editable_last_n, changed
                )
                if not positions:
                    break
                embeds = model.get_input_embeddings()(current_ids).detach().requires_grad_(True)
                out = model(
                    inputs_embeds=embeds,
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    use_cache=False,
                    return_dict=True,
                )
                loss = out.loss
                grad = torch.autograd.grad(loss, embeds, only_inputs=True)[0]
                proposals = self._candidate_replacements(
                    current_ids, grad, positions, candidate_top_k, verify_limit=6
                )
                if not proposals:
                    break
                # Verify a small number one-at-a-time. This is slower than a giant batch,
                # but avoids materializing [candidates, sequence, vocabulary] logits.
                best_row = -1
                best_loss = -float("inf")
                with torch.no_grad():
                    for row, (pos, replacement, _) in enumerate(proposals):
                        candidate_ids = current_ids.clone()
                        candidate_ids[0, pos] = replacement
                        cand_out = model(
                            input_ids=candidate_ids,
                            attention_mask=batch["attention_mask"],
                            use_cache=False,
                            return_dict=True,
                        )
                        exact = float(
                            self._sequence_label_loss(cand_out.logits, batch["labels"])[0].item()
                        )
                        if exact > best_loss:
                            best_loss = exact
                            best_row = row
                current_loss = float(loss.detach().item())
                if best_loss <= current_loss + 1e-8:
                    break
                pos, replacement, _ = proposals[best_row]
                current_ids[0, pos] = replacement
                changed.add(pos)
                losses.append(best_loss)
        finally:
            self._restore_params(model, flags)
            model.train(was_training)
        return current_ids.detach(), losses

    @staticmethod
    def _masked_kl(clean_logits: torch.Tensor, adv_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        clean = clean_logits[:, :-1, :].float()
        adv = adv_logits[:, :-1, :].float()
        response_mask = (labels[:, 1:] != -100).float()
        teacher = F.softmax(clean, dim=-1)
        student_log = F.log_softmax(adv, dim=-1)
        per_token = F.kl_div(student_log, teacher, reduction="none").sum(dim=-1)
        denom = response_mask.sum().clamp_min(1.0)
        return (per_token * response_mask).sum() / denom

    @staticmethod
    def _masked_js(clean_logits: torch.Tensor, adv_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Bounded Jensen-Shannon divergence over supervised response positions."""
        clean = clean_logits[:, :-1, :].float().detach()
        adv = adv_logits[:, :-1, :].float()
        response_mask = (labels[:, 1:] != -100).float()
        clean_log = F.log_softmax(clean, dim=-1)
        adv_log = F.log_softmax(adv, dim=-1)
        clean_p = clean_log.exp()
        adv_p = adv_log.exp()
        mix = 0.5 * (clean_p + adv_p)
        log_mix = torch.log(mix.clamp_min(1e-12))
        js = 0.5 * (clean_p * (clean_log - log_mix)).sum(dim=-1)
        js = js + 0.5 * (adv_p * (adv_log - log_mix)).sum(dim=-1)
        denom = response_mask.sum().clamp_min(1.0)
        return (js * response_mask).sum() / denom

    def _ensure_lora(self, rank: int, alpha: int, dropout: float) -> None:
        if isinstance(self.state.model, PeftModel):
            self.state.has_lora = True
            return
        cfg = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules="all-linear",
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.state.model = get_peft_model(self.state.model, cfg)
        self.state.model.to(self._device())
        self.state.model.config.use_cache = False
        self.state.has_lora = True

    def train_lora(
        self,
        request: dict[str, Any],
        progress: Callable[[dict[str, Any]], None] | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            self.ensure_loaded()
            random.seed(request["seed"])
            torch.manual_seed(request["seed"])
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(request["seed"])

            self._ensure_lora(request["lora_rank"], request["lora_alpha"], request["lora_dropout"])
            model = self.state.model
            model.train()
            trainable = [p for p in model.parameters() if p.requires_grad]
            optimizer = AdamW(trainable, lr=request["learning_rate"])
            examples = list(request["examples"])
            total_steps = request["epochs"] * len(examples)
            step_idx = 0
            running: list[float] = []

            def emit(payload: dict[str, Any]) -> None:
                if progress:
                    progress(payload)

            emit({"message": f"LoRA ready: {sum(p.numel() for p in trainable):,} trainable parameters", "progress": 0.0})

            for epoch in range(request["epochs"]):
                random.shuffle(examples)
                for ex in examples:
                    batch = self._prepare_supervised_example(ex["prompt"], ex["response"], request["max_length"])
                    # First find a high-loss perturbation with model parameters frozen.
                    if request.get("attack_mode", "embedding_pgd") == "token_hotflip":
                        adv_ids, pgd_losses = self._supervised_hotflip(
                            batch,
                            request["max_token_changes"],
                            request["candidate_top_k"],
                            request["editable_last_n"],
                        )
                        delta = None
                    else:
                        delta, pgd_losses = self._supervised_pgd(
                            batch,
                            request["epsilon"],
                            request["step_size"],
                            request["attack_steps"],
                            request["norm"],
                        )
                        adv_ids = None

                    embeds = model.get_input_embeddings()(batch["input_ids"])
                    goal = request.get("training_goal", "sensitive")
                    if goal == "sensitive":
                        # Keep the clean response correct, while making the perturbed response
                        # distribution different. JS is bounded, so this is much more stable
                        # than doing unconstrained gradient ascent on model weights.
                        clean_out = model(
                            inputs_embeds=embeds,
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                            use_cache=False,
                            return_dict=True,
                        )
                        if adv_ids is not None:
                            adv_out = model(
                                input_ids=adv_ids,
                                attention_mask=batch["attention_mask"],
                                use_cache=False,
                                return_dict=True,
                            )
                        else:
                            adv_out = model(
                                inputs_embeds=embeds + delta.detach(),
                                attention_mask=batch["attention_mask"],
                                use_cache=False,
                                return_dict=True,
                            )
                        ce = clean_out.loss
                        consistency = self._masked_js(clean_out.logits, adv_out.logits, batch["labels"])
                        total = ce - request["robust_lambda"] * consistency
                    else:
                        with torch.no_grad():
                            clean_out = model(
                                inputs_embeds=embeds.detach(),
                                attention_mask=batch["attention_mask"],
                                use_cache=False,
                                return_dict=True,
                            )
                            clean_logits = clean_out.logits.detach()
                        if adv_ids is not None:
                            adv_out = model(
                                input_ids=adv_ids,
                                attention_mask=batch["attention_mask"],
                                labels=batch["labels"],
                                use_cache=False,
                                return_dict=True,
                            )
                        else:
                            adv_out = model(
                                inputs_embeds=embeds + delta.detach(),
                                attention_mask=batch["attention_mask"],
                                labels=batch["labels"],
                                use_cache=False,
                                return_dict=True,
                            )
                        ce = adv_out.loss
                        consistency = self._masked_kl(clean_logits, adv_out.logits, batch["labels"])
                        total = ce + request["robust_lambda"] * consistency

                    optimizer.zero_grad(set_to_none=True)
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()

                    step_idx += 1
                    loss_value = float(total.detach().cpu())
                    running.append(loss_value)
                    emit({
                        "message": (
                            f"epoch {epoch + 1}/{request['epochs']} step {step_idx}/{total_steps} "
                            f"loss={loss_value:.4f} adv_ce={float(ce.detach().cpu()):.4f} "
                            f"div={float(consistency.detach().cpu()):.4f} "
                            f"goal={request.get('training_goal', 'sensitive')} "
                            f"attack={request.get('attack_mode', 'embedding_pgd')} "
                            f"attack_max={max(pgd_losses) if pgd_losses else float(ce.detach().cpu()):.4f}"
                        ),
                        "progress": step_idx / max(total_steps, 1),
                        "loss": loss_value,
                    })

            model.eval()
            jid = job_id or uuid.uuid4().hex[:12]
            out_dir = self.adapter_root / jid
            out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out_dir)
            self.state.tokenizer.save_pretrained(out_dir)
            metrics = {
                "adapter_path": str(out_dir),
                "steps": total_steps,
                "mean_loss": sum(running) / max(len(running), 1),
                "final_loss": running[-1] if running else None,
                "trainable_parameters": sum(p.numel() for p in trainable),
            }
            with open(out_dir / "training_summary.json", "w", encoding="utf-8") as f:
                json.dump({"request": request, "metrics": metrics}, f, indent=2)
            emit({"message": f"Saved adapter to {out_dir}", "progress": 1.0, "done": True, "metrics": metrics})
            return metrics
