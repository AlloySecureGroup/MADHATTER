import gc
import os
import threading
import time
import uuid
from difflib import SequenceMatcher
from typing import Any, Literal

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from transformers import AutoModelForCausalLM, AutoTokenizer

from .model_registry import DEFAULT_MODEL_ID, MODEL_BY_ID, model_metadata, model_options


class LoadRequest(BaseModel):
    model_id: str = Field(default=DEFAULT_MODEL_ID, min_length=1, max_length=1000)


class EvaluateRequest(BaseModel):
    model_id: str | None = Field(default=None, max_length=1000)
    mode: Literal["text", "token_ids"] = "text"
    prompt: str | None = Field(default=None, max_length=12000)
    input_ids: list[int] | None = Field(default=None, max_length=12000)
    top_k: int = Field(default=8, ge=1, le=30)
    max_new_tokens: int = Field(default=48, ge=1, le=256)

    @model_validator(mode="after")
    def validate_source(self):
        if self.mode == "text" and not self.prompt:
            raise ValueError("prompt is required for text mode")
        if self.mode == "token_ids" and not self.input_ids:
            raise ValueError("input_ids are required for token_ids mode")
        return self


class CompareRequest(BaseModel):
    expected_model_id: str | None = Field(default=None, max_length=1000)
    original_prompt: str = Field(min_length=1, max_length=12000)
    adversarial_prompt: str = Field(min_length=1, max_length=12000)
    adversarial_input_ids: list[int] = Field(min_length=1, max_length=12000)
    top_k: int = Field(default=8, ge=1, le=30)
    max_new_tokens: int = Field(default=48, ge=1, le=256)


class ResilienceRequest(BaseModel):
    source_model_id: str | None = Field(default=None, max_length=1000)
    original_prompt: str = Field(min_length=1, max_length=12000)
    adversarial_prompt: str = Field(min_length=1, max_length=12000)
    model_ids: list[str] = Field(min_length=1, max_length=10)
    top_k: int = Field(default=8, ge=1, le=30)
    max_new_tokens: int = Field(default=32, ge=1, le=128)


class CleanModelRunner:
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.model_id = None
        self.device = "cpu"
        self.dtype = "float32"
        self.lock = threading.RLock()
        self.force_cpu = os.getenv("FORCE_CPU", "0") == "1"

    def info(self) -> dict[str, Any]:
        gpu_name = None
        if torch.cuda.is_available() and not self.force_cpu:
            gpu_name = torch.cuda.get_device_name(0)
        return {
            "loaded": self.model is not None,
            "model_id": self.model_id,
            "device": self.device,
            "dtype": self.dtype,
            "clean_instance": True,
            "adapters_loaded": False,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": gpu_name,
            "registry": model_metadata(self.model_id),
        }

    def unload(self) -> None:
        with self.lock:
            self.model = None
            self.tokenizer = None
            self.model_id = None
            self.device = "cpu"
            self.dtype = "float32"
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load(self, model_id: str | None = None) -> dict[str, Any]:
        with self.lock:
            model_id = model_id or os.getenv("MODEL_ID", DEFAULT_MODEL_ID)
            if self.model is not None and self.model_id == model_id:
                return self.info()
            self.unload()
            device = torch.device("cpu") if self.force_cpu or not torch.cuda.is_available() else torch.device("cuda")
            if device.type == "cuda":
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                dtype = torch.float32
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
            )
            model.to(device)
            model.eval()
            model.config.use_cache = True
            self.model = model
            self.tokenizer = tokenizer
            self.model_id = model_id
            self.device = str(device)
            self.dtype = str(dtype).replace("torch.", "")
            return self.info()

    def _chat_prompt(self, prompt: str) -> str:
        tok = self.tokenizer
        messages = [{"role": "user", "content": prompt}]
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            return tok.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except (TypeError, ValueError):
            try:
                return tok.apply_chat_template(messages, **kwargs)
            except Exception:
                return prompt

    def _top_rows(self, logits: torch.Tensor, top_k: int) -> list[dict[str, Any]]:
        probs = F.softmax(logits.float(), dim=-1)[0]
        values, ids = torch.topk(probs, k=min(top_k, probs.shape[-1]))
        return [
            {
                "id": int(token_id),
                "token": self.tokenizer.decode(
                    [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
                ),
                "prob": float(value),
            }
            for value, token_id in zip(values.tolist(), ids.tolist())
        ]

    def _evaluate_locked(self, request: EvaluateRequest, return_probs: bool = False):
        model = self.model
        tok = self.tokenizer
        device = next(model.parameters()).device
        if request.mode == "text":
            formatted = self._chat_prompt(request.prompt or "")
            batch = tok(formatted, return_tensors="pt", add_special_tokens=False)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            source_text = request.prompt or ""
        else:
            input_ids = torch.tensor([request.input_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            formatted = tok.decode(request.input_ids or [], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            source_text = formatted

        vocab_size = int(model.get_input_embeddings().weight.shape[0])
        if input_ids.numel() == 0:
            raise ValueError("Input token sequence is empty")
        if int(input_ids.min()) < 0 or int(input_ids.max()) >= vocab_size:
            raise ValueError(f"Token IDs are incompatible with {self.model_id} vocabulary (0..{vocab_size - 1}).")

        with torch.inference_mode():
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
            next_logits = out.logits[:, -1, :].float()
            probs = F.softmax(next_logits, dim=-1)
            top = self._top_rows(next_logits, request.top_k)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=request.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id,
            )
        continuation_ids = generated[0, input_ids.shape[1]:].detach().cpu().tolist()
        result = {
            "model": self.info(),
            "mode": request.mode,
            "source_text": source_text,
            "formatted_input": formatted,
            "input_ids": input_ids[0].detach().cpu().tolist(),
            "input_token_count": int(input_ids.shape[1]),
            "next_top": top,
            "next_argmax": top[0] if top else None,
            "generated_token_ids": continuation_ids,
            "generated_text": tok.decode(continuation_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
        }
        return (result, probs.detach().cpu()) if return_probs else result

    def evaluate(self, request: EvaluateRequest) -> dict[str, Any]:
        with self.lock:
            self.load(request.model_id)
            return self._evaluate_locked(request)

    def compare_text(self, model_id: str, original: str, adversarial: str, top_k: int, max_new_tokens: int) -> dict[str, Any]:
        with self.lock:
            self.load(model_id)
            a, pa = self._evaluate_locked(EvaluateRequest(
                model_id=model_id, mode="text", prompt=original, top_k=top_k, max_new_tokens=max_new_tokens
            ), return_probs=True)
            b, pb = self._evaluate_locked(EvaluateRequest(
                model_id=model_id, mode="text", prompt=adversarial, top_k=top_k, max_new_tokens=max_new_tokens
            ), return_probs=True)
            mix = 0.5 * (pa + pb)
            js = 0.5 * (pa * (torch.log(pa.clamp_min(1e-12)) - torch.log(mix.clamp_min(1e-12)))).sum()
            js += 0.5 * (pb * (torch.log(pb.clamp_min(1e-12)) - torch.log(mix.clamp_min(1e-12)))).sum()
            similarity = SequenceMatcher(None, a["generated_text"], b["generated_text"]).ratio()
            return {
                "model_id": model_id,
                "metadata": model_metadata(model_id),
                "device": self.device,
                "original": a,
                "adversarial": b,
                "next_argmax_changed": a["next_argmax"]["id"] != b["next_argmax"]["id"],
                "next_token_js": float(js),
                "generation_similarity": float(similarity),
                "generation_changed": a["generated_text"] != b["generated_text"],
                "token_count_delta": b["input_token_count"] - a["input_token_count"],
            }


clean = CleanModelRunner()
app = FastAPI(title="MadHatter Clean Multi-Model Validator", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

jobs: dict[str, dict[str, Any]] = {}
jobs_lock = threading.Lock()


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "MadHatter clean multi-model validator",
        "port": 8001,
        "purpose": "Run perturbations against untouched open-weight checkpoints and measure transfer/resilience.",
        "model": clean.info(),
        "models": model_options(),
    }


@app.get("/api/models")
def models() -> dict[str, Any]:
    return {"ok": True, "models": model_options()}


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "model": clean.info()}


@app.post("/api/load")
def load(req: LoadRequest) -> dict[str, Any]:
    try:
        return {"ok": True, "model": clean.load(req.model_id)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/api/evaluate")
def evaluate(req: EvaluateRequest) -> dict[str, Any]:
    try:
        return {"ok": True, "result": clean.evaluate(req)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/api/compare")
def compare(req: CompareRequest) -> dict[str, Any]:
    try:
        target = req.expected_model_id or os.getenv("MODEL_ID", DEFAULT_MODEL_ID)
        clean.load(target)
        original = clean.evaluate(EvaluateRequest(model_id=target, mode="text", prompt=req.original_prompt, top_k=req.top_k, max_new_tokens=req.max_new_tokens))
        adversarial_text = clean.evaluate(EvaluateRequest(model_id=target, mode="text", prompt=req.adversarial_prompt, top_k=req.top_k, max_new_tokens=req.max_new_tokens))
        adversarial_exact = clean.evaluate(EvaluateRequest(model_id=target, mode="token_ids", input_ids=req.adversarial_input_ids, top_k=req.top_k, max_new_tokens=req.max_new_tokens))
        return {"ok": True, "result": {
            "model": clean.info(),
            "original": original,
            "adversarial_text": adversarial_text,
            "adversarial_exact": adversarial_exact,
            "text_roundtrip_matches_exact": adversarial_text["input_ids"] == req.adversarial_input_ids,
            "original_vs_text_argmax_changed": original["next_argmax"]["id"] != adversarial_text["next_argmax"]["id"],
            "original_vs_exact_argmax_changed": original["next_argmax"]["id"] != adversarial_exact["next_argmax"]["id"],
        }}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


def _resilience_worker(job_id: str, req: ResilienceRequest) -> None:
    rows = []
    try:
        for idx, model_id in enumerate(req.model_ids):
            if model_id not in MODEL_BY_ID:
                raise ValueError(f"Resilience matrix only accepts curated model IDs; unknown: {model_id}")
            with jobs_lock:
                jobs[job_id]["status"] = "running"
                jobs[job_id]["current_model"] = model_id
                jobs[job_id]["progress"] = idx / len(req.model_ids)
            row = clean.compare_text(model_id, req.original_prompt, req.adversarial_prompt, req.top_k, req.max_new_tokens)
            rows.append(row)
            with jobs_lock:
                jobs[job_id]["rows"] = list(rows)
                jobs[job_id]["progress"] = (idx + 1) / len(req.model_ids)
        with jobs_lock:
            jobs[job_id]["status"] = "complete"
            jobs[job_id]["finished_at"] = time.time()
    except Exception as exc:
        with jobs_lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = f"{type(exc).__name__}: {exc}"
            jobs[job_id]["finished_at"] = time.time()


@app.post("/api/resilience/start")
def resilience_start(req: ResilienceRequest) -> dict[str, Any]:
    unique = list(dict.fromkeys(req.model_ids))
    req.model_ids = unique
    job_id = uuid.uuid4().hex[:12]
    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "progress": 0.0,
            "current_model": None,
            "source_model_id": req.source_model_id,
            "rows": [],
            "error": None,
            "created_at": time.time(),
        }
    threading.Thread(target=_resilience_worker, args=(job_id, req), daemon=True).start()
    return {"ok": True, "job_id": job_id}


@app.get("/api/resilience/{job_id}")
def resilience_status(job_id: str) -> dict[str, Any]:
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Unknown resilience job id")
        return {"ok": True, "job": dict(jobs[job_id])}
