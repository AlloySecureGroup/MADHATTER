import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .engine import AdversarialModelEngine
from .model_registry import model_options
from .schemas import AttackRequest, LoadRequest, TrainRequest

app = FastAPI(title="MadHatter Multi-Model Resilience Lab", version="3.0.0")
engine = AdversarialModelEngine()
STATIC = Path(__file__).parent / "static"

jobs: dict[str, dict[str, Any]] = {}
jobs_lock = threading.Lock()


def update_job(job_id: str, payload: dict[str, Any]) -> None:
    with jobs_lock:
        job = jobs[job_id]
        if "message" in payload:
            job["logs"].append(payload["message"])
            job["logs"] = job["logs"][-200:]
        for key in ("progress", "loss", "metrics"):
            if key in payload:
                job[key] = payload[key]
        if payload.get("done"):
            job["status"] = "complete"
            job["finished_at"] = time.time()


def training_worker(job_id: str, data: dict[str, Any]) -> None:
    try:
        update_job(job_id, {"message": "Training job started", "progress": 0.0})
        metrics = engine.train_lora(data, progress=lambda p: update_job(job_id, p), job_id=job_id)
        update_job(job_id, {"done": True, "progress": 1.0, "metrics": metrics})
    except Exception as exc:
        with jobs_lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = f"{type(exc).__name__}: {exc}"
            jobs[job_id]["logs"].append(jobs[job_id]["error"])
            jobs[job_id]["finished_at"] = time.time()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")




@app.get("/api/models")
def models() -> dict[str, Any]:
    return {"ok": True, "models": model_options()}


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "model": engine.info()}


@app.post("/api/load")
def load(req: LoadRequest) -> dict[str, Any]:
    try:
        info = engine.load_model(req.model_id)
        return {"ok": True, "model": info}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/api/analyze")
def analyze(req: AttackRequest) -> dict[str, Any]:
    try:
        result = engine.analyze_attack(**req.model_dump())
        return {"ok": True, "result": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/api/train")
def train(req: TrainRequest) -> dict[str, Any]:
    with jobs_lock:
        active = any(v["status"] in ("queued", "running") for v in jobs.values())
        if active:
            raise HTTPException(status_code=409, detail="A training job is already running.")
        job_id = uuid.uuid4().hex[:12]
        jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "progress": 0.0,
            "loss": None,
            "logs": [],
            "metrics": None,
            "error": None,
            "created_at": time.time(),
        }

    def runner() -> None:
        with jobs_lock:
            jobs[job_id]["status"] = "running"
        training_worker(job_id, req.model_dump())

    threading.Thread(target=runner, daemon=True).start()
    return {"ok": True, "job_id": job_id}


@app.get("/api/train/{job_id}")
def train_status(job_id: str) -> dict[str, Any]:
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Unknown job id")
        return {"ok": True, "job": dict(jobs[job_id])}
