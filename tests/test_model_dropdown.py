from pathlib import Path

from app.model_registry import (
    DEFAULT_MODEL_ID,
    MODEL_OPTIONS,
    enabled_model_ids,
    is_curated_model,
    model_options,
)


def test_curated_model_registry_is_unique_and_has_default():
    ids = [model["id"] for model in MODEL_OPTIONS]
    assert len(ids) == 9
    assert len(ids) == len(set(ids))
    assert DEFAULT_MODEL_ID in ids
    assert "HuggingFaceTB/SmolLM3-3B" in ids
    assert is_curated_model(DEFAULT_MODEL_ID)
    assert not is_curated_model("/models/operator-checkpoint")
    enabled = [model["id"] for model in model_options()]
    assert enabled == [
        "HuggingFaceTB/SmolLM2-360M-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct",
        DEFAULT_MODEL_ID,
    ]
    assert "HuggingFaceTB/SmolLM3-3B" not in enabled


def test_models_file_keeps_order_and_skips_comments(tmp_path, monkeypatch):
    listing = tmp_path / "models.txt"
    listing.write_text(
        "\n".join(
            [
                "# hidden",
                "Qwen/Qwen3-1.7B",
                "not-a-model",
                "",
                "Qwen/Qwen3-1.7B",
                "HuggingFaceTB/SmolLM2-360M-Instruct  # first small model",
            ]
        )
    )
    monkeypatch.setenv("MODELS_FILE", str(listing))

    assert enabled_model_ids() == [
        "Qwen/Qwen3-1.7B",
        "not-a-model",
        "HuggingFaceTB/SmolLM2-360M-Instruct",
    ]
    assert [model["id"] for model in model_options()] == [
        "Qwen/Qwen3-1.7B",
        "HuggingFaceTB/SmolLM2-360M-Instruct",
    ]
    assert is_curated_model("Qwen/Qwen3-1.7B")
    assert not is_curated_model(DEFAULT_MODEL_ID)


def test_empty_models_file_falls_back_to_default(tmp_path, monkeypatch):
    listing = tmp_path / "models.txt"
    listing.write_text("# nothing enabled\n\n")
    monkeypatch.setenv("MODELS_FILE", str(listing))

    assert [model["id"] for model in model_options()] == [DEFAULT_MODEL_ID]
    assert is_curated_model(DEFAULT_MODEL_ID)


def test_main_ui_loads_model_dropdown_from_api():
    html = Path("app/static/index.html").read_text()
    main = Path("app/main.py").read_text()

    assert '<select id="modelId">' in html
    assert "loadModelOptions()" in html
    assert "if(availableModels[0])$('#modelId').value=availableModels[0].id" in html
    assert "/api/models" in html
    assert '@app.get("/api/models")' in main
    assert 'id="modeNotice"' in html
    assert "Discrete mode ignores ε" in html
    assert "renderPromptDiff" in html
    assert "renderGenerationDiff" in html


def test_ui_loads_both_services_when_model_selection_changes():
    html = Path("app/static/index.html").read_text()
    clean = Path("app/clean_server.py").read_text()

    assert "$('#modelId').onchange=" in html
    assert "loadSelectedModel()" in html
    assert "api('/api/load'" in html
    assert "cleanApi('/api/load'" in html
    assert "Promise.allSettled([attack,validator])" in html
    assert ">Load model</button>" in html
    assert "Loading model…" in html
    assert "Load both" not in html
    assert "Loading both" not in html
    assert "Loading ${modelId}…" in html
    assert "${m.model_id} loaded · ${m.device}" in html
    assert "Failed to load ${modelId}" in html
    assert "Loading model for attack service :8000" in html
    assert "Loading model for clean validator :8001" in html
    assert "Model load failed for attack service :8000" in html
    assert "Model load failed for clean validator :8001" in html
    assert "Model load did not complete:" in html
    assert '@app.post("/api/load")' in clean
    assert "is_curated_model(req.model_id)" in clean


def test_ui_clears_stale_attack_results_on_model_switch():
    html = Path("app/static/index.html").read_text()

    assert "lastAttackResult=null" in html
    assert "$('#attackResults').classList.add('hidden')" in html
    assert "$('#cleanValidation').classList.add('hidden')" in html
    assert "$('#attackError').classList.add('hidden')" in html
    assert "if(sequence!==modelLoadSequence)return" in html


def test_clean_load_keeps_current_model_and_locks_compare():
    clean = Path("app/clean_server.py").read_text()

    assert "model_id = model_id or self.model_id or os.getenv(\"MODEL_ID\", DEFAULT_MODEL_ID)" in clean
    assert "self.unload()" in clean
    assert "with clean.lock:" in clean
    assert "if req.expected_model_id and req.expected_model_id != clean.model_id:" in clean
    assert "Clean validator only accepts curated model IDs" in clean


def test_compose_does_not_use_model_id_for_service_synchronization():
    dockerfile = Path("Dockerfile").read_text()
    assert "COPY models.txt ./models.txt" in dockerfile
    for compose_file in ("docker-compose.yml", "docker-compose.cpu.yml"):
        text = Path(compose_file).read_text()
        assert "MODEL_ID" not in text
        assert text.count("./models.txt:/workspace/models.txt:ro") == 2
