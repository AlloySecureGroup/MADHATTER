from pathlib import Path

from app.model_registry import DEFAULT_MODEL_ID, MODEL_OPTIONS


def test_curated_model_registry_is_unique_and_has_default():
    ids = [model["id"] for model in MODEL_OPTIONS]
    assert len(ids) == 9
    assert len(ids) == len(set(ids))
    assert DEFAULT_MODEL_ID in ids
    assert "HuggingFaceTB/SmolLM3-3B" in ids


def test_main_ui_loads_model_dropdown_from_api():
    html = Path("app/static/index.html").read_text()
    main = Path("app/main.py").read_text()

    assert '<select id="modelId">' in html
    assert "loadModelOptions()" in html
    assert "/api/models" in html
    assert '@app.get("/api/models")' in main
    assert 'id="modeNotice"' in html
    assert "Discrete mode ignores ε" in html
    assert "renderPromptDiff" in html
    assert "renderGenerationDiff" in html
