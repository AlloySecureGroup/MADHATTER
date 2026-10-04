from app.model_registry import DEFAULT_MODEL_ID, MODEL_OPTIONS, MODEL_BY_ID, model_metadata


def test_curated_model_count_and_uniqueness():
    assert 5 <= len(MODEL_OPTIONS) <= 10
    ids = [m["id"] for m in MODEL_OPTIONS]
    assert len(ids) == len(set(ids))
    assert DEFAULT_MODEL_ID in ids
    assert "HuggingFaceTB/SmolLM3-3B" in ids


def test_registry_contract():
    for model in MODEL_OPTIONS:
        assert model["params_b"] > 0
        assert model["license"] == "Apache-2.0"
        assert model["family"]
        assert model["name"]
        assert model_metadata(model["id"])["id"] == model["id"]
        assert MODEL_BY_ID[model["id"]]["name"] == model["name"]
