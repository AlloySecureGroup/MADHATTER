from pathlib import Path


def test_ui_exposes_model_and_resilience_controls():
    html = Path("app/static/index.html").read_text()
    assert 'id="modelId"' in html
    assert 'id="resilience"' in html
    assert 'id="modelChecklist"' in html
    assert '/api/resilience/start' in html
