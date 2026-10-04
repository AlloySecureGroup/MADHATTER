from pathlib import Path


def test_public_branding_describes_broader_resilience_lab():
    readme = Path("README.md").read_text()
    html = Path("app/static/index.html").read_text()

    assert readme.startswith("# MadHatter — Adversarial Model Resilience Lab")
    assert "Open-Weight Adversarial Resilience Lab" in html
    assert "Qwen Token Perturbation Lab" not in readme
    assert "Qwen Token Perturbation Lab" not in html
