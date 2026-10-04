from pathlib import Path


def test_readme_links_to_illustrated_tutorial():
    readme = Path("README.md").read_text()
    tutorial = Path("docs/TUTORIAL.md").read_text()

    assert "(docs/TUTORIAL.md)" in readme
    assert "images/discrete-attack.png" in tutorial
    assert "images/clean-validation.png" in tutorial
    assert Path("docs/images/discrete-attack.png").is_file()
    assert Path("docs/images/clean-validation.png").is_file()
