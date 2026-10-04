from app.research_metrics import text_diff


def test_text_diff_preserves_text_and_marks_changed_segments():
    result = text_diff(
        "The agreement lasts three years.",
        "The agreement lasts three transitions.",
    )

    assert "".join(part["text"] for part in result["original"]) == (
        "The agreement lasts three years."
    )
    assert "".join(part["text"] for part in result["adversarial"]) == (
        "The agreement lasts three transitions."
    )
    assert any(part["changed"] for part in result["original"])
    assert any(part["changed"] for part in result["adversarial"])
    assert result["exact_match"] is False
    assert 0.0 < result["similarity"] < 1.0


def test_identical_text_has_no_highlighted_segments():
    result = text_diff("same output", "same output")

    assert result["exact_match"] is True
    assert result["similarity"] == 1.0
    assert not any(part["changed"] for part in result["original"])
