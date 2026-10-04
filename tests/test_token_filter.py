from app.token_filter import is_latin_replacement


def test_root_engine_uses_latin_character_policy():
    assert is_latin_replacement("Readable café / 42")
    assert not is_latin_replacement("漢字")
    assert not is_latin_replacement("кириллица")
    assert not is_latin_replacement("🙂")
