from app.token_filter import is_latin_replacement, latin_replacement_token_ids


class FakeTokenizer:
    all_special_ids = [0]
    values = {
        0: "<special>",
        1: " hello",
        2: "café",
        3: "世界",
        4: "Привет",
        5: "مرحبا",
        6: "🙂",
        7: " /-42",
        8: "�",
    }

    def decode(self, token_ids, **_kwargs):
        return self.values[token_ids[0]]


def test_latin_replacement_character_policy():
    assert is_latin_replacement("Hello, café / 42")
    assert is_latin_replacement("\n\t")
    assert not is_latin_replacement("世界")
    assert not is_latin_replacement("カタカナ")
    assert not is_latin_replacement("한글")
    assert not is_latin_replacement("Привет")
    assert not is_latin_replacement("مرحبا")
    assert not is_latin_replacement("🙂")
    assert not is_latin_replacement("�")


def test_token_filter_excludes_special_and_non_latin_tokens():
    assert latin_replacement_token_ids(FakeTokenizer(), 9) == [1, 2, 7]
