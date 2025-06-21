from src.data.morph import analyze


def test_morph_pipeline():
    tokens = analyze("안녕, 세계!!! ㅋㅋㅋㅋ")
    assert tokens
    assert all(not t['pos'].startswith('S') for t in tokens)
