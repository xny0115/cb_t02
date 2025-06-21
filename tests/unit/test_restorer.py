from src.utils.restorer import restore_sentence


def test_restore_sentence_cases():
    cases = [
        (
            [{"lemma": "소고기", "pos": "NNG"}, {"lemma": "익히", "pos": "VV"}],
            "소고기를 익히다",
        ),
        (
            [{"lemma": "나", "pos": "NNG"}, {"lemma": "피곤하", "pos": "VA"}],
            "나는 피곤하다",
        ),
        (
            [
                {"lemma": "학생", "pos": "NNG"},
                {"lemma": "책", "pos": "NNG"},
                {"lemma": "읽", "pos": "VV"},
            ],
            "학생은 책을 읽다",
        ),
        (
            [{"lemma": "날씨", "pos": "NNG"}, {"lemma": "춥", "pos": "VA"}],
            "날씨는 춥다",
        ),
        (
            [{"lemma": "사과", "pos": "NNG"}, {"lemma": "맛있", "pos": "VA"}],
            "사과는 맛있다",
        ),
    ]
    for tokens, expected in cases:
        assert restore_sentence(tokens) == expected
