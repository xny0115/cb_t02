from src.utils.restorer import restore_sentence

# 10 example token lists

test_inputs = [
    [
        {"lemma": "사과", "pos": "NNG"},
        {"lemma": "는", "pos": "JX"},
        {"lemma": "몸", "pos": "NNG"},
        {"lemma": "에", "pos": "JKB"},
        {"lemma": "좋", "pos": "VA"},
        {"lemma": "아", "pos": "EF"},
    ],
    [
        {"lemma": "운동", "pos": "NNG"},
        {"lemma": "을", "pos": "JKO"},
        {"lemma": "규칙적", "pos": "XR"},
        {"lemma": "으로", "pos": "JKB"},
        {"lemma": "하", "pos": "VV"},
        {"lemma": "아", "pos": "EC"},
        {"lemma": "건강", "pos": "NNG"},
        {"lemma": "하", "pos": "XSA"},
        {"lemma": "아", "pos": "EF"},
    ],
    [
        {"lemma": "책", "pos": "NNG"},
        {"lemma": "은", "pos": "JX"},
        {"lemma": "읽", "pos": "VV"},
        {"lemma": "어야", "pos": "EC"},
        {"lemma": "지식", "pos": "NNG"},
        {"lemma": "이", "pos": "JKS"},
        {"lemma": "늘", "pos": "VV"},
        {"lemma": "어", "pos": "EF"},
    ],
    [
        {"lemma": "과학", "pos": "NNG"},
        {"lemma": "은", "pos": "JX"},
        {"lemma": "실험", "pos": "NNG"},
        {"lemma": "을", "pos": "JKO"},
        {"lemma": "통해", "pos": "VV"},
        {"lemma": "", "pos": "EC"},
        {"lemma": "발전", "pos": "NNG"},
        {"lemma": "하", "pos": "XSA"},
        {"lemma": "아", "pos": "EF"},
    ],
    [
        {"lemma": "음악", "pos": "NNG"},
        {"lemma": "은", "pos": "JX"},
        {"lemma": "들", "pos": "VV"},
        {"lemma": "어야", "pos": "EC"},
        {"lemma": "마음", "pos": "NNG"},
        {"lemma": "이", "pos": "JKS"},
        {"lemma": "편안", "pos": "VA"},
        {"lemma": "하", "pos": "XSA"},
        {"lemma": "아", "pos": "EF"},
    ],
    [
        {"lemma": "고기", "pos": "NNG"},
        {"lemma": "는", "pos": "JX"},
        {"lemma": "익", "pos": "VV"},
        {"lemma": "어야", "pos": "EC"},
        {"lemma": "안전", "pos": "NNG"},
        {"lemma": "하", "pos": "XSA"},
        {"lemma": "아", "pos": "EF"},
    ],
    [
        {"lemma": "역사", "pos": "NNG"},
        {"lemma": "공부", "pos": "NNG"},
        {"lemma": "는", "pos": "JX"},
        {"lemma": "계속", "pos": "MAG"},
        {"lemma": "되", "pos": "VV"},
        {"lemma": "어야", "pos": "EC"},
        {"lemma": "이해", "pos": "NNG"},
        {"lemma": "가", "pos": "JKS"},
        {"lemma": "깊", "pos": "VA"},
        {"lemma": "어", "pos": "EF"},
    ],
    [
        {"lemma": "여행", "pos": "NNG"},
        {"lemma": "은", "pos": "JX"},
        {"lemma": "계획", "pos": "NNG"},
        {"lemma": "하", "pos": "XSV"},
        {"lemma": "아야", "pos": "EC"},
        {"lemma": "즐겁", "pos": "VA"},
        {"lemma": "다", "pos": "EF"},
    ],
    [
        {"lemma": "치즈", "pos": "NNG"},
        {"lemma": "는", "pos": "JX"},
        {"lemma": "숙성", "pos": "NNG"},
        {"lemma": "되", "pos": "VV"},
        {"lemma": "어야", "pos": "EC"},
        {"lemma": "맛", "pos": "NNG"},
        {"lemma": "있", "pos": "VA"},
        {"lemma": "어", "pos": "EF"},
    ],
    [
        {"lemma": "언어", "pos": "NNG"},
        {"lemma": "는", "pos": "JX"},
        {"lemma": "연습", "pos": "NNG"},
        {"lemma": "하", "pos": "XSV"},
        {"lemma": "아야", "pos": "EC"},
        {"lemma": "능숙", "pos": "VA"},
        {"lemma": "하", "pos": "XSA"},
        {"lemma": "아", "pos": "EF"},
    ],
]


def test_restore_sentence_examples():
    for item in test_inputs:
        out = restore_sentence(item)
        assert isinstance(out, str)
        assert out
