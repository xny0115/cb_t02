from src.utils.restorer import restore_with_template


def test_restore_with_template_cases():
    cases = [
        (
            [{"lemma": "소고기", "pos": "NNG"}],
            ["단백질"],
            "음식",
            "소고기에는 단백질이 많아",
        ),
        (
            [{"lemma": "샐러드", "pos": "NNG"}],
            ["채식"],
            "음식",
            "샐러드는 채식 메뉴야",
        ),
        (
            [{"lemma": "제주도", "pos": "NNG"}],
            ["휴양"],
            "여행",
            "제주도에서 휴양을 즐길 수 있어",
        ),
        (
            [{"lemma": "요가", "pos": "NNG"}],
            ["운동"],
            "건강",
            "요가는 운동에 도움이 돼",
        ),
        (
            [{"lemma": "로봇", "pos": "NNG"}],
            ["인공지능"],
            "기술",
            "로봇에 인공지능이 적용돼",
        ),
        (
            [{"lemma": "테스트", "pos": "NNG"}],
            ["미정"],
            "스포츠",
            "적절한 문장을 찾을 수 없습니다.",
        ),
    ]
    for tokens, concepts, domain, expected in cases:
        assert restore_with_template(tokens, concepts, domain) == expected
