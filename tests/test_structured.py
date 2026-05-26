from rag_collection.structured import extract_structured_records


def test_program_requirements_skip_graduate_story_noise() -> None:
    document = {"id": 1, "url": "https://sist.shanghaitech.edu.cn/", "fetched_at": "2026-05-26T00:00:00+00:00"}
    text = "毕业生故事 | 王希元：在热爱与平常心中前行\n2026-03-01"

    records = extract_structured_records(document, text)

    assert records["program_requirements"] == []


def test_program_requirements_extract_credit_evidence() -> None:
    document = {
        "id": 1,
        "url": "https://sist.shanghaitech.edu.cn/pyfa/list.htm",
        "fetched_at": "2026-05-26T00:00:00+00:00",
    }
    text = "计算机科学与技术专业毕业要求至少修满 160 学分"

    records = extract_structured_records(document, text)

    assert records["program_requirements"][0]["min_credits"] == 160.0
    assert "160 学分" in records["program_requirements"][0]["evidence"]


def test_courses_extract_code_and_credits() -> None:
    document = {
        "id": 2,
        "url": "https://sist.shanghaitech.edu.cn/bkjx/list.htm",
        "fetched_at": "2026-05-26T00:00:00+00:00",
    }
    text = "CS101 数据结构 4 学分"

    records = extract_structured_records(document, text)

    assert records["courses"][0]["course_code"] == "CS101"
    assert records["courses"][0]["credits"] == 4.0
