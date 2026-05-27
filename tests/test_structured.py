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


def test_courses_do_not_extract_bare_course_codes_as_facts() -> None:
    document = {
        "id": 2,
        "url": "https://faculty.sist.shanghaitech.edu.cn/office/Academics/Graduate/Courses/table.htm",
        "category": "courses",
        "fetched_at": "2026-05-26T00:00:00+00:00",
    }
    text = "\n".join(
        [
            "MATH2103",
            "MATH2104",
            "CS101 Data Structures 4 credits",
        ]
    )

    records = extract_structured_records(document, text)

    assert [record["course_code"] for record in records["courses"]] == ["CS101"]


def test_faculty_extracts_person_name_instead_of_page_labels() -> None:
    document = {
        "id": 4,
        "url": "https://pmicc.sist.shanghaitech.edu.cn/faculty.html",
        "category": "faculty",
        "fetched_at": "2026-05-26T00:00:00+00:00",
    }
    text = "\n".join(
        [
            "Faculty",
            "Wenhan Cao",
            "Assistant Professor",
            "Room 3-336, SIST Building",
            "Introduction",
            "Baile Chen",
            "Associate Professor",
        ]
    )

    records = extract_structured_records(document, text)

    assert [record["name"] for record in records["faculty_members"]] == ["Wenhan Cao", "Baile Chen"]


def test_faculty_extracts_person_name_from_inline_center_listing() -> None:
    document = {
        "id": 4,
        "url": "https://pmicc.sist.shanghaitech.edu.cn/faculty.html",
        "category": "faculty",
        "fetched_at": "2026-05-26T00:00:00+00:00",
    }
    text = "Faculty Wenhan Cao Assistant Professor Room 3-336, SIST Building Introduction"

    records = extract_structured_records(document, text)

    assert records["faculty_members"][0]["name"] == "Wenhan Cao"


def test_faculty_extraction_ignores_advisor_lines_in_cv_context() -> None:
    document = {
        "id": 4,
        "url": "https://faculty.sist.shanghaitech.edu.cn/chenjh",
        "category": "faculty",
        "fetched_at": "2026-05-26T00:00:00+00:00",
    }
    text = "\n".join(
        [
            "Chen, Jiahao 陈 嘉豪 Assistant Professor",
            "ShanghaiTech University",
            "Biography of the PI",
            "Advisor: Professor Christopher H. T. Lee",
            "Advisor: Professor Jin Huang",
        ]
    )

    records = extract_structured_records(document, text)

    assert [(record["name"], record["title"]) for record in records["faculty_members"]] == [
        ("Chen Jiahao", "Assistant Professor")
    ]


def test_events_on_archive_list_pages_skip_stale_rows() -> None:
    document = {
        "id": 6,
        "url": "https://sist.shanghaitech.edu.cn/2713/list3.htm",
        "category": "career",
        "fetched_at": "2026-05-26T00:00:00+00:00",
    }
    text = "\n".join(
        [
            "紫光展锐2020年实习生/应届生招聘",
            "2019-09-12",
            "2026届春季招聘信息",
            "2026-03-01",
        ]
    )

    records = extract_structured_records(document, text)

    assert [(record["title"], record["published_at"]) for record in records["events"]] == [
        ("2026届春季招聘信息", "2026-03-01")
    ]


def test_structured_extraction_does_not_promote_unrelated_page_snippets() -> None:
    admission_document = {
        "id": 3,
        "url": "https://admission.shanghaitech.edu.cn/",
        "category": "admission",
        "fetched_at": "2026-05-26T00:00:00+00:00",
    }
    homepage_document = {
        "id": 4,
        "url": "https://sist.shanghaitech.edu.cn/",
        "category": "school_info",
        "fetched_at": "2026-05-26T00:00:00+00:00",
    }
    news_document = {
        "id": 5,
        "url": "https://www.shanghaitech.edu.cn/2026/0525/c1001a1122931/page.htm",
        "category": "news",
        "fetched_at": "2026-05-26T00:00:00+00:00",
    }

    admission_records = extract_structured_records(admission_document, "咨询邮箱：admission@shanghaitech.edu.cn")
    homepage_records = extract_structured_records(homepage_document, "研究生培养\n毕业答辩流程\n培养方案")
    news_records = extract_structured_records(news_document, "ICECS2023 conference was held at ShanghaiTech")

    assert admission_records["faculty_members"] == []
    assert homepage_records["program_requirements"] == []
    assert news_records["courses"] == []
