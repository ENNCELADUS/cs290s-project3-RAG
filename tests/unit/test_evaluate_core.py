from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from evaluate.judge import judge_answer
from evaluate.metrics import normalize_url, source_matches, source_metrics
from evaluate.schema import load_questions


def test_load_questions_parses_json_columns(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.csv"
    _write_question_csv(questions_path)

    questions = load_questions(questions_path)

    assert len(questions) == 1
    assert questions[0].id == "q1"
    assert questions[0].acceptable_source_urls == ["https://example.edu/source"]
    assert questions[0].required_facts == ["office 3-530", "wanghy@example.edu"]


def test_load_questions_rejects_missing_required_columns(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.csv"
    questions_path.write_text("id,query\nq1,hello\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        load_questions(questions_path)


def test_source_metrics_are_rank_aware_and_url_prefix_based() -> None:
    observed = [
        "http://example.edu/noise/",
        "https://example.edu/source/detail",
        "https://example.edu/other",
    ]
    expected = ["https://example.edu/source", "https://example.edu/other"]

    metrics = source_metrics(observed, expected)

    assert normalize_url("http://EXAMPLE.edu/source/") == "https://example.edu/source"
    assert source_matches("https://example.edu/source/detail", "https://example.edu/source")
    assert not source_matches("https://example.edu/source/detail", "https://example.edu/")
    assert metrics["source_hit@1"] == 0.0
    assert metrics["source_hit@5"] == 1.0
    assert metrics["source_recall@5"] == 1.0
    assert metrics["mrr@5"] == 0.5
    assert metrics["precision@5"] == 0.4
    assert 0.0 < metrics["ndcg@5"] < 1.0


def test_source_metrics_match_sist_template_and_query_variants() -> None:
    expected = [
        "https://sist.shanghaitech.edu.cn/2024/0115/c7339a1097189/page.htm",
        "https://sist.shanghaitech.edu.cn/list.htm",
    ]
    observed = [
        "https://sist.shanghaitech.edu.cn/_t335/2024/0115/c7339a1097189/page.htm?from=nav#section",
        "https://sist.shanghaitech.edu.cn/list.htm?lang=en",
    ]

    metrics = source_metrics(observed, expected)

    assert source_matches(observed[0], expected[0])
    assert source_matches(observed[1], expected[1])
    assert metrics["source_hit@1"] == 1.0
    assert metrics["source_recall@5"] == 1.0


def test_source_metrics_match_sist_same_article_id_across_columns() -> None:
    observed = "https://sist.shanghaitech.edu.cn/2026/0327/c2863a1120270/page.htm"
    expected = "https://sist.shanghaitech.edu.cn/2026/0327/c7339a1120270/page.htm"

    assert source_matches(observed, expected)
    assert source_metrics([observed], [expected])["source_hit@1"] == 1.0


def test_source_metrics_match_sist_profile_root_main_and_list_aliases() -> None:
    assert source_matches(
        "https://sist.shanghaitech.edu.cn/zxy1/list.htm",
        "https://sist.shanghaitech.edu.cn/zxy1/main.htm",
    )
    assert source_matches(
        "https://sist.shanghaitech.edu.cn/shiye/list.htm",
        "https://sist.shanghaitech.edu.cn/shiye/main.htm",
    )
    assert source_matches(
        "https://sist.shanghaitech.edu.cn/tukw/list.htm",
        "https://sist.shanghaitech.edu.cn/tukw/main.htm",
    )
    assert source_matches(
        "https://sist.shanghaitech.edu.cn/_t335/zxy1/list.htm",
        "https://sist.shanghaitech.edu.cn/zxy1/",
    )
    assert source_metrics(
        ["https://sist.shanghaitech.edu.cn/tukw/list.htm"],
        ["https://sist.shanghaitech.edu.cn/tukw/main.htm"],
    )["source_hit@1"] == 1.0


def test_source_metrics_do_not_article_alias_unrelated_or_non_sist_urls() -> None:
    assert not source_matches(
        "https://sist.shanghaitech.edu.cn/2026/0327/c2863a1120270/page.htm",
        "https://sist.shanghaitech.edu.cn/2026/0327/c2863a1120271/page.htm",
    )
    assert not source_matches(
        "https://example.edu/2026/0327/c2863a1120270/page.htm",
        "https://example.edu/2026/0327/c7339a1120270/page.htm",
    )
    assert not source_matches(
        "https://sist.shanghaitech.edu.cn/zxy1/list.htm",
        "https://sist.shanghaitech.edu.cn/shiye/main.htm",
    )
    assert not source_matches(
        "https://sist.shanghaitech.edu.cn/zxy1/list.htm",
        "https://sist.shanghaitech.edu.cn/zxy1_0/list.htm",
    )
    assert not source_matches(
        "https://sist.shanghaitech.edu.cn/2858/list85.htm",
        "https://sist.shanghaitech.edu.cn/2858/list.htm",
    )


def test_judge_exact_required_manual_and_forbidden_cases(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.csv"
    _write_question_csv(questions_path)
    question = load_questions(questions_path)[0]

    assert judge_answer(question, "The office is office 3-530 and email is wanghy@example.edu.").is_correct == 1

    required_question = question.__class__(
        **{
            **question.__dict__,
            "judge_type": "required_facts_match",
            "acceptable_answers": [],
        }
    )
    assert judge_answer(required_question, "office 3-530 only").status == "incorrect"
    assert judge_answer(required_question, "office 3-530, wanghy@example.edu").status == "correct"

    manual_question = question.__class__(**{**question.__dict__, "judge_type": "required_facts_with_manual_review"})
    assert judge_answer(manual_question, "office 3-530, wanghy@example.edu").status == "manual_review"
    no_facts_question = question.__class__(
        **{
            **question.__dict__,
            "gt_answer": "",
            "required_facts": [],
            "acceptable_answers": [],
            "judge_type": "required_facts_with_manual_review",
        }
    )
    assert judge_answer(no_facts_question, "This needs review.").status == "manual_review"
    assert judge_answer(question, "wrong forbidden fact").status == "incorrect"


def test_judge_uses_cited_expected_source_for_loose_auto_decisions(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.csv"
    _write_question_csv(questions_path)
    question = load_questions(questions_path)[0]
    manual_question = question.__class__(**{**question.__dict__, "judge_type": "required_facts_with_manual_review"})

    correct = judge_answer(
        manual_question,
        "The office is office 3-530 and email is wanghy@example.edu. [1]",
        cited_expected_source_hit=True,
    )
    assert correct.status == "correct"
    assert correct.is_correct == 1

    incorrect = judge_answer(
        manual_question,
        "The office is office 3-530. [1]",
        cited_expected_source_hit=True,
    )
    assert incorrect.status == "incorrect"
    assert incorrect.is_correct == 0


def test_judge_loose_manual_review_matches_office_email_atoms_across_labels(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.csv"
    _write_question_csv(questions_path)
    question = load_questions(questions_path)[0]
    manual_question = question.__class__(
        **{
            **question.__dict__,
            "gt_answer": "他的办公室在信息学院3-530，工作邮箱是 wanghy@shanghaitech.edu.cn。",
            "required_facts": ["他的办公室在信息学院3-530，工作邮箱是 wanghy@shanghaitech.edu.cn。"],
            "acceptable_answers": [],
            "judge_type": "required_facts_with_manual_review",
        }
    )

    result = judge_answer(
        manual_question,
        "office: 信息学院3-530; email: wanghy@shanghaitech.edu.cn [3].",
        cited_expected_source_hit=True,
    )

    assert result.status == "correct"
    assert result.is_correct == 1


def test_judge_loose_manual_review_matches_credit_atoms_without_source_hit(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.csv"
    _write_question_csv(questions_path)
    question = load_questions(questions_path)[0]
    manual_question = question.__class__(
        **{
            **question.__dict__,
            "gt_answer": "人文社科通识板块要求45学分，自然科学通识板块要求32学分。",
            "required_facts": ["人文社科通识板块要求45学分，自然科学通识板块要求32学分。"],
            "acceptable_answers": [],
            "judge_type": "required_facts_with_manual_review",
        }
    )

    result = judge_answer(
        manual_question,
        "人文社科通识板块：45学分；自然科学通识板块：32学分 [1]",
        cited_expected_source_hit=False,
    )

    assert result.status == "correct"
    assert result.is_correct == 1


def test_judge_loose_manual_review_matches_credit_atoms_across_sentence_boundaries(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.csv"
    _write_question_csv(questions_path)
    question = load_questions(questions_path)[0]
    manual_question = question.__class__(
        **{
            **question.__dict__,
            "gt_answer": "人文社科通识板块要求修满45学分，自然科学通识板块要求修满32学分。",
            "required_facts": ["人文社科通识板块要求修满45学分，自然科学通识板块要求修满32学分。"],
            "acceptable_answers": [],
            "judge_type": "required_facts_with_manual_review",
        }
    )

    result = judge_answer(
        manual_question,
        "2025级本科生培养方案EE专业：人文社科通识板块（45学分），自然科学通识板块（32学分）。 [1]",
        cited_expected_source_hit=True,
    )

    assert result.status == "correct"
    assert result.is_correct == 1


def test_judge_loose_manual_review_normalizes_date_atoms(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.csv"
    _write_question_csv(questions_path)
    question = load_questions(questions_path)[0]
    manual_question = question.__class__(
        **{
            **question.__dict__,
            "gt_answer": "2025级硕士、博士培养方案的发布日期是2025年09月04日。",
            "required_facts": ["2025级硕士、博士培养方案的发布日期是2025年09月04日。"],
            "acceptable_answers": [],
            "judge_type": "required_facts_with_manual_review",
        }
    )

    result = judge_answer(
        manual_question,
        "培养方案列表包括：2025级硕士、博士培养方案 2025-09-04。 [1]",
        cited_expected_source_hit=True,
    )

    assert result.status == "correct"
    assert result.is_correct == 1


def test_judge_content_coverage_can_pass_without_expected_source_hit(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.csv"
    _write_question_csv(questions_path)
    question = load_questions(questions_path)[0]
    manual_question = question.__class__(
        **{
            **question.__dict__,
            "gt_answer": "申乔木，北京理工大学（珠海），李权，2026年4月28日上午10:15，创管学院106。",
            "required_facts": [
                "演讲者是申乔木",
                "单位是北京理工大学（珠海）",
                "邀请人是李权",
                "时间是2026年4月28日上午10:15",
                "地点是创管学院106",
            ],
            "acceptable_answers": [],
            "judge_type": "required_facts_with_manual_review",
        }
    )

    result = judge_answer(
        manual_question,
        "演讲者:申乔木，北京理工大学（珠海）；时间:2026年4月28日，上午10:15；"
        "邀请人:李权；地点:创管学院106。 [3]",
        cited_expected_source_hit=False,
        has_citation=True,
    )

    assert result.status == "correct"
    assert result.is_correct == 1


def test_judge_loose_manual_review_matches_chinese_and_iso_dates(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.csv"
    _write_question_csv(questions_path)
    question = load_questions(questions_path)[0]
    manual_question = question.__class__(
        **{
            **question.__dict__,
            "gt_answer": "报名截止日期是2025年09月04日。",
            "required_facts": ["报名截止日期是2025年09月04日。"],
            "acceptable_answers": [],
            "judge_type": "required_facts_with_manual_review",
        }
    )

    result = judge_answer(manual_question, "报名截止日期为 2025-09-04。 [1]")

    assert result.status == "correct"
    assert result.is_correct == 1


def test_judge_exact_or_alias_can_be_content_correct_without_source_hit(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.csv"
    _write_question_csv(questions_path)
    question = load_questions(questions_path)[0]
    exact_question = question.__class__(
        **{
            **question.__dict__,
            "gt_answer": "他的办公室在信息学院3-530，工作邮箱是 wanghy@shanghaitech.edu.cn。",
            "required_facts": ["他的办公室在信息学院3-530，工作邮箱是 wanghy@shanghaitech.edu.cn。"],
            "acceptable_answers": ["他的办公室在信息学院3-530，工作邮箱是 wanghy@shanghaitech.edu.cn。"],
            "judge_type": "exact_or_alias_match",
        }
    )

    result = judge_answer(
        exact_question,
        "office: 信息学院3-530; email: wanghy@shanghaitech.edu.cn. [1]",
        cited_expected_source_hit=False,
    )

    assert result.status == "correct"
    assert result.is_correct == 1


def test_judge_matches_q002_office_email_paraphrase_without_expected_source_hit(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.csv"
    _write_question_csv(questions_path)
    question = load_questions(questions_path)[0]
    exact_question = question.__class__(
        **{
            **question.__dict__,
            "gt_answer": "他的办公室在信息学院3-530，工作邮箱是 wanghy@shanghaitech.edu.cn。",
            "required_facts": ["他的办公室在信息学院3-530，工作邮箱是 wanghy@shanghaitech.edu.cn。"],
            "acceptable_answers": ["他的办公室在信息学院3-530，工作邮箱是 wanghy@shanghaitech.edu.cn。"],
            "judge_type": "exact_or_alias_match",
        }
    )

    result = judge_answer(
        exact_question,
        "王浩宇教授办公地点为 SIST 3-530，邮箱 wanghy@shanghaitech.edu.cn。[1]",
        cited_expected_source_hit=False,
    )

    assert result.status == "correct"
    assert result.is_correct == 1


def test_judge_matches_q019_acl_date_location_aliases(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.csv"
    _write_question_csv(questions_path)
    question = load_questions(questions_path)[0]
    required_question = question.__class__(
        **{
            **question.__dict__,
            "gt_answer": (
                "该成果发表在第64届计算语言学协会年会（ACL 2026）上。会议将于2026年7月2日至7日"
                "在美国加利福尼亚州圣迭戈（San Diego, California, USA）举行。"
            ),
            "required_facts": [
                "该成果发表在第64届计算语言学协会年会（ACL 2026）上。会议将于2026年7月2日至7日"
                "在美国加利福尼亚州圣迭戈（San Diego, California, USA）举行。"
            ],
            "acceptable_answers": [],
            "judge_type": "required_facts_match",
        }
    )

    result = judge_answer(
        required_question,
        "GiLT 被 ACL 2026 录用；会议时间是 2026年7月2日至7日，地点为美国加州圣地亚哥。[2]",
        cited_expected_source_hit=False,
    )

    assert result.status == "correct"
    assert result.is_correct == 1


def test_judge_matches_q092_device_translation_aliases(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.csv"
    _write_question_csv(questions_path)
    question = load_questions(questions_path)[0]
    required_question = question.__class__(
        **{
            **question.__dict__,
            "gt_answer": (
                "页面列出的设备包括 Semiconductor Analyzer、Microelectronic Measurement System、"
                "Low Temperature Measurement System、DLTS、Power Device Measurement System、"
                "Load-Pull System、Impedance Analyzer 和 OTF-1200X开启式管式炉。"
            ),
            "required_facts": [
                "Semiconductor Analyzer",
                "Microelectronic Measurement System",
                "Low Temperature Measurement System",
                "DLTS",
                "Power Device Measurement System",
                "Load-Pull System",
                "Impedance Analyzer",
                "OTF-1200X开启式管式炉",
            ],
            "acceptable_answers": [],
            "judge_type": "required_facts_match",
        }
    )

    result = judge_answer(
        required_question,
        "该页面列出半导体分析仪、微电子测量系统、低温测量系统、DLTS、功率器件测量系统、"
        "负载牵引系统、阻抗分析仪和 OTF-1200X open tube furnace。[1]",
        cited_expected_source_hit=False,
    )

    assert result.status == "correct"
    assert result.is_correct == 1


def test_judge_rejects_missing_required_numeric_fact_even_with_citation(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.csv"
    _write_question_csv(questions_path)
    question = load_questions(questions_path)[0]
    exact_question = question.__class__(
        **{
            **question.__dict__,
            "gt_answer": "毕业至少需要修满145学分，其中任选课占9学分。",
            "required_facts": ["毕业至少需要修满145学分，其中任选课占9学分。"],
            "acceptable_answers": ["毕业至少需要修满145学分，其中任选课占9学分。"],
            "judge_type": "exact_or_alias_match",
        }
    )

    result = judge_answer(exact_question, "毕业至少需要修满145学分。[1]", cited_expected_source_hit=False)

    assert result.status == "incorrect"
    assert result.is_correct == 0


def test_judge_rejects_wrong_required_numeric_fact_even_with_loose_match() -> None:
    questions = {
        question.id: question for question in load_questions(Path("data/test/question_final_structured_100.csv"))
    }

    result = judge_answer(
        questions["q067"],
        "学生毕业至少需要修满145总学分，其中CS专业课程板块必修学分为20，选修学分为145，合计学分为59。[1]",
        cited_expected_source_hit=True,
        has_citation=True,
    )

    assert result.status == "incorrect"
    assert result.is_correct == 0


def test_judge_accepts_targeted_evaluator_patch_questions() -> None:
    questions = {
        question.id: question for question in load_questions(Path("data/test/question_final_structured_100.csv"))
    }

    answers = {
        "q006": "屠可伟教授获得过 ACL 2023 杰出论文奖、SemEval 2022 最佳系统论文奖和 SemEval 2023 最佳系统论文奖。",
        "q034": "傅旻帆教授建议提前自学的课程是专业选修课《电力电子》，这门课已被录制成视频。[1]",
        "q047": "在专业课程板块中，学生至少需要修满选修 27 学分。[1]",
        "q075": (
            "未达到 B- 不能计入专业课学分和门数；可重修或修读其他课程达到 B- 及以上。"
            "本科期间研究生课程认定要求 3 学分及以上、成绩至少 B+，且不得超过 2 门。[1]"
        ),
        "q078": "基本学制为 5-7 年；总学分不低于 42 个总学分，课程学分不低于 40 个学分，课程实践不少于 8 学分。[1]",
        "q087": (
            "计算机科学与技术02-中科院联培院所招生、电子科学与技术、信息与通信工程总分均为320分；"
            "电子信息总分300分；单科线为满分100分科目35分、满分大于100分科目53分。[1]"
        ),
    }

    for question_id, answer in answers.items():
        result = judge_answer(questions[question_id], answer, has_citation=True)
        assert result.status == "correct", f"{question_id}: {result.reason}"
        assert result.is_correct == 1


def test_q081_required_facts_match_question_target() -> None:
    questions = {
        question.id: question for question in load_questions(Path("data/test/question_final_structured_100.csv"))
    }

    question = questions["q081"]

    assert question.required_facts == [
        "挑战赛是Deep Past Challenge（深邃历史挑战赛）",
        "参赛队伍为2673支",
        "开发者为3311名",
        "累计提交方案6.8万份",
    ]
    assert "全球总冠军" not in " ".join(question.required_facts)


def test_judge_rejects_wrong_q081_chinese_count_units() -> None:
    questions = {
        question.id: question for question in load_questions(Path("data/test/question_final_structured_100.csv"))
    }

    result = judge_answer(
        questions["q081"],
        "队伍参加的是Deep Past Challenge（深邃历史挑战赛），比赛共有2672支队伍、"
        "3310名开发者参与，并累计提交6.7万份方案。[1]",
        cited_expected_source_hit=True,
        has_citation=True,
    )

    assert result.status == "incorrect"
    assert result.is_correct == 0


def test_judge_matches_q067_professional_course_credit_summary() -> None:
    questions = {
        question.id: question for question in load_questions(Path("data/test/question_final_structured_100.csv"))
    }

    result = judge_answer(
        questions["q067"],
        "学生毕业至少需要修满145总学分，其中CS专业课程板块必修学分为20，选修学分为39，合计学分为59。[1]",
        cited_expected_source_hit=True,
        has_citation=True,
    )

    assert result.status == "correct", result.reason
    assert result.is_correct == 1


def test_judge_matches_q067_markdown_numeric_atoms_from_generation_run() -> None:
    questions = {
        question.id: question for question in load_questions(Path("data/test/question_final_structured_100.csv"))
    }

    result = judge_answer(
        questions["q067"],
        "根据2025级计算机科学与技术专业本科生培养方案，学生毕业至少需要修满**145**总学分。"
        "其中，专业课程板块的学分要求如下：必修**20**学分，选修**39**学分，合计**59**学分 [1]。",
        cited_expected_source_hit=True,
        has_citation=True,
    )

    assert result.status == "correct", result.reason
    assert result.is_correct == 1


def test_judge_matches_q009_markdown_comparative_credit_atoms_from_generation_run() -> None:
    questions = {
        question.id: question for question in load_questions(Path("data/test/question_final_structured_100.csv"))
    }

    result = judge_answer(
        questions["q009"],
        "根据提供的2025级EE专业本科培养方案[1]：\n\n"
        "*   **人文社科通识板块**：必修学分为**30**学分，选修学分为**15**学分，合计**45**学分。\n"
        "*   **自然科学通识板块**：必修学分为**16**学分，选修学分为**16**学分，合计**32**学分。",
        cited_expected_source_hit=True,
        has_citation=True,
    )

    assert result.status == "correct", result.reason
    assert result.is_correct == 1


def test_judge_matches_q068_markdown_list_numeric_atoms_from_generation_run() -> None:
    questions = {
        question.id: question for question in load_questions(Path("data/test/question_final_structured_100.csv"))
    }

    result = judge_answer(
        questions["q068"],
        "根据提供的2025级本科生培养方案，2025级CS专业人工智能荣誉班与普通CS专业在自然科学通识板块"
        "和专业课程板块的总学分要求对比如下：\n\n"
        "**1. 2025级CS专业人工智能荣誉班**\n"
        "*   **自然科学通识板块**：总学分为 **28** 学分（必修12学分 + 选修16学分）[1]。\n"
        "*   **专业课程板块**：总学分为 **68** 学分（必修42学分 + 选修26学分）[1]。\n\n"
        "**2. 2025级CS专业（普通班）**\n"
        "*   **自然科学通识板块**：总学分为 **32** 学分（必修16学分 + 选修16学分）[2]。\n"
        "*   **专业课程板块**：总学分为 **59** 学分（必修20学分 + 选修39学分）[2]。",
        cited_expected_source_hit=True,
        has_citation=True,
    )

    assert result.status == "correct", result.reason
    assert result.is_correct == 1


def test_judge_rejects_wrong_markdown_numeric_atoms_from_generation_contexts() -> None:
    questions = {
        question.id: question for question in load_questions(Path("data/test/question_final_structured_100.csv"))
    }

    wrong_q067 = judge_answer(
        questions["q067"],
        "根据2025级计算机科学与技术专业本科生培养方案，学生毕业至少需要修满**145**总学分。"
        "其中，专业课程板块的学分要求如下：必修**20**学分，选修**38**学分，合计**59**学分 [1]。",
        cited_expected_source_hit=True,
        has_citation=True,
    )
    wrong_q068 = judge_answer(
        questions["q068"],
        "*   **CS专业人工智能荣誉班自然科学通识板块**：总学分为 **28** 学分。\n"
        "*   **CS专业人工智能荣誉班专业课程板块**：总学分为 **67** 学分。\n"
        "*   **普通CS专业自然科学通识板块**：总学分为 **32** 学分。\n"
        "*   **普通CS专业专业课程板块**：总学分为 **59** 学分。[1]",
        cited_expected_source_hit=True,
        has_citation=True,
    )

    assert wrong_q067.status == "incorrect"
    assert wrong_q067.is_correct == 0
    assert wrong_q068.status == "incorrect"
    assert wrong_q068.is_correct == 0


def test_judge_matches_q052_committee_chair_summary() -> None:
    questions = {
        question.id: question for question in load_questions(Path("data/test/question_final_structured_100.csv"))
    }

    result = judge_answer(
        questions["q052"],
        "学术委员会主任由哈亚军担任，学位委员会主任由寇煦丰担任。[1]",
        cited_expected_source_hit=True,
        has_citation=True,
    )

    assert result.status == "correct", result.reason
    assert result.is_correct == 1


def test_judge_matches_q090_date_title_pairings_with_chinese_dates() -> None:
    questions = {
        question.id: question for question in load_questions(Path("data/test/question_final_structured_100.csv"))
    }

    result = judge_answer(
        questions["q090"],
        "2026年4月22日对应上海创芯学院&上海科技大学2026年电子信息工程博士招生简介；"
        "2026年4月15日对应上海科技大学-北京通用人工智能研究院2026年联合培养博士生专项计划"
        "（“通计划”）-第二轮报名；"
        "2026年3月17日对应上海科技大学信息科学与技术学院2026年全日制工程类专业学位博士报名通知"
        "（第二轮）。[1]",
        cited_expected_source_hit=True,
        has_citation=True,
    )

    assert result.status == "correct", result.reason
    assert result.is_correct == 1


def test_judge_rejects_q075_off_topic_answer_after_evaluator_patch() -> None:
    questions = {
        question.id: question for question in load_questions(Path("data/test/question_final_structured_100.csv"))
    }

    result = judge_answer(
        questions["q075"],
        "公共基础课包括思政类课程和英语类课程，要求至少修满 8 学分。[1]",
        cited_expected_source_hit=True,
        has_citation=True,
    )

    assert result.status == "incorrect"
    assert result.is_correct == 0


def _write_question_csv(path: Path) -> None:
    row = {
        "id": "q1",
        "category": "Factual",
        "language": "en",
        "query": "Where is the office?",
        "gt_answer": "office 3-530, wanghy@example.edu",
        "primary_source_url": "https://example.edu/source",
        "acceptable_source_urls": json.dumps(["https://example.edu/source"]),
        "evidence_snippet": "office 3-530",
        "required_facts": json.dumps(["office 3-530", "wanghy@example.edu"]),
        "acceptable_answers": json.dumps(["office 3-530 and email is wanghy@example.edu"]),
        "forbidden_facts": json.dumps(["forbidden fact"]),
        "grading_notes": "test row",
        "judge_type": "exact_or_alias_match",
        "complexity": "Low",
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
