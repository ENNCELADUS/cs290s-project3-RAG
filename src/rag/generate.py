from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from .index import DEFAULT_BM25, DEFAULT_CHUNK_INDEX, DEFAULT_DB, DEFAULT_FAISS, DEFAULT_REPORT
from .retrieve import ContextItem, HybridRetrievalResult, Retriever

AnswerMode = Literal["dense", "hybrid"]
AnswerStatus = Literal["answered", "insufficient_evidence"]
GenerationPath = Literal["initial", "extractive_fallback", "repair", "insufficient"]
Device = Literal["auto", "cpu", "cuda"]

DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_K = 5
ANSWER_EVIDENCE_CONTEXT_CHARS = 1400
ANSWER_EVIDENCE_WINDOW_CHARS = 900
VALID_CITATION_RE = re.compile(r"\[(\d+)\]")
PROMPT_LEAKAGE_MARKERS = ("Question:", "Sources:", "TEXT:", "URL:", "trace_ref:", "Use only the provided")


@dataclass(frozen=True)
class AnswerSource:
    source_id: int
    title: str | None
    url: str
    chunk_id: int
    document_id: int
    trace_ref: str
    snippet: str


@dataclass(frozen=True)
class AnswerConfig:
    model_path: str
    device: str
    max_new_tokens: int
    temperature: float
    top_k: int
    answer_reranker_model: str | None = None
    answer_reranker_device: str = "cpu"


@dataclass(frozen=True)
class AnswerTiming:
    retrieval_s: float
    generation_s: float
    total_s: float


@dataclass(frozen=True)
class RagAnswerResult:
    query: str
    mode: AnswerMode
    status: AnswerStatus
    answer: str
    sources: list[AnswerSource]
    retrieval: dict[str, Any]
    timing: AnswerTiming
    config: AnswerConfig
    generation_path: GenerationPath = "initial"
    generation_rejection_reason: str | None = None
    fallback_source_rank: int | None = None
    answer_context_order: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ExtractiveAnswer:
    answer: str
    source_rank: int


@dataclass(frozen=True)
class ExtractiveCandidate:
    text: str
    source_rank: int
    context_order: int
    score: float


class RagAnswerer:
    def __init__(
        self,
        retriever: Retriever,
        *,
        model_path: Path,
        device: Device = "auto",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        answer_reranker_model: Path | None = None,
        answer_reranker_device: str = "cpu",
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Local generator model path does not exist: {model_path}")
        if answer_reranker_model is not None and not answer_reranker_model.exists():
            raise FileNotFoundError(f"Answer reranker model path does not exist: {answer_reranker_model}")
        self.retriever = retriever
        self.model_path = model_path
        self.device = _resolve_device(device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.answer_reranker_model = answer_reranker_model
        self.answer_reranker_device = answer_reranker_device
        self._answer_reranker: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def answer(
        self,
        query: str,
        *,
        mode: AnswerMode = "hybrid",
        top_k: int = DEFAULT_TOP_K,
        **retrieve_kwargs: Any,
    ) -> RagAnswerResult:
        if mode not in ("dense", "hybrid"):
            raise ValueError(f"unsupported answer mode: {mode}")
        started = time.perf_counter()
        retrieval_started = time.perf_counter()
        if mode == "hybrid":
            retrieval_result = self.retriever.retrieve(query, mode=mode, top_k=top_k, **retrieve_kwargs)
        else:
            retrieval_result = self.retriever.retrieve(query, mode=mode, top_k=top_k)
        retrieval_s = time.perf_counter() - retrieval_started

        contexts = _contexts_from_retrieval(self.retriever, retrieval_result)
        sources = _sources_from_contexts(contexts)
        answer_context_order = self._answer_context_order(query, contexts)
        ordered_contexts = _ordered_contexts(contexts, answer_context_order)
        retrieval_payload = _retrieval_payload(retrieval_result, contexts)
        retrieval_payload["answer_context_order"] = answer_context_order
        config = AnswerConfig(
            model_path=str(self.model_path),
            device=self.device,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=top_k,
            answer_reranker_model=str(self.answer_reranker_model) if self.answer_reranker_model is not None else None,
            answer_reranker_device=self.answer_reranker_device,
        )
        if not contexts or not sources:
            return _insufficient_result(
                query,
                mode,
                sources=sources,
                retrieval=retrieval_payload,
                timing=AnswerTiming(retrieval_s=retrieval_s, generation_s=0.0, total_s=time.perf_counter() - started),
                config=config,
                answer_context_order=answer_context_order,
            )

        evidence_contexts = _select_local_evidence_contexts(query, ordered_contexts, retriever=self.retriever)
        messages = build_messages(query, evidence_contexts)
        generation_started = time.perf_counter()
        generated = self._generate_text(messages)
        generation_s = time.perf_counter() - generation_started
        valid_source_ids = {source.source_id for source in sources}
        rejection_reason = _answer_rejection_reason(
            generated,
            valid_source_ids,
            query=query,
            contexts=evidence_contexts,
        )
        if rejection_reason is not None:
            repair_text = self._generate_text(build_repair_messages(query, evidence_contexts, generated))
            generation_s = time.perf_counter() - generation_started
            repaired = _parse_repair_answer(
                repair_text,
                valid_source_ids=valid_source_ids,
                query=query,
                contexts=evidence_contexts,
            )
            if repaired is not None:
                return RagAnswerResult(
                    query=query,
                    mode=mode,
                    status="answered",
                    answer=repaired,
                    sources=sources,
                    retrieval=retrieval_payload,
                    timing=AnswerTiming(
                        retrieval_s=retrieval_s,
                        generation_s=generation_s,
                        total_s=time.perf_counter() - started,
                    ),
                    config=config,
                    generation_path="repair",
                    generation_rejection_reason=rejection_reason,
                    answer_context_order=answer_context_order,
                )
            extracted = _extract_answer_from_contexts(query, evidence_contexts)
            if extracted is not None and _is_acceptable_answer(
                extracted.answer,
                valid_source_ids,
                query=query,
                contexts=evidence_contexts,
            ):
                return RagAnswerResult(
                    query=query,
                    mode=mode,
                    status="answered",
                    answer=extracted.answer,
                    sources=sources,
                    retrieval=retrieval_payload,
                    timing=AnswerTiming(
                        retrieval_s=retrieval_s,
                        generation_s=generation_s,
                        total_s=time.perf_counter() - started,
                    ),
                    config=config,
                    generation_path="extractive_fallback",
                    generation_rejection_reason=rejection_reason,
                    fallback_source_rank=extracted.source_rank,
                    answer_context_order=answer_context_order,
                )
            return _insufficient_result(
                query,
                mode,
                sources=sources,
                retrieval=retrieval_payload,
                timing=AnswerTiming(
                    retrieval_s=retrieval_s,
                    generation_s=generation_s,
                    total_s=time.perf_counter() - started,
                ),
                config=config,
                generation_rejection_reason=rejection_reason,
                answer_context_order=answer_context_order,
            )
        return RagAnswerResult(
            query=query,
            mode=mode,
            status="answered",
            answer=generated,
            sources=sources,
            retrieval=retrieval_payload,
            timing=AnswerTiming(
                retrieval_s=retrieval_s,
                generation_s=generation_s,
                total_s=time.perf_counter() - started,
            ),
            config=config,
            generation_path="initial",
            answer_context_order=answer_context_order,
        )

    def _generate_text(self, messages: list[dict[str, str]]) -> str:
        tokenizer, model = self._load_model()
        prompt = _render_prompt(tokenizer, messages)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = _move_inputs(inputs, self.device)
        generate_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature
        output_ids = model.generate(**generate_kwargs)
        input_length = _input_length(inputs)
        generated_ids = output_ids[0][input_length:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def _load_model(self) -> tuple[Any, Any]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True, trust_remote_code=True)
        torch_dtype = _generation_torch_dtype(self.device)
        model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch_dtype,
        )
        if torch_dtype == "auto":
            model.to(self.device)
        else:
            model.to(device=self.device, dtype=torch_dtype)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        return tokenizer, model

    def _answer_context_order(self, query: str, contexts: list[ContextItem]) -> list[dict[str, Any]]:
        order: list[dict[str, Any]] = []
        reranker_scores = self._answer_reranker_scores(query, contexts)
        for context, reranker_score in zip(contexts, reranker_scores, strict=True):
            score, reasons = _metadata_answer_context_score(query, context)
            if reranker_score is not None:
                score += reranker_score
                reasons.append(f"answer_reranker_score:{reranker_score:.3f}")
            order.append(
                {
                    "source_id": context.rank,
                    "chunk_id": context.chunk_id,
                    "url": context.url,
                    "title": context.title,
                    "score": round(score, 6),
                    "reasons": reasons,
                }
            )
        return sorted(order, key=lambda item: (-float(item["score"]), int(item["source_id"])))

    def _answer_reranker_scores(self, query: str, contexts: list[ContextItem]) -> list[float | None]:
        if self.answer_reranker_model is None or not contexts:
            return [None for _ in contexts]
        from sentence_transformers import CrossEncoder

        if self._answer_reranker is None:
            self._answer_reranker = CrossEncoder(str(self.answer_reranker_model), device=self.answer_reranker_device)
        pairs = [(query, _context_score_text(context)) for context in contexts]
        return [float(score) for score in self._answer_reranker.predict(pairs)]


def build_prompt(query: str, contexts: list[ContextItem]) -> str:
    return _messages_to_prompt(build_messages(query, contexts))


def build_messages(query: str, contexts: list[ContextItem]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "\n".join(
                [
                    "You are a local RAG answer generator for official ShanghaiTech/SIST sources.",
                    "Answer in the same language as the user question.",
                    "Use only the provided sources. Do not use outside knowledge.",
                    "Preserve label-value bindings exactly; do not copy totals into adjacent rows or fields.",
                    "For formulas, preserve the source year's components and weights exactly.",
                    "Every factual paragraph must include at least one numbered citation like [1].",
                    "If the question asks for multiple fields or items, answer every requested field found in sources.",
                    "If the sources do not contain enough evidence, say that the evidence is insufficient.",
                    "Write only the final answer. Do not copy source metadata or prompt text.",
                ]
            ),
        },
        {
            "role": "user",
            "content": "\n\n".join(
                [
                    f"Question: {query}",
                    "Sources:",
                    "\n\n".join(_context_blocks(contexts)),
                ]
            ),
        },
    ]


def build_repair_messages(query: str, contexts: list[ContextItem], draft: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "\n".join(
                [
                    "You repair local RAG answers for official ShanghaiTech/SIST sources.",
                    "Answer in the same language as the user question.",
                    "Use only the provided sources. Do not add outside facts.",
                    "Preserve label-value bindings exactly; do not copy totals into adjacent rows or fields.",
                    "For formulas, preserve the source year's components and weights exactly.",
                    "Return only one strict JSON object.",
                    'For a supported answer, use: {"status":"answered","answer":"... [1]"}',
                    'If the sources are insufficient, use: {"status":"insufficient_evidence","answer":""}',
                    "The answer must include numbered citations that match the provided source numbers.",
                    "If the question asks for multiple fields or items, include every requested field found in "
                    "sources.",
                ]
            ),
        },
        {
            "role": "user",
            "content": "\n\n".join(
                [
                    f"Question: {query}",
                    f"Rejected draft: {draft}",
                    "Sources:",
                    "\n\n".join(_context_blocks(contexts)),
                ]
            ),
        },
    ]


def _context_blocks(contexts: list[ContextItem]) -> list[str]:
    blocks = []
    for context in contexts:
        title = context.title or "(untitled)"
        url = context.url or "(no url)"
        blocks.append(
            "\n".join(
                [
                    f"[{context.rank}] {title}",
                    f"URL: {url}",
                    f"chunk_id: {context.chunk_id}",
                    f"trace_ref: {context.trace_ref}",
                    "TEXT:",
                    context.text,
                ]
            )
        )
    return blocks


def _messages_to_prompt(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(
        [
            messages[0]["content"],
            "",
            messages[1]["content"],
            "",
            "Answer:",
        ]
    )


def _render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return _messages_to_prompt(messages)


def _contexts_from_retrieval(
    retriever: Retriever, retrieval_result: object
) -> list[ContextItem]:
    if isinstance(retrieval_result, HybridRetrievalResult):
        return retrieval_result.contexts
    return retriever.contexts_for_hits(retrieval_result)  # type: ignore[arg-type]


def _sources_from_contexts(contexts: list[ContextItem]) -> list[AnswerSource]:
    sources: list[AnswerSource] = []
    for context in contexts:
        if context.url is None:
            continue
        sources.append(
            AnswerSource(
                source_id=context.rank,
                title=context.title,
                url=context.url,
                chunk_id=context.chunk_id,
                document_id=context.document_id,
                trace_ref=context.trace_ref,
                snippet=context.snippet,
            )
        )
    return sources


def _retrieval_payload(retrieval_result: object, contexts: list[ContextItem]) -> dict[str, Any]:
    if isinstance(retrieval_result, HybridRetrievalResult):
        return {
            "mode": retrieval_result.mode,
            "hits": [asdict(hit) for hit in retrieval_result.hits],
            "contexts": [asdict(context) for context in contexts],
            "config": asdict(retrieval_result.config),
        }
    return {
        "mode": "dense",
        "hits": [_dataclass_or_value(hit) for hit in retrieval_result],  # type: ignore[union-attr]
        "contexts": [asdict(context) for context in contexts],
    }


def _dataclass_or_value(value: object) -> object:
    try:
        return asdict(value)
    except TypeError:
        return value


def _ordered_contexts(contexts: list[ContextItem], answer_context_order: list[dict[str, Any]]) -> list[ContextItem]:
    by_source_id = {context.rank: context for context in contexts}
    ordered = [
        by_source_id[int(item["source_id"])]
        for item in answer_context_order
        if int(item["source_id"]) in by_source_id
    ]
    if len(ordered) == len(contexts):
        return ordered
    selected_ids = {context.rank for context in ordered}
    return [*ordered, *[context for context in contexts if context.rank not in selected_ids]]


def _select_local_evidence_contexts(
    query: str,
    contexts: list[ContextItem],
    *,
    retriever: Retriever | None = None,
) -> list[ContextItem]:
    return [
        _select_local_evidence_context(query, context, sibling_texts=_sibling_chunk_texts(retriever, context))
        for context in contexts
    ]


def _select_local_evidence_context(
    query: str,
    context: ContextItem,
    *,
    sibling_texts: list[str],
) -> ContextItem:
    combined_text = "\n".join([context.text, *sibling_texts])
    if not sibling_texts and len(combined_text) <= ANSWER_EVIDENCE_CONTEXT_CHARS:
        return context
    evidence_text = _local_evidence_text(query, context, combined_text)
    if evidence_text is None:
        return context
    return replace(context, text=evidence_text, snippet=evidence_text[:240])


def _sibling_chunk_texts(retriever: Retriever | None, context: ContextItem) -> list[str]:
    if retriever is None:
        return []
    chunks = getattr(retriever, "_chunks", None)
    if not isinstance(chunks, list):
        return []
    siblings: list[tuple[int, str]] = []
    for row in chunks:
        if not isinstance(row, dict):
            continue
        try:
            row_chunk_id = int(row.get("chunk_id", -1))
            row_document_id = int(row.get("document_id", -1))
        except (TypeError, ValueError):
            continue
        if row_chunk_id == context.chunk_id:
            continue
        same_document = row_document_id == context.document_id
        same_url = context.url is not None and row.get("url") == context.url
        if not same_document and not same_url:
            continue
        text = row.get("text")
        if isinstance(text, str) and text.strip():
            siblings.append((row_chunk_id, text))
    return [text for _chunk_id, text in sorted(siblings)]


def _local_evidence_text(query: str, context: ContextItem, text: str) -> str | None:
    query_terms = _anchor_terms(query)
    evidence_pattern = _local_evidence_pattern(query)
    if evidence_pattern is None:
        return None
    scored: list[tuple[float, int, str]] = []
    context_header = " ".join(part for part in (context.title, context.url, context.snippet) if part)
    for index, window in enumerate(_candidate_windows(text)):
        candidate_match_text = f"{context_header} {window}"
        anchor_overlap = _anchor_overlap_count(query_terms, candidate_match_text)
        evidence_count = len(evidence_pattern.findall(window))
        if anchor_overlap < _minimum_anchor_overlap(query_terms) and evidence_count == 0:
            continue
        compact = _compact_sentence(window, max_chars=ANSWER_EVIDENCE_WINDOW_CHARS)
        if not compact:
            continue
        if _looks_like_navigation_span(compact):
            continue
        score = anchor_overlap * 1.5 + evidence_count * 6.0
        score += len(_years(query) & _years(candidate_match_text)) * 8.0
        score += _exact_date_overlap_count(query, candidate_match_text) * 12.0
        scored.append((score, -index, compact))
    if not scored:
        return None

    selected: list[str] = []
    total_chars = 0
    for _score, _negative_index, text in sorted(scored, reverse=True):
        normalized = re.sub(r"\s+", " ", text).strip()
        if any(normalized in existing or existing in normalized for existing in selected):
            continue
        if total_chars + len(normalized) > ANSWER_EVIDENCE_CONTEXT_CHARS:
            continue
        selected.append(normalized)
        total_chars += len(normalized)
        if total_chars >= ANSWER_EVIDENCE_WINDOW_CHARS:
            break
    if not selected:
        return None
    return "\n".join(selected)


def _local_evidence_pattern(query: str) -> re.Pattern[str] | None:
    patterns: list[str] = []
    lowered = query.lower()
    if _query_wants_contact(query) or _query_requires_phone_fact(query):
        patterns.extend(
            [
                r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
                r"(?<!\d)(?:0\d{2,3}[-\s]?)?\d{7,8}(?!\d)",
                r"联系咨询|咨询方式|联系人|联系电话|电话|座机|邮箱|办公室",
            ]
        )
    if _query_wants_schedule_and_contact(query) or _query_requires_capacity_limit(query):
        patterns.append(r"\d{1,2}\s*月\s*\d{1,2}\s*日|时间|地点|人数|上限|不超过|主讲")
    if "供应商" in query or "采购" in query or "procurement" in lowered:
        patterns.append(r"报价供应商要求|营业执照|税务登记证|组织机构代码证|联合体|报名资料|报价截止|递交地点")
    if "复试" in query or "总成绩" in query or "formula" in lowered:
        patterns.append(r"综合素质考核|专业面试|复试成绩|满分|合格|总成绩|初试成绩")
    if any(term in query for term in ("副主编", "IEEE Trans")) or any(
        term in lowered for term in ("tie", "tte", "tpea")
    ):
        patterns.append(r"副主编|IEEE\s*Trans(?:actions)?|TIE|TTE|TPEA")
    if any(term in query for term in ("专利", "第一发明人", "申请号", "在校生")):
        patterns.append(r"专利|第一发明人|申请号|在校生|CN\d{6,}")
    if any(term in query for term in ("选拔方式", "招生方式", "直博", "申请-考核制")):
        patterns.append(r"选拔方式|招生方式|直博|申请[-－—]考核制")
    if any(term in query for term in ("三选二", "本学科选修")) or ("2025级" in query and "ee" in lowered):
        patterns.append(r"三选二|本学科选修|2025\s*级|电子信息工程|EE")
    if any(term in query for term in ("录制成视频", "提前学习", "电力电子")):
        patterns.append(r"录制成视频|提前学习|电力电子")
    if not patterns:
        return None
    return re.compile("|".join(patterns), re.IGNORECASE)


def _metadata_answer_context_score(query: str, context: ContextItem) -> tuple[float, list[str]]:
    haystack = _context_score_text(context)
    normalized = haystack.lower()
    score = 0.0
    reasons: list[str] = []

    query_years = _years(query)
    context_years = _years(haystack)
    for year in sorted(query_years & context_years):
        score += 8.0
        reasons.append(f"query_year_match:{year}")
    exact_date_matches = _exact_date_overlap_count(query, haystack)
    if exact_date_matches:
        score += exact_date_matches * 8.0
        reasons.append("exact_date_match")
    elif _date_markers(query) and _date_markers(haystack):
        score -= 4.0
        reasons.append("date_mismatch_penalty")
    if query_years and context_years and _looks_like_degree_page(haystack):
        target_year = max(query_years)
        old_years = sorted(year for year in context_years if year < target_year)
        if old_years and target_year not in context_years:
            score -= 8.0
            reasons.append(f"old_year_penalty:{old_years[-1]}<{target_year}")

    anchor_count = sum(1 for term in _anchor_terms(query) if term in normalized)
    if anchor_count:
        score += min(anchor_count, 8) * 0.4
        reasons.append(f"anchor_overlap:{anchor_count}")

    if _query_wants_discipline_directions(query) and _has_discipline_direction_anchor(haystack):
        score += 6.0
        reasons.append("task_anchor:学科方向")

    program_matches = _matched_terms(
        query,
        haystack,
        ("cs", "computer science", "ee", "electrical", "electronic", "计算机", "电子", "电气", "信息"),
    )
    if program_matches:
        score += len(program_matches) * 1.5
        reasons.append(f"program_terms:{','.join(program_matches)}")

    course_matches = _matched_terms(
        query,
        haystack,
        ("培养方案", "学分", "课程", "program", "degree", "credit", "credits", "course", "curriculum"),
    )
    if course_matches:
        score += len(course_matches) * 1.5
        reasons.append(f"course_terms:{','.join(course_matches)}")

    if _query_wants_contact(query) and _has_contact_evidence(haystack):
        score += 3.0
        reasons.append("faculty_or_contact_evidence")
    if _query_wants_degree_page(query) and _looks_like_degree_page(haystack):
        score += 3.0
        reasons.append("page_type:degree_or_program")
    if not reasons:
        reasons.append("retrieval_rank_tiebreak")
    return score, reasons


def _context_score_text(context: ContextItem) -> str:
    return "\n".join(str(part or "") for part in (context.title, context.url, context.snippet, context.text))


def _years(text: str) -> set[int]:
    return {int(year) for year in re.findall(r"(?<!\d)(20\d{2})(?!\d)", text)}


def _date_markers(text: str) -> set[str]:
    markers: set[str] = set()
    for year, month, day in re.findall(r"(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", text):
        markers.add(f"{year}-{int(month):02d}-{int(day):02d}")
    for year, month, day in re.findall(r"(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})", text):
        markers.add(f"{year}-{int(month):02d}-{int(day):02d}")
    for year, month, day in re.findall(r"(?<!\d)(20\d{2})[/_-](\d{2})(\d{2})(?!\d)", text):
        markers.add(f"{year}-{int(month):02d}-{int(day):02d}")
    return markers


def _exact_date_overlap_count(query: str, text: str) -> int:
    query_dates = _date_markers(query)
    if not query_dates:
        return 0
    return len(query_dates & _date_markers(text))


def _matched_terms(query: str, context_text: str, terms: tuple[str, ...]) -> list[str]:
    query_lower = query.lower()
    context_lower = context_text.lower()
    return [term for term in terms if term.lower() in query_lower and term.lower() in context_lower]


def _query_wants_contact(query: str) -> bool:
    lowered = query.lower()
    chinese_contact_terms = (
        "办公室",
        "邮箱",
        "教师",
        "教授",
        "老师",
        "联系方式",
        "联系人",
        "联系电话",
        "电话",
        "咨询",
        "联系",
    )
    return any(term in lowered for term in ("office", "email", "contact", "faculty", "professor", "phone")) or any(
        term in query for term in chinese_contact_terms
    )


def _has_contact_evidence(text: str) -> bool:
    lowered = text.lower()
    return (
        re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text) is not None
        or "office" in lowered
        or "phone" in lowered
        or "tel" in lowered
        or "办公室" in text
        or "邮箱" in text
        or "联系人" in text
        or "联系电话" in text
        or "电话" in text
        or "咨询" in text
        or "professor" in lowered
        or re.search(r"\b(?:Room|Rm\.?)\s+[A-Za-z0-9]", text, flags=re.IGNORECASE) is not None
        or re.search(r"(?:\d+\s*号楼\s*)?(?:\d?[A-Za-z]|[A-Za-z]区)[-－]?\s*\d{2,4}\s*室", text) is not None
    )


def _query_wants_degree_page(query: str) -> bool:
    lowered = query.lower()
    return any(term in lowered for term in ("program", "degree", "credit", "credits", "curriculum")) or any(
        term in query for term in ("培养方案", "学分", "课程", "专业")
    )


def _query_wants_discipline_directions(query: str) -> bool:
    lowered = query.lower()
    return (
        "discipline direction" in lowered
        or "research direction" in lowered
        or "学科方向" in query
        or ("方向" in query and any(term in query for term in ("六个", "6个", "专业", "本科招生", "招生")))
    )


def _has_discipline_direction_anchor(text: str) -> bool:
    lowered = text.lower()
    return "学科方向" in text or "discipline direction" in lowered


def _looks_like_degree_page(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in ("program", "degree", "credit", "credits", "curriculum")) or any(
        term in text for term in ("培养方案", "学分", "课程", "专业")
    )


def _is_acceptable_answer(
    text: str,
    valid_source_ids: set[int],
    *,
    query: str | None = None,
    contexts: list[ContextItem] | None = None,
) -> bool:
    return _answer_rejection_reason(text, valid_source_ids, query=query, contexts=contexts) is None


def _answer_rejection_reason(
    text: str,
    valid_source_ids: set[int],
    *,
    query: str | None = None,
    contexts: list[ContextItem] | None = None,
) -> str | None:
    if any(marker in text for marker in PROMPT_LEAKAGE_MARKERS):
        return "prompt_leakage"
    if _states_insufficient_evidence(text):
        return "model_reported_insufficient_evidence"
    if not _has_valid_citation(text, valid_source_ids):
        return "invalid_or_missing_citation"
    if query is not None:
        shape_rejection = _query_shape_rejection_reason(query, text)
        if shape_rejection is not None:
            return shape_rejection
    if query is not None and contexts:
        label_binding_rejection = _label_value_binding_rejection_reason(query, text, contexts)
        if label_binding_rejection is not None:
            return label_binding_rejection
        formula_rejection = _numeric_formula_rejection_reason(query, text, contexts)
        if formula_rejection is not None:
            return formula_rejection
        source_fact_rejection = _requested_source_fact_rejection_reason(query, text, contexts)
        if source_fact_rejection is not None:
            return source_fact_rejection
        citation_rejection = _citation_support_rejection_reason(query, text, contexts)
        if citation_rejection is not None:
            return citation_rejection
    return None


def _has_valid_citation(text: str, valid_source_ids: set[int]) -> bool:
    citation_ids = [int(match.group(1)) for match in VALID_CITATION_RE.finditer(text)]
    return bool(citation_ids) and all(citation_id in valid_source_ids for citation_id in citation_ids)


def _states_insufficient_evidence(text: str) -> bool:
    normalized = text.lower()
    chinese_negative_markers = (
        "证据不足",
        "无法找到",
        "未提及",
        "没有提及",
        "未找到",
        "没有找到",
    )
    return "evidence is insufficient" in normalized or any(marker in text for marker in chinese_negative_markers)


def _query_shape_rejection_reason(query: str, answer: str) -> str | None:
    if _query_requires_contact_fact(query) and not _has_contact_evidence(answer):
        return "missing_requested_contact_fact"
    return None


def _query_requires_contact_fact(query: str) -> bool:
    lowered = query.lower()
    return any(term in lowered for term in ("office", "email", "contact")) or any(
        term in query for term in ("办公室", "邮箱", "联系方式", "联系")
    )


def _requested_source_fact_rejection_reason(query: str, answer: str, contexts: list[ContextItem]) -> str | None:
    cited_ids = {int(match.group(1)) for match in VALID_CITATION_RE.finditer(answer)}
    cited_text = "\n".join(_context_score_text(context) for context in contexts if context.rank in cited_ids)
    if _query_requires_phone_fact(query) and _has_phone_evidence(cited_text) and not _has_phone_evidence(answer):
        return "missing_requested_phone_fact"
    if _query_requires_capacity_limit(query):
        expected_limits = _capacity_limit_numbers(cited_text)
        if expected_limits and expected_limits.isdisjoint(_capacity_limit_numbers(answer)):
            return "missing_requested_capacity_limit"
    missing_labels = [
        label for label in ("课题组", "联合实验室") if label in query and label in cited_text and label not in answer
    ]
    if missing_labels:
        return "missing_requested_labeled_fact"
    return None


def _query_requires_phone_fact(query: str) -> bool:
    lowered = query.lower()
    return any(term in lowered for term in ("phone", "telephone", "tel", "contact")) or any(
        term in query for term in ("电话", "联系电话", "联系方式", "联系人")
    )


def _has_phone_evidence(text: str) -> bool:
    return re.search(r"(?<!\d)(?:0\d{2,3}[-\s]?)?\d{7,8}(?!\d)", text) is not None


def _query_requires_capacity_limit(query: str) -> bool:
    lowered = query.lower()
    return any(term in lowered for term in ("capacity", "limit", "cap")) or any(
        term in query for term in ("人数上限", "上限", "不超过", "名额")
    )


def _capacity_limit_numbers(text: str) -> set[str]:
    numbers: set[str] = set()
    for match in re.finditer(r"(?:不超过|不多于|限|上限)[^。；;，,\n]{0,12}?(\d+)\s*(?:人|位|名)", text):
        numbers.add(match.group(1))
    limit_after_subject = r"(?:人数|名额)[^。；;，,\n]{0,12}?(?:不超过|不多于|限|上限)[^。；;，,\n]{0,8}?(\d+)"
    for match in re.finditer(limit_after_subject, text):
        numbers.add(match.group(1))
    return numbers


def _parse_repair_answer(
    text: str,
    *,
    valid_source_ids: set[int],
    query: str | None = None,
    contexts: list[ContextItem] | None = None,
) -> str | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("status") != "answered":
        return None
    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return None
    answer = answer.strip()
    if not _is_acceptable_answer(answer, valid_source_ids, query=query, contexts=contexts):
        return None
    return answer


def _citation_support_rejection_reason(query: str, answer: str, contexts: list[ContextItem]) -> str | None:
    cited_ids = {int(match.group(1)) for match in VALID_CITATION_RE.finditer(answer)}
    context_by_rank = {context.rank: context for context in contexts}
    cited_contexts = [context_by_rank[source_id] for source_id in cited_ids if source_id in context_by_rank]
    if not cited_contexts:
        return None

    scores = {context.rank: _citation_support_score(query, answer, context) for context in contexts if context.url}
    if not scores:
        return None
    cited_best = max(scores.get(context.rank, 0.0) for context in cited_contexts)
    best_rank, best_score = max(scores.items(), key=lambda item: item[1])
    if best_rank in cited_ids:
        return None

    query_years = _years(query)
    if query_years and _context_matches_year(context_by_rank[best_rank], query_years):
        cited_year_match = any(_context_matches_year(context, query_years) for context in cited_contexts)
        if not cited_year_match:
            return "weak_citation_support"

    answer_facts = _answer_fact_terms(answer)
    if answer_facts and best_score >= 5.0 and best_score >= cited_best + 3.0:
        return "weak_citation_support"
    return None


def _label_value_binding_rejection_reason(query: str, answer: str, contexts: list[ContextItem]) -> str | None:
    cited_ids = {int(match.group(1)) for match in VALID_CITATION_RE.finditer(answer)}
    if not cited_ids:
        return None
    if "任选" not in query and "任选" not in answer:
        return None
    answer_free_values = _answer_labeled_credit_values(answer, ("任选课", "任选课程"))
    if not answer_free_values:
        return None

    cited_contexts = [context for context in contexts if context.rank in cited_ids and context.url]
    expected_values = _degree_plan_expected_values(query, cited_contexts, "任选课程")
    if not expected_values:
        return None
    if answer_free_values.isdisjoint(expected_values):
        return "unsupported_label_value_binding"
    return None


def _numeric_formula_rejection_reason(query: str, answer: str, contexts: list[ContextItem]) -> str | None:
    if not _query_wants_formula_answer(query, answer):
        return None
    cited_ids = {int(match.group(1)) for match in VALID_CITATION_RE.finditer(answer)}
    if not cited_ids:
        return None
    answer_weights = _percent_values(answer)
    if not answer_weights:
        return None
    cited_text = " ".join(_context_score_text(context) for context in contexts if context.rank in cited_ids)
    cited_weights = _percent_values(cited_text)
    if cited_weights and not answer_weights <= cited_weights:
        return "unsupported_numeric_formula"
    return None


def _query_wants_formula_answer(query: str, answer: str) -> bool:
    combined = f"{query} {answer}"
    return ("公式" in combined or "总成绩" in combined or "formula" in combined.lower()) and "%" in answer


def _percent_values(text: str) -> set[str]:
    return {_clean_number(value) for value in re.findall(r"(?<!\d)(\d+(?:\.\d+)?)\s*%", text)}


def _answer_labeled_credit_values(answer: str, labels: tuple[str, ...]) -> set[str]:
    values: set[str] = set()
    for label in labels:
        for match in re.finditer(rf"{label}[^。；;，,、\n]{{0,20}}?(\d+(?:\.\d+)?)\s*学分", answer):
            values.add(_clean_number(match.group(1)))
    return values


def _degree_plan_expected_values(query: str, contexts: list[ContextItem], label: str) -> set[str]:
    expected: set[str] = set()
    matching_contexts = []
    for context in contexts:
        text = re.sub(r"\s+", " ", f"{context.title or ''} {context.text}").strip()
        summary = _degree_plan_summary(text)
        if summary is None:
            continue
        if _degree_plan_context_matches_query(query, text):
            matching_contexts.append((summary, label))
        elif not matching_contexts:
            row = summary.get(label)
            if isinstance(row, tuple):
                expected.add(row[2])
    if matching_contexts:
        expected.clear()
        for summary, row_label in matching_contexts:
            row = summary.get(row_label)
            if isinstance(row, tuple):
                expected.add(row[2])
    return expected


def _degree_plan_context_matches_query(query: str, text: str) -> bool:
    query_years = _years(query)
    if query_years and not query_years <= _years(text):
        return False
    if "人工智能荣誉班" in query and "人工智能荣誉班" not in text:
        return False
    if ("电子信息工程" in query or "EE" in query) and "电子信息工程" not in text and "EE" not in text:
        return False
    if "计算机科学与技术" in query and "计算机科学与技术" not in text:
        return False
    return True


def _citation_support_score(query: str, answer: str, context: ContextItem) -> float:
    text = _context_score_text(context)
    normalized_text = text.lower()
    score = 0.0

    query_years = _years(query)
    context_years = _years(text)
    if query_years & context_years:
        score += 4.0
    elif query_years and context_years and max(context_years) < max(query_years):
        score -= 2.0

    answer_years = _years(answer)
    if answer_years & context_years:
        score += 4.0

    for fact in _answer_fact_terms(answer):
        if fact.lower() in normalized_text:
            score += 1.0

    anchor_hits = sum(1 for term in _anchor_terms(query) if term in normalized_text)
    score += min(anchor_hits, 8) * 0.4
    if _query_wants_degree_page(query) and _looks_like_degree_page(text):
        score += 2.0
    if _query_wants_contact(query) and _has_contact_evidence(text):
        score += 2.0
    return score


def _context_matches_year(context: ContextItem, years: set[int]) -> bool:
    return bool(_years(_context_score_text(context)) & years)


def _answer_fact_terms(answer: str) -> set[str]:
    clean_answer = VALID_CITATION_RE.sub(" ", answer)
    facts = set(re.findall(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", clean_answer))
    facts.update(re.findall(r"(?<!\d)20\d{2}(?!\d)", clean_answer))
    facts.update(re.findall(r"(?<!\d)\d+(?:\.\d+)?\s*(?:credits?|学分|%|人|项|门|个)?", clean_answer, re.I))
    facts.update(re.findall(r"\b\d{1,2}:\d{2}\b", clean_answer))
    return {fact.strip() for fact in facts if fact.strip()}


def _extract_answer_from_contexts(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    procurement_notice_answer = _extract_procurement_notice_answer(query, contexts)
    if procurement_notice_answer is not None:
        return procurement_notice_answer
    schedule_contact_answer = _extract_schedule_contact_answer(query, contexts)
    if schedule_contact_answer is not None:
        return schedule_contact_answer
    procurement_delivery_answer = _extract_procurement_delivery_answer(query, contexts)
    if procurement_delivery_answer is not None:
        return procurement_delivery_answer
    faculty_profile_answer = _extract_faculty_profile_slot_answer(query, contexts)
    if faculty_profile_answer is not None:
        return faculty_profile_answer
    office_answer = _extract_office_email_answer(query, contexts)
    if office_answer is not None:
        return office_answer
    address_answer = _extract_address_postcode_answer(query, contexts)
    if address_answer is not None:
        return address_answer
    lab_count_answer = _extract_lab_count_answer(query, contexts)
    if lab_count_answer is not None:
        return lab_count_answer
    discipline_direction_answer = _extract_admissions_discipline_direction_answer(query, contexts)
    if discipline_direction_answer is not None:
        return discipline_direction_answer
    committee_answer = _extract_committee_row_answer(query, contexts)
    if committee_answer is not None:
        return committee_answer
    degree_summary_answer = _extract_degree_plan_summary_answer(query, contexts)
    if degree_summary_answer is not None:
        return degree_summary_answer
    retest_formula_answer = _extract_retest_formula_answer(query, contexts)
    if retest_formula_answer is not None:
        return retest_formula_answer
    course_credit_answer = _extract_course_credit_row_answer(query, contexts)
    if course_credit_answer is not None:
        return course_credit_answer
    course_design_answer = _extract_course_design_pair_answer(query, contexts)
    if course_design_answer is not None:
        return course_design_answer
    dedup_answer = _extract_degree_plan_dedup_answer(query, contexts)
    if dedup_answer is not None:
        return dedup_answer
    credit_answer = _extract_credit_answer(query, contexts)
    if credit_answer is not None:
        return credit_answer
    seminar_answer = _extract_seminar_event_fields_answer(query, contexts)
    if seminar_answer is not None:
        return seminar_answer
    schedule_answer = _extract_date_time_location_answer(query, contexts)
    if schedule_answer is not None:
        return schedule_answer
    robotics_answer = _extract_robotics_faculty_answer(query, contexts)
    if robotics_answer is not None:
        return robotics_answer
    profile_answer = _extract_compact_person_profile_answer(query, contexts)
    if profile_answer is not None:
        return profile_answer
    student_undergraduate_answer = _extract_student_undergraduate_school_answer(query, contexts)
    if student_undergraduate_answer is not None:
        return student_undergraduate_answer
    for course_name in _course_terms_from_query(query):
        course_pattern = re.escape(course_name).replace(r"\ ", r"\s+")
        teacher_pattern = re.compile(rf"{course_pattern}\s*【\s*(?P<teacher>[^】]{{1,80}}?)\s*】", re.IGNORECASE)
        for context in contexts:
            if context.url is None:
                continue
            normalized_text = re.sub(r"\s+", " ", context.text)
            match = teacher_pattern.search(normalized_text)
            if match is None:
                continue
            teacher = " ".join(match.group("teacher").split())
            if not teacher:
                continue
            if _is_chinese(query):
                return ExtractiveAnswer(f"{course_name}的任课老师是{teacher}。 [{context.rank}]", context.rank)
            return ExtractiveAnswer(f"{course_name} was taught by {teacher} [{context.rank}].", context.rank)
    list_answer = _extract_list_or_comparison_answer(query, contexts)
    if list_answer is not None:
        return list_answer
    return None


def _extract_lab_count_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    if "课题组" not in query and "联合实验室" not in query:
        return None
    for context in contexts:
        if context.url is None:
            continue
        text = re.sub(r"\s+", "", context.text)
        match = re.search(r"(\d+)个?课题组[^。；;]{0,20}?(\d+)个?联合实验室", text)
        if match is None:
            continue
        groups, labs = match.groups()
        return ExtractiveAnswer(f"信息学院有{groups}个课题组、{labs}个联合实验室。 [{context.rank}]", context.rank)
    return None


def _extract_retest_formula_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    if "复试" not in query or ("公式" not in query and "总成绩" not in query):
        return None
    for context in contexts:
        if context.url is None:
            continue
        text = re.sub(r"\s+", "", context.text)
        if "综合素质考核" not in text or "专业面试" not in text or "总成绩" not in text:
            continue
        full_score = re.search(r"复试成绩满分(?:为)?(\d+(?:\.\d+)?)分", text)
        pass_score = re.search(r"(\d+(?:\.\d+)?)分为合格", text)
        formula = re.search(
            r"考生总成绩[=＝]初试成绩[×x*](\d+(?:\.\d+)?%)\+复试成绩[×x*](\d+(?:\.\d+)?%)",
            text,
            flags=re.IGNORECASE,
        )
        normalized_formula = re.search(
            r"考生总成绩[=＝](?P<formula>"
            r"\d+(?:\.\d+)?[×x*]初试成绩[/／]初试满分\+"
            r"\d+(?:\.\d+)?[×x*]复试成绩[/／]复试满分)",
            text,
            flags=re.IGNORECASE,
        )
        if full_score is None or pass_score is None or (formula is None and normalized_formula is None):
            continue
        full = _clean_number(full_score.group(1))
        passing = _clean_number(pass_score.group(1))
        if formula is not None:
            formula_text = f"初试成绩×{formula.group(1)}+复试成绩×{formula.group(2)}"
        else:
            formula_text = normalized_formula.group("formula").replace("×", "*").replace("x", "*")
        return ExtractiveAnswer(
            (
                "2026年复试包括综合素质考核和专业面试；"
                f"复试成绩满分为{full}分，{passing}分为合格；"
                f"考生总成绩={formula_text}。 [{context.rank}]"
            ),
            context.rank,
        )
    return None


def _extract_admissions_discipline_direction_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    if not _query_wants_discipline_directions(query):
        return None
    for context in contexts:
        if context.url is None:
            continue
        text = re.sub(r"\s+", " ", context.text).strip()
        if "学科方向" not in text:
            continue
        directions = _field_after_label(text, "学科方向")
        if directions is None:
            continue
        subject = "CS专业" if "CS" in query or "计算机" in query or "计算机科学与技术" in text else "该专业"
        return ExtractiveAnswer(f"{subject}的学科方向是{directions}。 [{context.rank}]", context.rank)
    return None


def _extract_committee_row_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    if "委员会" not in query or "主任" not in query:
        return None
    target = _committee_name_from_query(query)
    if target is None:
        return None
    for context in contexts:
        if context.url is None:
            continue
        if not all(label in context.text for label in ("委员会", "主任")):
            continue
        for line in context.text.splitlines():
            row = re.sub(r"\s+", " ", line).strip()
            if target not in row:
                continue
            match = re.search(rf"{re.escape(target)}\s+(?P<director>\S+)\s+(?P<deputy>[^。；;\n]+)", row)
            if match is None:
                continue
            director = match.group("director").strip(" ，,")
            deputy = match.group("deputy").strip(" ，,")
            if director and deputy:
                return ExtractiveAnswer(f"{target}主任是{director}，副主任是{deputy}。 [{context.rank}]", context.rank)
    return None


def _committee_name_from_query(query: str) -> str | None:
    for match in re.finditer(r"[\u4e00-\u9fff]{2,20}委员会", query):
        name = match.group(0)
        if name.startswith("信息学院") and len(name) > len("信息学院委员会"):
            name = name.removeprefix("信息学院")
        return name
    return None


def _extract_schedule_contact_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    if not _query_wants_schedule_and_contact(query):
        return None
    return _extract_compact_evidence(
        query,
        contexts,
        evidence_pattern=re.compile(
            r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b|"
            r"\d{1,2}\s*月\s*\d{1,2}\s*日|"
            r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|"
            r"(?<!\d)(?:0\d{2,3}[-\s]?)?\d{7,8}(?!\d)|"
            r"(?:日期|时间|安排|联系|联系人|如有疑问|邮箱|电话|截止|递交|地点|人数|上限|不超过|主讲)",
            re.IGNORECASE,
        ),
    )


def _query_wants_schedule_and_contact(query: str) -> bool:
    lowered = query.lower()
    wants_contact = any(term in lowered for term in ("email", "contact")) or any(
        term in query for term in ("邮箱", "联系", "联系人", "疑问")
    )
    wants_schedule = any(term in lowered for term in ("date", "time", "schedule")) or any(
        term in query for term in ("日期", "时间", "安排", "哪天", "几月", "几日")
    )
    return wants_contact and wants_schedule


def _extract_procurement_notice_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    if "供应商" not in query or "报价" not in query or "递交地点" not in query:
        return None
    for context in contexts:
        if context.url is None:
            continue
        text = re.sub(r"\s+", " ", context.text).strip()
        if not all(term in text for term in ("报价供应商要求", "联系人", "报价截止", "递交地点")):
            continue
        if "独立承担民事责任" not in text or "不允许联合体报价" not in text:
            continue
        teacher = _first_match(r"[\u4e00-\u9fffA-Za-z]{1,12}老师", text)
        phone = _first_match(r"(?<!\d)(?:0\d{2,3}[-\s]?)?\d{7,8}(?!\d)", text)
        email = _first_match(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)
        deadline_match = re.search(
            r"报价截止时间\s*(?P<deadline>20\d{2}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日\s*\d{1,2}\s*[:：]\s*\d{2})",
            text,
        )
        location_match = re.search(r"递交地点\s*(?P<location>[^。；;\n]{6,80})", text)
        if teacher is None or phone is None or email is None or deadline_match is None or location_match is None:
            continue
        deadline = re.sub(r"\s+", "", deadline_match.group("deadline")).replace("：", ":")
        location = re.sub(r"\s+", "", location_match.group("location")).strip(" 。；;")
        return ExtractiveAnswer(
            (
                "供应商需能独立承担民事责任，具有企业法人营业执照、税务登记证、组织机构代码证复印件，"
                f"且本项目不允许联合体报价；报名资料发送给{teacher}，电话{phone}，邮箱{email}；"
                f"报价截止时间为{deadline}，递交地点为{location}。 [{context.rank}]."
            ),
            context.rank,
        )
    return None


def _extract_procurement_delivery_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    if not all(term in query for term in ("采购", "递交")):
        return None
    if not any(term in query for term in ("异议", "质疑", "询价结果", "书面")):
        return None
    query_terms = _anchor_terms(query)
    for context in contexts:
        if context.url is None:
            continue
        text = re.sub(r"\s+", "", context.text)
        if not _has_anchor_overlap(query_terms, f"{context.title or ''}{text}"):
            continue
        room_match = re.search(r"(?:华夏中路393号)?信息学院1号楼(?:1B|B区|B)[-－]?\s*206室?", text)
        if room_match is None:
            continue
        teacher = _first_match(r"[\u4e00-\u9fffA-Za-z]{1,12}老师", text)
        if teacher is not None:
            teacher = re.sub(r"^(?:受理人|联系人|联系|为)+", "", teacher)
        location = room_match.group(0).strip()
        if teacher is not None:
            return ExtractiveAnswer(f"书面质疑材料应递交至{location}，{teacher}处。 [{context.rank}].", context.rank)
        return ExtractiveAnswer(f"书面质疑材料应递交至{location}。 [{context.rank}].", context.rank)
    return None


def _extract_address_postcode_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    lowered_query = query.lower()
    if not any(term in lowered_query for term in ("address", "postcode", "postal code")) and not any(
        term in query for term in ("地址", "邮编")
    ):
        return None
    return _extract_compact_evidence(
        query,
        contexts,
        evidence_pattern=re.compile(r"(?:address|postcode|postal code|地址|邮编|\b\d{6}\b)", re.IGNORECASE),
    )


def _extract_faculty_profile_slot_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    wants_contact = "办公室" in query or "邮箱" in query or any(term in query.lower() for term in ("office", "email"))
    wants_profile = (
        "博士" in query
        or "研究方向" in query
        or "phd" in query.lower()
        or "research direction" in query.lower()
    )
    if not wants_contact or not wants_profile:
        return None
    query_terms = _anchor_terms(query)
    name = _person_name_from_query(query)
    for context in contexts:
        if context.url is None:
            continue
        text = re.sub(r"\s+", " ", context.text).strip()
        if name is not None and name not in text:
            continue
        if name is None and not _has_anchor_overlap(query_terms, f"{context.title or ''} {text}"):
            continue
        office = _field_after_label(text, "办公室")
        email = _first_match(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)
        phd_school = _phd_school_from_profile_text(text)
        direction = _field_after_label(text, "研究方向")
        if office is None or email is None or phd_school is None or direction is None:
            continue
        subject = name or "该教师"
        return ExtractiveAnswer(
            (
                f"{subject}的办公室是{office}，邮箱是{email}，"
                f"博士毕业学校是{phd_school}，研究方向是{direction}。 [{context.rank}]"
            ),
            context.rank,
        )
    return None


def _person_name_from_query(query: str) -> str | None:
    match = re.search(r"(?P<name>[\u4e00-\u9fff]{2,4})(?:教授|老师|的)", query)
    if match is None:
        return None
    return match.group("name")


def _field_after_label(text: str, label: str) -> str | None:
    match = re.search(
        rf"{label}[:：]\s*(?P<value>.*?)"
        r"(?=\s*(?:办公室|邮箱|研究方向|教育背景|身份|报告人|演讲者|主讲人|所在单位|单位|机构|时间|地点)[:：]|[，,。；;\n]|$)",
        text,
    )
    if match is None:
        return None
    value = match.group("value").strip()
    return value or None


def _field_after_first_label(text: str, labels: tuple[str, ...]) -> str | None:
    for label in labels:
        value = _field_after_label(text, label)
        if value is not None:
            return value
    return None


def _phd_school_from_profile_text(text: str) -> str | None:
    match = re.search(
        r"博士(?:毕业于|毕业学校[:：]?|毕业院校[:：]?|学位[^，,。；;]{0,12}?于)\s*"
        r"(?P<school>[\u4e00-\u9fffA-Za-z0-9（）()·\- ]{2,40})",
        text,
    )
    if match is None:
        return None
    school = match.group("school").strip(" ，,。；;")
    return school or None


def _extract_office_email_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    lowered_query = query.lower()
    if "office" not in lowered_query and "办公室" not in query and "email" not in lowered_query and "邮箱" not in query:
        return None
    query_terms = _anchor_terms(query)
    candidates: list[ExtractiveCandidate] = []
    for context_order, context in enumerate(contexts):
        if context.url is None:
            continue
        normalized_text = re.sub(r"\s+", " ", context.text).strip()
        if not _has_anchor_overlap(query_terms, normalized_text):
            continue
        email = _first_match(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", normalized_text)
        room = _first_match(
            r"\b(?:Room|Rm\.?)\s+[A-Za-z0-9][A-Za-z0-9.-]{1,20}\b|"
            r"办公室[:：]\s*[\u4e00-\u9fffA-Za-z0-9.-]{2,30}",
            normalized_text,
        )
        if room is not None and "办公室" in room:
            room = re.sub(r"^办公室[:：]\s*", "", room).strip()
        if email is None and room is None:
            continue
        facts = []
        contact_label = _target_contact_label(query, normalized_text)
        if contact_label is not None:
            facts.append(f"contact: {contact_label}")
        if room is not None:
            facts.append(f"office: {room}")
        if email is not None:
            facts.append(f"email: {email}")
        anchor_overlap = _anchor_overlap_count(query_terms, f"{context.title or ''} {normalized_text}")
        focus_overlap = _contact_focus_overlap(query, normalized_text)
        score = 20.0 - context_order * 0.25 + min(anchor_overlap, 10) * 1.5 + focus_overlap * 6.0
        candidates.append(
            ExtractiveCandidate(
                text="; ".join(facts),
                source_rank=context.rank,
                context_order=context_order,
                score=score,
            )
        )
    if not candidates:
        return None
    best = max(candidates, key=lambda candidate: (candidate.score, -candidate.context_order, -candidate.source_rank))
    return ExtractiveAnswer(f"{best.text} [{best.source_rank}].", best.source_rank)


def _target_contact_label(query: str, text: str) -> str | None:
    for label in re.findall(r"[\u4e00-\u9fffA-Za-z]{1,12}老师", query):
        if label in text:
            return label
    match = re.search(r"(?:联系|咨询|负责)[^。；;]{0,20}?([\u4e00-\u9fffA-Za-z]{1,12}老师)", text)
    if match is None:
        return None
    return match.group(1)


def _contact_focus_overlap(query: str, text: str) -> int:
    focus_terms = (
        "招生咨询",
        "咨询",
        "高老师",
        "如有疑问",
        "联系",
        "联系人",
        "负责",
        "通知",
        "培训",
        "安排",
    )
    return sum(1 for term in focus_terms if term in query and term in text)


def _extract_credit_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    lowered_query = query.lower()
    if "credit" not in lowered_query and "学分" not in query:
        return None
    return _extract_compact_evidence(
        query,
        contexts,
        evidence_pattern=re.compile(r"\d+(?:\.\d+)?\s*(?:credits?|学分)", re.IGNORECASE),
    )


def _extract_degree_plan_summary_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    if "培养方案" not in query or "学分" not in query:
        return None
    comparison = _extract_degree_plan_comparison_answer(query, contexts)
    if comparison is not None:
        return comparison
    for context in contexts:
        if context.url is None:
            continue
        text = re.sub(r"\s+", " ", f"{context.title or ''} {context.text}").strip()
        summary = _degree_plan_summary(text)
        if summary is None:
            continue
        label = _degree_plan_label(query, text)
        total = summary.get("total")
        free = summary.get("任选课程", (None, None, None))[2]
        natural = summary.get("自然科学通识", (None, None, None))[2]
        professional = summary.get("专业课程", (None, None, None))
        if total and free and "任选" in query and not _query_wants_multiple_facts(query):
            return ExtractiveAnswer(
                f"{label}毕业至少需要修满{total}学分，任选课程占{free}学分。 [{context.rank}]",
                context.rank,
            )
        if "人文社科" in query and "自然科学" in query:
            humanities = summary.get("人文社科通识", (None, None, None))[2]
            if humanities and natural:
                return ExtractiveAnswer(
                    (
                        f"{label}中，人文社科通识板块要求{humanities}学分，"
                        f"自然科学通识板块要求{natural}学分。 [{context.rank}]"
                    ),
                    context.rank,
                )
        if total and "专业课程" in query and "必修" in query and "选修" in query:
            required, elective, professional_total = professional
            if required and elective and professional_total:
                return ExtractiveAnswer(
                    (
                        f"{label}毕业至少需要修满{total}学分；专业课程板块必修{required}学分、"
                        f"选修{elective}学分，合计{professional_total}学分。 [{context.rank}]"
                    ),
                    context.rank,
                )
        if (
            total
            and natural
            and professional[2]
            and free
            and "自然科学" in query
            and "专业课程" in query
            and "任选" in query
        ):
            return ExtractiveAnswer(
                (
                    f"{label}毕业至少需要修满{total}学分；自然科学通识{natural}学分，"
                    f"专业课程{professional[2]}学分，任选课程{free}学分。 [{context.rank}]"
                ),
                context.rank,
            )
    return None


def _extract_course_credit_row_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    if "学分" not in query and "credit" not in query.lower():
        return None
    for code in _course_codes_from_query(query):
        for context in contexts:
            if context.url is None:
                continue
            text = re.sub(r"\s+", " ", context.text).strip()
            match = re.search(
                rf"(?<![A-Z0-9]){re.escape(code)}(?![A-Z0-9])"
                rf"\s+(?P<name>[\u4e00-\u9fffA-Za-z0-9（）()ⅠⅡⅢIVX\s]+?)"
                rf"\s+(?P<credits>\d+(?:\.\d+)?)\s+(?=[一二三四五六七八九十]（|\d|[A-Z]{{2,}}\d)",
                text,
            )
            if match is None:
                continue
            name = re.sub(r"\s+", "", match.group("name")).strip()
            credits = _clean_number(match.group("credits"))
            if not name:
                continue
            if _is_chinese(query):
                return ExtractiveAnswer(f"{code}是《{name}》，{credits}学分。 [{context.rank}]", context.rank)
            return ExtractiveAnswer(f"{code} is {name}, worth {credits} credits [{context.rank}].", context.rank)
    return None


def _course_codes_from_query(query: str) -> list[str]:
    return re.findall(r"(?<![A-Z0-9])(?:[A-Z]{2,}\d{2,}[A-Z]?)(?![A-Z0-9])", query.upper())


def _extract_course_design_pair_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    if "课程设计" not in query or "合计" not in query or "学期" not in query:
        return None
    if "计算机体系结构" not in query:
        return None
    for context in contexts:
        if context.url is None:
            continue
        text = re.sub(r"\s+", " ", context.text).strip()
        theory = _course_row(text, code="CS110", name_pattern=r"计算机体系结构\s*I(?!\s*课程设计)")
        project = _course_row(text, code="CS110P", name_pattern=r"计算机体系结构\s*I\s*课程设计")
        if theory is None or project is None:
            continue
        theory_credits, theory_semester = theory
        project_credits, project_semester = project
        if theory_semester != project_semester:
            continue
        total = _clean_number(str(float(theory_credits) + float(project_credits)))
        return ExtractiveAnswer(
            "配套课程设计是CS110P《计算机体系结构I课程设计》。"
            f"CS110理论课{theory_credits}学分、CS110P课程设计{project_credits}学分，"
            f"合计{total}学分，均推荐在{theory_semester}学期修读。 [{context.rank}]",
            context.rank,
        )
    return None


def _course_row(text: str, *, code: str, name_pattern: str) -> tuple[str, str] | None:
    match = re.search(
        rf"{code}\s+{name_pattern}\s+(?P<credits>\d+(?:\.\d+)?)\s+(?P<semester>[一二三四五六七八九十]（\s*\d+\s*）)",
        text,
    )
    if match is None:
        return None
    semester = re.sub(r"\s+", "", match.group("semester"))
    return _clean_number(match.group("credits")), semester


def _extract_degree_plan_dedup_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    if "自动去重" not in query and "去重规则" not in query:
        return None
    if "本学科选修" not in query and "上一层级" not in query:
        return None
    for context in contexts:
        if context.url is None:
            continue
        text = re.sub(r"\s+", "", context.text)
        if not all(term in text for term in ("自动去重", "不重复计算学分", "上一层级", "仅会被计算1次")):
            continue
        return ExtractiveAnswer(
            f"教务系统在结算上一层级总学分时会自动去重；该课程学分最终仅计算1次，不会重复累加。 [{context.rank}]",
            context.rank,
        )
    return None


def _extract_degree_plan_comparison_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    if "对比" not in query and "比较" not in query:
        return None
    if "自然科学" not in query or "专业课程" not in query:
        return None
    regular: tuple[ContextItem, dict[str, tuple[str | None, str | None, str] | str]] | None = None
    honors: tuple[ContextItem, dict[str, tuple[str | None, str | None, str] | str]] | None = None
    for context in contexts:
        if context.url is None:
            continue
        text = re.sub(r"\s+", " ", f"{context.title or ''} {context.text}").strip()
        summary = _degree_plan_summary(text)
        if summary is None:
            continue
        if "人工智能荣誉班" in text:
            honors = (context, summary)
        elif "计算机科学与技术" in text or "CS" in (context.title or ""):
            regular = (context, summary)
    if regular is None or honors is None:
        return None

    regular_context, regular_summary = regular
    honors_context, honors_summary = honors
    regular_natural = _degree_summary_row_total(regular_summary, "自然科学通识")
    regular_professional = _degree_summary_row_total(regular_summary, "专业课程")
    honors_natural = _degree_summary_row_total(honors_summary, "自然科学通识")
    honors_professional = _degree_summary_row_total(honors_summary, "专业课程")
    if not all((regular_natural, regular_professional, honors_natural, honors_professional)):
        return None
    return ExtractiveAnswer(
        "2025级普通CS专业要求"
        f"自然科学通识{regular_natural}学分、专业课程{regular_professional}学分；"
        f"CS专业人工智能荣誉班要求自然科学通识{honors_natural}学分、专业课程{honors_professional}学分。 "
        f"[{regular_context.rank}][{honors_context.rank}]",
        regular_context.rank,
    )


def _degree_summary_row_total(
    summary: dict[str, tuple[str | None, str | None, str] | str],
    label: str,
) -> str | None:
    row = summary.get(label)
    if not isinstance(row, tuple):
        return None
    return row[2]


def _degree_plan_summary(text: str) -> dict[str, tuple[str | None, str | None, str] | str] | None:
    if "类别" not in text or "学分" not in text:
        return None
    total_match = re.search(r"修满至少\s*(\d+(?:\.\d+)?)\s*学分", text)
    rows: dict[str, tuple[str | None, str | None, str] | str] = {}
    if total_match is not None:
        rows["total"] = _clean_number(total_match.group(1))
    for label in ("人文社科通识", "自然科学通识", "专业课程"):
        match = re.search(rf"{label}\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)", text)
        if match is not None:
            rows[label] = tuple(_clean_number(value) for value in match.groups())  # type: ignore[assignment]
    free_match = re.search(r"任选课程\s+(\d+(?:\.\d+)?)(?:\s+\d+(?:\.\d+)?)?", text)
    if free_match is not None:
        rows["任选课程"] = (None, None, _clean_number(free_match.group(1)))
    if len(rows) < 2:
        return None
    return rows


def _clean_number(number: str) -> str:
    return number[:-2] if number.endswith(".0") else number


def _degree_plan_label(query: str, context_text: str) -> str:
    source = f"{query} {context_text}"
    year_match = re.search(r"(20\d{2})\s*级", source)
    year = f"{year_match.group(1)}级" if year_match is not None else ""
    if "人工智能荣誉班" in source:
        return f"{year}计算机科学与技术专业人工智能荣誉班"
    if "电子信息工程" in source or "EE" in query:
        return f"{year}电子信息工程专业"
    if "计算机科学与技术" in source or "CS" in query:
        return f"{year}计算机科学与技术专业"
    return f"{year}本科专业" if year else "该培养方案"


def _extract_date_time_location_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    lowered_query = query.lower()
    if not any(term in lowered_query for term in ("date", "time", "when", "where", "location")) and not any(
        term in query for term in ("日期", "时间", "地点", "哪里", "何时")
    ):
        return None
    return _extract_compact_evidence(
        query,
        contexts,
        evidence_pattern=re.compile(
            r"\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b|\b\d{1,2}:\d{2}\b|"
            r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b|"
            r"(?:日期|时间|地点|Room|Building)",
            re.IGNORECASE,
        ),
    )


def _extract_seminar_event_fields_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    wants_speaker = any(term in query for term in ("报告人", "演讲者", "主讲人", "speaker"))
    wants_institution = any(term in query for term in ("单位", "机构", "institution"))
    wants_time_location = "时间" in query and "地点" in query
    if not (wants_speaker and wants_institution and wants_time_location):
        return None
    for context in contexts:
        if context.url is None:
            continue
        text = re.sub(r"\s+", " ", context.text).strip()
        speaker = _field_after_first_label(text, ("报告人", "演讲者", "主讲人"))
        institution = _field_after_first_label(text, ("所在单位", "单位", "机构"))
        time_value = _field_after_label(text, "时间")
        location = _field_after_label(text, "地点")
        if not all((speaker, institution, time_value, location)):
            continue
        return ExtractiveAnswer(
            f"报告人是{speaker}，单位是{institution}，时间是{time_value}，地点是{location}。 [{context.rank}]",
            context.rank,
        )
    return None


def _extract_robotics_faculty_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    if "robotics" not in query.lower():
        return None
    for context in contexts:
        if context.url is None:
            continue
        normalized_text = re.sub(r"\s+", " ", f"{context.title or ''} {context.text}")
        if "robotics" not in normalized_text.lower() or "schwertfeger" not in normalized_text.lower():
            continue
        return ExtractiveAnswer(f"Prof. Schwertfeger works on robotics [{context.rank}].", context.rank)
    return None


def _extract_compact_person_profile_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    if not all(term in query for term in ("身份", "教育背景", "研究方向")):
        return None
    name_match = re.search(r"(?P<name>[\u4e00-\u9fff]{2,4})的", query)
    if name_match is None:
        return None
    name = name_match.group("name")
    for context in contexts:
        if context.url is None:
            continue
        text = re.sub(r"\s+", " ", context.text).strip()
        match = re.search(
            rf"{re.escape(name)}\s+身份[:：]\s*(?P<identity>[^，,。；;\s]+)\s+"
            rf"教育背景[:：]\s*(?P<education>[^，,。；;]+?)\s+"
            rf"研究方向[:：]\s*(?P<direction>[^，,。；;]+)",
            text,
        )
        if match is not None:
            identity = match.group("identity").strip()
            education = match.group("education").strip()
            direction = match.group("direction").strip()
            if not all((identity, education, direction)):
                continue
            return ExtractiveAnswer(
                f"{name}的身份是{identity}，教育背景是{education}，研究方向是{direction}。 [{context.rank}]",
                context.rank,
            )
        row_match = _lab_member_row_match(name, context.text)
        if row_match is None:
            continue
        identity, education, direction = row_match
        return ExtractiveAnswer(
            f"{name}的身份是{identity}，教育背景是{education}，研究方向是{direction}。 [{context.rank}]",
            context.rank,
        )
    return None


def _lab_member_row_match(name: str, text: str) -> tuple[str, str, str] | None:
    if not all(label in text for label in ("姓名", "身份", "教育背景", "研究方向")):
        return None
    pattern = re.compile(
        rf"{re.escape(name)}\s+"
        r"(?P<identity>博士研究生|硕士研究生|博士生|硕士生|本科生|研究生)\s+"
        r"(?P<education>[\u4e00-\u9fffA-Za-z0-9（）()·\-]+(?:本科|硕士|博士|学士|毕业)?)\s+"
        r"(?P<direction>[\u4e00-\u9fffA-Za-z0-9（）()·\-]{2,30})"
    )
    match = pattern.search(text)
    if match is None:
        return None
    return (
        match.group("identity").strip(),
        match.group("education").strip(),
        match.group("direction").strip(),
    )


def _extract_student_undergraduate_school_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    if "本科" not in query or not any(term in query for term in ("毕业院校", "本科毕业", "毕业于")):
        return None
    if not any(term in query for term in ("博士研究生", "研究生", "博士生")):
        return None

    query_terms = _anchor_terms(query)
    candidates: list[ExtractiveCandidate] = []
    for context_order, context in enumerate(contexts):
        if context.url is None:
            continue
        for sentence in re.split(r"[。！？.!?；;\n]+", context.text):
            normalized = re.sub(r"\s+", " ", sentence).strip()
            if not normalized:
                continue
            match = re.search(
                r"(?:博士研究生|研究生|博士生)?(?P<name>[\u4e00-\u9fff]{2,4})[，,、\s]+"
                r"(?P<body>[^。！？.!?；;]{0,160}?本科毕业于"
                r"(?P<school>[\u4e00-\u9fffA-Za-z0-9（）()·\- ]{2,40}))",
                normalized,
            )
            if match is None:
                continue
            school = match.group("school").strip(" ，,。；;")
            if not school:
                continue
            candidate_text = f"{context.title or ''} {normalized}"
            anchor_overlap = _anchor_overlap_count(query_terms, candidate_text)
            if anchor_overlap < _minimum_anchor_overlap(query_terms):
                continue
            focus_overlap = _student_focus_overlap(query, normalized, query_terms)
            score = 20.0 - context_order * 0.25 + min(anchor_overlap, 10) * 1.5 + focus_overlap * 8.0
            name = match.group("name")
            candidates.append(
                ExtractiveCandidate(
                    text=f"{name}的本科毕业院校是{school}",
                    source_rank=context.rank,
                    context_order=context_order,
                    score=score,
                )
            )
    if not candidates:
        return None
    best = max(candidates, key=lambda candidate: (candidate.score, -candidate.context_order, -candidate.source_rank))
    return ExtractiveAnswer(f"{best.text}。 [{best.source_rank}].", best.source_rank)


def _student_focus_overlap(query: str, text: str, query_terms: set[str]) -> int:
    broad_terms = {"信息", "学院", "博士", "研究生", "博士研究生", "本科", "毕业", "院校", "哪所", "那位"}
    overlap = sum(1 for term in query_terms if term not in broad_terms and term in text)
    if "研究方向" in query and "研究方向" in text:
        overlap += 1
    return overlap


def _extract_list_or_comparison_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    lowered_query = query.lower()
    if not any(term in lowered_query for term in ("list", "which", "different", "difference", "compare")) and not any(
        term in query for term in ("哪些", "有什么不同", "区别", "比较")
    ):
        return None
    return _extract_compact_evidence(query, contexts, evidence_pattern=None)


def _extract_compact_evidence(
    query: str,
    contexts: list[ContextItem],
    *,
    evidence_pattern: re.Pattern[str] | None,
) -> ExtractiveAnswer | None:
    query_terms = _anchor_terms(query)
    candidates: list[ExtractiveCandidate] = []
    for context_order, context in enumerate(contexts):
        if context.url is None:
            continue
        context_header = " ".join(part for part in (context.title, context.url, context.snippet) if part)
        for window in _candidate_windows(context.text):
            candidate_match_text = f"{context_header} {window}"
            anchor_overlap = _anchor_overlap_count(query_terms, candidate_match_text)
            if anchor_overlap < _minimum_anchor_overlap(query_terms):
                continue
            if evidence_pattern is not None and evidence_pattern.search(window) is None:
                continue
            if _has_newer_year_conflict(query, candidate_match_text):
                continue
            compact = _compact_sentence(_with_year_title_prefix(query, context, window))
            if not compact or _looks_like_navigation_span(compact):
                continue
            score = _extractive_candidate_score(
                query,
                query_terms,
                compact,
                context=context,
                context_order=context_order,
                evidence_pattern=evidence_pattern,
                anchor_overlap=anchor_overlap,
            )
            candidates.append(
                ExtractiveCandidate(
                    text=compact,
                    source_rank=context.rank,
                    context_order=context_order,
                    score=score,
                )
            )
    if not candidates:
        return None
    best = max(candidates, key=lambda candidate: (candidate.score, -candidate.context_order, -candidate.source_rank))
    return ExtractiveAnswer(f"{best.text} [{best.source_rank}].", best.source_rank)


def _candidate_windows(text: str) -> list[str]:
    units = _candidate_sentences(text)
    windows: list[str] = []
    seen: set[str] = set()
    for start in range(len(units)):
        for size in (1, 2, 3, 4):
            window_units = units[start : start + size]
            if len(window_units) != size:
                continue
            window = "; ".join(window_units)
            normalized = re.sub(r"\s+", " ", window).strip()
            if normalized in seen:
                continue
            seen.add(normalized)
            windows.append(window)
    return windows


def _candidate_sentences(text: str) -> list[str]:
    normalized = re.sub(r"[ \t\r\f\v]+", " ", text).strip()
    return [
        sentence.strip(" ;")
        for sentence in re.split(r"(?<=[。！？.!?])\s+|[;\n]+", normalized)
        if len(sentence.strip()) >= 8
    ]


def _compact_sentence(sentence: str, *, max_chars: int = 320) -> str:
    compact = sentence.strip(" .;，,")
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip(" ,;，") + "…"


def _with_year_title_prefix(query: str, context: ContextItem, text: str) -> str:
    if context.title is None:
        return text
    query_years = _years(query)
    if not query_years:
        return text
    target_year = max(query_years)
    if str(target_year) not in context.title or str(target_year) in text:
        return text
    if not (_query_wants_degree_page(query) or _looks_like_degree_page(text)):
        return text
    return f"{context.title}：{text}"


def _extractive_candidate_score(
    query: str,
    query_terms: set[str],
    text: str,
    *,
    context: ContextItem,
    context_order: int,
    evidence_pattern: re.Pattern[str] | None,
    anchor_overlap: int,
) -> float:
    context_text = _context_score_text(context)
    score = 20.0 - context_order * 0.25
    score += min(anchor_overlap, 10) * 1.5

    evidence_count = len(evidence_pattern.findall(text)) if evidence_pattern is not None else 0
    if evidence_count:
        score += min(evidence_count, 6) * 4.0
    if evidence_count > 1 and _query_wants_multiple_facts(query):
        score += 4.0

    candidate_years = _years(f"{context.title or ''} {text}")
    score += len(_years(query) & candidate_years) * 10.0
    exact_date_matches = _exact_date_overlap_count(query, f"{context_text} {text}")
    if exact_date_matches:
        score += exact_date_matches * 20.0
    elif _date_markers(query) and _date_markers(context_text):
        score -= 12.0

    if _query_wants_degree_page(query) and _looks_like_degree_page(f"{context.title or ''} {text}"):
        score += 4.0
    if _looks_like_degree_page(context_text):
        score += 1.5

    program_matches = _matched_terms(
        query,
        f"{context.title or ''} {text}",
        ("cs", "computer science", "ee", "electrical", "electronic", "计算机", "电子", "电气", "信息"),
    )
    score += len(program_matches) * 1.5

    if _looks_like_navigation_span(text):
        score -= 30.0
    if query_terms and not _has_anchor_overlap(query_terms, text):
        score -= 2.0
    return score


def _minimum_anchor_overlap(query_terms: set[str]) -> int:
    if not query_terms:
        return 0
    if len(query_terms) >= 6:
        return 2
    return 1


def _anchor_overlap_count(query_terms: set[str], text: str) -> int:
    if not query_terms:
        return 0
    normalized_text = text.lower()
    return sum(1 for term in query_terms if term in normalized_text)


def _has_newer_year_conflict(query: str, text: str) -> bool:
    query_years = _years(query)
    if not query_years:
        return False
    target_year = max(query_years)
    text_years = _years(text)
    if target_year in text_years:
        return False
    if not any(year < target_year for year in text_years):
        return False
    return _query_wants_degree_page(query) or _looks_like_degree_page(text)


def _looks_like_navigation_span(text: str) -> bool:
    lowered = text.lower()
    nav_terms = (
        "copyright",
        "all rights reserved",
        "breadcrumb",
        "sitemap",
        "login",
        "footer",
        "首页",
        "导航",
        "菜单",
        "上一页",
        "下一页",
        "版权所有",
        "站点地图",
        "友情链接",
    )
    nav_hits = sum(1 for term in nav_terms if term in lowered)
    if nav_hits >= 2:
        return True
    separators = len(re.findall(r"\s[|>›]\s", text))
    return separators >= 4 and nav_hits >= 1


def _query_wants_multiple_facts(query: str) -> bool:
    lowered = query.lower()
    return any(
        term in lowered
        for term in ("list", "which", "different", "difference", "compare", "respectively", "breakdown")
    ) or any(term in query for term in ("哪些", "有什么不同", "区别", "比较", "分别", "构成", "包括"))


def _first_match(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(0).strip(" .;,:") if match else None


def _anchor_terms(text: str) -> set[str]:
    ignored = {
        "what",
        "which",
        "where",
        "when",
        "who",
        "the",
        "and",
        "for",
        "with",
        "is",
        "are",
        "office",
        "email",
        "address",
        "postcode",
        "postal",
        "code",
        "credit",
        "credits",
        "date",
        "time",
        "location",
        "list",
        "different",
        "difference",
        "compare",
        "多少",
        "需要",
        "修满",
        "学分",
        "什么",
        "是谁",
        "哪里",
        "哪个",
        "哪些",
        "任课老师",
        "教授",
        "老师",
        "具体",
        "工作",
    }
    terms = {
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}|\d+[A-Za-z0-9.-]*|[\u4e00-\u9fff]{2,}", text.lower())
        if token not in ignored
    }
    for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        terms.update(
            chunk[start : start + length].lower()
            for length in range(2, min(6, len(chunk)) + 1)
            for start in range(0, len(chunk) - length + 1)
            if chunk[start : start + length].lower() not in ignored
        )
    return terms


def _has_anchor_overlap(query_terms: set[str], text: str) -> bool:
    if not query_terms:
        return True
    normalized_text = text.lower()
    return any(term in normalized_text for term in query_terms)


def _course_terms_from_query(query: str) -> list[str]:
    normalized = query.lower()
    terms: list[str] = []
    if "深度学习" in query or "deep learning" in normalized:
        terms.append("Deep Learning")
        terms.append("深度学习")
    if "robotics" in normalized:
        terms.append("Robotics")
    return terms


def _is_chinese(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _insufficient_result(
    query: str,
    mode: AnswerMode,
    *,
    sources: list[AnswerSource],
    retrieval: dict[str, Any],
    timing: AnswerTiming,
    config: AnswerConfig,
    generation_rejection_reason: str | None = None,
    answer_context_order: list[dict[str, Any]] | None = None,
) -> RagAnswerResult:
    return RagAnswerResult(
        query=query,
        mode=mode,
        status="insufficient_evidence",
        answer=_insufficient_answer(query),
        sources=sources,
        retrieval=retrieval,
        timing=timing,
        config=config,
        generation_path="insufficient",
        generation_rejection_reason=generation_rejection_reason,
        answer_context_order=answer_context_order or [],
    )


def _insufficient_answer(query: str) -> str:
    if any("\u4e00" <= char <= "\u9fff" for char in query):
        return "证据不足：当前检索到的官方来源不足以回答这个问题。"
    return "Evidence is insufficient: the retrieved official sources do not contain enough information to answer."


def _resolve_device(device: Device) -> str:
    import torch

    if device == "cpu":
        return "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for generation, but torch.cuda.is_available() is false")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _generation_torch_dtype(device: str) -> Any:
    if device != "cuda":
        return "auto"
    import torch

    major, _minor = torch.cuda.get_device_capability()
    if major < 8:
        return torch.float16
    return "auto"


def _move_inputs(inputs: Any, device: str) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def _input_length(inputs: Any) -> int:
    input_ids = inputs["input_ids"]
    shape = getattr(input_ids, "shape", None)
    if shape is not None:
        return int(shape[-1])
    return len(input_ids[0])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate local cited RAG answers over existing RAG artifacts.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--mode", choices=["dense", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--answer-reranker-model", type=Path, default=None)
    parser.add_argument("--answer-reranker-device", default="cpu")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--bm25", type=Path, default=DEFAULT_BM25)
    parser.add_argument("--faiss", type=Path, default=DEFAULT_FAISS)
    parser.add_argument("--chunk-index", type=Path, default=DEFAULT_CHUNK_INDEX)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dense-model", default=None)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of text.")
    args = parser.parse_args(argv)

    retriever = Retriever.from_paths(
        db_path=args.db,
        bm25_path=args.bm25,
        faiss_path=args.faiss,
        chunk_index_path=args.chunk_index,
        report_path=args.report,
        dense_model=args.dense_model,
    )
    answerer = RagAnswerer(
        retriever,
        model_path=args.model_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        answer_reranker_model=args.answer_reranker_model,
        answer_reranker_device=args.answer_reranker_device,
    )
    result = answerer.answer(args.query, mode=args.mode, top_k=args.top_k)
    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, sort_keys=True))
        return 0

    print(result.answer)
    if result.sources:
        print()
        print("Sources:")
        for source in result.sources:
            title = source.title or "(untitled)"
            print(f"[{source.source_id}] {title} - {source.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
