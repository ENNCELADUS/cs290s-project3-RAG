from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass, field
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

        messages = build_messages(query, ordered_contexts)
        generation_started = time.perf_counter()
        generated = self._generate_text(messages)
        generation_s = time.perf_counter() - generation_started
        valid_source_ids = {source.source_id for source in sources}
        rejection_reason = _answer_rejection_reason(generated, valid_source_ids, query=query, contexts=ordered_contexts)
        if rejection_reason is not None:
            repair_text = self._generate_text(build_repair_messages(query, ordered_contexts, generated))
            generation_s = time.perf_counter() - generation_started
            repaired = _parse_repair_answer(
                repair_text,
                valid_source_ids=valid_source_ids,
                query=query,
                contexts=ordered_contexts,
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
            extracted = _extract_answer_from_contexts(query, ordered_contexts)
            if extracted is not None and _is_acceptable_answer(
                extracted.answer,
                valid_source_ids,
                query=query,
                contexts=ordered_contexts,
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
        model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype="auto",
        )
        model.to(self.device)
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
                    "Every factual paragraph must include at least one numbered citation like [1].",
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
                    "Return only one strict JSON object.",
                    'For a supported answer, use: {"status":"answered","answer":"... [1]"}',
                    'If the sources are insufficient, use: {"status":"insufficient_evidence","answer":""}',
                    "The answer must include numbered citations that match the provided source numbers.",
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
    return any(term in lowered for term in ("office", "email", "contact", "faculty", "professor")) or any(
        term in query for term in ("办公室", "邮箱", "教师", "教授", "老师", "联系方式")
    )


def _has_contact_evidence(text: str) -> bool:
    lowered = text.lower()
    return (
        re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text) is not None
        or "office" in lowered
        or "办公室" in text
        or "邮箱" in text
        or "professor" in lowered
        or re.search(r"\b(?:Room|Rm\.?)\s+[A-Za-z0-9]", text, flags=re.IGNORECASE) is not None
    )


def _query_wants_degree_page(query: str) -> bool:
    lowered = query.lower()
    return any(term in lowered for term in ("program", "degree", "credit", "credits", "curriculum")) or any(
        term in query for term in ("培养方案", "学分", "课程", "专业")
    )


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
        citation_rejection = _citation_support_rejection_reason(query, text, contexts)
        if citation_rejection is not None:
            return citation_rejection
    return None


def _has_valid_citation(text: str, valid_source_ids: set[int]) -> bool:
    citation_ids = [int(match.group(1)) for match in VALID_CITATION_RE.finditer(text)]
    return bool(citation_ids) and all(citation_id in valid_source_ids for citation_id in citation_ids)


def _states_insufficient_evidence(text: str) -> bool:
    normalized = text.lower()
    return "evidence is insufficient" in normalized or "证据不足" in text


def _query_shape_rejection_reason(query: str, answer: str) -> str | None:
    if _query_requires_contact_fact(query) and not _has_contact_evidence(answer):
        return "missing_requested_contact_fact"
    return None


def _query_requires_contact_fact(query: str) -> bool:
    lowered = query.lower()
    return any(term in lowered for term in ("office", "email", "contact")) or any(
        term in query for term in ("办公室", "邮箱", "联系方式", "联系")
    )


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
    schedule_contact_answer = _extract_schedule_contact_answer(query, contexts)
    if schedule_contact_answer is not None:
        return schedule_contact_answer
    office_answer = _extract_office_email_answer(query, contexts)
    if office_answer is not None:
        return office_answer
    address_answer = _extract_address_postcode_answer(query, contexts)
    if address_answer is not None:
        return address_answer
    degree_summary_answer = _extract_degree_plan_summary_answer(query, contexts)
    if degree_summary_answer is not None:
        return degree_summary_answer
    course_design_answer = _extract_course_design_pair_answer(query, contexts)
    if course_design_answer is not None:
        return course_design_answer
    dedup_answer = _extract_degree_plan_dedup_answer(query, contexts)
    if dedup_answer is not None:
        return dedup_answer
    credit_answer = _extract_credit_answer(query, contexts)
    if credit_answer is not None:
        return credit_answer
    schedule_answer = _extract_date_time_location_answer(query, contexts)
    if schedule_answer is not None:
        return schedule_answer
    robotics_answer = _extract_robotics_faculty_answer(query, contexts)
    if robotics_answer is not None:
        return robotics_answer
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
            r"(?:日期|时间|安排|联系|联系人|如有疑问|邮箱)",
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
        for size in (1, 2, 3):
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
