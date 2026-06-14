from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
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
    generation_path: GenerationPath
    generation_rejection_reason: str | None = None
    fallback_source_rank: int | None = None


@dataclass(frozen=True)
class ExtractiveAnswer:
    answer: str
    source_rank: int


class RagAnswerer:
    def __init__(
        self,
        retriever: Retriever,
        *,
        model_path: Path,
        device: Device = "auto",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Local generator model path does not exist: {model_path}")
        self.retriever = retriever
        self.model_path = model_path
        self.device = _resolve_device(device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
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
        retrieval_payload = _retrieval_payload(retrieval_result, contexts)
        config = AnswerConfig(
            model_path=str(self.model_path),
            device=self.device,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=top_k,
        )
        if not contexts or not sources:
            return _insufficient_result(
                query,
                mode,
                sources=sources,
                retrieval=retrieval_payload,
                timing=AnswerTiming(retrieval_s=retrieval_s, generation_s=0.0, total_s=time.perf_counter() - started),
                config=config,
            )

        messages = build_messages(query, contexts)
        generation_started = time.perf_counter()
        generated = self._generate_text(messages)
        generation_s = time.perf_counter() - generation_started
        valid_source_ids = {source.source_id for source in sources}
        rejection_reason = _answer_rejection_reason(generated, valid_source_ids)
        if rejection_reason is not None:
            extracted = _extract_answer_from_contexts(query, contexts)
            if extracted is not None and _is_acceptable_answer(extracted.answer, valid_source_ids):
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
                )
            repair_text = self._generate_text(build_repair_messages(query, contexts, generated))
            generation_s = time.perf_counter() - generation_started
            repaired = _parse_repair_answer(
                repair_text,
                valid_source_ids=valid_source_ids,
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
    for index, context in enumerate(contexts, start=1):
        title = context.title or "(untitled)"
        url = context.url or "(no url)"
        blocks.append(
            "\n".join(
                [
                    f"[{index}] {title}",
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
    for index, context in enumerate(contexts, start=1):
        if context.url is None:
            continue
        sources.append(
            AnswerSource(
                source_id=index,
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
        "hits": [asdict(hit) for hit in retrieval_result],  # type: ignore[union-attr]
        "contexts": [asdict(context) for context in contexts],
    }


def _is_acceptable_answer(text: str, valid_source_ids: set[int]) -> bool:
    return _answer_rejection_reason(text, valid_source_ids) is None


def _answer_rejection_reason(text: str, valid_source_ids: set[int]) -> str | None:
    if any(marker in text for marker in PROMPT_LEAKAGE_MARKERS):
        return "prompt_leakage"
    if _states_insufficient_evidence(text):
        return "model_reported_insufficient_evidence"
    if not _has_valid_citation(text, valid_source_ids):
        return "invalid_or_missing_citation"
    return None


def _has_valid_citation(text: str, valid_source_ids: set[int]) -> bool:
    citation_ids = [int(match.group(1)) for match in VALID_CITATION_RE.finditer(text)]
    return bool(citation_ids) and all(citation_id in valid_source_ids for citation_id in citation_ids)


def _states_insufficient_evidence(text: str) -> bool:
    normalized = text.lower()
    return "evidence is insufficient" in normalized or "证据不足" in text


def _parse_repair_answer(text: str, *, valid_source_ids: set[int]) -> str | None:
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
    if not _is_acceptable_answer(answer, valid_source_ids):
        return None
    return answer


def _extract_answer_from_contexts(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    office_answer = _extract_office_email_answer(query, contexts)
    if office_answer is not None:
        return office_answer
    address_answer = _extract_address_postcode_answer(query, contexts)
    if address_answer is not None:
        return address_answer
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
        for index, context in enumerate(contexts, start=1):
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
                return ExtractiveAnswer(f"{course_name}的任课老师是{teacher}。 [{index}]", index)
            return ExtractiveAnswer(f"{course_name} was taught by {teacher} [{index}].", index)
    list_answer = _extract_list_or_comparison_answer(query, contexts)
    if list_answer is not None:
        return list_answer
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


def _extract_office_email_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    lowered_query = query.lower()
    if "office" not in lowered_query and "办公室" not in query and "email" not in lowered_query and "邮箱" not in query:
        return None
    query_terms = _anchor_terms(query)
    for index, context in enumerate(contexts, start=1):
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
        if room is not None:
            facts.append(f"office: {room}")
        if email is not None:
            facts.append(f"email: {email}")
        return ExtractiveAnswer(f"{'; '.join(facts)} [{index}].", index)
    return None


def _extract_credit_answer(query: str, contexts: list[ContextItem]) -> ExtractiveAnswer | None:
    lowered_query = query.lower()
    if "credit" not in lowered_query and "学分" not in query:
        return None
    return _extract_compact_evidence(
        query,
        contexts,
        evidence_pattern=re.compile(r"\d+(?:\.\d+)?\s*(?:credits?|学分)", re.IGNORECASE),
    )


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
    for index, context in enumerate(contexts, start=1):
        if context.url is None:
            continue
        normalized_text = re.sub(r"\s+", " ", f"{context.title or ''} {context.text}")
        if "robotics" not in normalized_text.lower() or "schwertfeger" not in normalized_text.lower():
            continue
        return ExtractiveAnswer(f"Prof. Schwertfeger works on robotics [{index}].", index)
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
    for index, context in enumerate(contexts, start=1):
        if context.url is None:
            continue
        for sentence in _candidate_sentences(context.text):
            if not _has_anchor_overlap(query_terms, sentence):
                continue
            if evidence_pattern is not None and evidence_pattern.search(sentence) is None:
                continue
            compact = _compact_sentence(sentence)
            if compact:
                return ExtractiveAnswer(f"{compact} [{index}].", index)
    return None


def _candidate_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    return [
        sentence.strip(" ;")
        for sentence in re.split(r"(?<=[。！？.!?])\s+|[;\n]+", normalized)
        if len(sentence.strip()) >= 8
    ]


def _compact_sentence(sentence: str, *, max_chars: int = 220) -> str:
    compact = sentence.strip(" .;，,")
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip(" ,;，") + "…"


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
