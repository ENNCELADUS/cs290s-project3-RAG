from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .answer_context import (
    _context_score_text,
    _contexts_from_retrieval,
    _metadata_answer_context_score,
    _ordered_contexts,
    _retrieval_payload,
    _select_local_evidence_contexts,
    _sources_from_contexts,
)
from .answer_prompts import _render_prompt, build_messages, build_prompt, build_repair_messages
from .answer_recovery import (
    _answer_rejection_reason,
    _extract_answer_from_contexts,
    _insufficient_result,
    _is_acceptable_answer,
    _parse_repair_answer,
)
from .answer_runtime import _generation_torch_dtype, _input_length, _move_inputs, _resolve_device
from .answer_types import (
    AnswerConfig,
    AnswerMode,
    AnswerSource,
    AnswerStatus,
    AnswerTiming,
    Device,
    GenerationPath,
    RagAnswerResult,
)
from .index import DEFAULT_BM25, DEFAULT_CHUNK_INDEX, DEFAULT_DB, DEFAULT_FAISS, DEFAULT_REPORT
from .retrieve import ContextItem, Retriever

DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_K = 5

__all__ = [
    "AnswerConfig",
    "AnswerMode",
    "AnswerSource",
    "AnswerStatus",
    "AnswerTiming",
    "Device",
    "GenerationPath",
    "RagAnswerResult",
    "RagAnswerer",
    "build_messages",
    "build_prompt",
    "build_repair_messages",
    "main",
]


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
