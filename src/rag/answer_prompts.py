from __future__ import annotations

from typing import Any

from .retrieve import ContextItem


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
