from __future__ import annotations

import re
from collections.abc import Iterator


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 120) -> list[str]:
    clean_text = normalize_text(text)
    if not clean_text:
        return []
    if len(clean_text) <= max_chars:
        return [clean_text]

    paragraphs = [paragraph.strip() for paragraph in clean_text.split("\n\n") if paragraph.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_split_long_text(paragraph, max_chars, overlap))
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())
        current = paragraph

    if current:
        chunks.append(current.strip())

    return _apply_overlap(chunks, overlap, max_chars)


def _split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    step = max(1, max_chars - overlap)
    return [text[start : start + max_chars].strip() for start in range(0, len(text), step)]


def _apply_overlap(chunks: list[str], overlap: int, max_chars: int) -> list[str]:
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]
    for previous, current in zip(chunks, chunks[1:], strict=False):
        prefix = previous[-overlap:].strip()
        candidate = f"{prefix}\n\n{current}".strip()
        overlapped.append(candidate[-max_chars:] if len(candidate) > max_chars else candidate)
    return overlapped


def iter_chunk_records(
    document_id: int,
    title: str | None,
    url: str,
    category: str | None,
    language: str,
    text: str,
    max_chars: int = 1200,
    overlap: int = 120,
) -> Iterator[dict[str, object]]:
    for index, chunk in enumerate(chunk_text(text, max_chars=max_chars, overlap=overlap)):
        yield {
            "document_id": document_id,
            "chunk_index": index,
            "title": title,
            "url": url,
            "category": category,
            "language": language,
            "text": chunk,
            "char_count": len(chunk),
        }
