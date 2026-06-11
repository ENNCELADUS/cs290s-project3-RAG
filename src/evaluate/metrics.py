from __future__ import annotations

import math
from urllib.parse import unquote, urlsplit, urlunsplit


def normalize_url(url: str) -> str:
    parsed = urlsplit(unquote(url.strip()))
    scheme = "https" if parsed.scheme in {"http", "https"} else parsed.scheme
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((scheme, parsed.netloc.lower(), path, parsed.query, ""))


def source_matches(observed_url: str, expected_url: str) -> bool:
    observed = normalize_url(observed_url)
    expected = normalize_url(expected_url)
    return observed == expected or observed.startswith(f"{expected}/")


def source_metrics(
    observed_urls: list[str], expected_urls: list[str], *, k_values: tuple[int, ...] = (1, 5)
) -> dict[str, float]:
    expected = [url for url in expected_urls if url]
    metrics: dict[str, float] = {}
    for k in k_values:
        relevance = _binary_relevance(observed_urls[:k], expected)
        metrics[f"source_hit@{k}"] = 1.0 if any(relevance) else 0.0
        metrics[f"source_recall@{k}"] = _source_recall(observed_urls[:k], expected)
        metrics[f"mrr@{k}"] = _mrr(relevance)
        metrics[f"ndcg@{k}"] = _ndcg(relevance, relevant_count=len(expected), k=k)
        metrics[f"precision@{k}"] = sum(relevance) / k if k else 0.0
    return metrics


def _binary_relevance(observed_urls: list[str], expected_urls: list[str]) -> list[int]:
    matched_expected: set[int] = set()
    relevance: list[int] = []
    for observed in observed_urls:
        matched_index = next(
            (
                index
                for index, expected in enumerate(expected_urls)
                if index not in matched_expected and source_matches(observed, expected)
            ),
            None,
        )
        if matched_index is None:
            relevance.append(0)
            continue
        matched_expected.add(matched_index)
        relevance.append(1)
    return relevance


def _source_recall(observed_urls: list[str], expected_urls: list[str]) -> float:
    if not expected_urls:
        return 0.0
    matched = {
        index
        for index, expected in enumerate(expected_urls)
        if any(source_matches(observed, expected) for observed in observed_urls)
    }
    return len(matched) / len(expected_urls)


def _mrr(relevance: list[int]) -> float:
    for rank, is_relevant in enumerate(relevance, start=1):
        if is_relevant:
            return 1.0 / rank
    return 0.0


def _ndcg(relevance: list[int], *, relevant_count: int, k: int) -> float:
    if relevant_count == 0:
        return 0.0
    dcg = sum(value / math.log2(rank + 1) for rank, value in enumerate(relevance, start=1))
    ideal_count = min(relevant_count, k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal if ideal else 0.0
