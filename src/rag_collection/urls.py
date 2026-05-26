from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

TRACKING_PARAMS = {
    "utm_campaign",
    "utm_content",
    "utm_medium",
    "utm_source",
    "utm_term",
}


@dataclass(frozen=True)
class SeedUrl:
    url: str
    category: str
    depth_limit: int
    priority: int
    notes: str = ""


def canonicalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    scheme = (parts.scheme or "https").lower()
    host = parts.netloc.lower()
    path = quote(parts.path or "/", safe="/:@!$&'()*+,;=%")
    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in TRACKING_PARAMS
    ]
    query = urlencode(sorted(query_pairs), doseq=True)
    return urlunsplit((scheme, host, path, query, ""))


def is_official_url(url: str) -> bool:
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"}:
        return False
    host = parts.netloc.lower().split("@")[-1].split(":")[0]
    return host == "shanghaitech.edu.cn" or host.endswith(".shanghaitech.edu.cn")


def same_or_subdomain(url: str, base_host: str) -> bool:
    host = urlsplit(url).netloc.lower().split(":")[0]
    base_host = base_host.lower()
    return host == base_host or host.endswith(f".{base_host}")


def infer_category(url: str, title: str | None = None, seed_category: str | None = None) -> str:
    haystack = f"{url} {title or ''}".lower()
    if any(token in haystack for token in ("course", "bkjx", "yjsjx", "curriculum")):
        return "courses"
    if any(token in haystack for token in ("pyfa", "training", "培养方案", "degree")):
        return "program_requirements"
    if any(token in haystack for token in ("admission", "zs", "招生")):
        return "admission"
    if any(token in haystack for token in ("career", "sxjy", "job", "employment")):
        return "career"
    if any(token in haystack for token in ("research", "kxyj", "lab")):
        return "research"
    if any(token in haystack for token in ("news", "xw")):
        return "news"
    if any(token in haystack for token in ("event", "tzgg", "notice", "lecture")):
        return "events"
    if any(token in haystack for token in ("faculty", "szdw", "professor", "teacher")):
        return "faculty"
    if any(token in haystack for token in ("academics", "school", "college")):
        return "program"
    if seed_category:
        return seed_category
    return "general"


def infer_language(text: str, url: str = "") -> str:
    chinese_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    ascii_letters = sum(1 for char in text if char.isascii() and char.isalpha())
    if chinese_chars >= 20 and chinese_chars >= ascii_letters * 0.15:
        return "zh"
    if "/eng" in url.lower() or "/en" in url.lower() or ascii_letters >= 30:
        return "en"
    return "unknown"
