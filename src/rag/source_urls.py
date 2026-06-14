from __future__ import annotations

import re
from urllib.parse import unquote, urlsplit, urlunsplit

SIST_HOST = "sist.shanghaitech.edu.cn"
_SIST_TEMPLATE_SEGMENT = re.compile(r"_t\d+")
_SIST_ARTICLE_SEGMENT = re.compile(r"^c\d+a(?P<article_id>\d+)$")
_SIST_PROFILE_SLUG = re.compile(r"[a-z][a-z0-9_]*")
_SIST_PROFILE_PAGE_NAMES = {"main.htm", "list.htm"}


def normalize_url(url: str) -> str:
    parsed = urlsplit(unquote(url.strip()))
    scheme = "https" if parsed.scheme in {"http", "https"} else parsed.scheme
    netloc = parsed.netloc.lower()
    path = _canonical_path(parsed.path, netloc)
    return urlunsplit((scheme, netloc, path, "", ""))


def sist_article_id(url: str) -> str | None:
    parsed = urlsplit(normalize_url(url))
    if parsed.netloc != SIST_HOST:
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2 or parts[-1] != "page.htm":
        return None
    match = _SIST_ARTICLE_SEGMENT.fullmatch(parts[-2])
    if match is None:
        return None
    return match.group("article_id")


def sist_profile_slug(url: str) -> str | None:
    parsed = urlsplit(normalize_url(url))
    if parsed.netloc != SIST_HOST:
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) == 1:
        slug = parts[0]
    elif len(parts) == 2 and parts[1] in _SIST_PROFILE_PAGE_NAMES:
        slug = parts[0]
    else:
        return None
    return slug if _SIST_PROFILE_SLUG.fullmatch(slug) else None


def _canonical_path(path: str, netloc: str) -> str:
    canonical = path.rstrip("/") or "/"
    if netloc != SIST_HOST:
        return canonical
    parts = [part for part in canonical.split("/") if part and not _SIST_TEMPLATE_SEGMENT.fullmatch(part)]
    return "/" + "/".join(parts) if parts else "/"
