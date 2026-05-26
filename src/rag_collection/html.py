from __future__ import annotations

import re
from html.parser import HTMLParser
from urllib.parse import urljoin

from .chunking import normalize_text


class TextAndLinkParser(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[str] = []
        self._tag_stack: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        self._tag_stack.append(tag)
        if tag == "title":
            self._in_title = True
        if tag in {"p", "div", "li", "tr", "h1", "h2", "h3", "h4", "br"}:
            self.text_parts.append("\n")
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(urljoin(self.base_url, href))

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
        if tag in {"p", "div", "li", "tr", "h1", "h2", "h3", "h4"}:
            self.text_parts.append("\n")
        if self._tag_stack:
            self._tag_stack.pop()

    def handle_data(self, data: str) -> None:
        if self._is_ignored_context():
            return
        clean = data.strip()
        if not clean:
            return
        if self._in_title:
            self.title_parts.append(clean)
        self.text_parts.append(clean)

    def _is_ignored_context(self) -> bool:
        return any(tag in {"script", "style", "noscript", "svg"} for tag in self._tag_stack)


CHARSET_RE = re.compile(rb"""charset=["']?\s*([A-Za-z0-9._-]+)""", re.I)


def extract_html(html_bytes: bytes, base_url: str, encoding: str | None = None) -> tuple[str | None, str, list[str]]:
    text = html_bytes.decode(_detect_encoding(html_bytes, encoding), errors="replace")
    parser = TextAndLinkParser(base_url)
    parser.feed(text)
    title = normalize_text(" ".join(parser.title_parts)) or None
    body = normalize_text("\n".join(parser.text_parts))
    return title, body, parser.links


def _detect_encoding(html_bytes: bytes, encoding: str | None) -> str:
    declared = encoding or _declared_charset(html_bytes)
    if not declared:
        return "utf-8"
    declared = declared.lower()
    if declared in {"gb2312", "gbk", "gb18030"}:
        return "gb18030"
    return declared


def _declared_charset(html_bytes: bytes) -> str | None:
    match = CHARSET_RE.search(html_bytes[:4096])
    if not match:
        return None
    try:
        return match.group(1).decode("ascii")
    except UnicodeDecodeError:
        return None
