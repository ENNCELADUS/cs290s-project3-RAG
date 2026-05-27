from __future__ import annotations

import hashlib
import http.client
import mimetypes
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self

from .chunking import normalize_text
from .html import extract_html
from .io import write_jsonl, write_manifest
from .office import OFFICE_CONTENT_TYPES, OFFICE_EXTENSIONS, extract_office_text
from .pdf import extract_pdf_text
from .quality import quality_flags, write_quality_report
from .seeds import load_seed_urls
from .structured import extract_structured_records
from .urls import SeedUrl, canonicalize_url, infer_category, infer_language, is_official_url, same_or_subdomain

USER_AGENT = "cs290s-rag-collector/0.1 (+official-source student project)"
CHARSET_RE = re.compile(r"charset=([A-Za-z0-9._-]+)", re.I)
SUPPORTED_TEXT_EXTENSIONS = {".html", ".htm", ".psp", ".txt", ".text", ".csv", ".md"}
UNSUPPORTED_BINARY_EXTENSIONS = {
    ".7z",
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".rar",
    ".svg",
    ".webp",
    ".zip",
}
UNSUPPORTED_BINARY_CONTENT_PREFIXES = ("image/", "audio/", "video/")


class ProgressReporter(Protocol):
    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    def update(self, amount: int) -> None: ...


ProgressFactory = Callable[[int], ProgressReporter]


@dataclass(frozen=True)
class CollectorConfig:
    seeds_path: Path
    run_dir: Path
    max_pages: int = 50
    request_delay_seconds: float = 1.0
    timeout_seconds: float = 20.0
    retries: int = 1
    dry_run: bool = False
    respect_robots: bool = True
    chunk_chars: int = 1200
    chunk_overlap: int = 120
    known_urls: frozenset[str] = frozenset()
    same_host_only: bool = False
    progress_factory: ProgressFactory | None = None


class OfficialCollector:
    def __init__(self, config: CollectorConfig):
        self.config = config
        self.seeds = load_seed_urls(config.seeds_path)
        self.robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}
        self.last_request_at: dict[str, float] = {}
        self.failures: list[str] = []

    def run(self) -> dict[str, int]:
        documents: list[dict[str, object]] = []
        chunks: list[dict[str, object]] = []
        manifest_rows: list[dict[str, object]] = []
        structured: dict[str, list[dict[str, object]]] = {
            "courses": [],
            "faculty_members": [],
            "program_requirements": [],
            "events": [],
        }

        if self.config.dry_run:
            manifest_rows = [self._dry_manifest_row(seed) for seed in self.seeds]
            self._write_outputs(documents, chunks, manifest_rows, structured)
            return {"documents": 0, "chunks": 0, "manifest_rows": len(manifest_rows)}

        queue: deque[tuple[SeedUrl, str, int, str | None]] = deque((seed, seed.url, 0, None) for seed in self.seeds)
        seen: set[str] = set()
        next_document_id = 1

        with self._progress() as progress:
            while queue and len(documents) < self.config.max_pages:
                seed, url, depth, parent_url = queue.popleft()
                canonical_url = canonicalize_url(url)
                if canonical_url in seen or not is_official_url(canonical_url):
                    continue
                seen.add(canonical_url)

                if self.config.respect_robots and not self._allowed_by_robots(canonical_url):
                    self.failures.append(f"robots disallowed: {canonical_url}")
                    continue

                try:
                    fetched = self._fetch(canonical_url)
                except RuntimeError as error:
                    self.failures.append(f"{canonical_url}: {error}")
                    continue

                parsed = self._parse_response(fetched, seed.category)
                if canonical_url in self.config.known_urls:
                    if depth < seed.depth_limit:
                        self._enqueue_links(queue, seed, parsed["links"], depth, canonical_url, seen)
                    continue

                document_id = next_document_id
                next_document_id += 1

                raw_path = self._write_raw(fetched["body"], fetched["sha256"], parsed["extension"])
                text_path = self._write_text(parsed["text"], fetched["sha256"])

                document = {
                    "id": document_id,
                    "run_id": self.config.run_dir.name,
                    "url": canonical_url,
                    "canonical_url": canonical_url,
                    "title": parsed["title"],
                    "host": urllib.parse.urlsplit(canonical_url).netloc.lower(),
                    "category": parsed["category"],
                    "language": parsed["language"],
                    "content_type": fetched["content_type"],
                    "status_code": fetched["status_code"],
                    "fetched_at": fetched["fetched_at"],
                    "source_published_at": None,
                    "valid_from": fetched["fetched_at"][:10],
                    "valid_until": None,
                    "validity_note": "fetched_at_no_explicit_validity",
                    "raw_path": _relative_to_run(raw_path, self.config.run_dir),
                    "text_path": _relative_to_run(text_path, self.config.run_dir),
                    "sha256": fetched["sha256"],
                    "depth": depth,
                    "parent_url": parent_url,
                    "text_chars": len(parsed["text"]),
                    "parser": parsed["parser"],
                    "ocr_used": parsed["ocr_used"],
                }
                flags = quality_flags(parsed["text"], fetched["status_code"], parsed["parser"])
                flags.extend(parsed["flags"])

                manifest_rows.append(
                    {
                        "url": canonical_url,
                        "canonical_url": canonical_url,
                        "title": parsed["title"] or "",
                        "host": document["host"],
                        "category": parsed["category"],
                        "language": parsed["language"],
                        "content_type": fetched["content_type"],
                        "status_code": fetched["status_code"],
                        "fetched_at": fetched["fetched_at"],
                        "raw_path": document["raw_path"],
                        "text_path": document["text_path"],
                        "sha256": fetched["sha256"],
                        "depth": depth,
                        "parent_url": parent_url or "",
                        "parser": parsed["parser"],
                        "ocr_used": parsed["ocr_used"],
                        "text_chars": len(parsed["text"]),
                        "quality_flags": flags,
                    }
                )
                documents.append(document)
                progress.update(1)

                chunks.extend(
                    {
                        "id": len(chunks) + 1,
                        **chunk,
                    }
                    for chunk in _chunk_document(
                        document,
                        parsed["text"],
                        self.config.chunk_chars,
                        self.config.chunk_overlap,
                    )
                )

                extracted = extract_structured_records(document, parsed["text"])
                for name, rows in extracted.items():
                    structured[name].extend(rows)

                if depth < seed.depth_limit:
                    self._enqueue_links(queue, seed, parsed["links"], depth, canonical_url, seen)

        self._write_outputs(documents, chunks, manifest_rows, structured)
        return {"documents": len(documents), "chunks": len(chunks), "manifest_rows": len(manifest_rows)}

    def _write_outputs(
        self,
        documents: list[dict[str, object]],
        chunks: list[dict[str, object]],
        manifest_rows: list[dict[str, object]],
        structured: dict[str, list[dict[str, object]]],
    ) -> None:
        jsonl_dir = self.config.run_dir / "jsonl"
        write_jsonl(jsonl_dir / "documents.jsonl", documents)
        write_jsonl(jsonl_dir / "chunks.jsonl", chunks)
        structured_counts: dict[str, int] = {}
        for name, rows in structured.items():
            structured_counts[f"{name}.jsonl"] = write_jsonl(jsonl_dir / f"{name}.jsonl", rows)
        write_jsonl(self.config.run_dir / "eval_seed_candidates.jsonl", build_eval_seed_candidates(documents))
        write_manifest(self.config.run_dir / "source_manifest.csv", manifest_rows)
        write_quality_report(self.config.run_dir, documents, manifest_rows, structured_counts, self.failures)

    def _progress(self) -> ProgressReporter:
        if self.config.progress_factory is None:
            return _NoopProgress()
        return self.config.progress_factory(self.config.max_pages)

    def _enqueue_links(
        self,
        queue: deque[tuple[SeedUrl, str, int, str | None]],
        seed: SeedUrl,
        links: list[str],
        depth: int,
        parent_url: str,
        seen: set[str],
    ) -> None:
        for link in links:
            link_url = canonicalize_url(link)
            seed_host = urllib.parse.urlsplit(canonicalize_url(seed.url)).netloc.lower()
            if self.config.same_host_only and not same_or_subdomain(link_url, seed_host):
                continue
            if link_url not in seen and is_official_url(link_url):
                queue.append((seed, link_url, depth + 1, parent_url))

    def _dry_manifest_row(self, seed: SeedUrl) -> dict[str, object]:
        canonical_url = canonicalize_url(seed.url)
        return {
            "url": canonical_url,
            "canonical_url": canonical_url,
            "title": "",
            "host": urllib.parse.urlsplit(canonical_url).netloc.lower(),
            "category": seed.category,
            "language": "unknown",
            "content_type": "",
            "status_code": "",
            "fetched_at": "",
            "raw_path": "",
            "text_path": "",
            "sha256": "",
            "depth": 0,
            "parent_url": "",
            "parser": "dry_run",
            "ocr_used": False,
            "text_chars": 0,
            "quality_flags": ["dry_run"],
        }

    def _fetch(self, url: str) -> dict[str, object]:
        host = urllib.parse.urlsplit(url).netloc.lower()
        elapsed = time.monotonic() - self.last_request_at.get(host, 0)
        if elapsed < self.config.request_delay_seconds:
            time.sleep(self.config.request_delay_seconds - elapsed)

        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        last_error: Exception | None = None
        for _ in range(self.config.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    body = response.read()
                    raw_content_type = response.headers.get("Content-Type", "")
                    self.last_request_at[host] = time.monotonic()
                    return {
                        "url": url,
                        "body": body,
                        "status_code": getattr(response, "status", 200),
                        "content_type": raw_content_type.split(";")[0].lower(),
                        "encoding": _encoding_from_content_type(raw_content_type),
                        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
                        "sha256": hashlib.sha256(body).hexdigest(),
                    }
            except (
                urllib.error.URLError,
                TimeoutError,
                OSError,
                http.client.InvalidURL,
                http.client.RemoteDisconnected,
            ) as error:
                last_error = error
                time.sleep(0.5)
        raise RuntimeError(str(last_error))

    def _parse_response(self, fetched: dict[str, object], seed_category: str) -> dict[str, object]:
        body = fetched["body"]
        assert isinstance(body, bytes)
        content_type = str(fetched["content_type"])
        sha256 = str(fetched["sha256"])
        extension = _extension_for_response(str(fetched["url"]), content_type)

        if content_type == "application/pdf" or extension == ".pdf":
            temp_pdf = self.config.run_dir / "raw" / f"{sha256}.pdf.tmp"
            temp_pdf.write_bytes(body)
            result = extract_pdf_text(temp_pdf)
            temp_pdf.unlink(missing_ok=True)
            text = result.text
            return {
                "title": None,
                "text": text,
                "links": [],
                "category": infer_category(str(fetched["url"]), seed_category=seed_category),
                "language": infer_language(text),
                "parser": result.parser,
                "ocr_used": result.ocr_used,
                "flags": result.flags,
                "extension": ".pdf",
            }

        if content_type in OFFICE_CONTENT_TYPES or extension in OFFICE_EXTENSIONS:
            temp_path = self.config.run_dir / "raw" / f"{sha256}{extension}.tmp"
            result = extract_office_text(body, extension, temp_path)
            text = result.text
            return {
                "title": None,
                "text": text,
                "links": [],
                "category": infer_category(str(fetched["url"]), seed_category=seed_category),
                "language": infer_language(text),
                "parser": result.parser,
                "ocr_used": False,
                "flags": result.flags,
                "extension": extension,
            }

        if content_type in {"text/html", "application/xhtml+xml"} or extension in {".html", ".htm"}:
            encoding = fetched.get("encoding")
            title, text, links = extract_html(
                body,
                base_url=str(fetched["url"]),
                encoding=encoding if isinstance(encoding, str) else None,
            )
            return {
                "title": title,
                "text": text,
                "links": links,
                "category": infer_category(str(fetched["url"]), title, seed_category),
                "language": infer_language(text),
                "parser": "html",
                "ocr_used": False,
                "flags": [],
                "extension": ".html",
            }

        if _is_unsupported_binary_response(content_type, extension):
            return {
                "title": None,
                "text": "",
                "links": [],
                "category": infer_category(str(fetched["url"]), seed_category=seed_category),
                "language": "unknown",
                "parser": "unsupported_binary",
                "ocr_used": False,
                "flags": ["unsupported_binary"],
                "extension": extension or ".bin",
            }

        text = normalize_text(body.decode("utf-8", errors="replace"))
        return {
            "title": None,
            "text": text,
            "links": [],
            "category": infer_category(str(fetched["url"]), seed_category=seed_category),
            "language": infer_language(text),
            "parser": "text",
            "ocr_used": False,
            "flags": [],
            "extension": extension or ".txt",
        }

    def _write_raw(self, body: bytes, sha256: str, extension: str) -> Path:
        path = self.config.run_dir / "raw" / f"{sha256}{extension}"
        path.write_bytes(body)
        return path

    def _write_text(self, text: str, sha256: str) -> Path:
        path = self.config.run_dir / "texts" / f"{sha256}.txt"
        path.write_text(text, encoding="utf-8")
        return path

    def _allowed_by_robots(self, url: str) -> bool:
        parts = urllib.parse.urlsplit(url)
        root = f"{parts.scheme}://{parts.netloc}"
        parser = self.robots_cache.get(root)
        if parser is None:
            parser = urllib.robotparser.RobotFileParser()
            parser.set_url(f"{root}/robots.txt")
            try:
                parser.read()
            except Exception:
                return True
            self.robots_cache[root] = parser
        return parser.can_fetch(USER_AGENT, url)


class _NoopProgress:
    def __enter__(self) -> _NoopProgress:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def update(self, amount: int) -> None:
        return None


def build_eval_seed_candidates(documents: list[dict[str, object]]) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    prompts = {
        "school_info": "University overview factual question",
        "program": "Schools, institutes, or program comparison question",
        "program_requirements": "Graduation credit or degree requirement question",
        "courses": "Course instructor, credit, or curriculum question",
        "faculty": "Faculty background or research direction question",
        "research": "Research center, lab, or direction question",
        "admission": "Admissions policy or deadline question",
        "career": "Internship, career, or job posting question",
        "news": "Recent news factual question",
        "events": "Latest notice, lecture, or time-sensitive question",
    }
    seen_categories: set[str] = set()
    for document in documents:
        category = str(document.get("category") or "general")
        if category in seen_categories or category not in prompts:
            continue
        seen_categories.add(category)
        candidates.append(
            {
                "category": category,
                "question_theme": prompts[category],
                "source_url": document.get("url"),
                "source_title": document.get("title"),
                "evidence_hint": f"Use `{document.get('title') or document.get('url')}` as ground-truth source.",
            }
        )
    return candidates


def _chunk_document(document: dict[str, object], text: str, max_chars: int, overlap: int) -> list[dict[str, object]]:
    from .chunking import iter_chunk_records

    return list(
        iter_chunk_records(
            document_id=int(document["id"]),
            title=document.get("title") if isinstance(document.get("title"), str) else None,
            url=str(document["url"]),
            category=str(document.get("category") or "general"),
            language=str(document.get("language") or "unknown"),
            text=text,
            max_chars=max_chars,
            overlap=overlap,
        )
    )


def _extension_for_response(url: str, content_type: str) -> str:
    path_suffix = Path(urllib.parse.urlsplit(url).path).suffix.lower()
    if path_suffix in {
        ".pdf",
        *SUPPORTED_TEXT_EXTENSIONS,
        *OFFICE_EXTENSIONS,
        *UNSUPPORTED_BINARY_EXTENSIONS,
    }:
        return path_suffix
    if content_type in OFFICE_CONTENT_TYPES:
        return OFFICE_CONTENT_TYPES[content_type]
    if content_type == "application/pdf":
        return ".pdf"
    if content_type in {"text/html", "application/xhtml+xml"}:
        return ".html"
    guessed = mimetypes.guess_extension(content_type)
    return guessed or ".bin"


def _is_unsupported_binary_response(content_type: str, extension: str) -> bool:
    return extension in UNSUPPORTED_BINARY_EXTENSIONS or content_type.startswith(UNSUPPORTED_BINARY_CONTENT_PREFIXES)


def _encoding_from_content_type(content_type: str) -> str | None:
    match = CHARSET_RE.search(content_type)
    return match.group(1) if match else None


def _relative_to_run(path: Path, run_dir: Path) -> str:
    return path.relative_to(run_dir).as_posix()
