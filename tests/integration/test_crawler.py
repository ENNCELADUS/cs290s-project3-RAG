from __future__ import annotations

import http.client
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

from rag_collection.crawler import CollectorConfig, OfficialCollector
from rag_collection.io import prepare_run_dir, read_jsonl


class FakeResponse:
    status = 200

    def __init__(self, body: bytes, content_type: str = "text/html; charset=utf-8"):
        self._body = body
        self.headers = {"Content-Type": content_type}

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


class RecordingProgress:
    def __init__(self) -> None:
        self.total: int | None = None
        self.updates: list[int] = []
        self.closed = False

    def __enter__(self) -> RecordingProgress:
        return self

    def __exit__(self, *args: object) -> None:
        self.closed = True

    def update(self, amount: int) -> None:
        self.updates.append(amount)


def test_collector_progress_updates_once_per_collected_document(tmp_path: Path, monkeypatch) -> None:
    seeds_path = tmp_path / "seeds.csv"
    seeds_path.write_text(
        "\n".join(
            [
                "url,category,depth_limit,priority,notes",
                "https://sist.shanghaitech.edu.cn/a.htm,program,0,1,",
                "https://sist.shanghaitech.edu.cn/b.htm,program,0,2,",
            ]
        ),
        encoding="utf-8",
    )
    pages = {
        "https://sist.shanghaitech.edu.cn/a.htm": b"<html><title>A</title><body>alpha page</body></html>",
        "https://sist.shanghaitech.edu.cn/b.htm": b"<html><title>B</title><body>beta page</body></html>",
    }

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
        assert timeout == 20.0
        return FakeResponse(pages[request.full_url])

    progress = RecordingProgress()

    def progress_factory(total: int) -> RecordingProgress:
        progress.total = total
        return progress

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    config = CollectorConfig(
        seeds_path=seeds_path,
        run_dir=prepare_run_dir(tmp_path / "runs", "progress"),
        max_pages=2,
        request_delay_seconds=0,
        respect_robots=False,
        progress_factory=progress_factory,
    )

    stats = OfficialCollector(config).run()

    assert stats["documents"] == 2
    assert progress.total == 2
    assert progress.updates == [1, 1]
    assert progress.closed


def test_collector_records_invalid_url_fetch_errors(tmp_path: Path, monkeypatch) -> None:
    seeds_path = tmp_path / "seeds.csv"
    seeds_path.write_text(
        "\n".join(
            [
                "url,category,depth_limit,priority,notes",
                "https://sist.shanghaitech.edu.cn/bad.htm,program,0,1,",
            ]
        ),
        encoding="utf-8",
    )

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
        raise http.client.InvalidURL("bad url")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    config = CollectorConfig(
        seeds_path=seeds_path,
        run_dir=prepare_run_dir(tmp_path / "runs", "invalid"),
        max_pages=1,
        request_delay_seconds=0,
        respect_robots=False,
    )

    stats = OfficialCollector(config).run()

    assert stats["documents"] == 0
    assert "bad url" in (config.run_dir / "quality_report.md").read_text(encoding="utf-8")


def test_collector_uses_html_charset_from_content_type(tmp_path: Path, monkeypatch) -> None:
    seeds_path = tmp_path / "seeds.csv"
    seeds_path.write_text(
        "\n".join(
            [
                "url,category,depth_limit,priority,notes",
                "https://sist.shanghaitech.edu.cn/gbk.htm,program,0,1,",
            ]
        ),
        encoding="utf-8",
    )
    body = "<html><title>学院新闻</title><body>信息科学与技术学院</body></html>".encode("gb18030")

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
        return FakeResponse(body, "text/html; charset=gb2312")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    config = CollectorConfig(
        seeds_path=seeds_path,
        run_dir=prepare_run_dir(tmp_path / "runs", "gbk"),
        max_pages=1,
        request_delay_seconds=0,
        respect_robots=False,
    )

    stats = OfficialCollector(config).run()

    documents = read_jsonl(config.run_dir / "jsonl" / "documents.jsonl")
    text = (config.run_dir / documents[0]["text_path"]).read_text(encoding="utf-8")
    manifest = (config.run_dir / "source_manifest.csv").read_text(encoding="utf-8")
    assert stats["documents"] == 1
    assert documents[0]["title"] == "学院新闻"
    assert "信息科学与技术学院" in text
    assert "replacement_chars" not in manifest


def test_collector_extracts_docx_text_without_garbled_quality_flags(tmp_path: Path, monkeypatch) -> None:
    seeds_path = tmp_path / "seeds.csv"
    seeds_path.write_text(
        "\n".join(
            [
                "url,category,depth_limit,priority,notes",
                "https://sist.shanghaitech.edu.cn/office/Academics/course.docx,courses,0,1,",
            ]
        ),
        encoding="utf-8",
    )
    body = _minimal_docx(
        [
            "CS101 Introduction to Computer Science",
            "Instructor: ShanghaiTech SIST",
        ]
    )

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
        return FakeResponse(
            body,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    config = CollectorConfig(
        seeds_path=seeds_path,
        run_dir=prepare_run_dir(tmp_path / "runs", "docx"),
        max_pages=1,
        request_delay_seconds=0,
        respect_robots=False,
    )

    stats = OfficialCollector(config).run()

    assert stats["documents"] == 1
    documents = read_jsonl(config.run_dir / "jsonl" / "documents.jsonl")
    manifest = (config.run_dir / "source_manifest.csv").read_text(encoding="utf-8")
    text = (config.run_dir / documents[0]["text_path"]).read_text(encoding="utf-8")
    assert "CS101 Introduction to Computer Science" in text
    assert documents[0]["parser"] == "docx"
    assert "replacement_chars" not in manifest
    assert "possibly_garbled" not in manifest


def test_collector_marks_image_binary_as_unsupported_without_garbled_text(tmp_path: Path, monkeypatch) -> None:
    seeds_path = tmp_path / "seeds.csv"
    seeds_path.write_text(
        "\n".join(
            [
                "url,category,depth_limit,priority,notes",
                "https://sist.shanghaitech.edu.cn/_upload/article/images/example.jpg,program,0,1,",
            ]
        ),
        encoding="utf-8",
    )

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
        return FakeResponse(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x02", "image/jpeg")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    config = CollectorConfig(
        seeds_path=seeds_path,
        run_dir=prepare_run_dir(tmp_path / "runs", "image"),
        max_pages=1,
        request_delay_seconds=0,
        respect_robots=False,
    )

    stats = OfficialCollector(config).run()

    assert stats["documents"] == 1
    documents = read_jsonl(config.run_dir / "jsonl" / "documents.jsonl")
    manifest = (config.run_dir / "source_manifest.csv").read_text(encoding="utf-8")
    text = (config.run_dir / documents[0]["text_path"]).read_text(encoding="utf-8")
    assert documents[0]["parser"] == "unsupported_binary"
    assert text == ""
    assert "unsupported_binary" in manifest
    assert "replacement_chars" not in manifest
    assert "possibly_garbled" not in manifest


def test_collector_can_keep_link_discovery_on_seed_host(tmp_path: Path, monkeypatch) -> None:
    seeds_path = tmp_path / "seeds.csv"
    seeds_path.write_text(
        "\n".join(
            [
                "url,category,depth_limit,priority,notes",
                "https://sist.shanghaitech.edu.cn/start.htm,events,1,1,",
            ]
        ),
        encoding="utf-8",
    )
    pages = {
        "https://sist.shanghaitech.edu.cn/start.htm": b"""
            <html><body>
              <a href="https://sist.shanghaitech.edu.cn/detail.htm">SIST detail</a>
              <a href="https://openinfo.shanghaitech.edu.cn/detail.htm">Open info detail</a>
            </body></html>
        """,
        "https://sist.shanghaitech.edu.cn/detail.htm": b"<html><body>SIST detail 2026-05-01</body></html>",
    }

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
        return FakeResponse(pages[request.full_url])

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    config = CollectorConfig(
        seeds_path=seeds_path,
        run_dir=prepare_run_dir(tmp_path / "runs", "same-host"),
        max_pages=5,
        request_delay_seconds=0,
        respect_robots=False,
        same_host_only=True,
    )

    stats = OfficialCollector(config).run()

    documents = read_jsonl(config.run_dir / "jsonl" / "documents.jsonl")
    assert stats["documents"] == 2
    assert [document["url"] for document in documents] == [
        "https://sist.shanghaitech.edu.cn/start.htm",
        "https://sist.shanghaitech.edu.cn/detail.htm",
    ]


def test_collector_can_limit_link_discovery_to_allowed_hosts(tmp_path: Path, monkeypatch) -> None:
    seeds_path = tmp_path / "seeds.csv"
    seeds_path.write_text(
        "\n".join(
            [
                "url,category,depth_limit,priority,notes",
                "https://sist.shanghaitech.edu.cn/start.htm,events,1,1,",
            ]
        ),
        encoding="utf-8",
    )
    pages = {
        "https://sist.shanghaitech.edu.cn/start.htm": b"""
            <html><body>
              <a href="https://sist.shanghaitech.edu.cn/detail.htm">SIST detail</a>
              <a href="https://faculty.sist.shanghaitech.edu.cn/profile.htm">Faculty detail</a>
              <a href="https://star-center.shanghaitech.edu.cn/research.htm">STAR detail</a>
              <a href="https://openinfo.shanghaitech.edu.cn/detail.htm">Open info</a>
              <a href="https://oaa.shanghaitech.edu.cn/detail.htm">OAA</a>
              <a href="https://jobs.shanghaitech.edu.cn/detail.htm">Jobs</a>
              <a href="https://career.shanghaitech.edu.cn/detail.htm">Career</a>
              <a href="https://yanzhao.shanghaitech.edu.cn/detail.htm">Yanzhao</a>
            </body></html>
        """,
        "https://sist.shanghaitech.edu.cn/detail.htm": b"<html><body>SIST detail 2026-05-01</body></html>",
        "https://faculty.sist.shanghaitech.edu.cn/profile.htm": b"<html><body>Faculty profile</body></html>",
        "https://star-center.shanghaitech.edu.cn/research.htm": b"<html><body>STAR research</body></html>",
    }

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
        return FakeResponse(pages[request.full_url])

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    config = CollectorConfig(
        seeds_path=seeds_path,
        run_dir=prepare_run_dir(tmp_path / "runs", "allowed-hosts"),
        max_pages=10,
        request_delay_seconds=0,
        respect_robots=False,
        allowed_hosts=frozenset(
            {
                "sist.shanghaitech.edu.cn",
                "faculty.sist.shanghaitech.edu.cn",
                "star-center.shanghaitech.edu.cn",
            }
        ),
    )

    stats = OfficialCollector(config).run()

    documents = read_jsonl(config.run_dir / "jsonl" / "documents.jsonl")
    assert stats["documents"] == 4
    assert [document["url"] for document in documents] == [
        "https://sist.shanghaitech.edu.cn/start.htm",
        "https://sist.shanghaitech.edu.cn/detail.htm",
        "https://faculty.sist.shanghaitech.edu.cn/profile.htm",
        "https://star-center.shanghaitech.edu.cn/research.htm",
    ]


def test_collector_expands_list_page_pagination(tmp_path: Path, monkeypatch) -> None:
    seeds_path = tmp_path / "seeds.csv"
    seeds_path.write_text(
        "\n".join(
            [
                "url,category,depth_limit,priority,notes",
                "https://sist.shanghaitech.edu.cn/zpxx/list.htm,career,1,1,",
            ]
        ),
        encoding="utf-8",
    )
    pages = {
        "https://sist.shanghaitech.edu.cn/zpxx/list.htm": "<html><body>页码 1/10</body></html>".encode(),
        **{
            f"https://sist.shanghaitech.edu.cn/zpxx/list{page}.htm": f"<html><body>page {page}</body></html>".encode()
            for page in range(2, 11)
        },
    }

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
        return FakeResponse(pages[request.full_url])

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    config = CollectorConfig(
        seeds_path=seeds_path,
        run_dir=prepare_run_dir(tmp_path / "runs", "pagination"),
        max_pages=20,
        request_delay_seconds=0,
        respect_robots=False,
        expand_list_pages=True,
    )

    stats = OfficialCollector(config).run()

    documents = read_jsonl(config.run_dir / "jsonl" / "documents.jsonl")
    assert stats["documents"] == 10
    assert [document["url"] for document in documents] == [
        "https://sist.shanghaitech.edu.cn/zpxx/list.htm",
        *[f"https://sist.shanghaitech.edu.cn/zpxx/list{page}.htm" for page in range(2, 11)],
    ]


def test_collector_does_not_expand_non_list_pages(tmp_path: Path, monkeypatch) -> None:
    seeds_path = tmp_path / "seeds.csv"
    seeds_path.write_text(
        "\n".join(
            [
                "url,category,depth_limit,priority,notes",
                "https://sist.shanghaitech.edu.cn/2026/0525/c5005a1122924/page.htm,news,1,1,",
            ]
        ),
        encoding="utf-8",
    )
    pages = {
        "https://sist.shanghaitech.edu.cn/2026/0525/c5005a1122924/page.htm": (
            b"<html><body>article text with misleading page count: page 1/10</body></html>"
        ),
    }

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
        return FakeResponse(pages[request.full_url])

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    config = CollectorConfig(
        seeds_path=seeds_path,
        run_dir=prepare_run_dir(tmp_path / "runs", "no-pagination"),
        max_pages=20,
        request_delay_seconds=0,
        respect_robots=False,
        expand_list_pages=True,
    )

    stats = OfficialCollector(config).run()

    documents = read_jsonl(config.run_dir / "jsonl" / "documents.jsonl")
    assert stats["documents"] == 1
    assert [document["url"] for document in documents] == [
        "https://sist.shanghaitech.edu.cn/2026/0525/c5005a1122924/page.htm"
    ]


def test_collector_skips_unindexable_and_malformed_discovered_links(tmp_path: Path, monkeypatch) -> None:
    seeds_path = tmp_path / "seeds.csv"
    seeds_path.write_text(
        "\n".join(
            [
                "url,category,depth_limit,priority,notes",
                "https://faculty.sist.shanghaitech.edu.cn/faculty/songfu/,faculty,1,1,",
            ]
        ),
        encoding="utf-8",
    )
    pages = {
        "https://faculty.sist.shanghaitech.edu.cn/faculty/songfu/": b"""
            <html><body>
              <a href="profile.htm">Profile</a>
              <a href="Projects/SCInfer/qmsInfer-master.zip">Code archive</a>
              <a href="Projects/ysecure-poster.jpg">Poster</a>
              <a href="Google Scholar: https:/scholar.google.com/citations?user=abc">Scholar</a>
            </body></html>
        """,
        "https://faculty.sist.shanghaitech.edu.cn/faculty/songfu/profile.htm": (
            b"<html><body>Song Fu professor profile</body></html>"
        ),
    }
    requested_urls: list[str] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
        requested_urls.append(request.full_url)
        return FakeResponse(pages[request.full_url])

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    config = CollectorConfig(
        seeds_path=seeds_path,
        run_dir=prepare_run_dir(tmp_path / "runs", "skip-bad-links"),
        max_pages=10,
        request_delay_seconds=0,
        respect_robots=False,
        allowed_hosts=frozenset({"faculty.sist.shanghaitech.edu.cn"}),
    )

    stats = OfficialCollector(config).run()

    documents = read_jsonl(config.run_dir / "jsonl" / "documents.jsonl")
    assert stats["documents"] == 2
    assert [document["url"] for document in documents] == [
        "https://faculty.sist.shanghaitech.edu.cn/faculty/songfu/",
        "https://faculty.sist.shanghaitech.edu.cn/faculty/songfu/profile.htm",
    ]
    assert requested_urls == [document["url"] for document in documents]


def test_quality_report_uses_seed_categories_as_run_coverage_scope(tmp_path: Path, monkeypatch) -> None:
    seeds_path = tmp_path / "seeds.csv"
    seeds_path.write_text(
        "\n".join(
            [
                "url,category,depth_limit,priority,notes",
                "https://sist.shanghaitech.edu.cn/szdwx/list.htm,faculty,0,1,",
            ]
        ),
        encoding="utf-8",
    )

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
        return FakeResponse(b"<html><body>Faculty listing</body></html>")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    config = CollectorConfig(
        seeds_path=seeds_path,
        run_dir=prepare_run_dir(tmp_path / "runs", "seed-coverage"),
        max_pages=1,
        request_delay_seconds=0,
        respect_robots=False,
    )

    OfficialCollector(config).run()

    report = (config.run_dir / "quality_report.md").read_text(encoding="utf-8")
    assert "| faculty | 1 | covered |" in report
    assert "school_info" not in report


def _minimal_docx(paragraphs: list[str]) -> bytes:
    xml = "".join(f"<w:p><w:r><w:t>{paragraph}</w:t></w:r></w:p>" for paragraph in paragraphs)
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{xml}</w:body>"
        "</w:document>"
    )
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", document)
    return buffer.getvalue()
