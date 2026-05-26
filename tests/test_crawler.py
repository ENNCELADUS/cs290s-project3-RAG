from __future__ import annotations

import http.client
import urllib.request
from pathlib import Path

from rag_collection.crawler import CollectorConfig, OfficialCollector
from rag_collection.io import prepare_run_dir


class FakeResponse:
    status = 200

    def __init__(self, body: bytes):
        self._body = body
        self.headers = {"Content-Type": "text/html; charset=utf-8"}

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
