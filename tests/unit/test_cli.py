from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from rag_collection import cli
from rag_collection.io import write_jsonl


def test_doctor_command_prints_dependency_warnings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    seeds_path = tmp_path / "seeds.csv"
    monkeypatch.setattr(cli, "load_seed_urls", lambda path: ["https://example.edu/a", "https://example.edu/b"])
    monkeypatch.setattr(
        cli,
        "ocr_environment_status",
        lambda: {
            "pdftotext": True,
            "pdftoppm": True,
            "tesseract": True,
            "tesseract_languages": {"eng"},
            "has_chi_sim": False,
        },
    )
    monkeypatch.setattr(
        cli,
        "office_environment_status",
        lambda: {
            "has_openpyxl": True,
            "has_python_docx": True,
            "has_python_pptx": True,
            "has_textutil": False,
            "has_xlrd": True,
        },
    )

    exit_code = cli.main(["doctor", "--seeds", str(seeds_path)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "seed_count=2" in output
    assert "warning=tesseract chi_sim language data is missing" in output
    assert "warning=office extraction dependency missing: has_textutil" in output


def test_collect_command_builds_collector_config_and_reports_skip_known(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    existing_jsonl = tmp_path / "jsonl"
    previous_run = tmp_path / "runs" / "previous" / "jsonl"
    existing_jsonl.mkdir()
    previous_run.mkdir(parents=True)
    write_jsonl(existing_jsonl / "documents.jsonl", [{"url": "https://known.example/a"}])
    write_jsonl(previous_run / "documents.jsonl", [{"canonical_url": "https://known.example/b"}])
    captured: dict[str, object] = {}

    class FakeCollector:
        def __init__(self, config) -> None:
            captured["config"] = config

        def run(self) -> dict[str, int]:
            return {"documents": 1, "chunks": 2}

    monkeypatch.setattr(cli, "OfficialCollector", FakeCollector)

    exit_code = cli.main(
        [
            "collect",
            "--seeds",
            str(tmp_path / "seeds.csv"),
            "--collection-runs",
            str(tmp_path / "runs"),
            "--run-name",
            "current",
            "--max-pages",
            "3",
            "--delay",
            "0",
            "--timeout",
            "5",
            "--retries",
            "2",
            "--dry-run",
            "--ignore-robots",
            "--same-host-only",
            "--allowed-hosts",
            "https://sist.shanghaitech.edu.cn, example.edu:443",
            "--skip-known",
            "--existing-jsonl",
            str(existing_jsonl),
        ]
    )

    config = captured["config"]
    output = capsys.readouterr().out
    assert exit_code == 0
    assert config.max_pages == 3
    assert config.timeout_seconds == 5
    assert config.retries == 2
    assert config.dry_run is True
    assert config.respect_robots is False
    assert config.same_host_only is True
    assert config.allowed_hosts == frozenset({"sist.shanghaitech.edu.cn", "example.edu"})
    assert config.known_urls == frozenset({"https://known.example/a", "https://known.example/b"})
    assert "documents=1" in output
    assert "known_urls_skipped=2" in output


def test_merge_command_prints_merge_stats(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    captured: dict[str, Path] = {}

    def fake_merge(existing_jsonl: Path, run_jsonl: Path, output: Path) -> dict[str, int]:
        captured["existing_jsonl"] = existing_jsonl
        captured["run_jsonl"] = run_jsonl
        captured["output"] = output
        return {"documents": 2, "chunks": 4}

    monkeypatch.setattr(cli, "merge_existing_with_run", fake_merge)

    exit_code = cli.main(
        [
            "merge",
            "--existing-jsonl",
            str(tmp_path / "existing"),
            "--run-jsonl",
            str(tmp_path / "run"),
            "--output",
            str(tmp_path / "merged"),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert captured["run_jsonl"] == tmp_path / "run"
    assert "documents=2" in output
    assert "chunks=4" in output


def test_reparse_command_passes_flags_url_filter_and_chunk_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    url_file = tmp_path / "urls.txt"
    url_file.write_text("# comment\nhttps://sist.shanghaitech.edu.cn/a.htm#frag\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_reparse(
        source_run: Path,
        run_dir: Path,
        seeds_path: Path,
        only_flags: set[str] | None,
        url_filter: set[str] | None,
        limit: int | None,
        chunk_chars: int,
        chunk_overlap: int,
    ) -> dict[str, int]:
        captured.update(
            source_run=source_run,
            run_dir=run_dir,
            seeds_path=seeds_path,
            only_flags=only_flags,
            url_filter=url_filter,
            limit=limit,
            chunk_chars=chunk_chars,
            chunk_overlap=chunk_overlap,
        )
        return {"documents": 1}

    monkeypatch.setattr(cli, "reparse_run", fake_reparse)

    exit_code = cli.main(
        [
            "reparse",
            "--source-run",
            str(tmp_path / "source"),
            "--seeds",
            str(tmp_path / "seeds.csv"),
            "--collection-runs",
            str(tmp_path / "runs"),
            "--run-name",
            "reparse",
            "--only-flag",
            "garbled_text",
            "--url-file",
            str(url_file),
            "--limit",
            "5",
            "--chunk-chars",
            "400",
            "--chunk-overlap",
            "40",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert captured["only_flags"] == {"garbled_text"}
    assert captured["url_filter"] == {"https://sist.shanghaitech.edu.cn/a.htm"}
    assert captured["limit"] == 5
    assert captured["chunk_chars"] == 400
    assert captured["chunk_overlap"] == 40
    assert "documents=1" in output


def test_main_returns_two_for_unreachable_command(monkeypatch: pytest.MonkeyPatch) -> None:
    parser_args = argparse.Namespace(command="unknown")
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", lambda self, argv=None: parser_args)

    assert cli.main([]) == 2
