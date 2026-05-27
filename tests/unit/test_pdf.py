from __future__ import annotations

import subprocess
import types
from pathlib import Path

from rag_collection import pdf


def test_extract_pdf_text_uses_pdftotext_when_text_density_is_high(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(pdf.shutil, "which", lambda name: f"/usr/bin/{name}" if name == "pdftotext" else None)

    def fake_run(*args, **kwargs) -> types.SimpleNamespace:
        return types.SimpleNamespace(returncode=0, stdout="alpha " * 80)

    monkeypatch.setattr(pdf.subprocess, "run", fake_run)

    result = pdf.extract_pdf_text(tmp_path / "sample.pdf", min_text_chars=100)

    assert result.parser == "pdftotext"
    assert result.ocr_used is False
    assert "alpha" in result.text


def test_extract_pdf_text_falls_back_to_ocr_when_pdftotext_is_sparse(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(pdf.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(pdf, "_tesseract_languages", lambda: {"chi_sim", "eng"})

    def fake_run(command, *args, **kwargs) -> types.SimpleNamespace:
        if command[0] == "pdftotext":
            return types.SimpleNamespace(returncode=0, stdout="short")
        if command[0] == "pdftoppm":
            Path(command[-1]).with_name("page-1.png").write_bytes(b"png")
            return types.SimpleNamespace(returncode=0, stdout="")
        if command[0] == "tesseract":
            return types.SimpleNamespace(returncode=0, stdout="信息科学与技术学院")
        raise AssertionError(command)

    monkeypatch.setattr(pdf.subprocess, "run", fake_run)

    result = pdf.extract_pdf_text(tmp_path / "sample.pdf", min_text_chars=100)

    assert result.parser == "ocr"
    assert result.ocr_used is True
    assert result.text == "信息科学与技术学院"
    assert result.flags == ["low_pdf_text_density"]


def test_extract_pdf_text_reports_unavailable_tools(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pdf.shutil, "which", lambda name: None)

    result = pdf.extract_pdf_text(tmp_path / "sample.pdf")

    assert result.parser == "pdftotext"
    assert result.text == ""
    assert result.flags == ["pdftotext_unavailable", "low_pdf_text_density", "pdftoppm_unavailable"]


def test_tesseract_languages_handles_command_failure(monkeypatch) -> None:
    monkeypatch.setattr(pdf.shutil, "which", lambda name: "/usr/bin/tesseract")

    def raise_timeout(*args, **kwargs) -> None:
        raise subprocess.TimeoutExpired("tesseract", 10)

    monkeypatch.setattr(pdf.subprocess, "run", raise_timeout)

    assert pdf._tesseract_languages() == set()
