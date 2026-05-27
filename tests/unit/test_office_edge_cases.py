from __future__ import annotations

import types
import zipfile
from io import BytesIO
from pathlib import Path

from rag_collection import office


def test_extract_docx_falls_back_to_document_xml() -> None:
    body = BytesIO()
    with zipfile.ZipFile(body, "w") as archive:
        archive.writestr(
            "word/document.xml",
            """
            <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
              <w:body><w:p><w:r><w:t>Fallback text</w:t></w:r></w:p></w:body>
            </w:document>
            """,
        )

    result = office.extract_office_text(body.getvalue(), ".docx", Path("unused"))

    assert result.parser == "docx"
    assert result.text == "Fallback text"


def test_extract_legacy_doc_uses_textutil_when_available(monkeypatch, tmp_path: Path) -> None:
    def fake_run(command, *args, **kwargs) -> types.SimpleNamespace:
        Path(command[4]).write_text("Converted legacy text", encoding="utf-8")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(office.subprocess, "run", fake_run)

    result = office.extract_office_text(b"legacy", ".doc", tmp_path / "legacy")

    assert result.parser == "doc_textutil"
    assert result.text == "Converted legacy text"
    assert result.flags == []
    assert not (tmp_path / "legacy.doc").exists()
    assert not (tmp_path / "legacy.txt").exists()


def test_extract_legacy_ppt_reports_unsupported_when_textutil_fails(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(office.subprocess, "run", lambda *args, **kwargs: types.SimpleNamespace(returncode=1))

    result = office.extract_office_text(b"legacy", ".ppt", tmp_path / "legacy")

    assert result.parser == "ppt_unsupported"
    assert result.text == ""
    assert result.flags == ["unsupported_office_binary"]


def test_extract_xls_reports_unsupported_when_xlrd_and_textutil_fail(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(office, "_extract_xls", lambda body: "")
    monkeypatch.setattr(office.subprocess, "run", lambda *args, **kwargs: types.SimpleNamespace(returncode=1))

    result = office.extract_office_text(b"legacy", ".xls", tmp_path / "legacy")

    assert result.parser == "xls_unsupported"
    assert result.flags == ["unsupported_office_binary"]


def test_extract_unknown_office_extension_reports_unsupported(tmp_path: Path) -> None:
    result = office.extract_office_text(b"legacy", ".rtf", tmp_path / "legacy")

    assert result.parser == "office_unsupported"
    assert result.flags == ["unsupported_office_binary"]


def test_cell_to_text_formats_blank_integer_and_string_values() -> None:
    assert office._cell_to_text(None) == ""
    assert office._cell_to_text(4.0) == "4"
    assert office._cell_to_text(" CS101 ") == "CS101"
