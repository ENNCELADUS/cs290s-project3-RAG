from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .chunking import normalize_text


@dataclass(frozen=True)
class PdfExtractionResult:
    text: str
    parser: str
    ocr_used: bool
    flags: list[str]


def extract_pdf_text(pdf_path: Path, min_text_chars: int = 240, max_ocr_pages: int = 3) -> PdfExtractionResult:
    flags: list[str] = []
    text = _extract_with_pdftotext(pdf_path, flags)
    if len(text) >= min_text_chars:
        return PdfExtractionResult(text=text, parser="pdftotext", ocr_used=False, flags=flags)

    flags.append("low_pdf_text_density")
    ocr_text = _extract_with_tesseract(pdf_path, max_ocr_pages=max_ocr_pages, flags=flags)
    if ocr_text.strip():
        return PdfExtractionResult(text=ocr_text, parser="ocr", ocr_used=True, flags=flags)
    return PdfExtractionResult(text=text, parser="pdftotext", ocr_used=False, flags=flags)


def ocr_environment_status() -> dict[str, object]:
    languages = _tesseract_languages()
    return {
        "pdftotext": shutil.which("pdftotext") is not None,
        "pdftoppm": shutil.which("pdftoppm") is not None,
        "tesseract": shutil.which("tesseract") is not None,
        "tesseract_languages": languages,
        "has_chi_sim": "chi_sim" in languages,
    }


def _extract_with_pdftotext(pdf_path: Path, flags: list[str]) -> str:
    if shutil.which("pdftotext") is None:
        flags.append("pdftotext_unavailable")
        return ""
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        flags.append("pdftotext_failed")
        return ""
    if result.returncode != 0:
        flags.append("pdftotext_failed")
    return normalize_text(result.stdout)


def _extract_with_tesseract(pdf_path: Path, max_ocr_pages: int, flags: list[str]) -> str:
    if shutil.which("pdftoppm") is None:
        flags.append("pdftoppm_unavailable")
        return ""
    if shutil.which("tesseract") is None:
        flags.append("tesseract_unavailable")
        return ""

    languages = _tesseract_languages()
    if "chi_sim" not in languages:
        flags.append("tesseract_chi_sim_missing")
        return ""
    language_arg = "chi_sim+eng" if "eng" in languages else "chi_sim"

    with tempfile.TemporaryDirectory() as tmpdir:
        output_prefix = Path(tmpdir) / "page"
        try:
            conversion = subprocess.run(
                [
                    "pdftoppm",
                    "-r",
                    "200",
                    "-png",
                    "-f",
                    "1",
                    "-l",
                    str(max_ocr_pages),
                    str(pdf_path),
                    str(output_prefix),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=90,
            )
        except (OSError, subprocess.TimeoutExpired):
            flags.append("pdftoppm_failed")
            return ""
        if conversion.returncode != 0:
            flags.append("pdftoppm_failed")
            return ""

        page_text: list[str] = []
        for image_path in sorted(Path(tmpdir).glob("page-*.png")):
            try:
                ocr = subprocess.run(
                    ["tesseract", str(image_path), "stdout", "-l", language_arg, "--psm", "6"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
            except (OSError, subprocess.TimeoutExpired):
                flags.append("tesseract_failed")
                continue
            if ocr.returncode != 0:
                flags.append("tesseract_failed")
                continue
            page_text.append(ocr.stdout)
        return normalize_text("\n\n".join(page_text))


def _tesseract_languages() -> set[str]:
    if shutil.which("tesseract") is None:
        return set()
    try:
        result = subprocess.run(
            ["tesseract", "--list-langs"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()
    return {line.strip() for line in result.stdout.splitlines()[1:] if line.strip()}
