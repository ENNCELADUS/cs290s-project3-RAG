from __future__ import annotations

import argparse
import json
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from .io import atomic_json_dump, read_jsonl

DEFAULT_INPUT = Path("data/merged/all-collection-runs-clean-2026-05-27")
DEFAULT_OUTPUT = Path("data/rag/sist_merged_2026-05-27.sqlite")
DEFAULT_REPORT = Path("data/rag/build_report_2026-05-27.json")

DOCUMENT_COLUMNS = [
    "id",
    "run_id",
    "url",
    "canonical_url",
    "title",
    "host",
    "category",
    "language",
    "content_type",
    "status_code",
    "fetched_at",
    "source_published_at",
    "valid_from",
    "valid_until",
    "validity_note",
    "raw_path",
    "text_path",
    "sha256",
    "depth",
    "parent_url",
    "parser",
    "ocr_used",
    "text_chars",
    "reparsed_at",
    "reparsed_from_document_id",
    "reparsed_from_run",
]
CHUNK_COLUMNS = [
    "id",
    "document_id",
    "chunk_index",
    "title",
    "url",
    "category",
    "language",
    "text",
    "char_count",
]
STRUCTURED_TABLES = {
    "courses": "courses.jsonl",
    "faculty_members": "faculty_members.jsonl",
    "program_requirements": "program_requirements.jsonl",
    "events": "events.jsonl",
}
INVALID_FACULTY_NAMES = {
    "all",
    "faculty",
    "people",
    "home",
    "homepage",
    "introduction",
    "profile",
    "teacher",
    "teachers",
    "师资队伍",
    "常任教授",
    "特聘教授",
    "研究人员",
    "支撑人员",
    "行政人员",
}


def build_database(input_dir: Path, output_path: Path, report_path: Path | None = None) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    if tmp_output_path.exists():
        tmp_output_path.unlink()

    documents = read_jsonl(input_dir / "documents.jsonl")
    chunks = read_jsonl(input_dir / "chunks.jsonl")
    report: dict[str, object] = {
        "built_at": datetime.now(UTC).isoformat(),
        "input_dir": str(input_dir),
        "sqlite_path": str(output_path),
        "input_rows": {
            "documents": len(documents),
            "chunks": len(chunks),
            **{table: len(read_jsonl(input_dir / filename)) for table, filename in STRUCTURED_TABLES.items()},
        },
        "filtered_rows": {},
        "output_rows": {},
    }

    valid_documents, document_filter_counts = _valid_documents(documents)
    document_ids = {int(row["id"]) for row in valid_documents}
    valid_chunks, chunk_filter_counts = _valid_chunks(chunks, document_ids)
    report["filtered_rows"] = {
        "documents": document_filter_counts,
        "chunks": chunk_filter_counts,
    }

    try:
        with sqlite3.connect(tmp_output_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            _create_schema(conn)
            _insert_documents(conn, valid_documents)
            _insert_chunks(conn, valid_chunks)

            structured_counts: dict[str, int] = {}
            structured_filter_counts: dict[str, dict[str, int]] = {}
            for table, filename in STRUCTURED_TABLES.items():
                rows, filter_counts = _valid_structured_rows(table, read_jsonl(input_dir / filename), document_ids)
                _insert_structured_rows(conn, table, rows)
                structured_counts[table] = len(rows)
                structured_filter_counts[table] = filter_counts
            conn.commit()

            report["filtered_rows"] = {
                **dict(report["filtered_rows"]),
                **structured_filter_counts,
            }
            report["output_rows"] = {
                "documents": _count_rows(conn, "documents"),
                "chunks": _count_rows(conn, "chunks"),
                **{table: _count_rows(conn, table) for table in STRUCTURED_TABLES},
            }
            report["foreign_key_errors"] = _foreign_key_errors(conn)
            report["sample_sources"] = _sample_sources(conn)
        tmp_output_path.replace(output_path)
    except Exception:
        if tmp_output_path.exists():
            tmp_output_path.unlink()
        raise

    if report_path is not None:
        existing_report = _load_report(report_path)
        atomic_json_dump(report_path, {**existing_report, "database": report})
    return report


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            run_id TEXT,
            url TEXT NOT NULL,
            canonical_url TEXT,
            title TEXT,
            host TEXT,
            category TEXT,
            language TEXT,
            content_type TEXT,
            status_code INTEGER,
            fetched_at TEXT,
            source_published_at TEXT,
            valid_from TEXT,
            valid_until TEXT,
            validity_note TEXT,
            raw_path TEXT,
            text_path TEXT,
            sha256 TEXT,
            depth INTEGER,
            parent_url TEXT,
            parser TEXT,
            ocr_used INTEGER,
            text_chars INTEGER,
            reparsed_at TEXT,
            reparsed_from_document_id INTEGER,
            reparsed_from_run TEXT,
            extra_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            document_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            title TEXT,
            url TEXT,
            category TEXT,
            language TEXT,
            text TEXT NOT NULL,
            char_count INTEGER NOT NULL,
            extra_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (document_id) REFERENCES documents(id)
        );

        CREATE TABLE courses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_document_id INTEGER,
            source_url TEXT,
            course_code TEXT,
            course_name TEXT,
            credits REAL,
            observed_at TEXT,
            evidence TEXT,
            confidence REAL,
            extra_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (source_document_id) REFERENCES documents(id)
        );

        CREATE TABLE faculty_members (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_document_id INTEGER,
            source_url TEXT,
            name TEXT,
            title TEXT,
            email TEXT,
            observed_at TEXT,
            evidence TEXT,
            confidence REAL,
            extra_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (source_document_id) REFERENCES documents(id)
        );

        CREATE TABLE program_requirements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_document_id INTEGER,
            source_url TEXT,
            program_name TEXT,
            requirement_type TEXT,
            requirement_text TEXT,
            min_credits REAL,
            observed_at TEXT,
            evidence TEXT,
            confidence REAL,
            extra_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (source_document_id) REFERENCES documents(id)
        );

        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_document_id INTEGER,
            source_url TEXT,
            title TEXT,
            published_at TEXT,
            event_type TEXT,
            language TEXT,
            org TEXT,
            observed_at TEXT,
            evidence TEXT,
            confidence REAL,
            extra_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (source_document_id) REFERENCES documents(id)
        );

        CREATE INDEX idx_chunks_document_id ON chunks(document_id);
        CREATE INDEX idx_chunks_category ON chunks(category);
        CREATE INDEX idx_documents_url ON documents(url);
        CREATE INDEX idx_documents_host ON documents(host);
        """
    )


def _insert_documents(conn: sqlite3.Connection, rows: list[dict[str, object]]) -> None:
    placeholders = ", ".join("?" for _ in [*DOCUMENT_COLUMNS, "extra_json"])
    conn.executemany(
        f"INSERT INTO documents ({', '.join([*DOCUMENT_COLUMNS, 'extra_json'])}) VALUES ({placeholders})",
        [_row_values(row, DOCUMENT_COLUMNS) for row in rows],
    )


def _insert_chunks(conn: sqlite3.Connection, rows: list[dict[str, object]]) -> None:
    placeholders = ", ".join("?" for _ in [*CHUNK_COLUMNS, "extra_json"])
    conn.executemany(
        f"INSERT INTO chunks ({', '.join([*CHUNK_COLUMNS, 'extra_json'])}) VALUES ({placeholders})",
        [_row_values(row, CHUNK_COLUMNS) for row in rows],
    )


def _insert_structured_rows(conn: sqlite3.Connection, table: str, rows: list[dict[str, object]]) -> None:
    columns_by_table = {
        "courses": [
            "source_document_id",
            "source_url",
            "course_code",
            "course_name",
            "credits",
            "observed_at",
            "evidence",
            "confidence",
        ],
        "faculty_members": [
            "source_document_id",
            "source_url",
            "name",
            "title",
            "email",
            "observed_at",
            "evidence",
            "confidence",
        ],
        "program_requirements": [
            "source_document_id",
            "source_url",
            "program_name",
            "requirement_type",
            "requirement_text",
            "min_credits",
            "observed_at",
            "evidence",
            "confidence",
        ],
        "events": [
            "source_document_id",
            "source_url",
            "title",
            "published_at",
            "event_type",
            "language",
            "org",
            "observed_at",
            "evidence",
            "confidence",
        ],
    }
    columns = columns_by_table[table]
    placeholders = ", ".join("?" for _ in [*columns, "extra_json"])
    conn.executemany(
        f"INSERT INTO {table} ({', '.join([*columns, 'extra_json'])}) VALUES ({placeholders})",
        [_row_values(row, columns) for row in rows],
    )


def _row_values(row: dict[str, object], columns: list[str]) -> tuple[object, ...]:
    extras = {key: value for key, value in row.items() if key not in columns}
    return (
        *(_sqlite_value(row.get(column)) for column in columns),
        json.dumps(extras, ensure_ascii=False, sort_keys=True),
    )


def _sqlite_value(value: object) -> object:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, dict | list):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _valid_documents(rows: Iterable[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, int]]:
    valid: list[dict[str, object]] = []
    seen_ids: set[int] = set()
    counts = {"missing_id": 0, "duplicate_id": 0, "missing_url": 0}
    for row in rows:
        document_id = _optional_int(row.get("id"))
        if document_id is None:
            counts["missing_id"] += 1
            continue
        if document_id in seen_ids:
            counts["duplicate_id"] += 1
            continue
        if not str(row.get("url") or row.get("canonical_url") or "").strip():
            counts["missing_url"] += 1
            continue
        valid.append(row)
        seen_ids.add(document_id)
    return valid, counts


def _valid_chunks(
    rows: Iterable[dict[str, object]], document_ids: set[int]
) -> tuple[list[dict[str, object]], dict[str, int]]:
    valid: list[dict[str, object]] = []
    seen_ids: set[int] = set()
    counts = {"missing_id": 0, "duplicate_id": 0, "missing_document": 0, "empty_text": 0}
    for row in rows:
        chunk_id = _optional_int(row.get("id"))
        document_id = _optional_int(row.get("document_id"))
        if chunk_id is None:
            counts["missing_id"] += 1
            continue
        if chunk_id in seen_ids:
            counts["duplicate_id"] += 1
            continue
        if document_id not in document_ids:
            counts["missing_document"] += 1
            continue
        text = _normalize_chunk_text(row.get("text"))
        if not text:
            counts["empty_text"] += 1
            continue
        if row.get("char_count") is not None:
            int(row["char_count"])
        normalized = {**row, "text": text, "char_count": len(text)}
        valid.append(normalized)
        seen_ids.add(chunk_id)
    return valid, counts


def _normalize_chunk_text(value: object) -> str:
    text = str(value or "")
    text = "".join(char for char in text if char in "\n\r\t" or ord(char) >= 32)
    return text.strip()


def _valid_structured_rows(
    table: str, rows: Iterable[dict[str, object]], document_ids: set[int]
) -> tuple[list[dict[str, object]], dict[str, int]]:
    valid: list[dict[str, object]] = []
    counts = {"missing_document": 0, "invalid_row": 0}
    for row in rows:
        source_document_id = _optional_int(row.get("source_document_id"))
        if source_document_id is not None and source_document_id not in document_ids:
            counts["missing_document"] += 1
            continue
        if table == "faculty_members":
            name = str(row.get("name") or "").strip()
            if not name or name.lower() in INVALID_FACULTY_NAMES:
                counts["invalid_row"] += 1
                continue
        valid.append(row)
    return valid, counts


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _count_rows(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])


def _foreign_key_errors(conn: sqlite3.Connection) -> list[list[object]]:
    return [list(row) for row in conn.execute("PRAGMA foreign_key_check").fetchall()]


def _sample_sources(conn: sqlite3.Connection) -> list[dict[str, object]]:
    rows = conn.execute(
        """
        SELECT c.id, c.title, c.url, d.host
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        ORDER BY c.id
        LIMIT 5
        """
    ).fetchall()
    return [{"chunk_id": row[0], "title": row[1], "url": row[2], "host": row[3]} for row in rows]


def _load_report(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the RAG SQLite database from merged JSONL.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    report = build_database(args.input_dir, args.output, args.report)
    print(f"sqlite_path={args.output}")
    for table, count in dict(report["output_rows"]).items():
        print(f"{table}={count}")
    if report["foreign_key_errors"]:
        print(f"foreign_key_errors={len(report['foreign_key_errors'])}")
        return 1
    print("foreign_key_errors=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
