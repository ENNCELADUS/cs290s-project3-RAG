#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def run() -> int:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from rag_collection.cli import main

    return main()


if __name__ == "__main__":
    raise SystemExit(run())
