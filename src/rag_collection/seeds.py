from __future__ import annotations

import csv
from pathlib import Path

from .urls import SeedUrl, canonicalize_url, is_official_url


def load_seed_urls(path: Path) -> list[SeedUrl]:
    seeds: list[SeedUrl] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            url = canonicalize_url(row["url"])
            if not is_official_url(url):
                raise ValueError(f"Seed URL is outside official domains: {url}")
            seeds.append(
                SeedUrl(
                    url=url,
                    category=row.get("category") or "general",
                    depth_limit=int(row.get("depth_limit") or 1),
                    priority=int(row.get("priority") or 5),
                    notes=row.get("notes") or "",
                )
            )
    return sorted(seeds, key=lambda seed: (seed.priority, seed.url))
