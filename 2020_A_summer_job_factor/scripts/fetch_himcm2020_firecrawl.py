#!/usr/bin/env python3
"""Fetch HiMCM2020/HiMCM_2020 raw files through Firecrawl (GitHub raw is often RST).

Requires env FIRECRAWL_API_KEY. Never commit the key.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Final

PACK_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
OUT_DIR: Final[Path] = PACK_ROOT / "data" / "raw" / "himcm2020_github"
RAW_BASE: Final[str] = "https://raw.githubusercontent.com/HiMCM2020/HiMCM_2020/main/"
API: Final[str] = "https://api.firecrawl.dev/v1/scrape"

FILES: Final[tuple[str, ...]] = (
    "README.md",
    "OriginalData.txt",
    "ReplacedData.txt",
    "ReplacedIndex.txt",
    "work_choice.txt",
    "SingleSample.txt",
    "Models.py",
    "entropy.py",
    "W_matrix.txt",
    "entropy_vector.txt",
)


def scrape_markdown(url: str, api_key: str) -> str:
    """POST /v1/scrape and return markdown/text body."""
    payload = json.dumps({"url": url, "formats": ["markdown"]}).encode("utf-8")
    request = urllib.request.Request(
        API,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "GoHiMCM-2020A-firecrawl/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Firecrawl HTTP {exc.code} for {url}") from exc
    if not body.get("success"):
        raise RuntimeError(f"Firecrawl failure for {url}: {body}")
    data = body.get("data") or {}
    text = data.get("markdown") or ""
    if not text.strip():
        raise RuntimeError(f"empty Firecrawl body for {url}")
    return text


def main() -> int:
    """Download the GitHub mirror into data/raw/himcm2020_github/."""
    api_key = os.environ.get("FIRECRAWL_API_KEY", "").strip()
    if not api_key:
        print("Set FIRECRAWL_API_KEY in the environment.", file=sys.stderr)
        return 2
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        url = RAW_BASE + name
        print(f"Firecrawl {name}")
        text = scrape_markdown(url, api_key)
        (OUT_DIR / name).write_text(text, encoding="utf-8")
        time.sleep(0.35)
    print(f"wrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
