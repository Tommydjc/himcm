#!/usr/bin/env python3
"""Firecrawl BLS OOH + O*NET pages for occupations named in 2020 HiMCM Problem A.

The official PDF contains no URLs. Occupations cited in the prompt (cashier,
lifeguard, wait staff, data analysis, office administration, research, plus
walk/bike/drive/virtual work) are mapped to BLS OOH and O*NET summary pages.

Requires FIRECRAWL_API_KEY. Never commit the key.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Final

PACK_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
OUT_MD: Final[Path] = PACK_ROOT / "data" / "raw" / "bls_onet_pages"
API: Final[str] = "https://api.firecrawl.dev/v1/scrape"

# Problem-A occupations → public handbook / O*NET URLs (not present as hyperlinks in the PDF).
SOURCES: Final[list[dict[str, str]]] = [
    {
        "job_id": "lifeguard",
        "pdf_mention": "lifeguarding",
        "url": "https://www.bls.gov/ooh/protective-service/lifeguards-ski-patrol-and-other-recreational-protective-service-workers.htm",
    },
    {
        "job_id": "cashier",
        "pdf_mention": "cashier at a store",
        "url": "https://www.bls.gov/ooh/sales/cashiers.htm",
    },
    {
        "job_id": "waiter",
        "pdf_mention": "wait staff at a restaurant",
        "url": "https://www.bls.gov/ooh/food-preparation-and-serving/waiters-and-waitresses.htm",
    },
    {
        "job_id": "retail_sales",
        "pdf_mention": "cashier at a store / store",
        "url": "https://www.bls.gov/ooh/sales/retail-sales-workers.htm",
    },
    {
        "job_id": "fast_food",
        "pdf_mention": "wait staff / food service",
        "url": "https://www.bls.gov/ooh/food-preparation-and-serving/food-and-beverage-serving-and-related-workers.htm",
    },
    {
        "job_id": "recreation_worker",
        "pdf_mention": "recreation activities",
        "url": "https://www.bls.gov/ooh/personal-care-and-service/recreation-workers.htm",
    },
    {
        "job_id": "animal_care",
        "pdf_mention": "physical activity jobs",
        "url": "https://www.bls.gov/ooh/personal-care-and-service/animal-care-and-service-workers.htm",
    },
    {
        "job_id": "self_enrichment_teacher",
        "pdf_mention": "analytical skills / teaching-like",
        "url": "https://www.bls.gov/ooh/education-training-and-library/self-enrichment-teachers.htm",
    },
    {
        "job_id": "office_clerk",
        "pdf_mention": "office administration",
        "url": "https://www.bls.gov/ooh/office-and-administrative-support/general-office-clerks.htm",
    },
    {
        "job_id": "data_scientist",
        "pdf_mention": "data analysis",
        "url": "https://www.bls.gov/ooh/math/data-scientists.htm",
    },
    {
        "job_id": "survey_researcher",
        "pdf_mention": "research",
        "url": "https://www.bls.gov/ooh/life-physical-and-social-science/survey-researchers.htm",
    },
    {
        "job_id": "computer_support",
        "pdf_mention": "work from home virtually/electronically",
        "url": "https://www.bls.gov/ooh/computer-and-information-technology/computer-support-specialists.htm",
    },
    {
        "job_id": "onet_lifeguard",
        "pdf_mention": "lifeguarding",
        "url": "https://www.onetonline.org/link/summary/33-9092.00",
    },
    {
        "job_id": "onet_cashier",
        "pdf_mention": "cashier at a store",
        "url": "https://www.onetonline.org/link/summary/41-2011.00",
    },
    {
        "job_id": "onet_waiter",
        "pdf_mention": "wait staff at a restaurant",
        "url": "https://www.onetonline.org/link/summary/35-3031.00",
    },
    {
        "job_id": "onet_retail",
        "pdf_mention": "cashier at a store",
        "url": "https://www.onetonline.org/link/summary/41-2031.00",
    },
    {
        "job_id": "onet_fast_food",
        "pdf_mention": "wait staff / food",
        "url": "https://www.onetonline.org/link/summary/35-3023.00",
    },
    {
        "job_id": "onet_recreation",
        "pdf_mention": "recreation",
        "url": "https://www.onetonline.org/link/summary/39-9032.00",
    },
    {
        "job_id": "onet_animal",
        "pdf_mention": "physical activity",
        "url": "https://www.onetonline.org/link/summary/39-2021.00",
    },
    {
        "job_id": "onet_tutor",
        "pdf_mention": "analytical skills",
        "url": "https://www.onetonline.org/link/summary/25-3041.00",
    },
    {
        "job_id": "onet_office",
        "pdf_mention": "office administration",
        "url": "https://www.onetonline.org/link/summary/43-9061.00",
    },
    {
        "job_id": "onet_data",
        "pdf_mention": "data analysis",
        "url": "https://www.onetonline.org/link/summary/15-2051.00",
    },
    {
        "job_id": "onet_research_asst",
        "pdf_mention": "research",
        "url": "https://www.onetonline.org/link/summary/19-4061.00",
    },
    {
        "job_id": "onet_computer_support",
        "pdf_mention": "work from home virtually/electronically",
        "url": "https://www.onetonline.org/link/summary/15-1232.00",
    },
    {
        "job_id": "onet_work_context_db",
        "pdf_mention": "O*NET Work Context database (job-side factor structure)",
        "url": "https://www.onetcenter.org/database.html",
    },
]


def scrape_markdown(url: str, api_key: str) -> str:
    """POST Firecrawl /v1/scrape and return markdown."""
    payload = json.dumps({"url": url, "formats": ["markdown"], "onlyMainContent": True}).encode(
        "utf-8"
    )
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
        err = exc.read().decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"Firecrawl HTTP {exc.code} for {url}: {err}") from exc
    if not body.get("success"):
        raise RuntimeError(f"Firecrawl failure for {url}: {body}")
    data = body.get("data") or {}
    text = str(data.get("markdown") or "")
    if not text.strip():
        raise RuntimeError(f"empty Firecrawl body for {url}")
    return text


def main() -> int:
    """Scrape all mapped URLs into data/raw/bls_onet_pages/."""
    api_key = os.environ.get("FIRECRAWL_API_KEY", "").strip()
    if not api_key:
        print("Set FIRECRAWL_API_KEY in the environment.", file=sys.stderr)
        return 2
    OUT_MD.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str | int]] = []
    for src in SOURCES:
        slug = re.sub(r"[^a-z0-9_]+", "_", src["job_id"])
        path = OUT_MD / f"{slug}.md"
        print(f"Firecrawl {src['job_id']}  {src['url']}", flush=True)
        try:
            text = scrape_markdown(src["url"], api_key)
            path.write_text(text, encoding="utf-8")
            status = "ok"
            n_bytes = path.stat().st_size
        except Exception as exc:  # noqa: BLE001
            status = f"fail:{type(exc).__name__}:{exc}"
            n_bytes = 0
            print("  FAIL", status)
        manifest.append(
            {
                "job_id": src["job_id"],
                "pdf_mention": src["pdf_mention"],
                "url": src["url"],
                "path": str(path.relative_to(PACK_ROOT)),
                "status": status,
                "bytes": n_bytes,
            }
        )
        time.sleep(0.4)
    man_path = OUT_MD / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    n_ok = sum(1 for row in manifest if row["status"] == "ok")
    print(f"wrote {man_path}  ok={n_ok}/{len(manifest)}")
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
