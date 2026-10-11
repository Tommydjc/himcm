#!/usr/bin/env python3
"""Evidence gate: high-signal numbers in paper should appear in results/*.csv.

Usage
-----
    python tools/evidence_check.py --slug 2025_A_evacuation
    python tools/evidence_check.py --slug 2024_A_olympic_sde --min-digits 3

Exit codes
----------
    0 : no missing hits (or only skipped soft tokens)
    1 : at least one paper number missing from CSV corpus
    2 : bad arguments / missing paths
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT: Path = Path(__file__).resolve().parents[1]

# Skip tiny integers that are usually section numbers / years / N= counts in prose.
_SOFT_INT_MAX: int = 20

_NUMBER_RE: re.Pattern[str] = re.compile(
    r"""
    (?<![\w.])                  # not mid-identifier
    (?:
        \d+\.\d+                # decimals
        | \d{2,}                # integers with 2+ digits
    )
    (?![\w.])
    """,
    re.VERBOSE,
)

_PAPER_GLOBS: tuple[str, ...] = (
    "paper/**/*.tex",
    "paper/**/*.md",
    "paper_figures/**/*.md",
    "results/**/*.md",
)


def _iter_files(root: Path, patterns: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(root.glob(pattern)))
    return [p for p in files if p.is_file()]


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"[warn] cannot read {path}: {exc}", file=sys.stderr)
        return ""


def _extract_numbers(text: str, min_digits: int) -> set[str]:
    """Extract normalized number tokens from text.

    Parameters
    ----------
    text :
        Raw file contents.
    min_digits :
        Minimum digit count for integers (decimals always kept if they have a
        fractional part). Soft-filters integers ``<= _SOFT_INT_MAX``.

    Returns
    -------
    set[str]
        Canonical string forms as they appear (no thousand separators assumed).
    """
    found: set[str] = set()
    for match in _NUMBER_RE.finditer(text):
        token: str = match.group(0)
        if "." in token:
            found.add(token)
            # Also keep trimmed trailing zeros variant for soft match later.
            continue
        if len(token) < min_digits:
            continue
        try:
            value: int = int(token)
        except ValueError:
            continue
        if value <= _SOFT_INT_MAX:
            continue
        found.add(token)
    return found


def _csv_corpus(slug_dir: Path) -> str:
    chunks: list[str] = []
    for csv_path in sorted(slug_dir.glob("results/**/*.csv")):
        chunks.append(_read_text(csv_path))
    return "\n".join(chunks)


def _paper_corpus(slug_dir: Path) -> tuple[str, list[Path]]:
    paths: list[Path] = _iter_files(slug_dir, _PAPER_GLOBS)
    # Prefer dedicated paper tree; fall back to top-level himcm_paper if present.
    extra: Path = slug_dir / "paper" / "himcm_paper.tex"
    if extra.is_file() and extra not in paths:
        paths.append(extra)
    text: str = "\n".join(_read_text(p) for p in paths)
    return text, paths


def _load_allowlist(slug_dir: Path) -> set[str]:
    """Load optional ``paper/evidence_allowlist.txt`` (one token per line).

    Lines starting with ``#`` and blank lines are ignored. Tokens are model
    constants, page geometry, or other non-result numbers that need not appear
    in ``results/*.csv``.
    """
    path: Path = slug_dir / "paper" / "evidence_allowlist.txt"
    if not path.is_file():
        return set()
    allowed: set[str] = set()
    for line in _read_text(path).splitlines():
        stripped: str = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        allowed.add(stripped)
    return allowed


def check_slug(slug: str, min_digits: int, strict: bool) -> int:
    """Run evidence check for one study package.

    Parameters
    ----------
    slug :
        Directory name under repo root, e.g. ``2025_A_evacuation``.
    min_digits :
        Integer digit threshold (default 2 via CLI).
    strict :
        If False (default), exit 0 when missing count is reported but treat the
        run as an advisory report unless ``--strict`` is set. If True, any
        missing token yields exit 1.

    Returns
    -------
    int
        Process exit code (0/1/2).
    """
    slug_dir: Path = REPO_ROOT / slug
    if not slug_dir.is_dir():
        print(f"[error] slug directory not found: {slug_dir}", file=sys.stderr)
        return 2

    results_dir: Path = slug_dir / "results"
    if not results_dir.is_dir():
        print(f"[error] missing results/: {results_dir}", file=sys.stderr)
        return 2

    paper_text, paper_files = _paper_corpus(slug_dir)
    if not paper_text.strip():
        print(f"[error] no paper/markdown sources under {slug}", file=sys.stderr)
        return 2

    csv_text: str = _csv_corpus(slug_dir)
    if not csv_text.strip():
        print(f"[error] no CSV under {results_dir}", file=sys.stderr)
        return 2

    allowlist: set[str] = _load_allowlist(slug_dir)
    paper_nums: set[str] = _extract_numbers(paper_text, min_digits=min_digits)
    missing: list[str] = sorted(
        token
        for token in paper_nums
        if token not in csv_text and token not in allowlist
    )

    print(f"slug: {slug}")
    print(f"paper files: {len(paper_files)}")
    print(f"allowlist tokens: {len(allowlist)}")
    print(f"numbers extracted from paper: {len(paper_nums)}")
    print(f"missing from results/**/*.csv: {len(missing)}")
    if missing:
        print("--- missing (sample up to 40) ---")
        for token in missing[:40]:
            print(f"  {token}")
        if len(missing) > 40:
            print(f"  ... and {len(missing) - 40} more")
        print(
            "hint: fix paper, re-run into results/, or add non-result constants "
            "to paper/evidence_allowlist.txt"
        )
        if strict:
            return 1
        print("mode: advisory (pass); re-run with --strict for gate failure")
        return 0

    print("OK: all extracted high-signal numbers appear in CSV corpus.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Check that paper numbers appear in results/*.csv"
    )
    parser.add_argument(
        "--slug",
        required=True,
        help="Study package directory name under repo root",
    )
    parser.add_argument(
        "--min-digits",
        type=int,
        default=2,
        help="Minimum digits for integer tokens (default: 2)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 when any paper number is missing from CSV/allowlist",
    )
    args: argparse.Namespace = parser.parse_args(argv)
    if args.min_digits < 1:
        print("[error] --min-digits must be >= 1", file=sys.stderr)
        return 2
    return check_slug(
        slug=args.slug, min_digits=args.min_digits, strict=bool(args.strict)
    )



if __name__ == "__main__":
    sys.exit(main())
