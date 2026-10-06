#!/usr/bin/env bash
# Compile HiMCM 2021 paper suite (keep the main PDF under 25 pages).
set -euo pipefail
cd "$(dirname "$0")"

export PATH="${HOME}/.local/bin:/Library/TeX/texbin:/Applications/MiKTeX Console.app/Contents/bin:${PATH}"

compile_one() {
  local tex="$1"
  if command -v tectonic >/dev/null 2>&1; then
    tectonic "$tex"
  elif command -v latexmk >/dev/null 2>&1 && command -v pdflatex >/dev/null 2>&1; then
    latexmk -pdf -interaction=nonstopmode "$tex"
  else
    echo "error: neither tectonic nor pdflatex found." >&2
    exit 1
  fi
}

for f in himcm_paper.tex summary_sheet.tex letter_to_homeowners.tex; do
  echo "==> $f"
  compile_one "$f"
done

if command -v python3 >/dev/null 2>&1; then
  python3 - <<'PY'
from pathlib import Path
try:
    import pypdf
    reader = pypdf.PdfReader("himcm_paper.pdf")
    n = len(reader.pages)
except Exception:
    try:
        from pypdf import PdfReader
        n = len(PdfReader("himcm_paper.pdf").pages)
    except Exception:
        n = -1
print(f"himcm_paper.pdf pages={n} (limit 25)")
if n > 25:
    raise SystemExit("paper exceeds 25 pages")
PY
fi

echo "done: $(ls -1 *.pdf 2>/dev/null | tr '\n' ' ')"
