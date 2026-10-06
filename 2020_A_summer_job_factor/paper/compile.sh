#!/usr/bin/env bash
# Compile HiMCM 2020 paper suite (main PDF under 25 pages).
set -euo pipefail
cd "$(dirname "$0")"

export PATH="${HOME}/.local/bin:/Library/TeX/texbin:/Applications/MiKTeX Console.app/Contents/bin:${PATH}"

compile_one() {
  local tex="$1"
  if command -v tectonic >/dev/null 2>&1; then
    tectonic "$tex"
  elif command -v latexmk >/dev/null 2>&1 && command -v pdflatex >/dev/null 2>&1; then
    latexmk -pdf -interaction=nonstopmode "$tex"
  elif command -v pdflatex >/dev/null 2>&1; then
    pdflatex -interaction=nonstopmode "$tex"
    pdflatex -interaction=nonstopmode "$tex"
  else
    echo "error: neither tectonic nor pdflatex found." >&2
    exit 1
  fi
}

for f in himcm_paper.tex summary_sheet.tex high_school_job_guide.tex; do
  echo "==> $f"
  compile_one "$f"
done

if command -v python >/dev/null 2>&1; then
  python - <<'PY'
from pathlib import Path
n = -1
try:
    from pypdf import PdfReader
    n = len(PdfReader("himcm_paper.pdf").pages)
except Exception:
    try:
        import pypdf
        n = len(pypdf.PdfReader("himcm_paper.pdf").pages)
    except Exception:
        pass
print(f"himcm_paper.pdf pages={n} (limit 25)")
if n > 25:
    raise SystemExit("paper exceeds 25 pages")
PY
fi

echo "done: $(ls -1 *.pdf 2>/dev/null | tr '\n' ' ')"
