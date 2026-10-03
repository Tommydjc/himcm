#!/usr/bin/env bash
# Compile HiMCM 2022 paper suite.
# Prefer tectonic (~/.local/bin); fall back to latexmk/pdflatex if TeX Live/MiKTeX is ready.
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
    echo "Install one of:" >&2
    echo "  curl -L https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-aarch64-apple-darwin.tar.gz | tar xz -C ~/.local/bin tectonic" >&2
    echo "  brew install --cask basictex   # then: export PATH=/Library/TeX/texbin:\$PATH" >&2
    echo "  Open 'MiKTeX Console' → finish setup → Links → install to a PATH directory" >&2
    exit 1
  fi
}

for f in himcm_paper.tex summary_sheet.tex grower_advisory_letter.tex ai_disclosure.tex; do
  echo "==> $f"
  compile_one "$f"
done
echo "done: $(ls -1 *.pdf 2>/dev/null | tr '\n' ' ')"
