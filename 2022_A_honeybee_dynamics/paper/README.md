# Paper suite (HiMCM 2022 Problem A)

## Why `latexmk` failed

Homebrew only installed `latexmk`. There is **no** `pdflatex` on PATH
(`/Library/TeX` missing; MiKTeX Console is present but first-time setup /
symlinks were never finished). Hence:

```text
sh: pdflatex: command not found
```

## Recommended compile (already set up)

A standalone engine is installed at `~/.local/bin/tectonic`. From this directory:

```bash
export PATH="$HOME/.local/bin:$PATH"
./compile.sh
# or one file:
tectonic himcm_paper.tex && open himcm_paper.pdf
```

## Alternative: finish a full TeX distro

1. **BasicTeX / MacTeX**
   ```bash
   brew install --cask basictex
   eval "$(/usr/libexec/path_helper)"
   export PATH="/Library/TeX/texbin:$PATH"
   latexmk -pdf himcm_paper.tex
   ```
2. **MiKTeX**: open *MiKTeX Console* → finish setup → *Settings → Directories / Links*
   so that `pdflatex` appears on PATH, then rerun `latexmk`.

Figures load from `../paper_figures/`. Numbers come from `../results/*.csv`.
