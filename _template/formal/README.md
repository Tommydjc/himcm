# formal/ — optional Lean 4 lemmas

Copy this folder into `YYYY_P_slug/formal/` when you need appendix-level formalization.

## Rules

- Prove only statements allowed by `wiki/concepts/formal-methods-boundary.md`.
- Empirical numbers in `results/*.csv` are **not** Lean theorems.
- Contest time budget: at most one short lemma (see `wiki/playbooks/lean-lemma-loop.md`).

## Suggested layout

```text
formal/
  README.md           # this file
  lakefile.toml       # add when initializing a Lean 4 package
  FormalLemmas.lean   # statements + proofs
```

Initialize with your local Lean 4 / Lake toolchain; this template intentionally ships no `lake` lockfile so the Studies hub stays lightweight.

## Paper citation pattern

> Under assumptions H1–Hk, Lemma N holds; machine-checked in `formal/FormalLemmas.lean`. Ranking / clearance / cover metrics remain those in `results/*.csv`.
