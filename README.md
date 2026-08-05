# ZGV Morse

This computation-only repository supports the paper *A Coefficient- and
Phase-Resolved Bridge across Noncommuting Limits of Lamb-Wave ZGV Splitting*.
It rebuilds seven qualified numerical artifacts, twelve publication figures,
their source data, generated manuscript values and tables, and both paper PDFs.

## Manuscript sources

`paper/main.tex` and `paper/supplement.tex` are the original journal-neutral
sources. `paper/main_prr.tex` and `paper/supplement_prr.tex` are the same
content in APS REVTeX 4.2 form for a Physical Review Research submission;
compile the supplement first, since the main text resolves its equation and
theorem numbers through `xr-hyper`:

```sh
cd paper
pdflatex supplement_prr && pdflatex supplement_prr
pdflatex main_prr && bibtex main_prr && pdflatex main_prr && pdflatex main_prr
```

Author and affiliation fields in both PRR files are placeholders.

## Required commands

From the repository root, sync once and then choose the smoke development
profile or the full release profile:

```sh
uv sync --frozen --all-extras
uv run python scripts/reproduce_all.py --profile smoke
uv run python scripts/reproduce_all.py --profile full
```

The smoke profile is not manuscript evidence; it is a development check. Each
profile overwrites the canonical `data/generated/`, `data/source_data/`,
`data/provenance_manifest.json`, `figures/main/`, and
`figures/supplementary/` outputs. Use a disposable checkout when those files
must remain untouched.

The full profile runs the complete tests; recomputes the isotropic,
sensitivity, critical-point, scaling, Green-response, convergence, and silicon
stages; validates isotropic convergence and the closed provenance manifest;
rebuilds all figures and supplementary tables; applies strict figure QA;
exports manuscript values; and compiles both PDFs. It is the release evidence.

Lower-level development interfaces remain available, but are not additional
release commands. These include `uv sync --python 3.12.13 --frozen --all-extras`,
`uv run python scripts/reproduce_all.py --profile smoke --skip-paper`,
`uv run python scripts/reproduce_all.py --profile full --skip-paper`, and
`uv run python scripts/export_manuscript_values.py`.

## Exact and semantic TeX gates

The reference gate, `uv run python scripts/compile_paper.py`, requires the exact
recorded TeX toolchain. It builds the Supplement before the main paper in two
clean cycles, audits both final logs and PDFs, and requires byte-identical PDF
bytes. The reference environment is macOS 26.4.1 (build 25E253), arm64, uv
0.11.19, Python 3.12.13, and TeX Live 2026 (latexmk 4.88, pdfTeX 1.40.29,
BibTeX 0.99e).

On a nonreference TeX host, `uv run python scripts/compile_paper_semantic.py`
provides the portability gate. It uses the same Supplement-first dependency,
reference, citation, warning, overfull-box, completeness, and deterministic
metadata checks, but makes no cross-toolchain byte-identity claim. Semantic
success is therefore not a substitute for the reference byte-reproduction
record.

## Clean and retained-state verification

`scripts/verify_clean_reproduction.py` refuses a divergent working tree, makes
a detached temporary worktree from `HEAD`, removes tracked generated outputs
and caches, and uses isolated uv and Matplotlib caches with a deterministic
single-thread environment. The cold run performs the frozen sync, all seven
workflow stages, figure and table generation, strict QA, macro export, exact
compilation, the full test suite, and the claim-evidence gate in explicit order.

The retained-state run then invokes the full profile again. All scientific
stages are recomputed; the verifier does not claim a persistent scientific
cache and records `persistent_scientific_cache` as false. It requires the
baseline, cold, and retained-state hashes of the complete scientific closure to
match before transactionally publishing `data/reproduction_report.json` and
the measured-results block below. A failed run changes neither file, and
temporary worktrees are removed on failure.

## Registered outputs

- `data/generated/`: seven NPZ artifacts and JSON sidecars, plus the isotropic
  validation JSON and convergence CSV.
- `data/provenance_manifest.json`: exact artifact, source-data, input, script,
  and figure-output hashes.
- `data/source_data/`: all 47 manifest-registered source CSVs.
- `figures/main/` and `figures/supplementary/`: six figures each in SVG, PDF,
  PNG, and TIFF (48 files).
- `paper/generated/`: results macros and two supplementary tables.
- `build/paper/main.pdf` and `build/paper/supplement.pdf`.
- `data/reproduction_report.json`: environment, commands, file counts, bytes,
  aggregate hash, per-file hashes, and cold/retained wall times. Timestamps,
  logs, and the report itself are excluded from the scientific hash.

`scripts/build_release.py` validates the online citation audit and its BibTeX
hash, verifies the reproduction report against current files, copies only the
registered allowlist, writes sorted `SHA256SUMS`, and creates a deterministic
archive without recursively including `release/`.

## Verified reproduction measurements

<!-- BEGIN VERIFIED REPRODUCTION RESULTS -->
Verified by the clean-room reproducer using measured values:

- Scientific closure: `117` files, `18288576` bytes.
- Aggregate SHA-256: `976b4903a413c1f6c7358695226ee9ee8f3609bdb3a55431e819f4768001594a`.
- Cold-run wall time: `1531.757092` seconds.
- Retained-state wall time: `1502.889776` seconds.
- Persistent scientific stage cache: `false`; all scientific stages were recomputed.
<!-- END VERIFIED REPRODUCTION RESULTS -->
