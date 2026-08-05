# Numerical reproducibility adversarial review

Date: 2026-07-13

## Review setup

- **Input scope.** Full local manuscript, Supplementary Information, reference
  configuration, numerical source, workflow scripts, tests, seven qualified NPZ/JSON
  artifact pairs, source-data CSV files, twelve figure bundles, provenance manifest,
  CI workflows, and the deterministic PDF compiler.
- **Assessment boundary.** This report evaluates numerical soundness, conditioning,
  provenance, and reproducibility only. It does not independently adjudicate literature
  priority, and it does not treat the silicon calculation as evidence for the
  weak-anisotropy theorem. No external CI run log or independent-machine rerun was
  supplied, so those points are explicitly marked not assessable.
- **Shared claim summary.** On one simple symmetric Lamb branch, an isotropic ZGV
  critical ring is unfolded by a registered cubic perturbation into four minima and four
  saddles; the same coefficient set controls the radial displacement, frequency
  splitting, and the transition from a ring response to fixed-Morse-point decay.
- **Evidence-status vocabulary.** **Supported** means directly backed by inspected code,
  qualified artifacts, and/or an executed check. **Weak** means the evidence is
  informative but does not establish the strength of the wording used. **Not
  assessable** means the required independent environment or execution record was not
  available.

## Overall assessment

The numerical package is unusually strong for a computation-only theory paper. The
authors separate the high-precision Rayleigh--Lamb regression from the thickness-resolved
GLL solver, track a mass-weighted mode rather than an eigenvalue ordinal, retain the
reduced-resolvent term in the mixed sensitivity, rebuild production Green surfaces at
three resolutions, expose fit masks and cancellation masks, and publish deterministic
source data with a closed artifact/figure hash graph. The committed evidence is internally
consistent: all seven qualified artifacts are full-profile records in one scientific
context, the manifest closes seven artifact and twelve figure records, and the current
source-tree hash matches the hash recorded in the sidecars.

The current numerical results are also well conditioned at the reported points. The
committed minimum gaps are about 0.19--0.23, the critical Hessian-to-uncertainty ratio is
about 5961, the response source/window sweep changes the RMS norm by only about -2.8% to
+1.5%, and the maximum reported production accumulated-phase estimate is about
1.29e-6 rad against a 0.05-rad gate. An additional read-only order-14/18 probe at all
eight reported epsilon=0.02 critical points found a maximum next-order frequency change
of 1.003e-12 versus a maximum recorded order-10/14 estimate of 3.701e-11; the analogous
gradient values were 6.453e-13 and 2.197e-10. Thus the concerns below are principally
about what has actually been certified and made portable, not evidence of a presently
wrong critical point or response curve.

I found no Critical defect. I do, however, recommend major revision before presenting
the numerical layer as an exact, portable certification. The central theoretical case
remains promising if the authors distinguish a finite-resolution numerical realization
from a proof of exhaustive enumeration, strengthen or rename the two-order error
estimates, and make the public clean-reproduction route consistent with the exact
toolchain gate.

## Interested readership

The numerical construction should interest researchers in guided elastic waves,
ultrasonics, spectral discretization of operator pencils, Morse--Bott bifurcation,
stationary-phase asymptotics, caustics, and symmetry breaking. The most transferable
parts are the coefficient-resolved bridge from a critical manifold to isolated Morse
points and the explicit phase-controlled comparison between direct branch quadrature and
uniform/fixed-point asymptotics.

## Major strengths

1. **Independent isotropic regression is real rather than cosmetic (Supported).** The
   production spectral branch is checked against a high-precision Rayleigh--Lamb root and
   curvature, and a two-element assembly is used as a separate discretization check
   (`paper/sections/07_methods.tex:20-29`,
   `tests/test_isotropic_validation.py:273-326`).
2. **Mode selection is structurally informed (Supported).** Mass MAC, phase alignment,
   connected eigenvalue clusters, principal subspace overlaps, angular closure, spectrum
   truncation rejection, and coarse/fine eigengaps are all implemented explicitly
   (`src/zgv_morse/mode_tracking.py:539-618`,
   `src/zgv_morse/dispersion.py:307-314`,
   `src/zgv_morse/dispersion.py:404-474`).
3. **Sensitivity validation is genuinely formulation-diverse (Supported).** The analytic
   Hellmann--Feynman/reduced-resolvent calculation is compared with centered physical
   finite differences and a tensor-product step sweep
   (`src/zgv_morse/workflows/sensitivity.py:156-250`,
   `src/zgv_morse/workflows/convergence.py:340-479`).
4. **The direct Green quadrature does not call the Bessel approximation (Supported).** It
   integrates tracked full-dispersion frequencies and modal amplitudes with the polar
   Jacobian, source weight, super-Gaussian window, angular trapezoid, and radial Simpson
   rule (`src/zgv_morse/green_response.py:309-385`). The normal-form and exact-Morse
   responses are constructed separately (`src/zgv_morse/workflows/green.py:409-465`).
5. **Fit choices and exclusion masks are unusually transparent (Supported).** Early and
   late masks are fixed in code, complete beat bins are stored, no amplitude/phase/carrier
   or time-shift fit is allowed, and the exact-Morse cancellation mask and normalization
   are written to the sidecar (`src/zgv_morse/workflows/green.py:230-256`,
   `src/zgv_morse/workflows/green.py:482-563`,
   `src/zgv_morse/workflows/green.py:723-760`).
6. **The artifact and figure closure is strong (Supported).** Deterministic NPZ members
   use sorted names and a fixed ZIP timestamp (`src/zgv_morse/artifacts.py:185-203`);
   stage metadata hashes configuration, source, code, and lock state
   (`src/zgv_morse/workflows/common.py:57-94`); downstream lineage is checked before
   publication (`src/zgv_morse/workflows/common.py:204-231`); and the manifest verifies
   dependency closure (`src/zgv_morse/provenance.py:437-465`) plus figure inputs, source
   CSVs, scripts, and four output formats (`src/zgv_morse/provenance.py:179-292`).
7. **The PDF build has a serious same-machine reproducibility gate (Supported).** It
   cleans only a registered safe directory, compiles Supplement before Main, rejects
   unstable metadata, and requires two clean PDF byte streams to be identical
   (`scripts/compile_paper.py:164-217`, `scripts/compile_paper.py:247-291`).

## Critical items

None found in the inspected numerical evidence.

## Major items

### M1. The finite-grid annular check cannot support an exact numerical exhaustion claim

- **Status:** Weak.
- **Evidence.** Candidate seeds are generated only when both sampled polar gradient
  components change sign within a cell (`src/zgv_morse/critical_points.py:292-329`). The
  coarse and fine searches then call the same local root solver, the same Hessian routine,
  and the same dispersion evaluator; their near-identical converged roots primarily show
  that the same basins were seeded, not that no other basin exists
  (`src/zgv_morse/workflows/critical_points.py:183-215`). Most importantly, the numerical
  routine itself states that the boundary/index check “does not by itself exclude
  unresolved canceling critical-point pairs in the annulus interior”
  (`src/zgv_morse/critical_points.py:478-492`). The Methods acknowledge the same limit
  (`paper/sections/07_methods.tex:104-119`), but claim C4 still says the tracked branch
  “realizes exactly” four minima and four saddles
  (`docs/manuscript/claim_evidence_matrix.csv:5`).
- **Why this matters.** Outer-minus-inner winding fixes only the sum of indices. An
  unseeded minimum--saddle pair contributes zero and can evade both index closure and a
  local solver. Grid doubling and an angular offset are good empirical controls, but are
  not an exhaustion proof.
- **Required resolution.** Either (a) replace “exactly/certified exhaustion” in the
  numerical claim with “the resolved set found by the registered finite-resolution
  search,” leaving exact cardinality to the analytic theorem under its stated remainder
  hypotheses, or (b) add a validated cellwise degree/interval-Newton or equivalent
  a-posteriori exclusion over the entire annulus. A synthetic adversarial field containing
  a narrow canceling pair between current nodes should be included as a negative test.
- **Verification command for the present, limited gate:**

  ```sh
  uv run pytest -q \
    tests/test_morse_splitting.py::test_candidate_resolution_doubling_is_stable_and_deduplicated \
    tests/test_morse_splitting.py::test_exhaustion_compares_point_index_with_outer_minus_inner_winding \
    tests/test_morse_splitting.py::test_exhaustion_report_detects_unresolved_boundary_uncertainty
  ```

### M2. Two-order differences are estimators, not rigorous absolute uncertainty bounds

- **Status:** Weak, with a large empirical safety margin at the reported critical points.
- **Evidence.** The evaluator defines relative eigenvalue “uncertainty” as the maximum of
  one order-p/order-(p+4) discrepancy and the two algebraic residuals
  (`src/zgv_morse/dispersion.py:290-296`). Absolute frequency and gradient uncertainties
  are simply fine-minus-coarse differences (`src/zgv_morse/dispersion.py:527-545`). The
  long-time phase gate then treats the maximum of these values and a common-node grid
  difference as an error bound (`src/zgv_morse/green_response.py:809-830`), and the
  manuscript calls the result a certificate (`paper/sections/07_methods.tex:47-58`,
  `docs/derivations/04_spectral_numerics.tex:610-641`). No theorem, monotonicity result,
  saturation test, or residual-based Hermitian eigenvalue enclosure is supplied that
  turns a single two-order difference into an upper bound on the exact discretization
  error.
- **Mitigating evidence.** A read-only p=14/18 probe performed for this review at all eight
  epsilon=0.02 critical points was much smaller than the committed p=10/14 differences:
  frequency 1.003e-12 versus 3.701e-11, and gradient 6.453e-13 versus 2.197e-10. The
  production phase estimate also lies roughly four orders of magnitude below its gate.
  These facts support accuracy, but the higher-order probe is not yet part of the
  qualified production surface or artifact.
- **Required resolution.** Use “nested-order error estimate” consistently unless a bound
  is established. For certification language, add at least a third order at every
  accepted production node, a demonstrated contraction/enclosure rule, and a failure
  path for saturation; an a-posteriori generalized-Hermitian eigenvalue bound would be
  preferable. Store the higher-order evidence and phase bound in the qualified artifact.
- **Verification command used for the manuscript-level gate:**

  ```sh
  uv run python - <<'PY'
  from pathlib import Path
  from zgv_morse.provenance import validate_manifest
  record = validate_manifest(Path('data/provenance_manifest.json'), require_figures=True)
  print(len(record['artifacts']), len(record['figures']))
  PY
  ```

  Observed: `7 12`.

### M3. The exact local PDF toolchain is not reproducibly provisioned by the full CI job

- **Status:** Not assessable for an actual Ubuntu run; the configuration mismatch itself
  is supported.
- **Evidence.** The compiler refuses anything except latexmk 4.88, pdfTeX 1.40.29/TeX
  Live 2026, and BibTeX 0.99e/TeX Live 2026
  (`scripts/compile_paper.py:27-37`, `scripts/compile_paper.py:146-160`). The full CI job
  uses the mutable `ubuntu-latest` image and installs unversioned distribution packages
  with `apt-get` immediately before invoking that exact-version gate
  (`.github/workflows/repro-full.yml:22-31`). The committed numerical sidecars document
  only the local macOS execution (`data/generated/green_crossover.json:152-154`); no
  successful Linux full-run record was supplied.
- **Why this matters.** A mutable runner plus unpinned system packages cannot guarantee a
  hard-coded byte-reproduction toolchain. The job can fail solely because the runner
  image advances, independently of the manuscript or code. The same issue prevents this
  review from treating cross-platform PDF reproduction as established.
- **Required resolution.** Provision TeX from a pinned container image or checksum-pinned
  archive that contains the exact required binaries, and record a successful clean Linux
  run. Alternatively, relax the exact-version requirement and define a semantic PDF gate,
  but then byte identity should be scoped to the reference image only.
- **Verification commands:**

  ```sh
  uv run python scripts/compile_paper.py
  latexmk -version
  pdflatex --version
  bibtex --version
  ```

  These establish the local toolchain only. A successful run of
  `.github/workflows/repro-full.yml` on the pinned reference image remains required.

### M4. The public reproduction entry point is missing from the README and profile safety is unclear

- **Status:** Supported.
- **Evidence.** The README still says that the production reproduction command “will be
  introduced” (`README.md:21-22`), although the command and paper compiler now exist.
  The actual runner writes every profile to the canonical `data/generated` directory
  because its stage commands do not pass a separate output location
  (`scripts/reproduce_all.py:48-83`). Consequently a local smoke run overwrites the
  committed full-profile artifacts and figures; the sidecars correctly label the result,
  but the public README neither gives the full command nor warns about this mutation. The
  manuscript contains the real command sequence (`paper/sections/07_methods.tex:165-213`),
  but a code release must not require readers to discover execution instructions inside
  the PDF source.
- **Why this matters.** “Runnable code” is not only an internal test property. A reader
  needs one documented clean command, expected duration/resources, TeX prerequisites,
  output locations, the smoke/full distinction, and a statement that smoke replaces
  canonical generated files unless `--output` isolation is used.
- **Required resolution.** Replace the stale README paragraph with the exact frozen sync,
  full reproduction, manuscript export, and compile commands; document the current local
  reference environment and expected outputs; and either make smoke use an isolated build
  root by default or prominently warn that it overwrites full evidence. Add a README/CLI
  contract test.
- **Verification command after correction:**

  ```sh
  uv sync --python 3.12.13 --frozen --all-extras
  uv run python scripts/reproduce_all.py --profile full --skip-paper
  uv run python scripts/export_manuscript_values.py
  uv run python scripts/compile_paper.py
  ```

## Minor items

### m1. Production-node overlap evidence is enforced weakly and not persisted

- **Status:** Weak in provenance; supported at the eight reported critical points by an
  additional probe.
- **Evidence.** The generic tracker accepts a candidate unless both scalar MAC and squared
  subspace overlap fall below the default 0.2 threshold
  (`src/zgv_morse/mode_tracking.py:560-604`). Arbitrary critical-search queries call this
  default directly (`src/zgv_morse/dispersion.py:494-524`). Critical-point artifacts store
  residuals and Hessians but no per-point MAC, subspace overlap, or gap
  (`src/zgv_morse/workflows/critical_points.py:364-383`). The reported >=0.99 tracking MAC
  is a separate isotropic radial sweep (`src/zgv_morse/workflows/convergence.py:587-626`),
  not a minimum over the anisotropic critical/Green production nodes.
- **Probe result.** Direct inspection at the eight reported epsilon=0.02 points found
  minimum coarse/fine scalar MACs of 0.99999905, minimum principal cosines of 0.99999953,
  and minimum relative gap 0.22404. Thus branch identity is strongly supported at those
  points, but this evidence is not stored and does not cover every surface node.
- **Requested correction.** Persist minimum scalar MAC and principal overlap by epsilon
  and grid, raise the production acceptance threshold to a value justified by the observed
  margin, and include these arrays in the artifact/claim graph.

### m2. The measured crossover-scaling test uses a theory-centred search bracket

- **Status:** Weak as an independent epsilon-scaling test; supported as a diagnostic
  consistent with the stronger Bessel-collapse comparison.
- **Evidence.** The “measured” crossing is searched only inside
  `[0.5 t_pred, 1.5 t_pred]`, where `t_pred` already scales as `1/|epsilon V4|`
  (`src/zgv_morse/workflows/green.py:131-147`). The numerical specification repeats this
  design (`docs/derivations/04_spectral_numerics.tex:604-608`). This does not force an
  exact -1 slope, but it constrains every accepted measurement to an epsilon-scaled
  interval before the slope is fitted.
- **Requested correction.** Report an unbracketed or fixed-global-window crossing where
  possible, add threshold sensitivity (for example 0.8, 0.9, 1.0), and present the present
  metric as a consistency diagnostic rather than independent evidence for the scaling
  law. The parameter-free complex Bessel collapse should remain the primary test.

### m3. Same-machine numerical byte reproducibility is tested at the writer/figure/PDF level, not for two complete scientific reruns

- **Status:** Not assessable for the seven expensive stages as a whole.
- **Evidence.** NPZ serialization itself is deterministic
  (`src/zgv_morse/artifacts.py:185-203`), figure tests exercise repeated byte-identical
  rendering, and the PDF compiler performs two clean builds. The public reproduction
  runner, however, executes the scientific stages once and then validates their content
  (`scripts/reproduce_all.py:90-110`); it does not repeat the full calculation and compare
  the seven NPZ/JSON hashes. The full CI likewise invokes the runner once
  (`.github/workflows/repro-full.yml:31`).
- **Requested correction.** Add an optional release-grade command that runs all seven
  full stages into two isolated output roots under identical deterministic environments
  and compares arrays exactly or under a declared numeric equivalence policy. Record both
  manifests. Exact byte equality may be scoped to one pinned platform; cross-platform
  reproducibility should use declared numerical tolerances.

### m4. Environment provenance omits the numerical backend that can affect eigensolvers

- **Status:** Weak.
- **Evidence.** Sidecars record Python, NumPy, SciPy, mpmath, and a platform string
  (`src/zgv_morse/workflows/common.py:254-260`), while CI fixes several thread counts
  (`.github/workflows/repro-full.yml:15-20`). They do not record BLAS/LAPACK vendor,
  library build, CPU architecture details beyond the platform string, compiler/runtime,
  or the `uv` version. These can affect low-order bits and, near degeneracy, eigenvector
  bases.
- **Requested correction.** Store `numpy.show_config()` or a normalized backend record,
  SciPy/NumPy build identifiers, `uv --version`, and the relevant deterministic thread
  environment in a release-level environment manifest.

## Circular-validation and hidden-choice audit

- **Direct Green versus Bessel:** Supported as noncircular at the quadrature level. The
  direct integration contains no Bessel call (`src/zgv_morse/green_response.py:309-385`).
  It does share `k0`, curvature, sensitivities, source, and modal amplitude with the
  asymptotic prediction, as it should for a parameter-free reduction test. This is not an
  independent solver comparison and should not be described as one.
- **Exact Morse sum versus direct response:** Supported as a no-fit same-branch comparison.
  Points, Hessians, modal amplitudes, and the direct surface all come from the same
  spectral formulation, but they enter different numerical constructions. The disclosed
  coherence exclusion and retained cancellation-region error prevent the most obvious
  post-selection bias (`src/zgv_morse/workflows/green.py:535-563`).
- **Sensitivity finite differences:** Supported as an independent differentiation route,
  not an independent PDE solver. Both routes share the GLL matrices and mode tracker.
- **Critical-point grid doubling:** Weak as an independence claim because the converged
  roots use the same local solver and field; it is a seed-coverage perturbation only.
- **Fit masks:** Supported as explicit and response-independent in source. Historical
  preregistration before any result was seen is not assessable from the final repository
  alone; “predeclared in the versioned implementation” is the defensible wording.
- **Stale artifacts:** No stale qualified artifact was detected. The current source-tree
  hash equals the recorded `code_hash`, all seven records are full profile, and the
  complete figure closure validates. Generated PDFs are outside the artifact manifest but
  are retained and byte-gated by the compiler.

## Verification record

### Executed successfully for this review

1. Full artifact and figure closure:

   ```sh
   uv run python - <<'PY'
   from pathlib import Path
   from zgv_morse.provenance import validate_manifest
   from zgv_morse.artifact_schema import validate_artifact
   root = Path('.')
   manifest = validate_manifest(root/'data/provenance_manifest.json', require_figures=True)
   profiles = {
       validate_artifact(path, path.with_suffix('.json'))[1]['profile']
       for path in (root/'data/generated').glob('*.npz')
   }
   print(len(manifest['artifacts']), len(manifest['figures']), profiles)
   PY
   ```

   Observed: `7 12 {'full'}`.

2. Current source/sidecar identity:

   ```sh
   uv run python - <<'PY'
   import json
   from pathlib import Path
   from zgv_morse.workflows.common import _tree_hash
   recorded = json.load(open('data/generated/isotropic_zgv.json'))['code_hash']
   current = _tree_hash(Path('src'))
   print(recorded, current, recorded == current)
   PY
   ```

   Observed: matching digest
   `2b23305385bade99b7f5d1393b01fc925393482e1202dfef73ef7cad805311e9`.

3. Focused adversarial unit checks:

   ```sh
   uv run pytest -q \
     tests/test_morse_splitting.py::test_candidate_resolution_doubling_is_stable_and_deduplicated \
     tests/test_morse_splitting.py::test_exhaustion_compares_point_index_with_outer_minus_inner_winding \
     tests/test_morse_splitting.py::test_exhaustion_report_detects_unresolved_boundary_uncertainty \
     tests/test_mode_tracking.py::test_tracking_follows_shape_when_eigenvalue_order_changes \
     tests/test_green_response.py::test_registered_grid_verifier_uses_nested_complex_response_differences
   ```

   Observed: `5 passed in 0.45s`.

4. Eight-point branch-conditioning probe using the evaluator's stored coarse and fine
   tracked-mode records:

   ```sh
   # Read-only probe at the eight committed epsilon=0.02 critical points.
   # Recreate order-10/14 evaluators, inspect mode.mac, subspace_overlap, and eigengap.
   ```

   Observed minima: coarse/fine MAC `0.99999905/0.99999905`, coarse/fine principal
   cosine `0.99999953/0.99999953`, relative gap `0.22404119`.

5. Higher-order critical-point accuracy probe:

   ```sh
   # Read-only probe: compare order-10/14 and order-14/18 evaluators at the same
   # eight committed epsilon=0.02 critical points.
   ```

   Observed maximum frequency differences: p10/14 `3.7006842e-11`, p14/18
   `1.0027534e-12`; gradient differences: p10/14 `2.1966607e-10`, p14/18
   `6.4529601e-13`.

### Recommended release checks not established by this review

```sh
uv run pytest -q
uv run python scripts/check_claim_evidence.py --require-supported
uv run python scripts/reproduce_all.py --profile full --skip-paper
uv run python scripts/export_manuscript_values.py
uv run python scripts/qa_figures.py --strict
uv run python scripts/compile_paper.py
```

A successful pinned-Linux execution, a second independent full scientific rerun, and a
complete release-level checksum/environment manifest were not available for assessment.

## Nature-style criteria

- **Originality:** Promising and technically distinctive at the level of the integrated
  analytic/numerical workflow. Priority relative to all prior ZGV/Morse literature is not
  assessable in this numerical-only review.
- **Scientific importance:** Potentially high for computation-led wave physics because the
  work connects symmetry breaking, critical-point topology, and temporal decay with no
  fitted transition curve. The importance claim depends on retaining the present scope
  discipline and avoiding overstatement of numerical certification.
- **Interdisciplinary readership:** Credible across applied mathematics, elasticity,
  spectral numerics, and asymptotic wave theory. The source/provenance design may also
  interest computational-science readers.
- **Technical soundness:** Supported for the reported registered family and observed
  points, with major qualifications M1--M3. Current numerical margins are strong; the
  logical and portability language is the weaker part.
- **Readability for nonspecialists:** The manuscript and Supplement are unusually explicit,
  but the repository-facing route is not readable or usable while the README remains
  stale. A simple workflow diagram and a concise “what is proved / what is computed / what
  is not claimed” table would materially help.

## Recommendation posture

**Major revision; technically supportive if the certification and reproduction issues are
resolved.** The registered numerical evidence supports the paper's local weak-anisotropy
mechanism, scaling, and response comparisons. It does not yet support an exact numerical
exhaustion claim or a portable full-build claim at the strength currently implied. These
are addressable without experiments and without changing the central theoretical result.
