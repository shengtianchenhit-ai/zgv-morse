# PRR-style significance and readability adversarial review

## Review setup

- **Input scope.** The review covers the complete main manuscript, abstract,
  Introduction, Results, Discussion, Methods, Conclusion, reproducibility
  statements, all six main figures and captions, the Supplementary Information
  architecture and theorem/proof package, the claim--evidence matrix, figure
  contracts, dated search protocol and log, novelty matrix, citation audit, and
  BibTeX database.
- **Assessment boundary.** This is a journal-style significance, originality,
  interdisciplinary-reach, and readability review. It is not an editorial
  decision, an experimental review, a fresh exhaustive priority search, or an
  independent reimplementation of the numerical solver. The dated local
  literature audit is assessed for internal consistency and claim calibration.
- **Shared manuscript claim.** For one simple, uniformly gapped Lamb branch and
  one weak [001] cubic perturbation family, the paper closes the angular and
  radial normal-form coefficients from full elasticity, proves a controlled
  Morse--Bott-to-Morse splitting, and connects the known isotropic ring response
  to the known isolated-point endpoint through a normalized
  `t^-1/2 J0(epsilon V4 t)` law and a fixed-anisotropy exact-Hessian Morse sum.
- **Visible evidence base.** Exact Rayleigh--Lamb differentiation, an independent
  thickness-resolved eigensolver, analytic generalized-eigenvalue sensitivities,
  centered full-wave finite differences, compensated perturbation sequences,
  Hessian and annular-index certificates, phase-bounded response quadrature,
  and machine-audited source data are all present. The claim--evidence gate
  reports seven supported rows.
- **Status labels.** **SUPPORTED** means established by the supplied manuscript
  and local checks; **WEAK** means plausible but not yet made persuasive at the
  claimed level of significance or readability; **NOT ASSESSABLE** means the
  computation-only packet cannot establish the point and does not pretend to.

### Read-only checks used

```text
uv run python scripts/check_claim_evidence.py
# claim-evidence gate passed: 7 supported claim rows

uv run python scripts/verify_bibliography.py --bib paper/references.bib
# entries=39 mismatch=0 duplicate_doi=0 invalid_doi=0
# manual_needed=0 missing_required=0

rg -io '\bregistered\b' paper/sections paper/figure_captions.tex | wc -l
# 57
```

## Reviewer 1 -- technical-evidence and claim-calibration emphasis

### Overall assessment

The within-model evidentiary chain is unusually complete. The manuscript does
not rely on a contour plot or a fitted normal form: it derives `V4` and `B`,
checks both independently, proves the local reduction with derivative-level
remainders, certifies the point set, and controls accumulated numerical phase.
The distinction between compact-`tau`, growing-`|tau|`, and fixed-anisotropy
limits is also technically explicit. I found no critical technical defect in
the claims as actually bounded.

The principal weakness is the visual presentation of the temporal evidence.
Figure 6c invites a stronger empirical reading than the underlying numerical
test supports: the early and late fitted slopes come from different anisotropy
rows, whereas the main text correctly states that the plotted late bins are not
a numerical joint-path test of the growing-`|tau|` theorem.

### Who would be interested, and why

Researchers in guided-wave mechanics, spectral perturbation, stationary-phase
asymptotics, wave localization, and computational elastodynamics would value the
constant- and phase-resolved bridge between two previously separate endpoint
descriptions. The annular index and accumulated-phase controls are particularly
useful as a reproducible template for singularly perturbed dispersion problems.

### Major strengths

- **SUPPORTED:** The paper separates theorem, coefficient computation, and
  numerical realization rather than treating visual agreement as proof
  (`paper/sections/03_morse_unfolding.tex:37`,
  `paper/sections/03_morse_unfolding.tex:74`,
  `paper/sections/05_numerical_verification.tex:16`).
- **SUPPORTED:** The temporal analysis explicitly prohibits extrapolating the
  compact-`tau` theorem to the fixed-anisotropy limit
  (`paper/sections/04_temporal_crossover.tex:43`,
  `paper/sections/06_discussion.tex:30`).
- **SUPPORTED:** The fixed-anisotropy sum uses exact frequencies, amplitudes,
  and Cartesian Hessians, not first-order surrogates
  (`paper/sections/04_temporal_crossover.tex:76`).
- **SUPPORTED:** The computation-only boundary is explicit and scientifically
  appropriate (`paper/sections/06_discussion.tex:127`,
  `paper/sections/08_conclusion.tex:27`).

### Major concern

- **M2 / WEAK -- Figure 6 visually conflates two noncommuting numerical
  regimes.** The panel title says “Observable decay: `t^-1/2 -> t^-1`”
  (`src/zgv_morse/figures/figure06_crossover.py:780`), and the figure contract
  calls the result a “universal” crossover and says panel c links the collapse
  to the observable exponent change
  (`docs/figures/figure_contracts.md:80`,
  `docs/figures/figure_contracts.md:87`). In fact, the caption states that the
  early slope is from `epsilon=0.005` and the late slope from `epsilon=0.08`
  (`paper/figure_captions.tex:118`), and the Results explicitly state that the
  late bins are a separate fixed-anisotropy endpoint rather than a numerical
  joint-path test (`paper/sections/04_temporal_crossover.tex:69`). The theory of
  the overlap is supported; the present panel does not, by itself, display a
  single-trajectory numerical crossover from one exponent to the other. The
  panel hierarchy and wording should make “two asymptotic slices plus a proved
  overlap” unmistakable.

### Technical failings that must be addressed before the presented case is clear

No new experiment is required to establish the stated mathematical/numerical
case. The presentation must, however, stop the early- and late-slope markers from
being read as a single direct numerical trajectory. If a single fixed-`epsilon`
trace cannot access both controlled windows, that limitation should be visible
in the panel itself, not only recoverable from the caption and Results prose.

### Nature-style axes

- **Originality:** supported as a coefficient- and phase-resolved Lamb-wave
  connection, not as a new Bessel or stationary-phase theorem.
- **Scientific importance:** strong within wave mechanics; broader importance
  depends on portability beyond the one demonstrated family.
- **Interdisciplinary reach:** plausible for singular perturbations and critical
  manifolds, but not yet foregrounded.
- **Technical soundness:** strong within the declared model and evidence packet.
- **Nonspecialist readability:** the limit structure is accurate in prose but
  not yet self-evident from the key temporal figure.

### Recommendation posture

Supportive after a major presentation revision to Figure 6 and its associated
claim hierarchy; no experimental condition should be imposed on the bounded
computation-only theorem.

## Reviewer 2 -- originality and scientific-importance emphasis

### Overall assessment

The novelty audit is candid and substantially strengthens the paper. It correctly
excludes priority for the ZGV ring, the four-minimum/four-saddle pattern,
anisotropic beating, the abstract Morse--Bott perturbation, the Bessel identity,
and ordinary two-dimensional stationary phase
(`docs/literature/novelty_matrix.md:16`). The defensible advance is therefore a
specific synthesis: full-elastic coefficient closure, controlled local
splitting, uniform temporal normalization, and constant/phase matching. That
synthesis is credible and useful, but its precision is not carried by the title
or the first half of the abstract strongly enough.

### Who would be interested, and why

The immediate audience is guided-wave and elastic-wave physics. A secondary
audience includes mathematical physicists studying coalescing stationary sets,
van Hove-type critical geometry, and broken continuous symmetry. Computational
mechanics readers may adopt the eigengap, sensitivity, index, and phase
certificates even outside Lamb waves.

### Major strengths

- **SUPPORTED:** The paper states the closest prior endpoints before presenting
  its own contribution (`paper/sections/01_introduction.tex:17`) and identifies
  the precise missing connection (`paper/sections/01_introduction.tex:32`).
- **SUPPORTED:** The novelty matrix records high-overlap areas and limits the
  strongest supportable sentence to the complete chain rather than any one
  standard component (`docs/literature/novelty_matrix.md:23`).
- **SUPPORTED:** The Discussion repeats the distinction from prior ZGV,
  Morse--Bott, Bessel, and stationary-phase work
  (`paper/sections/06_discussion.tex:42`,
  `paper/sections/06_discussion.tex:69`).
- **SUPPORTED:** The result is more than an exponent match: it retains elastic
  coefficients, source normalization, signature phases, and an explicit
  phase-validity window (`paper/sections/06_discussion.tex:75`).

### Major concerns

- **M1 / WEAK -- The title advertises the known mechanism more prominently than
  the new contribution.** “Weak Cubic Anisotropy Unfolds a Lamb-Wave ZGV Ring”
  (`paper/main.tex:22`) is close to phenomena and wording that the Introduction
  itself identifies as prior art, including the four-plus-four organization and
  “unfolding” (`paper/sections/01_introduction.tex:17`). The new distinction is
  the coefficient-resolved, full-elastic, uniformly matched connection
  (`paper/sections/01_introduction.tex:32`), but that distinction is absent from
  the title. A skeptical reader can therefore dismiss the paper before reaching
  its genuinely stronger result. The title and opening claim should foreground
  the controlled connection or the noncommuting temporal limits, not merely the
  already known endpoint change.

- **M3 / WEAK -- Broad physical importance is not yet demonstrated beyond one
  canonical branch and perturbation path.** The manuscript is admirably explicit
  that its evidence concerns one simple branch, one weak cubic family, one
  local annulus, and one nonnodal source/weight class
  (`paper/sections/01_introduction.tex:58`,
  `paper/sections/08_conclusion.tex:27`). The silicon calculation tests search
  and classification but lies outside the asymptotic evidence
  (`paper/sections/05_numerical_verification.tex:81`). This is sufficient for a
  strong, focused mechanics/waves contribution, but it leaves the broader PRR
  case weak: the packet does not show how often nonzero `V4`, a usable
  phase-validity window, or an observable crossover occurs across branches,
  material paths, or sources. This concern can be addressed computationally by
  a second controlled family/branch or, at minimum, by a sharper generality map
  showing which parts are theorem-level and which are instance-level. It is not
  a demand for laboratory data.

### Technical failings that must be addressed before the significance case is established

The manuscript must make the unit of novelty immediately legible: not “eight
points,” not “unfolding,” and not “a Bessel function,” but the elasticity-derived
coefficient and constant/phase-resolved bridge across noncommuting limits. It
must also distinguish a theorem that is structurally transferable from numerical
evidence obtained for only one branch.

### Nature-style axes

- **Originality:** high for the complete plate-specific chain; low for each
  endpoint or generic asymptotic ingredient in isolation.
- **Scientific importance:** credible and substantial for mechanics/waves;
  outstanding cross-physics importance is not yet demonstrated.
- **Interdisciplinary reach:** latent in the critical-manifold and symmetry-
  breaking structure, but underdeveloped in the title, abstract, and Discussion.
- **Technical soundness:** the supported within-model chain makes the originality
  claim unusually defensible.
- **Nonspecialist readability:** the paper requires readers to discover the
  novelty by subtraction from prior art rather than stating the positive advance
  in one plain sentence at the start.

### Recommendation posture

Promising and potentially strong for PRR or a high-level mechanics/waves journal,
but the significance case requires major reframing before submission. The local
scientific contribution is not incremental; its current top-level wording makes
it look more incremental than it is.

## Reviewer 3 -- interdisciplinary-reach and nonspecialist-readability emphasis

### Overall assessment

The Introduction begins accessibly, and Figure 1 provides an effective geometric
entry point. The main text is more readable than the Supplementary mathematics
needs to be, and the Discussion is unusually disciplined about scope. However,
the abstract rapidly turns into a numerical audit trail. A broad physics reader
is given nine model-specific scalar values before being told why the result
matters outside the selected branch. Repeated words such as “registered,”
“stored,” “predeclared,” and “artifact-backed” make the prose sound like an
internal verification dossier rather than a scientific narrative.

### Who would be interested, and why

Nonspecialists could care about the general principle that weak symmetry breaking
changes the dimension of a stationary set and creates noncommuting long-time
limits. Readers in wave localization, phononics, metamaterials, seismology,
optics, and semiclassical asymptotics could recognize this pattern. The current
abstract does not yet invite those readers in those terms.

### Major strengths

- **SUPPORTED:** The first Introduction paragraph explains ZGV localization and
  the geometric origin of the time exponent without assuming specialist notation
  (`paper/sections/01_introduction.tex:3`).
- **SUPPORTED:** Figure 1 establishes a ring-to-points visual grammar and keeps
  the conceptual panel explicitly separate from numerical evidence
  (`paper/figure_captions.tex:3`).
- **SUPPORTED:** The Discussion gives a concise explanation of the nonuniform
  `|epsilon V4|^-1/2` prefactor and noncommuting limits
  (`paper/sections/06_discussion.tex:17`).
- **SUPPORTED:** The limitations section is honest about nodal channels, loss,
  finite boundaries, defects, and eigengap closure
  (`paper/sections/06_discussion.tex:82`).

### Major concern

- **M4 / WEAK -- The abstract prioritizes validation metadata over the conceptual
  result and its audience.** Lines 9--21 enumerate `kappa0`, `Omega0`, `a`,
  `V4`, two perturbation slopes, two envelope slopes, and a phase bound
  (`paper/sections/00_abstract.tex:9`). These values are not interpretable without
  the model normalization and do not tell a broad reader why the bridge matters.
  The most arresting idea is instead the noncommutation of the symmetry-restoring
  and long-time limits, which appears only later in the paper
  (`paper/sections/06_discussion.tex:17`). The abstract should retain only
  numerical evidence that changes belief, state the new connection positively,
  and explain in plain language why symmetry-lifted stationary dimensions change
  transient localization.

### Technical failings that must be addressed before the readability case is established

The abstract and key figures must let a reader answer three questions without
opening the Supplement: what was already known, what exact bridge is new, and
why the bridge changes the interpretation of a transient response. At present
the answers exist, but are distributed among the Introduction, novelty audit,
Figure 6 caption, and Discussion.

### Nature-style axes

- **Originality:** precise once the full Introduction is read; insufficiently
  positive and memorable in the title/abstract.
- **Scientific importance:** the noncommuting-limit insight is potentially broad,
  but the practical or cross-domain consequence is not articulated early.
- **Interdisciplinary reach:** plausible, not yet demonstrated through a second
  system or a cross-domain consequence.
- **Technical soundness:** the audit vocabulary reflects real rigor, but too much
  of it is carried into reader-facing prose.
- **Nonspecialist readability:** good in the first Introduction paragraph and
  Figure 1; weak in the abstract and dense Figure 6.

### Recommendation posture

Major readability revision. The problem is not excessive mathematics in the
Supplement; it is that the main narrative does not consistently preserve the
simple geometric idea already achieved in Figure 1.

## Cross-review synthesis

### Overall assessment

This is a technically serious and likely publishable computation-only paper for
a strong mechanics/waves venue. Its best contribution is a controlled,
coefficient-resolved and phase-resolved connection between two known ZGV
endpoints. The manuscript is unusually strong on reproducibility, theorem
boundaries, and non-overclaiming. No critical issue was found. The submission is
not yet optimized for a selective broad-scope physics journal because the title,
abstract, and temporal hero figure do not make the precise unit of originality
and the two-limit evidence hierarchy immediately apparent.

### Interested readership

- guided elastic waves, Lamb waves, ZGV localization, and nondestructive
  evaluation;
- asymptotic analysis of coalescing stationary points and Morse--Bott critical
  manifolds;
- spectral perturbation and computational wave mechanics;
- phononic, acoustic, optical, or semiclassical researchers interested in weak
  symmetry breaking and transient critical-set geometry.

### Consensus strengths

1. **SUPPORTED:** exceptionally disciplined novelty boundaries;
2. **SUPPORTED:** coefficient closure from the full traction-free eigenproblem,
   independently checked rather than fitted;
3. **SUPPORTED:** exact Hessian signatures, index closure, and phase control;
4. **SUPPORTED:** a genuine constant- and phase-level Bessel--Morse match, not
   merely an exponent analogy;
5. **SUPPORTED:** computation-only scope is stated clearly and does not require
   an experiment to validate the mathematical claims.

### Issue ledger

#### Critical issues

None identified.

#### Major issues

1. **M1 / WEAK -- The title foregrounds a known endpoint change, not the new
   coefficient-resolved connection.** Evidence:
   `paper/main.tex:22`, `paper/sections/01_introduction.tex:17`,
   `paper/sections/01_introduction.tex:32`.
2. **M2 / WEAK -- Figure 6 visually presents a single observable exponent arrow
   although the early and late numerical slopes are different limit slices.**
   Evidence: `src/zgv_morse/figures/figure06_crossover.py:780`,
   `paper/figure_captions.tex:118`,
   `paper/sections/04_temporal_crossover.tex:69`,
   `docs/figures/figure_contracts.md:80`.
3. **M3 / WEAK -- The field-local importance is strong, but portability across
   branches/material families/sources is not demonstrated.** Evidence:
   `paper/sections/01_introduction.tex:58`,
   `paper/sections/05_numerical_verification.tex:81`,
   `paper/sections/08_conclusion.tex:27`.
4. **M4 / WEAK -- The abstract is dominated by model-specific values and audit
   vocabulary rather than the conceptual advance and broad relevance.**
   Evidence: `paper/sections/00_abstract.tex:9`,
   `paper/sections/00_abstract.tex:22`,
   `paper/sections/06_discussion.tex:17`.

#### Minor issues

1. **m1 / SUPPORTED discrepancy -- The literature narrative says 38 retained
   DOI records, but the current machine audit and BibTeX contain 39.** Evidence:
   `docs/literature/novelty_matrix.md:31`,
   `docs/literature/search_log.md:115`,
   `docs/literature/citation_audit.json:2231`. The read-only local audit reports
   `entries=39`. This is a documentation-drift issue, not a bibliographic
   metadata failure.
2. **m2 / WEAK terminology -- “Stationary-set topology controls temporal decay”
   is broader than the demonstrated mechanism.** The analysis shows that
   stationary-manifold dimension/codimension and nondegenerate Hessian rank
   control the stationary-phase exponent; topology alone does not.
   Evidence: `src/zgv_morse/figures/figure01_geometry.py:330`,
   `paper/sections/06_discussion.tex:12`,
   `paper/sections/08_conclusion.tex:12`.
3. **m3 / WEAK readability -- Protocol vocabulary is overexposed.** “Registered”
   appears 57 times across the main sections and caption registry, including
   three times in the abstract (`paper/sections/00_abstract.tex:13`,
   `paper/sections/00_abstract.tex:18`,
   `paper/sections/00_abstract.tex:23`); “artifact-backed” appears in reader-facing
   Results (`paper/sections/02_isotropic_ring.tex:107`). Keep the rigor, but move
   procedural provenance details to Methods/captions where possible.
4. **m4 / WEAK figure economy -- Figures 1a--b and 4a--b repeat essentially the
   same ring-to-eight-point contour comparison.** Evidence:
   `paper/figure_captions.tex:3`, `paper/figure_captions.tex:62`. Their distinct
   roles--conceptual mechanism versus certified full-wave realization--should be
   more visually differentiated.
5. **m5 / SUPPORTED but low-value panel -- Figure 3c is explicitly a perturbation-
   convention check rather than an independent full-wave comparison, yet it
   occupies a large fraction of the figure.** Evidence:
   `paper/figure_captions.tex:49`, `docs/figures/figure_contracts.md:44`. For a
   significance-led figure sequence, the independent `V`/`B` closure deserves
   greater visual priority.
6. **m6 / WEAK wording -- “Parameter-free” can be read as “contains no physical
   parameters,” although the intended statement is “no fitted alignment.”**
   Evidence: `src/zgv_morse/figures/figure06_crossover.py:632`,
   `paper/figure_captions.tex:113`. Prefer the latter phrase consistently.

### What is not assessable without experiments

- **NOT ASSESSABLE:** whether damping and finite-boundary return times leave a
  measurable interval wide enough to observe both asymptotic regimes in a real
  plate;
- **NOT ASSESSABLE:** whether a laboratory source/detector realizes the assumed
  nonnodal angular weight and sufficiently isolates the selected branch;
- **NOT ASSESSABLE:** whether the synthetic weak-cubic continuation maps onto an
  experimentally convenient material and thickness range with the required
  eigengap and phase window;
- **NOT ASSESSABLE:** absolute signal-to-noise, transducer bandwidth, and
  parameter-identification uncertainty.

These are physical-observability questions, not defects in the computation-only
theorem. The manuscript already acknowledges loss, boundaries, nodal channels,
and branch contamination (`paper/sections/06_discussion.tex:82`,
`paper/sections/06_discussion.tex:127`). Experiments would be necessary only if
the paper claimed laboratory observability or material-specific predictive
accuracy; it does not.

### Broad-interest/significance readout

- **For a strong mechanics/waves journal:** the case is credible now and would
  become strong after M1, M2, and M4 are resolved.
- **For PRR or another selective broad-scope physics venue:** the technical case
  is credible, but broad significance is presently **WEAK** because M3 remains
  largely a promise of transferability. A second fully computational example is
  the cleanest strengthening route; a sharper theorem-versus-instance map is the
  minimum route.
- **For a Nature-style bar:** originality is supported, technical soundness is
  strong, and the result is potentially arresting; outstanding importance,
  immediate/far-reaching implications, and interdisciplinary reach are not yet
  established from the supplied single-family evidence.

### Recommendation posture

**Major revision before broad-scope submission; encouraging posture after
revision.** The requested revisions are principally claim architecture, figure
logic, and computational generality. The absence of experiments should not be
treated as a rejection criterion for the bounded theoretical claims.

## Risk / unsupported claims

- **SUPPORTED:** a bounded negative literature result found no prior source
  combining the complete Lamb-ZGV chain, but this is not proof of priority
  (`docs/literature/search_log.md:128`). A fresh search immediately before
  submission remains necessary, as the novelty matrix itself states
  (`docs/literature/novelty_matrix.md:31`).
- **WEAK if used without qualifiers:** “universal crossover,” “topology controls
  decay,” or “parameter-free.” Each has a defensible narrow meaning in the
  technical packet, but a broader ordinary-language reading exceeds the shown
  evidence.
- **NOT ASSESSABLE:** finite-specimen, dissipative, experimental, and nonlinear
  behavior. The manuscript appropriately excludes all four.
- **Unsupported and correctly not claimed:** discovery of the eight anisotropic
  points, discovery of ZGV beating, a new Bessel identity, a new general
  stationary-phase theorem, a global dispersion-surface point count, or a
  nonlinear amplitude equation.
