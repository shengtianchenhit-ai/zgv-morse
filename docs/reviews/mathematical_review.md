# Mathematical adversarial review

## Review setup

- **Input scope.** Complete main manuscript, complete Supplementary Information, the four canonical derivation files, the theorem/equation labels imported into the main text, the analytic-identity checks, the focused Morse/asymptotic/Green tests, and the relevant implementation of the critical-point and response calculations.
- **Assessment boundary.** This is a technical-soundness review, not an independent literature search, experimental validation, or editorial decision. No reviewer identity, institution, or hidden expertise is assumed.
- **Central claim assessed.** A simple isotropic Lamb eigenbranch has a finite-radius Morse--Bott ZGV ring; a declared weak [001] cubic perturbation produces a pure first-order fourfold angular potential, which resolves the ring into four minima and four saddles; the associated local positive-frequency response crosses from a uniform (t^{-1/2}J_0(\varepsilon V_4t)) law to a fixed-anisotropy (t^{-1}) exact-Hessian Morse sum.
- **Evidence inspected.** Exact Rayleigh--Lamb determinant and high-precision root; generalized-Hermitian eigenvalue sensitivities; derivative-level normal form; Cartesian Hessian/Schur inertia; annular winding and critical-point search; Bessel and stationary-phase derivations; direct branch quadrature; committed artifacts and numerical gates.

## Overall assessment

The analytical core is unusually careful and, in the material inspected, internally consistent. I found no Critical flaw and no counterexample to the principal local theorems. In particular, the determinant desingularization, implicit curvature formula, cubic tensor contraction, complete differentiated-mode contribution to (B), polar-to-Cartesian Hessian, Morse signatures, Fourier normalization, Bessel constant, factor-of-two frequency convention, and Bessel--Morse phase matching all survive adversarial checks.

The present recommendation posture is nevertheless **major revision before the computational case is described as certified or exact at the registered finite perturbations**. Three gaps remain between the hypotheses of the analytical statements and the finite-resolution evidence used to instantiate them. None currently demonstrates that the reported result is false; one reviewer-side expanded-annulus calculation in fact supports the intended conclusion. The problem is that the committed evidence chain does not yet prove the stronger words “bound”, “certified”, and “exactly” in the senses in which they are used.

## Interested readership

The results should interest researchers in guided-wave physics, applied asymptotics, singularity/Morse methods, elastic-wave computation, and nondestructive ultrasonics. The most transferable point is not the already known four-plus-four pattern, but the coefficient-resolved matching of a Morse--Bott family, a symmetry-breaking angular potential, and the isolated-point stationary-phase endpoint.

## Major strengths

1. **The local geometry is formulated with the correct hypotheses.** The isotropic theorem explicitly requires a simple uniformly separated branch, (k_0>0), a nonzero determinant frequency derivative, and (a\ne0) (`docs/derivations/01_isotropic_rayleigh_lamb.tex:425`, `docs/derivations/01_isotropic_rayleigh_lamb.tex:430`, `docs/derivations/01_isotropic_rayleigh_lamb.tex:435`, `docs/derivations/01_isotropic_rayleigh_lamb.tex:438`). Its proof correctly factors (f'(r)), thereby excluding other critical radii in a sufficiently small annulus (`docs/derivations/01_isotropic_rayleigh_lamb.tex:468`, `docs/derivations/01_isotropic_rayleigh_lamb.tex:471`, `docs/derivations/01_isotropic_rayleigh_lamb.tex:476`).

2. **The perturbation coefficients are not obtained by fitting the final roots.** The arbitrary-normalization Hellmann--Feynman formula is correct (`docs/derivations/02_anisotropic_morse_unfolding.tex:54`), and the mixed radial coefficient includes the mode derivative/reduced-resolvent term (`docs/derivations/02_anisotropic_morse_unfolding.tex:188`, `docs/derivations/02_anisotropic_morse_unfolding.tex:200`). The explicit cubic contraction produces (V_0+V_4\cos4\theta) with (V_4>0) for the declared family (`docs/derivations/02_anisotropic_morse_unfolding.tex:363`, `docs/derivations/02_anisotropic_morse_unfolding.tex:382`, `docs/derivations/02_anisotropic_morse_unfolding.tex:393`).

3. **The normal form retains the term most often lost in informal reductions.** The derivative-level factorization and the radial relaxation term (-B^2/(2a)) are stated and used consistently (`docs/derivations/02_anisotropic_morse_unfolding.tex:440`, `docs/derivations/02_anisotropic_morse_unfolding.tex:449`, `docs/derivations/02_anisotropic_morse_unfolding.tex:484`; `docs/derivations/03_green_function_asymptotics.tex:191`).

4. **The Morse classification uses the Cartesian Hessian.** The connection terms in the polar representation are retained (`docs/derivations/02_anisotropic_morse_unfolding.tex:559`), and the exact Schur complement is used before the inertia is assigned (`docs/derivations/02_anisotropic_morse_unfolding.tex:587`, `docs/derivations/02_anisotropic_morse_unfolding.tex:598`). This correctly prevents a negative angular curvature from being misclassified as a Cartesian maximum when the radial curvature is positive.

5. **The temporal constants and signatures close.** The positive-frequency normalization is defined without an implicit factor of two (`docs/derivations/03_green_function_asymptotics.tex:60`, `docs/derivations/03_green_function_asymptotics.tex:68`, `docs/derivations/03_green_function_asymptotics.tex:95`). The radial Fresnel phase, (2\pi J_0) angular integral, large-argument Bessel constant, polar Jacobian cancellation, and minimum/saddle signature factors are mutually consistent (`docs/derivations/03_green_function_asymptotics.tex:210`, `docs/derivations/03_green_function_asymptotics.tex:293`, `docs/derivations/03_green_function_asymptotics.tex:568`, `docs/derivations/03_green_function_asymptotics.tex:655`, `docs/derivations/03_green_function_asymptotics.tex:669`).

6. **Limit regimes are not silently conflated.** The compact-\(\tau\) result, growing-\(|\tau|\) overlap, and fixed-\(\varepsilon\) Morse theorem are explicitly separated (`paper/sections/04_temporal_crossover.tex:43`, `paper/sections/04_temporal_crossover.tex:48`, `paper/sections/04_temporal_crossover.tex:74`; `paper/sections/06_discussion.tex:30`). The nonuniformity of the fixed-\(\varepsilon\) remainder is correctly acknowledged (`docs/derivations/03_green_function_asymptotics.tex:619`, `docs/derivations/03_green_function_asymptotics.tex:631`).

## Major concerns and technical failings

### Major 1 — Inter-discretization discrepancies are not proved upper bounds on continuum phase error

**Status: weakly supported, not mathematically certified.**

The manuscript defines absolute frequency “uncertainties” as differences between the (p) and (p+4) tracked answers (`docs/derivations/04_spectral_numerics.tex:214`). The production phase quantity then takes the maximum of a common-node grid discrepancy and the largest adjacent-order differences (`docs/derivations/04_spectral_numerics.tex:610`, `docs/derivations/04_spectral_numerics.tex:627`). The implementation does exactly this: `estimate_nested_frequency_error` is a maximum difference (`src/zgv_morse/green_response.py:388`, `src/zgv_morse/green_response.py:396`), and `verify_registered_grid_convergence` multiplies the maximum observed discrepancy by (t_{\max}) (`src/zgv_morse/green_response.py:825`, `src/zgv_morse/green_response.py:830`).

These quantities rigorously bound the difference *between the computed levels*. They do not, without an a posteriori theorem, a verified convergence ratio, interval enclosure, or a monotone error bracket, bound the difference from the continuum eigenfrequency. Spectral stagnation or correlated branch errors can make (\lvert\omega_{p+4}-\omega_p\rvert) small while the common error is not. The main text nevertheless says that this quantity “controls the longest reported time” (`paper/sections/05_numerical_verification.tex:47`) and the abstract calls it a “maximum numerical accumulated-phase bound” (`paper/sections/00_abstract.tex:21`). The helper docstring likewise calls it a “hard accumulated-phase error budget” (`src/zgv_morse/green_response.py:289`).

This is not a sign or factor error, and the reported value is very small. It is a logical overstatement of what the estimator establishes. Before the case is described as phase-certified, either:

- provide a verified continuum enclosure (for example, a validated eigenvalue bound or a posteriori spectral estimate with its hypotheses checked), or
- establish and use a conservative extrapolation bound from a longer order sequence, including a demonstrated asymptotic ratio and a safety factor, or
- consistently rename the quantity as an **accumulated inter-resolution phase-discrepancy estimator** and remove claims that it bounds the unknown continuum phase error.

### Major 2 — “Exactly eight” at the registered finite perturbation is not obtained from the unquantified small-parameter theorem or from the finite grid

**Status: the asymptotic theorem is supported; its finite-\(\varepsilon\) numerical instantiation is weak.**

The cubic splitting theorem proves an exact eight-point count only “for every sufficiently small nonzero (\varepsilon)” (`docs/derivations/02_anisotropic_morse_unfolding.tex:724`, `docs/derivations/02_anisotropic_morse_unfolding.tex:729`). Its proof is sound as a qualitative perturbation argument: it uses a positive lower bound for (\lvert V'\rvert) off eight neighborhoods and an (O_{C^1}(\varepsilon)) correction after division by (\varepsilon) (`docs/derivations/02_anisotropic_morse_unfolding.tex:772`, `docs/derivations/02_anisotropic_morse_unfolding.tex:775`). However, no numerical lower bound, remainder constant, or resulting admissible (\varepsilon_0) is supplied. Thus the theorem does not by itself certify that the registered (\varepsilon=0.02) calculation (`docs/derivations/04_spectral_numerics.tex:768`) lies within its exact-count interval.

The numerical route is strong finite-resolution evidence, but it explicitly cannot exclude an unresolved opposite-index pair (`docs/derivations/04_spectral_numerics.tex:398`, `docs/derivations/04_spectral_numerics.tex:400`). Candidate-grid doubling and symmetry-offset repetition reduce that risk but do not make it logically impossible. Index closure cannot see a newly created (+1/-1) pair. The main text is admirably candid about the limitation (`paper/sections/07_methods.tex:116`), yet elsewhere says the local search “resolved exactly” four minima and four saddles and defines “exactly eight” for the registered annulus (`paper/sections/03_morse_unfolding.tex:75`, `paper/sections/03_morse_unfolding.tex:85`).

To make the exact finite-\(\varepsilon\) claim follow, the authors should supply one of the following:

- an explicit numerical (C^1) remainder bound and the corresponding (\varepsilon_0\) in the proof of the cubic splitting theorem;
- an interval/Krawczyk certification of one root in each of eight boxes plus a global interval exclusion of zeros on the complement of those boxes in the annulus; or
- a weaker formulation such as “eight resolved roots, stable under the declared refinements,” reserving “exactly eight” for the qualitative sufficiently-small theorem.

### Major 3 — The exact-Morse comparison does not use a critical-point exhaustion over the complete response integration domain

**Status: missing committed control; reviewer-side calculation found no counterexample.**

The fixed-anisotropy theorem assumes that every stationary point on the support is isolated and nondegenerate (`docs/derivations/03_green_function_asymptotics.tex:602`, `docs/derivations/03_green_function_asymptotics.tex:607`). The committed critical search uses the annulus

\[
|q|/k_0\le f_A=0.15
\]

(`paper/sections/07_methods.tex:93`, `paper/sections/07_methods.tex:94`; `src/zgv_morse/workflows/green.py:312`), whereas the direct response integrates

\[
|q|\le1.5\sigma, \sigma/k_0=0.15, \text{hence }|q|/k_0\le0.225
\]

(`paper/sections/07_methods.tex:136`, `paper/sections/07_methods.tex:139`; `src/zgv_morse/workflows/green.py:296`, `src/zgv_morse/workflows/green.py:297`). Nevertheless, the Morse contributions are built only from the points found in the narrower annulus (`src/zgv_morse/workflows/green.py:426`, `src/zgv_morse/workflows/green.py:444`, `src/zgv_morse/workflows/green.py:459`). No committed certificate excludes additional stationary points in the two shoulders (0.15<|q|/k_0\le0.225). Such points, if present, would contribute at the same (t^{-1}) order and would have to appear in the exact-Morse sum.

As a reviewer-side read-only check, I repeated the same search and winding test on the expanded (0.225k_0) annulus for (\varepsilon=0.005,0.01,0.02,0.04,0.08). Every row returned four minima and four saddles, noncritical sampled boundaries, and index closure. This makes an actual hidden point unlikely and means the concern should be straightforward to fix. The expanded check is not presently a qualified artifact and remains subject to the same finite-resolution limitation discussed in Major 2.

The production workflow should use a search/certificate domain containing the complete numerical support (preferably with margin), store that result, and use those points in the fixed-Morse sum.

## Minor concern

### Minor 1 — The numerical radial truncation is not the smooth compact cutoff used to justify the stated fixed-\(\varepsilon\) remainder

**Status: leading-order conclusion supported; stated numerical remainder class not directly applicable.**

The asymptotic derivation says that the super-Gaussian is multiplied by a (C^\infty) bump that equals one on the effective numerical window and vanishes before the annular boundary, explicitly to remove endpoint contributions (`docs/derivations/03_green_function_asymptotics.tex:127`, `docs/derivations/03_green_function_asymptotics.tex:130`). The numerical integral instead stops at (q=\pm1.5\sigma) and uses the unmodified weight (\exp[-(q/\sigma)^8]) (`src/zgv_morse/workflows/green.py:369`, `src/zgv_morse/green_response.py:356`, `src/zgv_morse/green_response.py:378`). Its endpoint value is small but nonzero:

\[
\exp[-1.5^8]=7.4046995\times10^{-12}.
\]

Therefore the numerical integral is not literally the compactly supported smooth-amplitude integral for which the displayed (O_\varepsilon(t^{-2})) interior stationary-phase remainder is stated (`docs/derivations/03_green_function_asymptotics.tex:627`). The endpoint term is negligible over the reported window and does not challenge the leading (t^{-1}) result, but the manuscript should either implement the declared bump, quantify the endpoint term over the comparison window, or state that only the leading fixed-Morse term—not its (O(t^{-2})) remainder—is being transferred to the hard-truncated numerical integral.

## Supported, weak, and not-assessable claims

### Supported by the inspected material

- The high-precision isotropic ZGV location and positive radial curvature for the selected branch.
- The Morse--Bott classification of the finite-radius isotropic ring under the stated simplicity/gap hypotheses.
- The generalized-eigenvalue sensitivity formula and inclusion of the differentiated-mode term in (B).
- The pure first-order (V_0+V_4\cos4\theta) structure for the declared linear cubic family and the sign (V_4>0).
- The qualitative sufficiently-small-\(\varepsilon\) four-minimum/four-saddle theorem and role reversal under (\varepsilon\mapsto-\varepsilon).
- The polar-to-Cartesian Hessian/Jacobian conventions and Morse signatures.
- The compact-\(\tau\) Bessel law, the separately stated growing-\(|\tau|\) overlap, the constant-and-phase match to the first-order Morse sum, and the fixed-\(\varepsilon\) stationary-phase formula under their stated analytical hypotheses.
- The factor-of-two distinction between critical-frequency separation and modulation angular rate.

### Weak or overstated in the present evidence chain

- A rigorous continuum accumulated-phase *bound* rather than an inter-resolution discrepancy estimator.
- An exact continuum count at the specific registered finite perturbations.
- Exhaustion of stationary points over the complete domain used in the fixed-Morse direct-response comparison.
- Transfer of the fixed-\(\varepsilon\) (O(t^{-2})) remainder to the hard-truncated numerical integral.

### Not assessable from the provided material

- External priority and the full novelty distinction from all prior ZGV, broken-symmetry, and uniform-asymptotic literature. The manuscript itself frames the endpoint geometries and Bessel identity as prior work, but this review did not conduct an independent literature search.
- Experimental relevance in a finite, lossy specimen. The manuscript explicitly excludes that inference.
- Global completeness across all Lamb and shear-horizontal branches. The manuscript explicitly does not claim it.

## Reproducible verification commands

From the repository root:

```bash
uv run python scripts/check_analytic_identities.py
```

Observed result: `symbolic and asymptotic identities: PASS`, with the reported high-precision ZGV substitution and cubic coefficient closure.

```bash
uv run pytest \
  tests/test_derivation_identities.py \
  tests/test_morse_splitting.py \
  tests/test_asymptotics.py \
  tests/test_green_response.py -q
```

Observed result: `174 passed in 72.88s`.

```bash
uv run python scripts/check_latex_log.py build/paper/supplement.log
uv run python scripts/check_latex_log.py build/paper/main.log
```

Observed result: both log gates passed, so the theorem/equation references inspected here are not unresolved LaTeX references.

The expanded-response-domain check used for Major 3 is reproducible with:

```bash
uv run python - <<'PY'
from zgv_morse.elasticity import cubic_family
from zgv_morse.dispersion import RingAnchoredSpectralEvaluator
from zgv_morse.critical_points import Annulus, locate_critical_points, verify_annular_exhaustion

k0 = 0.8042173193715181
omega0 = 2.8517587749600901
annulus = Annulus(k0, 0.225 * k0)
for epsilon in (0.005, 0.01, 0.02, 0.04, 0.08):
    evaluator = RingAnchoredSpectralEvaluator(
        cubic_family(2.0, 1.0, 1.0, epsilon)[0],
        rho=1.0,
        half_thickness=1.0,
        k0=k0,
        target_omega=omega0,
        order=10,
        num_modes=12,
        angular_sectors=8,
    )
    points = locate_critical_points(
        evaluator, annulus, n_radial=9, n_theta=32, hessian_step=1.0e-3
    )
    report = verify_annular_exhaustion(evaluator, annulus, points, 32)
    print(
        epsilon,
        len(points),
        sum(point.kind == "minimum" for point in points),
        sum(point.kind == "saddle" for point in points),
        report.boundary_is_noncritical,
        report.index_closes,
    )
PY
```

Observed result for every row: `8 4 4 True True`.

## Nature-style criteria

- **Originality.** The manuscript makes a credible internal case that its intended advance is the coefficient-resolved bridge, not discovery of the endpoint geometries or of the Bessel identity. External priority remains not assessable in this technical-only review.
- **Scientific importance.** The local geometric/asymptotic connection is technically meaningful and likely important within guided-wave and applied-asymptotic research. “Outstanding” cross-field importance is not established solely by one branch and one perturbation family.
- **Interdisciplinary readership.** The Morse--Bott-to-Morse and noncommuting-limit structure is potentially interesting beyond Lamb-wave specialists, especially to wave asymptotics and singular perturbation readers. The current application scope remains narrow.
- **Technical soundness.** The analytical derivations are strong. The finite-resolution claims need the three evidence-chain corrections above before “certified”, “bound”, and finite-\(\varepsilon\) “exactly” are fully justified.
- **Readability for nonspecialists.** The main text is disciplined about scope and prior endpoints, and the geometric schematic helps. However, the proof of the central case lives largely in a long supplement; a nonspecialist will not readily distinguish a qualitative sufficiently-small theorem from the finite-resolution certificates unless the terminology is tightened.

## Recommendation posture

**Promising and analytically substantial, but major revision is required before the strongest computational-certification language is established.** The requested changes are primarily about closing hypothesis-to-computation links, not replacing the central theory. The reviewer-side expanded-annulus result and the passing analytic checks suggest that the main conclusions are likely to survive those corrections.
