# Main-figure contracts

These contracts fix the scientific argument before plotting. All drawing,
previewing, export, and visual quality assurance use Python/matplotlib with the
non-interactive Agg backend. Every panel is generated from a strictly validated
artifact and has a machine-readable source-data CSV; no numerical plot value is
entered by hand. The shared visual vocabulary is blue circles for minima and
orange diamonds for saddles, so Morse class remains legible without colour.

## Figure 1 — Conceptual geometry and mechanism overview

- **Core conclusion:** anisotropy lifts the tangential Hessian kernel, changing a one-dimensional stationary set into isolated points.
- **Archetype:** `schematic-led composite`.
- **Target/output:** Physical Review Research-ready, journal-neutral two-column composite; editable SVG master, PDF, PNG, and TIFF derivatives; raster exports at 600 dpi.
- **Final size:** 183 mm wide × 118 mm high.
- **Panel map:** a, isotropic local contours and the ZGV critical ring; b, weakly anisotropic local contours with four minimum circles and four saddle diamonds in the declared annulus; c, conceptual codimension map from a one-dimensional stationary ring to isolated stationary points and from `t^-1/2` to `t^-1` decay.
- **Hero evidence:** Panel a-to-b is the dominant mechanism overview: the continuous ring and the eight resolved local Morse points share the same axes, annulus, angular origin, and visual vocabulary.
- **Validation evidence:** Panel b uses the refined full-wave coordinates rather than points inferred from contour pixels, but Figure 1 does not carry the numerical-realization claim. Figure 4 reports the finite-resolution search, Hessian classification, index consistency, and stability evidence. Panel c is explicitly conceptual and does not substitute for the two numerical slices in Figure 6.
- **Source data:** `isotropic_zgv`, `angular_sensitivity`, and `critical_points`; `panel_a_ring.csv`, `panel_b_surface.csv`, and `panel_b_points.csv` contain exactly the arrays used by the axes.
- **Statistics:** The displayed resolved set has `n = 8` in the declared annulus (four minima and four saddles), total gradient index zero, Hessian signs and gradient-residual tolerances reported; no inferential significance test or fitted parameter is used.
- **Image-integrity:** Analytic geometry and contours are rendered directly from validated arrays. No point is moved for legibility, no count is inferred from a raster image, and colour is backed by circle/diamond marker redundancy.
- **Reviewer risk:** Do not imply discovery of the known four-minimum/four-saddle pattern, exhaustive finite-perturbation enumeration, or a global count. Describe the stationary-set dimension and Hessian-rank change, and keep this mechanism overview distinct from the full-wave realization in Figure 4.

## Figure 2 — Exact isotropic ZGV foundation

- **Core conclusion:** the chosen ZGV ring has a nondegenerate positive radial curvature.
- **Archetype:** `quantitative grid`.
- **Target/output:** Physical Review Research-ready, journal-neutral two-column composite; editable SVG master, PDF, PNG, and TIFF derivatives; raster exports at 600 dpi.
- **Final size:** 183 mm wide × 132 mm high.
- **Panel map:** a, the selected symmetric full-wave branch with the exact Rayleigh–Lamb `(k0, omega0)` marker; b, the local full-wave branch against `omega0 + a q^2/2`; c, `k0`, `omega0`, and curvature errors versus polynomial order; d, normalized displacement components and the squared-displacement proxy through `z/h`.
- **Hero evidence:** Panel b is the primary test: independently stored local branch values are compared with the coefficient-level quadratic prediction, with positive `a` and no plotting-time fit.
- **Validation evidence:** Panel c exposes convergence of the ZGV coordinates, curvature, eigen residual, Hermitian residual, mass orthogonality, and eigengap. Panels a and d verify branch identity and modal regularity at the selected point.
- **Source data:** `isotropic_zgv` and `convergence`; one source-data CSV for each panel, containing the stored branch, local, convergence, and through-thickness arrays used in the plot.
- **Statistics:** Report the final-two-order changes and registered numerical tolerances, the minimum relative eigengap, and the curvature uncertainty. The local quadratic is an a priori formula, not a regression, so no fit confidence interval is applicable.
- **Image-integrity:** Curves and profiles are drawn directly from validated machine-readable arrays. The quadratic array is checked against `omega0 + 0.5*a*q^2` before rendering; no smoothing, resampling, or manual ZGV placement is allowed.
- **Reviewer risk:** A skeptical reader may attribute the minimum to branch switching or polynomial truncation. Keep branch labels, eigengap, residuals, convergence, and the no-fit construction visible enough to rule out that interpretation.

## Figure 3 — Elastic perturbation and angular potential

- **Core conclusion:** cubic anisotropy produces the predicted fourfold unfolding potential.
- **Archetype:** `quantitative grid`.
- **Target/output:** Physical Review Research-ready, journal-neutral two-column composite; editable SVG master, PDF, PNG, and TIFF derivatives; raster exports at 600 dpi.
- **Final size:** 183 mm wide × 125 mm high.
- **Panel map:** a, hero polar comparison of `V(theta)-V0` and `V4 cos(4 theta)`; b, harmonic amplitudes with `m = 4` highlighted and leakage retained; c, the physical shift `epsilon V4` versus `Delta_C(epsilon)` with the predicted `Q4` slope; d, analytic sensitivity values `V, B` against stored centered finite differences.
- **Hero evidence:** Panel a makes the fourfold potential and angular phase immediately visible while panel b demonstrates that the reconstruction is supported by the harmonic content, not imposed by styling.
- **Validation evidence:** Panel d supplies an independent finite-difference check for both first-order coefficients and displays the convergence plateau. Panel c tests the physical perturbation convention without fitting a hidden amplitude or double-counting epsilon.
- **Source data:** `angular_sensitivity` and `convergence`; panel-level CSV files contain `theta`, `V`, `V_reconstruction`, harmonics, `epsilon`, `delta_c`, `physical_V4_shift`, `V_fd`, `B_fd`, and step-convergence errors exactly as plotted.
- **Statistics:** Report maximum or RMS analytic-versus-finite-difference errors, the resolved step plateau, harmonic leakage relative to `|V4|`, and the coefficient-level residual of `epsilon V4 = Q4 Delta_C`; no favourable subset may be selected at plotting time.
- **Image-integrity:** Polar and Cartesian panels reuse the same angular origin and semantic colours. No harmonic is removed, no finite difference is recomputed by plotting code, and no axis may label bare `V4` as the physical frequency shift.
- **Reviewer risk:** The principal risk is convention ambiguity between `V4`, `epsilon V4`, `Delta_C^(1)`, and `Delta_C(epsilon)`. Labels and source columns must make `V4 = Q4 Delta_C^(1)` and `epsilon V4 = Q4 Delta_C` unambiguous.

## Figure 4 — Full-wave numerical realization and Morse points

- **Core conclusion:** the full three-dimensional elastic calculation resolves a stable eight-root set with the Morse classes predicted by the sufficiently-small theorem.
- **Archetype:** `quantitative grid`.
- **Target/output:** Physical Review Research-ready, journal-neutral two-column composite; editable SVG master, PDF, PNG, and TIFF derivatives; raster exports at 600 dpi.
- **Final size:** 183 mm wide × 132 mm high.
- **Panel map:** a, isotropic local surface or contours with the critical ring; b, hero anisotropic contours with eight resolved roots in the declared local annulus; c, both Cartesian Hessian eigenvalues for every point, encoded by Morse colour and marker; d, full-wave minus perturbative location and frequency errors.
- **Hero evidence:** Panel b shows the resolved, refinement-stable local set on the full three-dimensional elastic branch; blue circles and orange diamonds establish the alternating geometric organization without relying on colour alone.
- **Validation evidence:** Panel c makes classification by Hessian inertia explicit, while panel d quantifies agreement with the perturbative theorem. Pre-render checks require alternating type, index sum zero, nonzero Hessian eigenvalues, bounded gradient residuals, a noncritical boundary, and the registered annulus width `0.15 k0`.
- **Source data:** `isotropic_zgv` and `critical_points`; source CSVs preserve the two local surfaces, contour axes, refined and predicted coordinates/frequencies, Hessian eigenvalues, Morse indices, kinds, and gradient residuals.
- **Statistics:** Report four minima, four saddles, total index zero, minimum Hessian-eigenvalue-to-uncertainty ratio, maximum normalized gradient residual, boundary-gradient margin, and full-versus-predicted coordinate and frequency errors.
- **Image-integrity:** Critical symbols are placed only at stored refined coordinates. Classification never comes from visual contour shape; no interpolation or cosmetic displacement changes a point, and markers remain redundant with colour.
- **Reviewer risk:** Say “eight resolved roots in the declared local annulus,” not “exactly eight at finite perturbation.” The figure supplies finite-resolution realization and Hessian classification; exact four-plus-four cardinality belongs to the sufficiently-small theorem.

## Figure 5 — Scaling and error laws

- **Core conclusion:** the computed unfolding is genuinely perturbative and quantitatively predicted.
- **Archetype:** `quantitative grid`.
- **Target/output:** Physical Review Research-ready, journal-neutral two-column composite; editable SVG master, PDF, PNG, and TIFF derivatives; raster exports at 600 dpi.
- **Final size:** 183 mm wide × 118 mm high.
- **Panel map:** a, minimum–saddle frequency splitting versus `|epsilon|` with the stored first-order prediction; b, compensated `q_min/epsilon` and `q_saddle/epsilon` limits; c, minimum and saddle frequency errors divided by `epsilon^2`; d, angular-role reversal under `epsilon -> -epsilon`.
- **Hero evidence:** Panels b and c are the primary evidence because converged compensated limits test the coefficient predictions directly; the log–log slope in panel a is a secondary diagnostic.
- **Validation evidence:** Stored full-wave and perturbative arrays are shown together across the predeclared epsilon sequence. The sign-reversal panel checks a symmetry consequence independent of the positive-epsilon scaling plots.
- **Source data:** `perturbation_scaling` and `angular_sensitivity`; panel CSVs include every epsilon, full and predicted splitting, radial shifts, frequency errors, compensated quantities, stored slopes, and role-reversal angles used by the axes.
- **Statistics:** Require `|slope_splitting - 1| <= 0.1` and `|slope_remainder - 2| <= 0.2`, quantify convergence of compensated limits and their difference from `-B(theta_j)/a`, and display both minimum and saddle remainders. Fits and masks are fixed upstream, not chosen by the plot.
- **Image-integrity:** Plotting code performs no regression, masking, extrapolation, or value entry. If the linear radial coefficient is unresolved or zero, the panel and claim must be replaced by the separately derived first nonzero order rather than visually forcing a linear trend.
- **Reviewer risk:** A reviewer may suspect cherry-picked log–log ranges. Lead with compensated limits, expose the complete registered epsilon sequence, distinguish predictions from full-wave points, and keep the sign-reversal control visible.

## Figure 6 — Two asymptotic slices and their proved overlap

- **Core conclusion:** a joint-limit Bessel slice and a fixed-anisotropy Morse slice are connected by the separately proved growing-`|tau|` overlap.
- **Archetype:** `asymmetric mixed-modality figure`.
- **Target/output:** Physical Review Research-ready, journal-neutral two-column composite; editable SVG master, PDF, PNG, and TIFF derivatives; raster exports at 600 dpi.
- **Final size:** 183 mm wide × 155 mm high.
- **Panel map:** a, hero real part and magnitude of the scaled full response against `J0(tau)` without fitted alignment; b, scaled absolute complex collapse error including Bessel-zero neighbourhoods; c, separate joint-limit and fixed-`epsilon` envelope slices; d, theory-centred crossover-time consistency diagnostic; e, fixed-`epsilon` full integral against the exact-Morse stationary sum; f, critical-frequency features and signed modulation.
- **Hero evidence:** Panel a is the primary numerical test: the full complex joint-limit response collapses across multiple epsilon values without fitted alignment. Panel c shows two different asymptotic slices, not one numerical trajectory; their connection is supplied by the proved growing-`|tau|` overlap.
- **Validation evidence:** Panel b quantifies the primary collapse accuracy. Panel d is secondary because its search bracket is centred on the predicted inverse-rate time. Panel e separately tests the fixed-`epsilon` exact-Morse limit; panel f connects the minimum–saddle feature separation `2|epsilon V4|` to the signed modulation angular rate `|epsilon V4|`.
- **Source data:** `green_crossover`, `angular_sensitivity`, and `convergence`; source CSVs contain all plotted time, epsilon, complex-response components, `tau`, `J0`, envelopes, masks, stored slopes, crossover times, spectrum coordinates, critical frequencies, and phase errors.
- **Statistics:** Require early and late slopes within 0.05 of `-1/2` and `-1`, respectively; report the maximum scaled absolute complex collapse error, exact-Morse RMS discrepancy, quadrature/interpolation convergence, and the accumulated inter-resolution phase-discrepancy estimator against its configured `0.05` threshold. The crossover-time slope is a consistency diagnostic, not an independent scaling measurement.
- **Image-integrity:** Complex values are exported as explicit real, imaginary, and magnitude columns. Bessel-zero neighbourhoods use scaled absolute complex error rather than ill-conditioned relative error; declared masks and fit windows are read from the artifact and never redrawn by eye.
- **Reviewer risk:** Make “two numerical slices plus a proved overlap” unmistakable. Do not imply that panel c is one direct numerical crossover trajectory or that the phase-discrepancy estimator bounds continuum error. Call panel-f peaks “critical-frequency features,” not unqualified spectral lines, and distinguish separation `2|epsilon V4|` from modulation rate `|epsilon V4|`.
