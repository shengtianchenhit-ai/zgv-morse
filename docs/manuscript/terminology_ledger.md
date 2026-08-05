# Manuscript terminology and claim-boundary ledger

Status: locked for the first-paper manuscript, supplement, captions, and
generated tables. This file is the terminology source of truth. A later rename
must be applied globally and recorded here before manuscript prose changes.

## One-sentence argument

In a lossless elastic plate, we show that weak cubic anisotropy unfolds an
isotropic ZGV Morse--Bott ring into four minima and four saddles and changes the
local temporal decay from \(t^{-1/2}\) to \(t^{-1}\), using full-elastic
sensitivities, controlled critical-point geometry, uniform Bessel asymptotics,
and resolution-checked spectral computations, within a declared local annulus and the
controlled weak-anisotropy regime.

## Canonical terminology

| Canonical term | First-use form / notation | Operational meaning | Mandatory boundary |
|---|---|---|---|
| zero-group-velocity (ZGV) | zero-group-velocity (ZGV) Lamb wave | A tracked Lamb-wave state satisfying \(\nabla_{\mathbf k}\omega=\mathbf 0\). After first use, use **ZGV**, not a changing mixture of “zero group velocity”, “zero-group speed”, and “stationary wave”. | ZGV is a dispersion property. It does not by itself imply a globally trapped mode, damping, or an experimental observation. |
| ZGV critical ring | ZGV critical ring \(\Gamma_0=\{\mathbf k:\lvert\mathbf k\rvert=k_0\}\) | The one-dimensional circle of critical points of the selected isotropic branch in the full two-dimensional wavevector plane. | It is one Morse--Bott critical manifold, not infinitely many unrelated isolated ZGV points. “Ring” refers to wavevector space, not a finite specimen. |
| Morse--Bott critical manifold | Morse--Bott critical manifold | A critical manifold whose Hessian kernel is exactly its tangent space and whose normal Hessian is nondegenerate. For \(\Gamma_0\), the radial eigenvalue is \(a>0\) and the tangential eigenvalue is zero. | This is a plate-specific verification and organizing classification, not a new general Morse--Bott theorem or discovery of the familiar isotropic ZGV circle. |
| Morse minimum | Cartesian Morse minimum | An isolated critical point whose Cartesian \((k_x,k_y)\) Hessian is positive definite; its gradient index is \(+1\). | Classify with the uncertainty-resolved Cartesian Hessian, never from a contour image or from angular curvature alone. |
| Morse saddle | Cartesian Morse saddle | An isolated critical point whose Cartesian \((k_x,k_y)\) Hessian has one positive and one negative eigenvalue; its gradient index is \(-1\). | Do not call the negative-angular-curvature point a maximum: the retained positive radial curvature makes it a saddle. |
| gradient index | gradient (Poincaré) index | The local vector-field index of \(\nabla_{\mathbf k}\omega\): \(+1\) for each resolved minimum and \(-1\) for each resolved saddle in this problem. | The zero index sum is checked against sampled boundary winding in the declared annulus. This finite-resolution check cannot exclude an unresolved opposite-index pair and must not be attributed solely to annular topology. |
| angular potential | angular potential \(V(\theta)=\partial_\varepsilon\omega(k_0,\theta;0)\) | The first-order frequency-sensitivity function on the isotropic ring. | \(V\) is a coefficient of \(\varepsilon\), not the finite-\(\varepsilon\) physical frequency shift. |
| fourfold coefficient | signed fourfold coefficient \(V_4\) in \(V(\theta)=V_0+V_4\cos4\theta\) | The coefficient obtained from the full-elastic generalized-eigenvalue sensitivity for the registered first-order cubic family. | Preserve its sign. Do not label bare \(V_4\) as a physical frequency shift and do not infer an eight-point count from \(C_{4v}\) symmetry alone. |
| physical fourfold shift | signed physical fourfold shift \(\varepsilon V_4\) | The leading finite-perturbation angular frequency shift; it controls the angular role assignment and the Bessel argument. | Count \(\varepsilon\) exactly once. Reversing \(\varepsilon V_4\) exchanges minimum and saddle angular roles. |
| radial sensitivity coefficient | radial sensitivity coefficient \(B(\theta)=\partial_k\partial_\varepsilon\omega(k_0,\theta;0)\) | The coefficient of \(\varepsilon q\) in the local normal form, including the differentiated-mode or reduced-resolvent contribution. | It is not a fixed-mode strain integral and is not fitted from the final critical-point locations. |
| minimum--saddle critical-frequency separation | nonnegative minimum--saddle critical-frequency separation \(\Delta\omega_{\mathrm{ms}}=\lvert\omega_{\mathrm{s}}-\omega_{\mathrm{m}}\rvert\) | The separation of the two critical-frequency features; \(\Delta\omega_{\mathrm{ms}}=2\lvert\varepsilon V_4\rvert+O(\varepsilon^2)\) in the controlled weak-anisotropy regime. | It is nonnegative and is not itself the modulation angular-frequency magnitude. |
| modulation angular-frequency magnitude | nonnegative modulation angular-frequency magnitude \(\Omega_{\mathrm{mod}}=\Delta\omega_{\mathrm{ms}}/2\) | The magnitude multiplying time in the two-family signed cosine modulation; at leading order \(\Omega_{\mathrm{mod}}=\lvert\varepsilon V_4\rvert\). | State the factor of two explicitly. Do not use unqualified “beat frequency”. |
| critical-frequency feature | critical-frequency feature | A feature associated with an exact minimum or saddle frequency in the branch-resolved infinite, lossless continuum. | Use this instead of “spectral line”; the calculation does not establish a discrete damped resonance line. |
| undoubled positive-frequency response | undoubled positive-frequency response \(G^{+}(t;\varepsilon)\) | The branch-projected positive-frequency contribution with the normalization derived in Eq. `eq:branch-projected-response`. | Do not call \(G^{+}\) an analytic signal. The physical real response is reconstructed as \(G_{\mathrm{phys}}=2\operatorname{Re}G^{+}\); the factor of two is not absorbed into \(G^{+}\). |
| declared local annulus | declared local annulus \(\mathcal N\) | The computational neighbourhood of the selected isotropic ZGV ring in which boundary noncriticality, Hessian classification, grid refinement, and index consistency are checked. | At finite \(\varepsilon\), say **eight resolved, refinement-stable roots in \(\mathcal N\)**. Reserve exact four-plus-four cardinality for the sufficiently-small theorem; never imply global enumeration. |
| controlled weak-anisotropy family | controlled one-parameter weak-cubic-anisotropy family | The bulk-modulus-preserving cubic perturbation used to test the asymptotic theorem as \(\lvert\varepsilon\rvert\ll1\). | The theorem requires the stated eigengap, regularity, nonzero leading harmonic, nonnodal response, and validity-window hypotheses. |
| uniform Bessel crossover | coefficient-resolved uniform Bessel crossover | The joint-limit law with \(t\to\infty\), \(\varepsilon\to0\), and \(\tau=\varepsilon V_4t=O(1)\), together with its controlled growing-\(\lvert\tau\rvert\) overlap. | “Universal” may be used only with the source/weight class and asymptotic window stated. The Bessel identity and general broken-symmetry uniformization are established mathematics. |
| silicon stress test | finite-anisotropy silicon stress test | A separate robustness calculation using a physical [001]-cut silicon stiffness, reported in Supplementary Figure S06. | Silicon is outside the proof role: it neither proves nor validates the weak-anisotropy theorem and supplies no global eight-point claim. |

## Prohibited or qualification-required language

| Prohibited wording | Required replacement or action |
|---|---|
| `spectral line` | Use **critical-frequency feature** and state that the model is an infinite lossless continuum. |
| unqualified `beat frequency` | Report \(\Delta\omega_{\mathrm{ms}}\) and \(\Omega_{\mathrm{mod}}=\Delta\omega_{\mathrm{ms}}/2\) separately. |
| unqualified `universal` | State the source/weight class, distinguished limit, and phase-validity window, or use **coefficient-resolved uniform crossover**. |
| `first ever`, `first observation`, `first computation`, `discovery`, or equivalent priority language | Delete. The literature already reports isolated anisotropic ZGV points, four-minimum/four-saddle patterns, splitting, and beating. |
| `new Bessel identity` | Delete. Claim only the coefficient-resolved Lamb-wave derivation and validation within the stated limit. |
| `topological phase transition` | Use **local Morse--Bott-to-Morse unfolding** or **change in local critical-set dimension and organization**. |
| `silicon proves the theorem` or `silicon validates the weak-anisotropy theorem` | Use **finite-anisotropy silicon stress test** and keep it outside theorem evidence. |
| `analytic signal` for \(G^{+}\) | Use **undoubled positive-frequency response** and state the real-response reconstruction. |
| `exactly eight` for a finite-resolution computation | Use **eight resolved, refinement-stable roots in the declared local annulus**. Reserve exact four-plus-four cardinality for the sufficiently-small theorem. |
| any nonlinear amplitude-equation, saturation, mode-coupling, or pattern-selection result | Remove from this paper; nonlinear amplitude equations belong to the separate second paper. |
| `experimental validation`, `measured`, or `observed here` | Use **computed**, **derived**, or **numerically verified** as appropriate; this paper contains no experiment. |

## Novelty boundary

The manuscript may claim a controlled plate-specific connection, full-elastic
coefficient closure, and a uniformly matched temporal crossover. It must not
claim discovery of anisotropic ZGV points, the known four-minimum/four-saddle
pattern, ZGV beating, isotropic \(t^{-1/2}\) decay, two-dimensional stationary
phase, or the general Bessel identity. The dated literature audit must be
refreshed immediately before submission; until then, no priority wording is
authorized.
