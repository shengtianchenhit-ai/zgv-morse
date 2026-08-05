# Live literature-search log

## Execution record

- Registered search executed: 2026-07-11 (Asia/Shanghai).
- Citation graphs and DOI metadata refreshed: 2026-07-13 before the
  bibliography audit. This refresh re-resolved the records discovered by the
  registered 2026-07-11 search; it did not silently move the registered-query
  cutoff.
- Sources: Zotero Scholar discovery, OpenAlex works/citation graph, Crossref
  Works metadata, publisher pages, author-hosted manuscripts, arXiv, and PMC.
- Registered-query horizon: records discoverable through 2026-07-11. A 2026 review and
  2026 citing records were screened; reviews and snippets were not used to
  establish technical priority.
- Evidence rule: every technical statement below was checked in a primary full
  text. OpenAlex counts are discovery coverage, not a count of independent
  peer-reviewed results.

## Exact-query execution

| Registered query | Closest retained results | Audit decision |
|---|---|---|
| `"zero group velocity" Lamb wave anisotropic plate` | Prada et al. 2009; Hussain & Ahmad 2012; Karous et al. 2019; Kiefer et al. 2023/2025 | Directional anisotropic ZGV shifts, multiple ZGV points, and the eight silicon points are prior art. |
| `("ZGV" OR "zero group velocity") anisotropy critical point stationary phase` | Kiefer et al. 2023/2025; Velichko & Wilcox 2007; Chapuis et al. 2010; Karmazin et al. 2013 | Ordinary isolated-point and anisotropic far-field stationary phase are prior art; retain only the coalescing critical-manifold temporal uniformization as a candidate distinction. |
| `Morse-Bott stationary phase critical manifold wave decay` | Bott 1954; Banyaga & Hurtubise 2009/2013; Creagh 1996 and Brack et al. 1999 | Morse--Bott theory, perturbative splitting, and family-to-isolated-contribution asymptotics are established mathematics. The manuscript must prove the plate-specific hypotheses and cannot claim the general theorem. |
| `Lamb wave van Hove singularity anisotropic plate` | Van Hove 1953 plus the ZGV local-response/decay papers | No primary Lamb-wave paper located by this wording supplied the proposed weak-anisotropy time bridge. The van Hove connection is background, not a priority claim. |
| `Bessel crossover stationary ring weak anisotropy` | Creagh 1996; Brack, Meier & Tanaka 1999; Brack & Roccia 2009 | A nearly isomorphic `J0` symmetry-breaking uniformization exists in semiclassical periodic-orbit theory. General Bessel uniformization is therefore explicitly excluded from the novelty claim. |
| `"zero group velocity" cubic silicon plate` | Prada et al. 2009; Kiefer, Mezil & Prada 2023/2025 | Cubic-silicon fourfold frequency variation, four minima, four saddles, and beating have already been reported. |

The four registered vocabulary variants for a ring of minima, degenerate ring,
Morse--Bott dispersion, and critical-circle symmetry breaking were also run.
They recovered critical-circle selection in other fields, notably Bressloff
2003, but no verified primary source containing the complete Lamb-ZGV chain
audited here. This is a bounded negative result, not proof of priority.

## Direct ZGV and Lamb-wave precedents

| DOI | Verified contribution | Consequence for this manuscript |
|---|---|---|
| [`10.1063/1.2128063`](https://doi.org/10.1063/1.2128063) | Prada, Balogun & Murray (2005) demonstrated laser generation and detection of finite-wavenumber ZGV Lamb waves. | ZGV localization and excitation are established. |
| [`10.1121/1.2918543`](https://doi.org/10.1121/1.2918543) | Prada, Clorennec & Royer (2008) developed the local plate-vibration description. | Cite for the isotropic local-resonance framework. |
| [`10.1016/j.wavemoti.2007.11.005`](https://doi.org/10.1016/j.wavemoti.2007.11.005) | Prada, Clorennec & Royer (2008), their Eqs. (3)--(6), reduced the isotropic axisymmetric wave-number integral to a spatial factor `J0(k0 r)` and temporal decay proportional to `(D t)^(-1/2)`. | Spatial ZGV Bessel structure, ZGV stationary phase, and `t^-1/2` decay are not new. The candidate result uses the different, temporal argument `J0(epsilon V4 t)`. |
| [`10.1016/j.wavemoti.2014.04.001`](https://doi.org/10.1016/j.wavemoti.2014.04.001) | Laurent, Royer & Prada (2014) independently confirmed the curvature-controlled `t^-1/2` ZGV decay. | Confirms the known isotropic endpoint of the proposed crossover. |
| [`10.1121/1.3167277`](https://doi.org/10.1121/1.3167277) | Prada et al. (2009) measured [001] cubic-silicon ZGV frequencies and amplitudes versus direction, with a 90-degree period. | Cubic silicon, directional ZGV shifts, and fourfold variation are prior art. |
| [`10.1121/1.4730891`](https://doi.org/10.1121/1.4730891) | Hussain & Ahmad (2012) found multiple ZGV points on Lamb branches in an orthotropic plate. | Multiple anisotropic ZGV points are established, although not as a weak splitting of a two-dimensional critical circle. |
| [`10.1139/cjp-2018-0348`](https://doi.org/10.1139/cjp-2018-0348) | Karous et al. (2019) computed multiple ZGV modes along different crystallographic axes. | Axis-dependent multiplicity is not a novelty claim. |
| [`10.1016/j.jsv.2021.116023`](https://doi.org/10.1016/j.jsv.2021.116023) | Glushkov & Glushkova (2021) studied multiple ZGV resonances in layered structures. | Multiple resonances alone do not distinguish the present work. |
| [`10.1121/10.0017252`](https://doi.org/10.1121/10.0017252) | Kiefer et al. (2023) gave globally and locally convergent algorithms for anisotropic ZGV points. | The present solver is validation infrastructure, not a standalone novelty. |
| [`10.1126/sciadv.adk6846`](https://doi.org/10.1126/sciadv.adk6846) | Kiefer, Mezil & Prada (2023) explicitly reported eight isolated silicon ZGV points--four minima and four saddles--and modeled two interfering four-wave sets with a 21.2-microsecond beat period. Searches within the primary full text found no Morse--Bott, perturbative `epsilon -> 0`, or temporal Bessel crossover analysis. | This is the closest direct precedent. The eight points, their type, their silicon realization, and their beating cannot be claimed as discoveries. |
| [`10.1103/PhysRevResearch.7.L012043`](https://doi.org/10.1103/PhysRevResearch.7.L012043) | Kiefer, Mezil & Prada (2025) again used four minima and four saddles and stationary-phase points to explain line-scan multiplicity; the article also uses the word “unfolding.” | Avoid an unqualified claim of a new unfolding or first use of stationary phase. Use the precise phrase “controlled Morse--Bott critical-circle unfolding.” |
| [`10.1063/5.0183620`](https://doi.org/10.1063/5.0183620) | Morales et al. (2024) measured stress-induced directional ZGV shifts for acoustoelastic characterization. | Weak directional shifts are not by themselves new. |
| [`10.1007/s00466-025-02656-8`](https://doi.org/10.1007/s00466-025-02656-8) | Plestenjak, Kiefer & Gravenkamp (2025) improved the computation of ZGV points with a Sylvester-equation method. | Does not supply the topology or uniform time law, but further removes solver novelty. |

## Stationary-phase and symmetry-breaking analogues

Anisotropic Lamb-wave Green functions and spatial far fields already use
stationary phase: Velichko & Wilcox 2007
([`10.1121/1.2390674`](https://doi.org/10.1121/1.2390674)), Chapuis,
Terrien & Royer 2010
([`10.1121/1.3263607`](https://doi.org/10.1121/1.3263607)), Karmazin et
al. 2013 ([`10.1016/j.ultras.2012.06.012`](https://doi.org/10.1016/j.ultras.2012.06.012)),
and Glushkov et al. 2014
([`10.1121/1.4829534`](https://doi.org/10.1121/1.4829534)). These works
bound the novelty of the method but treat spatial far fields, focusing, or
caustics, rather than the source-point joint limit `epsilon -> 0`, `t ->
infinity`, `epsilon t = O(1)`.

The most important cross-domain conflict is semiclassical broken-symmetry
theory. Creagh 1996
([`10.1006/aphy.1996.0051`](https://doi.org/10.1006/aphy.1996.0051))
uniformly connects continuous orbit families to isolated orbits. Brack, Meier
& Tanaka 1999
([`10.1088/0305-4470/32/2/009`](https://doi.org/10.1088/0305-4470/32/2/009))
exhibit a `J0(delta S / hbar)` symmetry-breaking modulation whose large-argument
limit recovers isolated-orbit contributions. Brack & Roccia 2009
([`10.1088/1751-8113/42/35/355210`](https://doi.org/10.1088/1751-8113/42/35/355210))
give another explicit angular-integral realization. Thus neither the Bessel
identity nor the abstract family-to-points bridge is new. The remaining
plate-specific distinction is to derive its argument and prefactor from the
full Lamb dispersion, prove the applicable error window, and validate the
matching to the eight elastic Morse points.

Bressloff 2003
([`10.1016/S0167-2789(03)00238-0`](https://doi.org/10.1016/S0167-2789(03)00238-0))
also breaks an isotropic critical circle by periodic anisotropy and derives
amplitude equations. This prevents any broad claim to invent “critical ring +
discrete symmetry breaking + amplitude equation.” The present first paper is
therefore confined to the linear Lamb-wave Green response; nonlinear amplitude
equations remain a separate project. The broader pattern-selection context was
also checked against Cross & Hohenberg 1993
([`10.1103/RevModPhys.65.851`](https://doi.org/10.1103/RevModPhys.65.851)).

## Forward-citation audit

The refreshed OpenAlex records reported:

- the 2023 Science Advances paper: 14 citing records. Three are SSRN or
  duplicate preprint records, so this is not “14 independent peer-reviewed
  papers.” The closest technical descendant is the 2025 Physical Review
  Research paper; the rest are solvers, material characterization, nonlinear
  generation, tapered-waveguide, thin-film, or inspection applications;
- the 2025 Physical Review Research paper: four citing records--an NDT&E
  application, a piezoelectric spectral-method paper, the Sylvester solver, and
  a 2026 review.

No screened citing record supplied a Morse--Bott critical-circle reduction or
the `epsilon t` temporal crossover. Aggregate OpenAlex records listed 53 and 27
reference links, respectively; the discovery interface exposed 43 and 25
metadata-bearing reference rows, all of which were screened. This coverage
difference is recorded rather than silently equating database identifiers with
resolved papers.

## Machine bibliography audit

On 2026-07-13, `paper/references.bib` contained 39 unique DOI-backed records.
The online Crossref audit resolved every DOI and compared title, year,
container title, and ordered author-family sequence after documented harmless
typography normalization. Its embedded SHA-256 digest binds the JSON to the
exact BibTeX bytes:

```text
entries=39 mismatch=0 duplicate_doi=0 invalid_doi=0 manual_needed=0 missing_required=0
```

The full per-entry values and comparisons are stored in
`docs/literature/citation_audit.json`.

## Search conclusion and allowed wording

The search did not verify a prior work combining all four plate-specific
elements: a Lamb-ZGV critical circle, full-elastic closure of the cubic `V4`
coefficient, controlled splitting into four minima and four saddles, and the
uniform temporal factor `J0(epsilon V4 t)`. This supports only the bounded
sentence:

> We connect the known isotropic ZGV critical ring and the known eight
> anisotropic silicon ZGV points through a controlled Morse--Bott unfolding,
> close the fourfold coefficient from the full elastodynamic problem, and
> derive and validate the associated uniform weak-anisotropy time crossover.

No occurrence of “first,” “first-ever,” “new Bessel identity,” or “discovery of
eight ZGV points” is authorized by this audit.
