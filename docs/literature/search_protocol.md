# Live literature-search protocol

## Registration

- Execution date: 2026-07-11 (Asia/Shanghai).
- Scope: zero-group-velocity (ZGV) Lamb waves, anisotropic ZGV points,
  symmetry breaking of an isotropic critical ring, Morse--Bott stationary
  phase, and the temporal crossover from a critical manifold to isolated
  Morse points.
- Databases and discovery interfaces: Crossref Works (DOI metadata), OpenAlex
  (backward/forward citation graphs), the Zotero Scholar discovery interface,
  and general web search for exact-query recall. The final bibliography is
  resolved independently through the Crossref REST API.
- Primary evidence route: publisher records and full texts, author-hosted
  accepted manuscripts, or corresponding arXiv preprints. Technical claims are
  accepted only after one of these primary records has been inspected.
- Secondary discovery route: backward and forward citation graphs from the
  2023 and 2025 Kiefer--Mezil--Prada papers. Search snippets are discovery aids,
  not evidence.

The coordinated academic-search workflow was executed with the sources above:
primary records first for claims, citation graphs for expansion, and Crossref
for a field-by-field metadata audit. Search-result snippets were never used as
the sole support for a technical or priority statement.

## Exact registered queries

```text
"zero group velocity" Lamb wave anisotropic plate
("ZGV" OR "zero group velocity") anisotropy critical point stationary phase
Morse-Bott stationary phase critical manifold wave decay
Lamb wave van Hove singularity anisotropic plate
Bessel crossover stationary ring weak anisotropy
"zero group velocity" cubic silicon plate
```

Additional analogue searches preserve the concepts while varying vocabulary:

```text
"ring of minima" anisotropy Bessel time decay dispersion
"degenerate ring" dispersion anisotropy stationary phase Bessel
"Morse-Bott" dispersion relation anisotropy wave
"critical circle" stationary phase Bessel symmetry breaking dispersion
```

## Inclusion and exclusion rules

Include a work when it treats at least one of the following at theorem,
derivation, numerical, or experimental level:

1. a finite-wavenumber ZGV mode or a two-dimensional guided-wave dispersion
   critical point;
2. anisotropic splitting or classification of ZGV extrema and saddles;
3. transient decay caused by a stationary ring or isolated two-dimensional
   stationary points;
4. a uniform asymptotic transition controlled by weak angular symmetry
   breaking;
5. a directly transferable Morse--Bott perturbation or clean stationary-phase
   result.

Exclude papers that use only a zero *phase* velocity, a cutoff at zero
wavenumber without relevance to the finite-wavenumber ring, or the words
"critical ring" in an unrelated geometric sense. Reviews may orient the
search, but do not establish novelty or technical claims.

## Citation-graph procedure

1. Verify the five seed DOIs in `seed_dois.txt`.
2. Inspect all references of DOI `10.1126/sciadv.adk6846` returned by the
   citation graph; retain every anisotropic-ZGV, transient-decay, spectral
   discretization, and silicon-elasticity work.
3. Inspect references of DOI `10.1103/PhysRevResearch.7.L012043` for
   stationary-phase and anisotropic-wave analogues.
4. Inspect every citing record returned by OpenAlex for both papers through the
   execution date; do not equate database coverage with all citations that may
   exist in the literature.
5. Run the registered cross-domain searches for Morse--Bott and Bessel
   analogues.
6. For each proposed novelty statement, record the closest verified overlap
   and narrow the statement if any earlier work contains the same result.

No unqualified priority phrase ("first", "first-ever", or equivalent) is
permitted solely because a query returned no hit.

## Reproducible bibliography audit

Every cited entry must contain author, title, journal, year, and a syntactically
valid unique DOI. The online audit resolves each normalized DOI with Crossref
and requires exact normalized agreement of title, publication year, and
container title and ordered author-family sequence. The audit embeds a SHA-256
digest of the exact BibTeX bytes and the test suite checks the complete
key/normalized-DOI set against the current file. A network error, incomplete
remote record, duplicate DOI, invalid DOI, field mismatch, or stale BibTeX hash
is a hard failure, never an implicit pass. The machine-readable execution
record is `citation_audit.json`.

The ranked search interfaces do not expose a stable total hit count or a
complete export for general-web queries. `search_candidates.csv` therefore
records the returned scope honestly, maps every retained closest result to its
query and DOI/OpenAlex identifier, and never represents a ranked sample as an
exhaustive database result. `exclusions.csv` records the screened close false
positives and duplicate/preprint records with reasons.
