# Coverage Report — PROVISIONAL 63-gene universe (review required, no training)

Date: 2026-10-11. Artifacts: `target_universe_draft_v2.csv` (63 rows x 30 cols),
`verified_pairs_inventory.csv` (13 rows), `target_universe_draft.csv` + `README.md`
(v1, superseded by v2 except selection criteria text). Raw API snapshots (audit):
temp `hgnc_63_raw.json`, `mygene_go_63_raw.json` (re-fetchable; sources below).

## 1. Identifier verification (Requirement 3) — COMPLETE

- Source: HGNC REST `rest.genenames.org/fetch/symbol/{SYM}` + MyGene.info v3
  (aggregates NCBI/HGNC/UniProt/GO), both queried 2026-10-11.
- 63/63 `id_status=verified`, `canonical_symbol_match=yes`; every row carries
  `hgnc_id` (e.g. EGFR HGNC:3236), `entrez_id` (e.g. 1956), `uniprot_swissprot`
  (e.g. P00533), `id_source`, `aliases`, `former_symbols`.
- Synonym verification (Requirement 2): VSIR prev=C10orf54 (HGNC:30085),
  NECTIN4 prev=PVRL4 (HGNC:19688), MS4A1 prev=CD20 (HGNC:7315), TNFRSF17
  prev=BCMA — all from HGNC records, not assumed. Canonical symbols retained;
  former symbols used only for Gene2Vec lookup (manifest updated accordingly).

## 2. Membrane / accessibility (Requirement 4) — DOCUMENTED per gene

- `go_cc_terms` lists verbatim MyGene GO-CC terms per gene; `membrane_go_class`
  from a fixed rule (plasma-membrane ID set in build script).
- 60/63 `membrane-go`; 0 rows with no annotation at all.
- Exceptions retained with mechanism + feature-compatibility notes
  (`exception_mechanism`, `target_accessibility`):
  - VEGFA, ANGPT2 = `secreted-ligand`: soluble ligands addressed by
    sequestration/blockade arms (faricimab/AK112 precedent). mRNA-based
    co-expression features stay computable but do NOT imply same-cell
    co-binding — interpret cautiously. Safety/pathway/embedding unaffected.
  - PMEL = `melanosomal-antigen`: premelanosome protein; clinical precedent is
    TCR-based tebentafusp (peptide-HLA), not antibody surface binding.
    Conventional BsAb formats unsuitable; feature set does not model HLA
    (recorded limitation).

## 3. Expression coverage (Requirement 5) — NOT claimed

- HNSCC: 63/63 observed in `data/processed/hnscc_raw.h5ad` (33,545-gene index).
- NSCLC: pending (matrix is R/RDS `GSE131907...matrix.rds.gz`, 866 MB; needs
  R-mapped gene list; annotation file confirms tumor+normal samples exist).
- Pancreatic: pending (`GSE205013_RAW.tar`, 2.1 GB 10x, unprocessed).
- Liver: pending (`GSE156625_HCCscanpyobj.h5ad.gz`, 839 MB, unprocessed).
- Union rule applied; the four-tumor check has NOT passed.

## 4. Clinical evidence (Requirement 6) — preserved and graded

- `clinical_source_url` kept for all 22 verified rows (FDA spotlight, evitria,
  PMC10618718, paper DOI); `clinical_status` in {verified-fda, verified-paper,
  provisional-discussed, none-claimed}. 41 rows claim nothing.

## 5. Training-data sufficiency (Requirement 7) — INSUFFICIENT, do not train

Available verified pair records (`verified_pairs_inventory.csv`): 13 rows but
only ranks {0, 1}: twelve approved (10 FDA incl. 1 self-pair ERBB2-ERBB2
unsuitable for diff-training, 1 non-oncology VEGFA-ANGPT2, 1 TCR-format
PMEL-CD3E; 2 China-NMPA) + ONE Phase-3 row (CD274-CTLA4, paper-stated,
unverified). Paper Table 6 scores/stages are extraction-ambiguous (its
"0 0 4 4 2 0 0 4 4" row conflicts with the text stating CD40+EGFR/CD3+ERBB3
have no clinical BsAbs) and are EXCLUDED as labels. 8/12 approved pairs share
the CD3E arm (arm-level confounding). Conclusion: no valid multi-rank
comparison set exists yet. Next step when approved: draft a verified
multi-stage clinical CSV (target1, target2, indication, stage, evidence ID)
for row-by-row verification — the 63-gene list is the feature universe, NOT
the label set.

## Review questions

1. Accept v2 (+3 exceptions documented) as the provisional universe?
2. Accept HGNC-verified former-symbol Gene2Vec mapping for VSIR/NECTIN4?
3. Proceed to draft the multi-stage clinical CSV next (no feature computation)?

## 6. Scope restriction: exactly four tumor types (added 2026-10-11)

- Supported: NSCLC, pancreatic cancer, HNSCC, liver cancer as hepatocellular
  carcinoma (HCC) — the subtype of the selected dataset GSE156625 (Sharma et
  al. 2020; 43 tumor + 14 adjacent samples, 14 patients; GEO
  https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156625). Fetal/mouse
  samples in that series are OUT of scope for BSPAI processing.
- Code audit 2026-10-11: no tumor-type list is hard-coded anywhere (`indication`
  is free text in `backend/app/schemas/targets.py` and
  `backend/app/models/database.py`; no tumor constants in `ml/`,
  `feature_engineering/`, `rag/`). No scope creep in code; scope is enforced
  in curation (this report + `clinical_pairs_draft.csv`).
- `clinical_pairs_draft.csv` (15 rows): every record carries `indication_scope`
  (`in-scope:NSCLC|pancreatic` vs `out-of-scope:<heme|SCLC-not-NSCLC|biliary|
  uveal-melanoma|cervical|non-oncology>`) and `eligible_training` (yes/no +
  reason). Out-of-scope oncology records are kept ONLY as identified
  clinical-development evidence, never as supported types. Non-oncology
  (faricimab), incompatible format (tebentafusp TCR), self-pair (zanidatamab),
  and unverified rows (KN046, ivonescimab) are excluded from eligibility.
- Impact of restriction on pairwise comparisons (computed, same method as
  `ml/data/pairwise.py`: distinct-rank combos):
  - Full 15 rows: 26 combos, but ALL rank contrast rests on 3 unverified rows.
  - In-scope 6 rows: 8 combos, same unverified dependence.
  - Eligible in-scope (3 rows: EGFR-MET NSCLC; ERBB2-ERBB3 NSCLC + pancreatic):
    2 distinct pairs, single rank (0) -> ZERO valid comparisons. Arms reduce to
    {EGFR, MET, ERBB2, ERBB3}; HNSCC and HCC/Liver contribute ZERO rows.
- Verdict: scope-compliant training is currently impossible. No training
  started. Proposed next step (needs approval): curate in-scope multi-stage
  BsAb records (NSCLC/HNSCC/PDAC/HCC trial-stage pairs) via ClinicalTrials.gov
  + literature search, drafted for row-by-row verification. HNSCC and HCC
  coverage is the critical gap (zero eligible records each).
