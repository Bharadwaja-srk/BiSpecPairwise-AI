# BSPAI Draft Target-Gene Universe (for review — NOT approved for training)

Files: `target_universe_draft.csv` (v1), `target_universe_draft_v2.csv` (current:
v1 + HGNC/Entrez/UniProt IDs, aliases, GO-CC terms, membrane classes, per-target
rationale — 63 rows x 30 cols), `verified_pairs_inventory.csv` (13 verified
pair records with evidence levels), `coverage_report.md` (verification results
+ sufficiency verdict: INSUFFICIENT for training), `clinical_pairs_draft.csv`
(15 rows with in-scope/out-of-scope + eligible_training flags; scope section 6
in coverage report: eligible in-scope set yields ZERO valid comparisons),
`clinical_four_indication_draft.csv` (26 in-scope rows, 14 eligible, 7 pairs,
ranks 0/1/2 across all 4 indications; 57 cross-rank combos) +
`clinical_quality_report.md` (v2: sufficiency verdict — exploratory pilot
defensible, generalization claim NOT; same-pair policy; LGR5 assessment). No feature generation, no training performed.

## Selection criteria

Union (not intersection) across tumors: a gene is eligible if it is observed in
>=1 of the four tumor datasets checked so far AND present in the Gene2Vec vocab.
Genes are grouped in strata (see `stratum` column):

- **A-paper (19)**: targets/pairs named in Zhang et al. 2024
  (`paper.txt`, DOI https://doi.org/10.1007/s00432-024-05740-3).
  Clinical note only repeats what the paper states (e.g. amivantamab approved
  NSCLC 2021; AK104 approved cervical; KN046 Phase 3). Nothing inferred.
- **B-fda (6)**: targets of FDA-approved bispecifics, verified against
  https://www.fda.gov/drugs/spotlight-cder-science/bispecific-antibodies-area-research-and-clinical-applications
  (2014–2023 list) and
  https://www.evitria.com/journal/bispecific-antibodies/fda-approved-bispecific-antibodies
  (2024–2025: DLL3, ERBB2xERBB3, GPRC5D) and
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10618718 (myeloma BCMA/GPRC5D).
  Non-oncology FIX/FX (emicizumab) excluded.
- **C-checkpoint (12), D-taa (16), E-heme (10)**: provisional candidates from
  established receptor families. `clinical_relevance` = "none claimed - do not
  infer from expression" for every one of these. No exceptions.

## Column semantics and status values

- `membrane_status`: `verified-CD` = member of the HGNC CD-molecules group
  (https://www.genenames.org/data/genegroup/#!/group/471);
  `provisional-family` = family-based, per-gene UniProt verification pending
  (query URL given in `membrane_source_url`, e.g.
  https://www.uniprot.org/uniprotkb?query=%28gene%3AEGFR%29+AND+%28organism_id%3A9606%29);
  `verified-exception` = secreted ligand (VEGFA, ANGPT2) kept deliberately per
  AK112/faricimab precedent, marked as NOT membrane.
- `clinical_status`: `verified-paper` / `verified-fda` = traceable claim with URL;
  `provisional-discussed` = mentioned in paper without a BsAb claim;
  `none-claimed` = no clinical statement (expression never used as evidence).
- `identifier_url` is a resolvable NCBI Gene query, not a curated numeric ID;
  numeric Entrez/HGNC IDs are unavailable pending annotation freeze.
- Coverage columns: real join results, never filled. `hnscc_coverage` from
  `data/processed/hnscc_raw.h5ad` var index (33,545 symbols, read 2026-10-11).
  NSCLC/pancreatic/liver are `pending` with reasons (RDS needs R mapping;
  10x TARs and HCC h5ad.gz unprocessed). The four-tumor coverage check has
  NOT passed.
- `gene2vec_coverage` from the staged vocab
  (`data/embeddings/gene2vec_dim_200_iter_9.txt`, SHA256 in
  `feature_engineering/biological_features/gene2vec_manifest.json`).
  `present-via-*` = former symbol (C10ORF54 for VSIR, PVRL4 for NECTIN4);
  verify at NCBI Gene before training use.

## Coverage summary (computed 2026-10-11)

- 63/63 observed in HNSCC raw; 63/63 Gene2Vec-resolved (61 direct + 2 via
  former symbol); 22/63 carry a verified clinical note; 41/63 claim nothing.
- NSCLC, pancreatic, liver: all pending. Do not treat this draft as
  four-tumor validated.

## Missing-data issues for your review

1. Four-tumor expression NOT verified (only HNSCC joined). Approval of this
   list does not substitute for the coverage check.
2. Per-gene UniProt membrane confirmation pending for all `provisional-family`
   rows (35 genes); CD-group rows are group-verified only.
3. VSIR/NECTIN4 resolve via former symbols — confirm you accept this or want
   them dropped (63 -> 61).
4. Numeric gene identifiers (Entrez/HGNC IDs) not yet attached.
5. Strata C–E have no clinical evidence by design; keep, trim, or replace with
   your preferred candidates.
