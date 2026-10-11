# Curation Draft — rank-3 gap + LGR5 candidate (PROPOSAL ONLY, 2026-10-11)

No CSV modified. No universe change. No labels changed. For review.

## 1. Rank-3 / preclinical search: NO verified records found

Searched ClinicalTrials.gov, sponsor PRs, ASCO/ESMO abstracts, peer reviews
for preclinical/IND-stage bispecifics in NSCLC, PDAC, HNSCC, HCC.

Near-misses, all rejected with reason (do NOT add without new evidence):
- ADCE-T02 (NCT06597721, PDAC/HNSCC expansion), IBI343 (PDAC BTD), YL201
  (HNSCC/PDAC cohorts): antibody-DRUG CONJUGATES, not bispecifics — excluded
  by modality.
- INCA33890 (TGFβR2×PD-1, ESMO 2025 Ph1): microsatellite-stable CRC only —
  out of the four indications.
- Cesalatamab/AI-081 (PD-1×VEGF, NCT06635785 verified RECRUITING Ph1/2,
  Fast Track NSCLC Sept 2026): basket "Advanced Solid Tumor" condition —
  NSCLC indication-specific status unclear (same class as cibisatamab row).
  Provisional; needs indication-specific evidence before drafting.
- HB0025 (PD-L1×VEGF NSCLC Ph2, ASCO 2026 abstract only): single-source.
- CT-95 (MSLN×CD3 PDAC Ph1 2025, review-cited only): single-source.
- Gap stands: rank 3 remains ZERO. Do not fabricate preclinical rows.

## 2. LGR5 verification dossier (expansion 63 -> 64 PROPOSED, not applied)

- HGNC:4504, symbol LGR5, Entrez 8549, UniProt O75473, prev GPR67/GPR49
  (source: rest.genenames.org, 2026-10-11). Canonical symbol, no alias needed.
- GO-CC (MyGene.info): plasma membrane (GO:0005886) -> membrane-go.
- Gene2Vec dim_200_iter_9: LGR5 present (direct, staged file verified).
- HNSCC: observed in hnscc_raw.h5ad var index (33,545-gene join).
- Clinical (in-scope HNSCC, all verified): petosemtamab (MCLA-158) EGFR×LGR5 —
  Ph1/2 NCT03526835 (AACR 2023 ORR 37.2%; Dec 2024 interim ORR 36%, DOR 6.2 mo;
  ASCO 2024/2025 +pembro 1L: ORR 63%, PFS 9 mo), FDA Breakthrough Therapy
  Designation (previously treated HNSCC), Ph3 LiGeR-HN1 NCT06525220 +
  LiGeR-HN2 NCT06496178. Mechanism: EGFR signaling inhibition + LGR5-mediated
  internalization/degradation + enhanced ADCC (Genmab; Nat Cancer 2022
  organoid screen). Sources: manifoldtimes 2024-12-07 (Merus PR),
  OncLive 2023-04-26 (AACR), ascopubs JCO 6014/6024/6025, Cancer Network BTD,
  Genmab 2026 portfolio.
- Proposed change (needs explicit approval): (a) verify LGR5 entry through
  the ID/GO/expression pipeline into target_universe (64 rows); (b) flip the
  2 petosemtamab HNSCC rows to eligible (rank 1, registry-verified) —
  adds 1 unique pair (EGFR-LGR5), rank-1 count 7 -> 9, combos recompute at
  build time. No other rows affected.

## 3. Decision asks

(a) Accept the rank-3 gap as documented (no action)?
(b) Approve LGR5 63 -> 64 + petosemtamab eligibility flip?
(c) Draft cesalatamab/HB0025/CT-95 as provisional rows pending registries?
