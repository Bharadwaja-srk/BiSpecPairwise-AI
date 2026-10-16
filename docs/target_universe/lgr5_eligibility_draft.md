# LGR5 Eligibility Draft (PROPOSAL ONLY, 2026-10-11)

Split from `rank3_lgr5_curation_draft.md` (preserved unchanged as source).
Universe stays at 63 genes. No CSV modified. No labels changed. For review.

## 1. Identifier verification (verified, not assumed)

- HGNC:4504, symbol LGR5, Entrez 8549, UniProt O75473, prev GPR67/GPR49
  (source: rest.genenames.org, 2026-10-11). Canonical symbol, no alias needed.

## 2. Target-class and data coverage (verified)

- GO-CC (MyGene.info): plasma membrane (GO:0005886) -> membrane-go.
- Gene2Vec dim_200_iter_9: LGR5 present (direct, staged file verified).
- HNSCC: observed in hnscc_raw.h5ad var index (33,545-gene join).

## 3. Clinical evidence (in-scope HNSCC, verified sources)

Petosemtamab (MCLA-158) EGFR×LGR5 —
Ph1/2 NCT03526835 (AACR 2023 ORR 37.2%; Dec 2024 interim ORR 36%, DOR 6.2 mo;
ASCO 2024/2025 +pembro 1L: ORR 63%, PFS 9 mo), FDA Breakthrough Therapy
Designation (previously treated HNSCC), Ph3 LiGeR-HN1 NCT06525220 +
LiGeR-HN2 NCT06496178. Mechanism: EGFR signaling inhibition + LGR5-mediated
internalization/degradation + enhanced ADCC (Genmab; Nat Cancer 2022
organoid screen). Sources: manifoldtimes 2024-12-07 (Merus PR),
OncLive 2023-04-26 (AACR), ascopubs JCO 6014/6024/6025, Cancer Network BTD,
Genmab 2026 portfolio.

## 4. Proposed change (needs explicit approval, NOT applied)

(a) Verify LGR5 entry through the ID/GO/expression pipeline into
target_universe (64 rows); (b) flip the 2 petosemtamab HNSCC rows to eligible
(rank 1, registry-verified) — adds 1 unique pair (EGFR-LGR5), rank-1 count
7 -> 9, combos recompute at build time. No other rows affected.

## 5. Open decision

Approve LGR5 63 -> 64 + petosemtamab eligibility flip?
