# Pilot Readiness Package (review only — NO fitting performed)

> HISTORICAL (superseded 2026-10-17): 16-row/v2-label verdict below. Current:
> v3 labels (18 eligible / 9 pairs / 104 combos), 3-tumor stores (HNSCC 10/11,
> HCC 11/11, NSCLC 11/11), exploratory LOPO in `docs/pilot_v3_lopo.json`
> (pairwise-XGB 0.823, pointwise-XGB 0.813, dummy 0.500 over 40 held-out
> comparisons). PDAC-only rows remain incomplete (no pancreatic store).

Date: 2026-10-11. Dataset: `docs/target_universe/clinical_four_indication_draft_v2.csv`
(28 rows; labels approved with 3-cell fix). No model fitted, no features generated
beyond previously reported HNSCC validation probes.

## 1. Dataset version for training

- 16 eligible rows / 8 unique pairs / ranks 0:3, 1:7, 2:5, 4:1.
- Indications: NSCLC 10, pancreatic 2, HNSCC 2, HCC 2. CD3E-arm rows: 0.
- Same-pair policy (approved design): group key = pair; zero-diff same-pair
  rows excluded from transform; one row per (pair, indication) for counting.

## 2. Feature availability per eligible row (verified, no new runs)

| Feature group | NSCLC rows (10) | PDAC (2) | HNSCC (2) | HCC (2) |
|---|---|---|---|---|
| Expression (activity/single/double/corr) | UNAVAILABLE (RDS unprocessed) | UNAVAILABLE (TAR unprocessed) | COMPUTABLE from `hnscc_raw.h5ad` (all arms present) + leiden mapping | UNAVAILABLE (h5ad.gz unprocessed) |
| Safety (tumor-vs-normal) | UNAVAILABLE (nLung unprocessed) | unavailable (no normal) | unavailable (no normal) | unavailable (no normal) |
| Pathway (KEGG/GO) | 0.0 by policy (no GMT staged — open acquisition) | same | same | same |
| Embedding (Gene2Vec) | available, all arms | available | available | available |

Additional verified constraint: `hnscc.h5ad` holds only 2,000 HVG genes and
just 4/11 eligible-pair arms (VEGFA, CTLA4, TNFRSF9, LAG3) — it is NOT a
feature source; HNSCC features must come from `hnscc_raw.h5ad` (33,545 genes)
joined to leiden labels (143k/151k cell-ID mapping, defined but unexecuted).

OPEN POLICY QUESTION (needs your decision before any build): which tumor's
expression features attach to each training row — indication-matched tumor
(requires all 4 processed) or a pooled/pan-tumor representation? The reference
methodology computes per-tumor features; cross-tumor substitution for
NSCLC/PDAC/HCC rows would be a methodological extension requiring approval.

## 3. Missing-value projection (indication-matched policy)

- HNSCC rows: ~9/11 real (safety NaN, pathways 0.0).
- All other rows: ~2/11 real (embeddings only). NOT trainable as-is.

## 4. Proposed evaluation design (for review, not executed)

- Leave-one-pair-out over 8 pairs (8 folds; test fold = all rows of held-out pair).
- Train: pairwise-XGB (diff vectors, best-stage reference on train folds only);
  baselines: stage-prior dummy + pointwise XGB (same features).
- Metrics: held-out pairwise concordance (primary — matches the ranking
  objective), AUC with single-class-fold NaN handling (secondary, reported
  with fold table, never as a lone number), NDCG where candidate sets allow.
- Leakage: pair-grouped splits; reference vector recomputed per fold;
  no test labels in tuning; per-fold results + variability, never aggregate-only.

## 5. Adopted policy: INDICATION-MATCHED ONLY (decision 2026-10-11)

Every clinical training row sources biological features from its corresponding
tumor type. No pooled features, no cross-tumor substitution without a separately
approved methodological extension. Rows with unavailable indication-matched
features are INCOMPLETE (see `docs/feature_availability_matrix.csv`: 16 rows,
best case 8/11 — the 2 HNSCC rows — remainder lower; safety missing everywhere,
pathways zero-by-policy). Verdict reaffirmed: NOT READY FOR FITTING.

Blockers in order: (1) processed matrices for NSCLC/PDAC/HCC (+HNSCC
leiden mapping) — Docker-gated; (2) feature-sourcing policy decision above;
(3) GMT acquisition or documented 0.0 acceptance; (4) rank-3 gap + n=8 pairs
(thin even when complete — pilot stays exploratory, leave-one-pair-out,
no generalization claim, no AUC target).
