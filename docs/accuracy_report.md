# Model Accuracy Report (2026-10-11 — NO Docker, NO real-model fitting)

> HISTORICAL (partially superseded 2026-10-17): "real accuracy not estimable"
> below held for v2 labels. Current exploratory pilot (`docs/pilot_v3_lopo.json`,
> v3 labels + HNSCC/HCC/NSCLC stores): pairwise-XGB 0.823, pointwise-XGB 0.813,
> dummy 0.500 over 40 same-space held-out comparisons (9 folds, 8 scored).
> Exploratory only — no generalization claim; PDAC rows still incomplete.

## What was actually tested

1. **Evaluation protocol on synthetic fixtures** (seed 7, 16 rows / 8 pairs /
   ranks {0:3, 1:7, 2:5, 4:1}, rank-correlated signal + noise; script in temp,
   NOT committed as evidence):
   - Leave-one-pair-out over 8 pairs; reference recomputed on train folds only.
   - Held-out concordance over 86 cross-rank row pairs:
     pairwise-XGB **0.686**, pointwise-XGB **0.756**, dummy-0.5 **0.500**.
   - The protocol executes correctly AND shows pairwise learning does NOT
     automatically beat pointwise — superiority must be earned on real data,
     never assumed from the paper.
2. **Real biological features (HNSCC raw, 151,680 cells):** all 8 eligible
   pairs computed end-to-end (expression 6 + embeddings 2 = 8/11 real;
   safety NaN, pathways 0.0). Double-positive ranges 0.0011–0.1425;
   correlations −0.0643 to +0.3405 — biologically plausible spreads, validated
   ranges, no fabrication.

## Accuracy verdict: NOT ESTIMABLE on real data

Under the adopted indication-matched policy, HNSCC features attach only to the
2 HNSCC clinical rows — same pair (PDCD1–CTLA4), single rank → **zero valid
comparisons**. The remaining 14 rows have no indication-matched expression
features (NSCLC/PDAC/HCC unprocessed — Docker-gated). Training a ranker on
mismatched-tumor features would violate policy and produce a misleading number,
so no real model was fitted and no real accuracy exists to report.

## What accuracy testing requires (in order)

1. Docker re-enabled; pancreatic/HCC/NSCLC processed with QC reports.
2. Indication-matched feature rows for all 16 eligible rows (+GMT decision).
3. Leave-one-pair-out pilot per `docs/pilot_readiness.md` (exploratory only).
4. Only then do concordance/AUC numbers become legitimate — reported with
   fold tables, baselines, and the n=8-pairs limitation. Paper's 89.29% is
   not our result under any circumstance.
