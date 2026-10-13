# PCA Investigation Report (2026-10-11/12 — DECISION: deferred)

## Verdict

Full pancreatic PCA (282,767 cells x 2,000 genes) is POSTPONED until a
higher-memory environment is available. No methodology change adopted.

## Evidence

1. **Controlled 5k-cell A/B** (seeded-random subset of valid
   `smoke1.stage_hvg.h5ad`; `-m 6g`; fresh `ab_*` outputs; exit 0):
   reference vs float32-precast arms produced IDENTICAL results because
   scanpy 1.11.5 upcasts both paths to dense float64 inside `scale`
   (per-arm dtype logs). Scaled diff 0.0; PC corr 1.000000 (50/50);
   variance-ratio diff 0.0; 15-NN Jaccard 1.000000; leiden ARI 1.000000
   (17 vs 17 clusters). Tolerances met trivially — the experiment
   demonstrates the float32 config buys NOTHING (no memory benefit).
2. **Full-scale memory math** (measured, not guessed): dense float64
   282,767x2000 = 4.53 GB + sparse base ~1.5 GB transient + ARPACK
   subspace ~1 GB + checkpoint write buffer ~1.4 GB + neighbors/leiden
   ~1 GB = ~9.5-11 GB peak vs 10g cap -> exit 137 observed. 12g is
   prohibited on the 12.5 GB VM (user decision).
3. **Code path exonerated**: full 89k-cell smoke run reached PASS with
   valid PCA/cluster artifacts; 5k diag passed twice. Deaths were
   cap-exceedance (+ same-day daemon/EIO instability), never logic.

## Preserved state (nothing deleted/overwritten)

Valid: `pancreatic_raw.h5ad`, batch{1,2,3} int64 + batch1 f32, f32 concat
inputs, all `pancreatic_f32.stage_{qc,filtered,normalized,hvg}.h5ad`,
HNSCC outputs, 3 `.bak` partials, `ab_*` + `diag*_` diagnostics.
Redundant but VALID: `pancreatic_f32_raw.batch{1,2,3}.h5ad` (checksummed
equal to originals; removal needs approval).

## Resume requirements (higher-memory machine)

- RAM: >= 20 GB host with >= 16g container cap (measured peak ~11 GB +
  headroom for write buffers and daemon overhead).
- Disk: >= 60 GB free (checkpoints ~15 GB + outputs + image).
- Docker: daemon healthy; `bspai-ml:step3` (scanpy 1.11.5 stack) present.
- Pre-flight: h5py metadata validation of `pancreatic_f32.stage_hvg.h5ad`
  (282,767 x 2000, markers qc/filtered/normalized/hvg); no other containers;
  pause OneDrive sync during multi-GB writes.
- Exact resume command (DO NOT EXECUTE without approval):
  `docker run --rm -m 12g --cpus=10 -v "<repo>:/work" -w /work bspai-ml:step3`
  `python -m feature_engineering.preprocessing.run_10x_dataset --tar
  data/raw/GSE205013_RAW.tar --name pancreatic --out-raw
  data/processed/pancreatic_f32_raw.h5ad --out data/processed/pancreatic_f32.h5ad
  --qc data/processed/qc_pancreatic_f32.json --log data/processed/log_pancreatic_f32.txt
  --resume-stage data/processed/pancreatic_f32.stage_hvg.h5ad`
  (markers skip qc->hvg; only pca+cluster run; atomic writes; RSS logged).
- Success criteria: exit 0, `pancreatic_f32.h5ad` readable with X_pca
  (N,50) + leiden + all six markers, `qc_pancreatic_f32.json` written.
