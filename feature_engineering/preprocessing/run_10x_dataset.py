"""Process one 10x MTX TAR dataset end-to-end (Phase 2).

Reads samples streaming via mtx10x (no full TAR extraction), builds per-sample
AnnData, concatenates, runs the repo Scanpy pipeline, writes raw + processed
checkpoints, QC JSON and a run log. Exclusions are recorded, never silent.

Usage (inside bspai-ml container, repo mounted at /work):
  python -m feature_engineering.preprocessing.run_10x_dataset \
    --tar data/raw/GSE205013_RAW.tar --name pancreatic \
    --out-raw data/processed/pancreatic_raw.h5ad \
    --out data/processed/pancreatic.h5ad \
    --qc data/processed/qc_pancreatic.json --log data/processed/log_pancreatic.txt
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import traceback


def log(msg: str, fh) -> None:
    line = f"{datetime.datetime.now(datetime.timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def atomic_write(adata, path: str) -> None:
    """Write AnnData atomically (tmp + rename) so kills never corrupt checkpoints."""
    import os as _os

    tmp = path + ".tmp"
    adata.write(tmp)
    _os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--qc", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--batch-size", type=int, default=9)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--resume-batches", default=None,
                    help="glob of existing batch h5ad files; skips TAR loading")
    ap.add_argument("--resume-raw", default=None,
                    help="existing raw concat h5ad; skips TAR loading and concat")
    ap.add_argument("--resume-stage", default=None,
                    help="existing stage checkpoint h5ad; starts the staged pipeline "
                         "from it (completed steps skip via uns markers)")
    ap.add_argument("--recast-batches", default=None,
                    help="glob of int64 batch h5ad files; rewrites each as float32 "
                         "<name>.f32.h5ad and exits (no pipeline run)")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow overwriting existing non-empty outputs; without it "
                         "the run refuses to touch existing --out-raw/--out/--qc")
    args = ap.parse_args()

    import os as _os

    guard_paths = (args.out, args.qc) if args.resume_stage else (args.out_raw, args.out, args.qc)
    for _p in guard_paths:
        if _os.path.exists(_p) and _os.path.getsize(_p) > 0 and not args.overwrite:
            print(f"REFUSE: output exists (use --overwrite to replace): {_p}", flush=True)
            return 2

    from feature_engineering.preprocessing import mtx10x
    from feature_engineering.preprocessing.pipeline import PreprocessConfig

    fh = open(args.log, "a", encoding="utf-8")
    rc = 0
    try:
        if args.recast_batches:
            import glob as _glob

            import anndata as _ad

            for bf in sorted(_glob.glob(args.recast_batches)):
                a = _ad.read_h5ad(bf)
                n, before = a.n_obs, str(a.X.dtype)
                a.X = mtx10x.ensure_float32_csr(a.X)
                out = bf.replace(".h5ad", ".f32.h5ad")
                atomic_write(a, out)
                log(f"recast {bf} -> {out}: {n} cells X {before} -> float32", fh)
                del a
            log("RECAST_DONE", fh)
            return 0

        log(f"START dataset={args.name} tar={args.tar}", fh)
        listing = mtx10x.list_10x_samples(args.tar)
        complete = listing["complete"]
        log(f"samples complete={len(complete)} excluded={len(listing['excluded'])}", fh)
        for e in listing["excluded"]:
            log(f"EXCLUDED {e['sample']}: {e['reason']}", fh)

        import anndata as ad
        import glob as _glob

        batch_files: list[str] = []
        if args.resume_batches:
            batch_files = sorted(_glob.glob(args.resume_batches))
            log(f"resuming from {len(batch_files)} batch files", fh)
        elif args.resume_raw:
            log(f"resuming from raw checkpoint {args.resume_raw} (TAR load skipped)", fh)
        elif args.resume_stage:
            log(f"resuming from stage checkpoint {args.resume_stage} (TAR load skipped)", fh)
        else:
            names = [s for s, _ in complete]
            if args.max_samples is not None:
                names = names[: args.max_samples]
                log(f"SMOKE-RUN limited to {len(names)} samples", fh)
            batch: list = []  # initialized before loop (legacy UnboundLocalError fix)
            for i, sample in enumerate(names, start=1):
                try:
                    mat = mtx10x.read_matrix(args.tar, sample)
                    genes = mtx10x.read_genes(args.tar, sample)
                    barcodes = mtx10x.read_barcodes(args.tar, sample)
                    a = mtx10x.to_anndata([mat], [sample], genes)
                    a.obs_names = [f"{sample}_{b}" for b in barcodes]
                    a.obs_names_make_unique()
                    batch.append(a)
                    del mat
                except Exception as e:  # noqa: BLE001 - recorded, not raised
                    log(f"EXCLUDED {sample}: read failed {type(e).__name__}: {e}", fh)
                    continue
                if len(batch) >= args.batch_size or i == len(names):
                    b = ad.concat(batch, join="outer", label="sample", index_unique="-")
                    bf = args.out_raw.replace(".h5ad", f".batch{i // args.batch_size}.h5ad")
                    atomic_write(b, bf)
                    batch_files.append(bf)
                    log(f"batch checkpoint {bf}: {b.n_obs} cells x {b.n_vars} genes", fh)
                    del batch, b
                    batch = []
                if i % 9 == 0:
                    log(f"loaded {i}/{len(names)}", fh)
            log(f"loaded {len(names)} requested; batches={len(batch_files)}", fh)

        parts = [ad.read_h5ad(bf) for bf in batch_files]
        if args.resume_stage:
            raw = ad.read_h5ad(args.resume_stage)
            raw_shape = [int(raw.n_obs), int(raw.n_vars)]
            log(f"resumed stage: {args.resume_stage} {raw_shape[0]} cells x {raw_shape[1]} genes", fh)
        elif parts:
            raw = ad.concat(parts, join="outer", label="sample", index_unique="-")
            del parts
            raw_shape = [int(raw.n_obs), int(raw.n_vars)]
            log(f"concat: {raw_shape[0]} cells x {raw_shape[1]} genes", fh)
            atomic_write(raw, args.out_raw)
            log(f"checkpoint saved: {args.out_raw}", fh)
        else:
            assert args.resume_raw, "no batches and no --resume-raw"
            raw = ad.read_h5ad(args.resume_raw)
            raw_shape = [int(raw.n_obs), int(raw.n_vars)]
            log(f"resumed raw: {raw_shape[0]} cells x {raw_shape[1]} genes", fh)

        before = str(raw.X.dtype)
        if before != "float32":
            # Single sparse astype allocation (no dense pass); old matrix freed
            # on reassignment. Prefer float32 batch files going forward.
            raw.X = mtx10x.ensure_float32_csr(raw.X)
            log(f"cast X {before} -> float32 (large-nnz scale needs it)", fh)
        import gc as _gc
        import resource as _resource
        import time as _time

        from feature_engineering.preprocessing.pipeline import (
            cluster,
            filter_cells_genes,
            highly_variable_genes,
            normalize,
            quality_control,
            run_pca,
        )

        def _sparse_tag() -> str:
            import scipy.sparse as _sp

            return ("sparse-csr" if _sp.issparse(raw.X) else "DENSE") + f":{raw.X.dtype}"

        def _rss(tag: str, t0: float) -> None:
            gb = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss / 1e6
            dt = _time.monotonic() - t0
            log(f"RSS {tag}: {gb:.2f} GB shape={raw.shape} X={_sparse_tag()} elapsed={dt:.0f}s", fh)

        cfg = PreprocessConfig()
        steps = [
            ("bspai_qc", "qc", quality_control),
            ("bspai_filtered", "filtered", filter_cells_genes),
            ("bspai_normalized", "normalized", normalize),
            ("bspai_hvg", "hvg", highly_variable_genes),
            ("bspai_pca", "pca", run_pca),
            ("bspai_clustered", "clustered", cluster),
        ]
        stage_prefix = args.out.replace(".h5ad", "")
        for marker, tag, fn in steps:
            if raw.uns.get(marker):
                log(f"step {tag}: cached, skipping", fh)
                continue
            t0 = _time.monotonic()
            raw = fn(raw, cfg)
            _gc.collect()
            ckpt = f"{stage_prefix}.stage_{tag}.h5ad"
            atomic_write(raw, ckpt)
            _rss(tag, t0)
            log(f"step {tag}: checkpoint {ckpt}", fh)
        atomic_write(raw, args.out)
        log(f"processed saved: {args.out}", fh)

        import numpy as np

        qc = {
            "dataset": args.name,
            "n_obs": int(raw.n_obs),
            "n_vars": int(raw.n_vars),
            "markers": {k: bool(v) for k, v in raw.uns.items() if k.startswith("bspai_")},
            "median_total_counts": float(np.median(raw.obs["total_counts"])) if "total_counts" in raw.obs else None,
            "median_mt_pct": float(np.median(raw.obs["pct_counts_mt"])) if "pct_counts_mt" in raw.obs else None,
            "n_clusters": int(raw.obs["leiden"].nunique()) if "leiden" in raw.obs else None,
            "n_samples": int(raw.obs["sample"].nunique()) if "sample" in raw.obs else None,
            "raw_shape": raw_shape,
        }
        with open(args.qc, "w", encoding="utf-8") as qf:
            json.dump(qc, qf, indent=2)
        log(f"QC: {json.dumps(qc)}", fh)
        log("STEP3_RESULT: PASS", fh)
    except Exception:
        traceback.print_exc(file=fh)
        log("STEP3_RESULT: FAIL", fh)
        rc = 1
    finally:
        log("DONE", fh)
        fh.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
