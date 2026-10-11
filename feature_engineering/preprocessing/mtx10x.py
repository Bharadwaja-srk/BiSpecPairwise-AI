"""Repo-owned 10x MTX TAR adapter (Phase 2).

Reads CellRanger-style (barcodes/features/matrix) triplets streaming from a
TAR archive without full extraction. Replaces the ephemeral
/tmpjob/step3_preprocess.py process_10x, which failed with
UnboundLocalError: 'batch' (see data/processed/log_pancreatic.txt): its batch
accumulator was referenced before assignment on the multi-batch path.

Design guards: every accumulator is initialized before its loop; incomplete
triplets are excluded with a recorded reason, never silently dropped; AnnData
is imported lazily so header/inventory work needs only scipy+pandas.
"""
from __future__ import annotations

import gzip
import tarfile
from typing import Any

SUFFIXES = ("barcodes.tsv.gz", "features.tsv.gz", "matrix.mtx.gz")


def _members(tar_path: str) -> list[str]:
    with tarfile.open(tar_path, "r") as t:
        return [m.name for m in t.getmembers() if m.isfile()]


def list_10x_samples(tar_path: str) -> dict:
    """Return {'complete': [(sample, {kind: member})], 'excluded': [{sample, reason}]}."""
    names = _members(tar_path)
    by_sample: dict[str, dict[str, str]] = {}
    for name in names:
        for suffix in SUFFIXES:
            if name.endswith(suffix):
                sample = name[: -len(suffix)].rstrip("_")
                by_sample.setdefault(sample, {})[suffix] = name
    complete: list[tuple[str, dict[str, str]]] = []
    excluded: list[dict[str, str]] = []
    for sample in sorted(by_sample):
        kinds = by_sample[sample]
        missing = [s for s in SUFFIXES if s not in kinds]
        if missing:
            excluded.append({"sample": sample,
                             "reason": f"incomplete triplet, missing: {', '.join(missing)}"})
        else:
            complete.append((sample, kinds))
    return {"complete": complete, "excluded": excluded}


def _open_member(tar_path: str, member: str):
    t = tarfile.open(tar_path, "r")
    f = t.extractfile(member)
    assert f is not None, f"missing member {member}"
    return t, f


def read_barcodes(tar_path: str, sample: str) -> list[str]:
    for s, kinds in list_10x_samples(tar_path)["complete"]:
        if s == sample:
            t, f = _open_member(tar_path, kinds["barcodes.tsv.gz"])
            try:
                with gzip.open(f, "rt") as g:
                    return [line.strip() for line in g if line.strip()]
            finally:
                t.close()
    raise KeyError(f"sample not complete: {sample}")


def read_genes(tar_path: str, sample: str) -> list[str]:
    for s, kinds in list_10x_samples(tar_path)["complete"]:
        if s == sample:
            t, f = _open_member(tar_path, kinds["features.tsv.gz"])
            try:
                with gzip.open(f, "rt") as g:
                    return [line.split("\t")[0].strip() for line in g if line.strip()]
            finally:
                t.close()
    raise KeyError(f"sample not complete: {sample}")


def read_matrix(tar_path: str, sample: str):
    """Return scipy sparse (genes x cells) matrix for one sample."""
    import io

    import scipy.io

    for s, kinds in list_10x_samples(tar_path)["complete"]:
        if s == sample:
            t, f = _open_member(tar_path, kinds["matrix.mtx.gz"])
            try:
                with gzip.open(f, "rb") as g:
                    buf = io.BytesIO(g.read())
                return scipy.io.mmread(buf).tocsr()
            finally:
                t.close()
    raise KeyError(f"sample not complete: {sample}")


def accumulate_batches(tar_path: str, batch_size: int = 8) -> tuple[list, list, list]:
    """Stream all complete samples into full batches + leftover.

    Returns (batches, leftover, excluded) where batches is a list of lists of
    per-sample CSR matrices. The accumulator is initialized before the loop
    (the legacy UnboundLocalError fix); leftovers never require a full batch.
    """
    batches: list[list] = []
    current: list = []
    excluded: list[dict[str, str]] = []
    listing = list_10x_samples(tar_path)
    excluded.extend(listing["excluded"])
    for sample, _kinds in listing["complete"]:
        try:
            mat = read_matrix(tar_path, sample)
        except Exception as e:  # noqa: BLE001 - recorded, not raised
            excluded.append({"sample": sample, "reason": f"read failed: {type(e).__name__}"})
            continue
        current.append(mat)
        if len(current) >= batch_size:
            batches.append(current)
            current = []
    leftover: list = current
    return batches, leftover, excluded


def ensure_float32_csr(mat):
    """Return CSR float32 matrix with identical values/nnz (no dense pass).

    Single astype allocation only when dtype differs; already-float32 input
    is returned untouched. Preserves sparsity and numerical values exactly
    for integer counts (all counts << 2**24).
    """
    import numpy as np
    import scipy.sparse

    m = mat.tocsr() if scipy.sparse.issparse(mat) else scipy.sparse.csr_matrix(mat)
    if str(m.dtype) == "float32":
        return m
    return m.astype(np.float32)


def to_anndata(matrices: list, sample_names: list, gene_names: list | None = None) -> Any:
    """Concatenate per-sample matrices into one AnnData (lazy anndata import)."""
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse

    assert len(matrices) == len(sample_names), "matrices/sample_names length mismatch"
    xs = [ensure_float32_csr(m).T.tocsr() for m in matrices]
    x = scipy.sparse.vstack(xs, format="csr")
    obs = pd.DataFrame({"sample": np.repeat(sample_names, [m.shape[1] for m in matrices])})
    var = pd.DataFrame(index=gene_names) if gene_names is not None else pd.DataFrame(
        index=[f"GENE{i}" for i in range(x.shape[1])])
    return ad.AnnData(X=x, obs=obs, var=var)


__all__ = ["SUFFIXES", "list_10x_samples", "read_barcodes", "read_genes",
           "read_matrix", "accumulate_batches", "ensure_float32_csr", "to_anndata"]
