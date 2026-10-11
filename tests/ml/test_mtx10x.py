"""Regression tests for the repo-owned 10x MTX adapter (mxt10x.py).

Covers the Phase-1 defect class: UnboundLocalError 'batch' in the ephemeral
/tmpjob/step3_preprocess.py process_10x, where the batch accumulator was
referenced before assignment on the multi-batch path (log_pancreatic.txt).
Uses synthetic MTX triplets only - no real datasets, no scanpy/anndata.
"""
import gzip
import tarfile

import pytest

pytest.importorskip("scipy")

from feature_engineering.preprocessing import mtx10x


def _write_triplet(base, prefix, n_genes=8, n_cells=5):
    base.mkdir(parents=True, exist_ok=True)
    barcodes = [f"{prefix}_CELL{i:03d}" for i in range(n_cells)]
    genes = [f"GENE{j:03d}" for j in range(n_genes)]
    (base / f"{prefix}_barcodes.tsv.gz").write_bytes(
        gzip.compress(("\n".join(barcodes) + "\n").encode()))
    (base / f"{prefix}_features.tsv.gz").write_bytes(
        gzip.compress(("\n".join(f"{g}\t{g}\tGene Expression" for g in genes) + "\n").encode()))
    nnz = n_genes * n_cells
    entries = "\n".join(f"{(i % n_genes) + 1} {(i % n_cells) + 1} 1" for i in range(nnz))
    mtx = f"%%MatrixMarket matrix coordinate integer general\n%\n{n_genes} {n_cells} {nnz}\n{entries}\n"
    (base / f"{prefix}_matrix.mtx.gz").write_bytes(gzip.compress(mtx.encode()))


def _make_tar(tmp_path, prefixes):
    src = tmp_path / "src"
    for p in prefixes:
        _write_triplet(src, p)
    tar_path = tmp_path / "test_10x.tar"
    with tarfile.open(tar_path, "w") as t:
        for f in sorted(src.iterdir()):
            t.add(f, arcname=f.name)
    return tar_path


def test_multi_batch_accumulation_keeps_all_samples(tmp_path):
    """5 samples, batch_size 2 -> 2 full batches + 1 leftover; none lost.

    The legacy defect dropped/failed here (accumulator used before assignment).
    """
    tar_path = _make_tar(tmp_path, [f"P{i:02d}" for i in range(5)])
    samples = mtx10x.list_10x_samples(tar_path)
    assert [s for s, _ in samples["complete"]] == [f"P{i:02d}" for i in range(5)]
    assert samples["excluded"] == []
    batches, leftover, excluded = mtx10x.accumulate_batches(tar_path, batch_size=2)
    assert excluded == []
    assert len(batches) == 2
    assert all(len(b) == 2 for b in batches)
    total_cells = sum(m.shape[1] for b in batches for m in b)
    total_cells += sum(m.shape[1] for m in leftover)
    assert total_cells == 5 * 5
    assert len(leftover) == 1


def test_single_sample_does_not_require_full_batch(tmp_path):
    tar_path = _make_tar(tmp_path, ["ONLY"])
    batches, leftover, excluded = mtx10x.accumulate_batches(tar_path, batch_size=4)
    assert batches == []
    assert len(leftover) == 1
    assert leftover[0].shape == (8, 5)
    assert excluded == []


def test_incomplete_triplet_excluded_with_reason(tmp_path):
    src = tmp_path / "src"
    _write_triplet(src, "GOOD")
    (src / "BAD_barcodes.tsv.gz").write_bytes(gzip.compress(b"CELL1\n"))
    tar_path = tmp_path / "incomplete.tar"
    with tarfile.open(tar_path, "w") as t:
        for f in sorted(src.iterdir()):
            t.add(f, arcname=f.name)
    samples = mtx10x.list_10x_samples(tar_path)
    assert [s for s, _ in samples["complete"]] == ["GOOD"]
    assert len(samples["excluded"]) == 1
    assert samples["excluded"][0]["sample"] == "BAD"
    assert "matrix" in samples["excluded"][0]["reason"]
    batches, leftover, excluded = mtx10x.accumulate_batches(tar_path, batch_size=10)
    assert [e["sample"] for e in excluded] == ["BAD"]
    assert sum(m.shape[1] for m in leftover) == 5


def test_gene_symbol_universe_readable_from_triplet(tmp_path):
    tar_path = _make_tar(tmp_path, ["S1"])
    genes = mtx10x.read_genes(tar_path, "S1")
    assert genes == [f"GENE{j:03d}" for j in range(8)]


def test_ensure_float32_csr_preserves_values_and_sparsity():
    import numpy as np
    import scipy.sparse

    m = scipy.sparse.csr_matrix(np.array([[0, 3, 0], [1, 0, 2]], dtype=np.int64))
    out = mtx10x.ensure_float32_csr(m)
    assert str(out.dtype) == "float32"
    assert scipy.sparse.issparse(out)
    assert out.nnz == m.nnz == 3
    assert (out.toarray() == m.toarray()).all()


def test_ensure_float32_csr_noop_on_float32():
    import numpy as np
    import scipy.sparse

    m = scipy.sparse.csr_matrix(np.array([[1.5, 0.0]], dtype=np.float32))
    out = mtx10x.ensure_float32_csr(m)
    assert out is m  # no copy when already compliant


def test_to_anndata_emits_float32_sparse(tmp_path):
    pytest.importorskip("anndata")
    import numpy as np
    import scipy.sparse

    m = scipy.sparse.csr_matrix(np.array([[0, 1], [2, 0], [0, 3]], dtype=np.int64))
    a = mtx10x.to_anndata([m], ["S"], ["G1", "G2"])
    assert str(a.X.dtype) == "float32"
    assert a.shape == (2, 2)
    assert list(a.obs["sample"]) == ["S", "S"]


def test_runner_refuses_to_overwrite_without_flag(tmp_path, monkeypatch):
    """ existing --out-raw/--out/--qc must never be silently replaced. """
    from feature_engineering.preprocessing import run_10x_dataset as runner

    raw = tmp_path / "r.h5ad"
    out = tmp_path / "o.h5ad"
    qc = tmp_path / "q.json"
    log = tmp_path / "l.txt"
    for p in (raw, out, qc):
        p.write_text("sentinel")
    monkeypatch.setattr(
        "sys.argv",
        ["run_10x_dataset", "--tar", "x.tar", "--name", "t",
         "--out-raw", str(raw), "--out", str(out), "--qc", str(qc),
         "--log", str(log), "--resume-batches", "nothing*.h5ad"],
    )
    assert runner.main() == 2
    for p in (raw, out, qc):
        assert p.read_text() == "sentinel"
    assert not log.exists()  # refused before touching anything


def test_resume_batches_never_extracts_tar(tmp_path, monkeypatch):
    pytest.importorskip("anndata")
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse

    from feature_engineering.preprocessing import run_10x_dataset as runner

    tiny = tmp_path / "tiny.h5ad"
    ad.AnnData(
        X=scipy.sparse.csr_matrix(np.ones((3, 2), dtype=np.float32)),
        obs=pd.DataFrame(index=["c1", "c2", "c3"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    ).write(tiny)

    def _boom(*a, **k):
        raise AssertionError("TAR extraction must not run on resume path")

    monkeypatch.setattr(mtx10x, "read_matrix", _boom)
    monkeypatch.setattr(mtx10x, "read_genes", _boom)
    monkeypatch.setattr(mtx10x, "read_barcodes", _boom)
    empty_tar = tmp_path / "empty.tar"
    import tarfile as _tarfile

    with _tarfile.open(empty_tar, "w"):
        pass
    out_raw = tmp_path / "r.h5ad"
    out = tmp_path / "o.h5ad"
    qc = tmp_path / "q.json"
    log = tmp_path / "l.txt"
    monkeypatch.setattr(
        "sys.argv",
        ["run_10x_dataset", "--tar", str(empty_tar), "--name", "t",
         "--out-raw", str(out_raw), "--out", str(out), "--qc", str(qc),
         "--log", str(log), "--overwrite",
         "--resume-batches", str(tiny)],
    )
    # Must fail only on the missing scanpy pipeline (lazy import), never on TAR reads.
    try:
        runner.main()
    except (ImportError, ModuleNotFoundError):
        pass
    assert "resuming from 1 batch files" in log.read_text()
