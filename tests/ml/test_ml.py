"""Synthetic-only ML tests (no large downloads, no secrets)."""
import numpy as np
import pandas as pd
import pytest

rng = np.random.default_rng(0)


def test_double_positive():
    from feature_engineering.biological_features.double_positive import (
        double_positive_proportion, single_proportions)
    a = np.array([1.0, 0.0, 2.0, 0.0])
    b = np.array([1.0, 1.0, 0.0, 0.0])
    assert double_positive_proportion(a, b, 0.5) == pytest.approx(0.25)
    s = single_proportions(a, b, 0.5)
    assert s["single_ratio_sum"] == pytest.approx(1.0)
    assert s["single_ratio_min"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        double_positive_proportion(a, b[:2])


def test_safety_paper_formula():
    from feature_engineering.biological_features.safety import safety_score
    # dx=2, dy=4 -> 8/6
    assert safety_score(3.0, 1.0, 5.0, 1.0) == pytest.approx(8 / 6)
    assert safety_score(1.0, 1.0, 1.0, 1.0) == 0.0  # dx+dy==0 edge


def test_pathway_no_fabrication():
    from feature_engineering.biological_features.pathway import (
        pathway_features)
    pw = {"KEGG001": {"A", "B"}, "GO001": {"B", "C"}}
    f = pathway_features("A", "B", pw)
    assert f["pathway_shared_count"] == 1.0
    assert 0.0 <= f["pathway_jaccard"] <= 1.0
    assert pathway_features("ZZZ1", "ZZZ2", pw)["pathway_similarity"] == 0.0


def test_embedding():
    from feature_engineering.biological_features.gene_embedding import (
        embedding_similarity, pair_similarity)
    e = embedding_similarity(np.zeros(4), np.zeros(4))
    assert e["gene_embedding_distance"] == 0.0
    assert e["gene_embedding_similarity"] == 1.0
    emb = {"G1": np.zeros(3), "G2": np.ones(3)}
    assert pair_similarity("G1", "G2", emb)["gene_embedding_distance"] > 0
    with pytest.raises(ValueError):
        pair_similarity("G1", "MISSING", emb)


def test_correlation_guards():
    from feature_engineering.biological_features.correlation import (
        pearson_correlation)
    x = rng.normal(size=50)
    assert pearson_correlation(x, x) == pytest.approx(1.0)
    assert pearson_correlation(np.ones(10), rng.normal(size=10)) == 0.0
    assert np.isnan(pearson_correlation(np.array([1.0]), np.array([1.0])))
    with pytest.raises(ValueError):
        pearson_correlation(x, x[:5])


def test_target_activity():
    from feature_engineering.biological_features.target_activity import (
        celltype_activity, positive_proportion)
    assert positive_proportion(np.array([0.0, 1.0, 2.0]), 0.5) == pytest.approx(2 / 3)
    act = celltype_activity(np.array([1.0, 0.0, 3.0, 0.0]),
                            ["T", "T", "M", "M"], 0.5)
    assert act["T"]["positive_proportion"] == pytest.approx(0.5)


def test_schema_and_builder():
    from feature_engineering.feature_builder import build_features
    from feature_engineering.feature_schema import FEATURE_ORDER, validate_feature_dict
    assert len(FEATURE_ORDER) >= 10
    out = build_features("CD274", "CTLA4",
                         rng.normal(1, 1, 60), rng.normal(1, 1, 60),
                         5.0, 1.0, 4.0, 1.0,
                         pathways={"P1": {"CD274", "CTLA4"}},
                         embeddings={"CD274": np.zeros(4),
                                     "CTLA4": np.ones(4)})
    assert out["target1"] == "CD274"
    assert len(out["vector"]) == len(FEATURE_ORDER)
    assert set(validate_feature_dict({})) == set(FEATURE_ORDER)


def test_clinical_and_pairwise():
    from ml.data.clinical_stages import process_clinical_df, stage_to_rank
    from ml.data.pairwise import build_pairwise
    assert stage_to_rank("Approved, NDA/BLA") == 0
    assert stage_to_rank("Phase 1, 2") == 2
    assert stage_to_rank("Preclinical, IND application") == 3
    with pytest.raises(ValueError):
        stage_to_rank("moon-phase 9")
    df = process_clinical_df(pd.DataFrame({
        "target1": ["A", "B", "C"], "target2": ["X", "Y", "Z"],
        "stage": ["Approved", "Phase 1, 2", "Preclinical"]}))
    assert list(df["stage_rank"]) == [0, 2, 3]
    built = build_pairwise(rng.normal(size=(4, 5)),
                           np.array([0, 1, 2, 3]), seed=0)
    assert built["n_pairs"] == 6  # all distinct ranks
    assert set(np.unique(built["y"])) == {0, 1}
    # ties skipped
    tied = build_pairwise(np.zeros((2, 3)), np.array([1, 1]))
    assert tied["n_pairs"] == 0


def test_preprocessing_markers():
    pytest.importorskip("scanpy", reason="scanpy optional for unit tests")
    import anndata as ad
    from feature_engineering.preprocessing.pipeline import (
        PreprocessConfig, run_pipeline)
    adata = ad.AnnData(rng.poisson(2, size=(60, 30)).astype(float))
    adata.var_names = [f"G{i}" for i in range(30)]
    cfg = PreprocessConfig(min_genes=1, min_cells=1, n_top_genes=10, n_pcs=5)
    out = run_pipeline(adata, cfg)
    assert out.uns.get("bspai_clustered") is True


def _synth_df(n=40, p=6, seed=0):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, p))
    logits = X[:, 0] * 2 + X[:, 1]
    y = (logits > np.median(logits)).astype(int)
    cols = [f"f{i}" for i in range(p)]
    df = pd.DataFrame(X, columns=cols)
    df["label"] = y
    df["stage_rank"] = np.where(y == 1, 0, 3)
    df["pair_id"] = [f"P{i}" for i in range(n)]
    return df, cols


def test_training_prediction_ranking_eval():
    pytest.importorskip("xgboost")
    pytest.importorskip("sklearn")
    from ml.evaluation.evaluate import evaluate_proba
    from ml.explainability.feature_importance import top_features
    from ml.prediction.predict import predict_target_pair
    from ml.prediction.ranking import rank_candidates
    from ml.training.train_pairwise import train_pairwise_df
    from ml.training.train_xgboost import train as train_xgb
    df, cols = _synth_df()
    m = train_xgb(df, cols, "label", seed=0)
    bundle = {"model": m, "features": cols, "kind": "xgboost_pointwise"}
    r = predict_target_pair("A", "B", {c: 0.5 for c in cols}, bundle)
    assert 0.0 <= r["prediction_score"] <= 1.0
    assert r["model"] == "xgboost_pointwise"
    ranked = rank_candidates(
        [{"target1": f"G{i}", "target2": "H",
          "features": {c: float(i) for c in cols}} for i in range(5)], bundle)
    assert [x["rank"] for x in ranked] == [1, 2, 3, 4, 5]
    scores = [x["prediction_score"] for x in ranked]
    assert scores == sorted(scores, reverse=True)
    ev = evaluate_proba(df["label"].to_numpy(),
                        m.predict_proba(
                            np.nan_to_num(df[cols].to_numpy()))[:, 1])
    assert 0.0 <= ev["auc"] <= 1.0
    assert "confusion_matrix" in ev
    assert len(top_features(m, cols, k=3)) == 3
    # pairwise bundle
    pb = train_pairwise_df(df, cols, seed=0)
    assert pb["kind"] == "xgboost_pairwise" and "reference" in pb
    rp = predict_target_pair("A", "B", {c: 0.5 for c in cols}, pb)
    assert 0.0 <= rp["prediction_score"] <= 1.0


def test_service_imports():
    pytest.importorskip("fastapi")
    import ml.service.app as app
    assert app.health()["status"] == "ok"
