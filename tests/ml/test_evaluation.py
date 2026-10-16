"""Evaluation-code fixtures: pairwise transform, leakage-safe splits, metrics.

Small controlled arrays only — no biological data, no trained artifacts.
These tests establish that the evaluation machinery is correct; they say
nothing about real predictive accuracy.
"""
import numpy as np
import pytest

pytest.importorskip("sklearn")


def test_build_pairwise_ties_skipped_and_mirrored():
    from ml.data.pairwise import build_pairwise

    X = np.array([[3.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    ranks = np.array([0, 1, 1, 2])
    out = build_pairwise(X, ranks, seed=0)
    # combos with unequal ranks: (0,1),(0,2),(0,3),(1,3),(2,3) = 5
    assert out["n_pairs"] == 5
    assert out["X_diff"].shape == (10, 2)
    assert set(np.unique(out["y"])) == {0, 1}
    # mirrored rows sum to zero diff
    assert np.allclose(out["X_diff"][0::2] + out["X_diff"][1::2], 0.0)
    assert (out["y"][0::2] == 1).all() or (out["y"][0::2] == 0).all()


def test_build_pairwise_max_pairs_reproducible():
    from ml.data.pairwise import build_pairwise

    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 4))
    ranks = np.array([0, 0, 1, 1, 2, 2, 3, 4])
    a = build_pairwise(X, ranks, max_pairs=6, seed=7)
    b = build_pairwise(X, ranks, max_pairs=6, seed=7)
    assert a["n_pairs"] == 6
    assert np.array_equal(a["X_diff"], b["X_diff"])


def test_group_split_keeps_pairs_together():
    from ml.data.pairwise import group_split

    ids = ["P1", "P1", "P2", "P2", "P3", "P3", "P4", "P4"]
    for tr, te in group_split(ids, n_splits=2, seed=0):
        assert set(ids[i] for i in tr).isdisjoint({ids[i] for i in te})


def test_cross_validate_grouped_runs_and_bounds():
    from sklearn.dummy import DummyClassifier

    from ml.evaluation.cross_validation import cross_validate

    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 3))
    y = (X[:, 0] > 0).astype(int)
    groups = [f"G{i // 4}" for i in range(20)]
    out = cross_validate(X, y, lambda: DummyClassifier(strategy="prior"),
                         groups=groups, n_splits=5, seed=0)
    assert len(out["auc_per_fold"]) == 5
    assert 0.0 <= out["auc_mean"] <= 1.0 or np.isnan(out["auc_mean"])
    assert out["auc_std"] >= 0.0


def test_evaluate_proba_perfect_and_report_keys():
    from ml.evaluation.evaluate import evaluate_proba

    y = np.array([0, 0, 1, 1])
    out = evaluate_proba(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert out["auc"] == 1.0
    assert out["accuracy"] == 1.0
    assert out["confusion_matrix"] == [[2, 0], [0, 2]]
    single = evaluate_proba(np.array([1, 1]), np.array([0.9, 0.8]))
    assert np.isnan(single["auc"])  # single-class -> NaN, never forced to 0


def test_ndcg_perfect_is_one_and_mrr_sane():
    from ml.evaluation.metrics import mrr, ndcg

    assert ndcg([3, 2, 1, 0]) == 1.0
    assert ndcg([0, 0, 3]) < 1.0
    assert mrr([1]) == 1.0
    assert mrr([2]) == 0.5
    assert mrr([]) == 0.0


def _mkrow(t1, t2, scope, rank):
    return {"target1": t1, "target2": t2, "indication_scope": scope,
            "stage_rank": rank,
            "pair": "__".join(sorted([t1, t2]))}


def _mkstore(*pairs):
    return {"dummy": {p: {"features": {k: 0.5 for k in
            ["target_activity_1", "target_activity_2", "single_ratio_max",
             "single_ratio_min", "double_positive", "safety_score",
             "pathway_similarity", "pathway_shared_count",
             "gene_embedding_similarity", "gene_embedding_distance",
             "pearson_correlation"]}, "missing_features": []} for p in pairs}}


def test_pilot_assemble_representative_and_indication_match():
    from ml.evaluation.pilot import SCOPE_STORES, assemble

    h_store, c_store = _mkstore("A__B"), _mkstore("C__D")
    stores = {SCOPE_STORES["in-scope:HNSCC"][1]: h_store["dummy"],
              SCOPE_STORES["in-scope:liver-HCC"][1]: c_store["dummy"]}
    rows = [_mkrow("A", "B", "in-scope:HNSCC", 2),
            _mkrow("A", "B", "in-scope:HNSCC", 1),  # advanced wins
            _mkrow("C", "D", "in-scope:liver-HCC", 2),
            _mkrow("E", "F", "in-scope:NSCLC", 0)]  # no store -> incomplete
    out = assemble(rows, stores)
    assert len(out["representatives"]) == 3
    assert len(out["attached"]) == 2
    assert len(out["incomplete"]) == 1
    assert out["comparisons"] == []  # distinct spaces, distinct ranks untouched


def test_pilot_assemble_counts_cross_rank_within_space():
    from ml.evaluation.pilot import SCOPE_STORES, assemble

    h_store = _mkstore("A__B", "C__D")
    stores = {SCOPE_STORES["in-scope:HNSCC"][1]: h_store["dummy"]}
    rows = [_mkrow("A", "B", "in-scope:HNSCC", 0),
            _mkrow("C", "D", "in-scope:HNSCC", 2)]
    out = assemble(rows, stores)
    assert len(out["comparisons"]) == 1


def _attached_row(pair, space, rank, val=0.5):
    return {"pair": pair, "space": space, "stage_rank": rank,
            "features": [val] * 11}


def test_pilot_lopo_holds_out_pair_across_indications():
    """Same pair in two spaces: neither may train while the other is held out."""
    from ml.evaluation.pilot import run_lopo

    attached = [_attached_row("A__B", "HNSCC", 0, 0.1),
                _attached_row("A__B", "HCC", 2, 0.9),
                _attached_row("C__D", "HNSCC", 2, 0.8),
                _attached_row("E__F", "HNSCC", 0, 0.2)]
    out = run_lopo(attached, comparisons=[("x", "y", "z")], seed=0)
    by_pair = {f["held_out_pair"]: f for f in out["folds"]}
    # HNSCC A__B vs HNSCC C__D only; HCC A__B has no same-space partner (excluded)
    assert by_pair["A__B"]["n_heldout_comparisons"] == 1
    assert by_pair["A__B"]["n_train_rows"] == 2  # no A__B row in train


def test_pilot_dummy_scores_half_and_marks_non_estimable():
    from ml.evaluation.pilot import _concordant, run_lopo

    assert _concordant(None, True) == 0.5
    assert _concordant(True, True) == 1.0
    assert _concordant(False, True) == 0.0
    # single rank everywhere -> no comparisons anywhere
    attached = [_attached_row("A__B", "HNSCC", 1),
                _attached_row("C__D", "HNSCC", 1)]
    out = run_lopo(attached, comparisons=[], seed=0)
    assert all("NOT ESTIMABLE" in f["status"] for f in out["folds"])
    assert out["summary"]["pairwise_xgb"]["mean_concordance"] is None


def test_pilot_pointwise_runs_on_two_ranks():
    from ml.evaluation.pilot import run_lopo

    attached = [_attached_row("A__B", "HNSCC", 0, 0.1),
                _attached_row("C__D", "HNSCC", 2, 0.9),
                _attached_row("E__F", "HNSCC", 0, 0.15)]
    out = run_lopo(attached, comparisons=[("x",)], seed=0)
    for f in out["folds"]:
        if f["status"] == "scored":
            for arm, v in f["models"].items():
                assert v is None or 0.0 <= v <= 1.0
    assert out["summary"]["dummy_constant"]["mean_concordance"] == 0.5
