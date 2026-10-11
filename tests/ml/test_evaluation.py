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
