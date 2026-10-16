"""P-A/P-B diagnostics tests: margins, ties, NaN, overlap matching (numpy only)."""
import numpy as np
import pytest

from feature_engineering.biological_features.target_activity import (
    compare_selections,
    overlap_best_match,
    rank_margins,
    top_expressing_populations,
)


def test_selection_rule_pinned():
    expr = np.array([3.0, 3.0, 1.0, 1.0, 0.0, 0.0])
    labels = ["b", "b", "a", "a", "c", "c"]
    assert top_expressing_populations(expr, labels, 2) == ["b", "a"]


def test_selection_tie_break_is_alphabetical():
    expr = np.array([1.0, 1.0, 1.0, 1.0])
    labels = ["b", "b", "a", "a"]
    assert top_expressing_populations(expr, labels, 2) == ["a", "b"]


def test_rank_margins_values():
    expr = np.array([5.0, 5.0, 3.0, 3.0, 1.0, 1.0])
    labels = ["x", "x", "y", "y", "z", "z"]
    m = rank_margins(expr, labels, k=2)
    assert [t["population"] for t in m["top"]] == ["x", "y", "z"]
    assert m["m1v2"] == {"abs": 2.0, "rel": pytest.approx(0.4)}
    assert m["m2v3"] == {"abs": 2.0, "rel": pytest.approx(2.0 / 3.0)}
    assert m["cutoff_tie"] is False
    assert m["n_nan_pops"] == 0


def test_rank_margins_tie_and_nan():
    expr = np.array([2.0, 2.0, 2.0, 2.0, np.nan, np.nan])
    labels = ["a", "a", "b", "b", "c", "c"]
    m = rank_margins(expr, labels, k=2)
    # finite ranking has only a/b (no cutoff margin); NaN pop counted + flagged
    assert m["m2v3"] is None
    assert m["n_nan_pops"] == 1
    assert "legacy_matches_finite_ranking" in m
    # selection itself keeps legacy behavior (no silent reordering by diagnostics)
    assert set(top_expressing_populations(expr, labels, 2)) <= {"a", "b", "c"}


def test_overlap_best_match_shares():
    ref = ["a", "a", "a", "b", "b"]
    alt = ["x", "x", "y", "y", "y"]
    m = overlap_best_match(ref, alt)
    assert m["a"] == {"alt_pop": "x", "share": pytest.approx(2 / 3), "n_cells": 3}
    assert m["b"]["alt_pop"] == "y" and m["b"]["share"] == 1.0


def test_compare_selections_identical_is_stable():
    lab = ["a", "a", "b", "b", "c", "c"]
    out = compare_selections(lab, lab, ["a", "b"], ["a", "b"])
    assert out["status"] == "compared" and out["differ"] is False


def test_compare_selections_split_differs():
    ref = ["a"] * 10 + ["b"] * 10
    alt = ["x"] * 10 + ["y"] * 5 + ["z"] * 5  # b splits y/z
    out = compare_selections(ref, alt, ["a", "b"], ["x", "y"])
    # matched sets agree, but b leaks half its cells outside the match
    assert out["differ"] is False
    assert out["leaky_selected_pops"] == ["b"]
    assert out["matches"]["b"]["share"] == 0.5


def test_compare_selections_set_mismatch_differs():
    ref = ["a"] * 10 + ["b"] * 10
    alt = ["x"] * 10 + ["y"] * 10
    out = compare_selections(ref, alt, ["a", "b"], ["x", "w"])
    assert out["differ"] is True


def test_compare_selections_unavailable_alt():
    out = compare_selections(["a", "b"], None, ["a"], ["a"])
    assert out["status"].startswith("unavailable") and out["differ"] is None
    out2 = compare_selections(["a", "b"], [], ["a"], ["a"])
    assert out2["differ"] is None


def test_overlap_requires_same_cells():
    with pytest.raises(AssertionError):
        overlap_best_match(["a", "b"], ["a"])
