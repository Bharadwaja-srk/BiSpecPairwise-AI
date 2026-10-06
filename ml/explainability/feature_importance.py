"""XGBoost feature importance (split-frequency / gain), machine-readable.

Output feeds the RAG/LLM agent: ranked list + binned natural-language
labels per the paper's Table 5 discretization.
"""
from __future__ import annotations

# Paper Table 5 bins: (upper_bound, label)
DISCRETIZATION_BINS: dict[str, list[tuple[float, str]]] = {
    "gene_embedding_similarity": [
        (3.67, "very similar"), (4.02, "somewhat similar"),
        (4.34, "similar"), (4.77, "dissimilar"), (float("inf"), "very dissimilar")],
    "safety_score": [
        (-0.0071, "unsafe"), (-0.000073, "moderately low safety"),
        (0.0012, "safe"), (0.013, "moderately high safety"),
        (float("inf"), "high safety")],
    "pearson_correlation": [
        (-0.020, "strongly negative correlation"),
        (-0.004, "weakly negative correlation"),
        (0.004, "uncorrelated"), (0.020, "weakly positive correlation"),
        (float("inf"), "strongly positive correlation")],
    "single_ratio_max": [
        (0.11, "low"), (0.24, "moderately low"), (0.45, "moderate"),
        (0.67, "moderately high"), (float("inf"), "high")],
    "double_positive": [
        (0.00019, "low"), (0.0015, "moderately low"), (0.0071, "moderate"),
        (0.35, "moderately high"), (float("inf"), "high")],
}


def discretize(feature: str, value: float) -> str:
    bins = DISCRETIZATION_BINS.get(feature)
    if not bins:
        return str(value)
    for upper, label in bins:
        if value <= upper:
            return label
    return bins[-1][1]


def importance_dict(model, feature_names: list[str],
                    kind: str = "weight") -> dict:
    """Return {feature: score} sorted desc. Supports XGB + sklearn models."""
    try:
        booster = model.get_booster()
        raw = booster.get_score(importance_type=kind)
        # xgb names features f0..fn when given numpy
        out = {}
        for i, name in enumerate(feature_names):
            out[name] = float(raw.get(f"f{i}", raw.get(name, 0.0)))
        total = sum(out.values()) or 1.0
        return dict(sorted(((k, v / total) for k, v in out.items()),
                           key=lambda kv: kv[1], reverse=True))
    except Exception:
        pass
    if hasattr(model, "feature_importances_"):
        vals = list(map(float, model.feature_importances_))
        total = sum(vals) or 1.0
        return dict(sorted(zip(feature_names, [v / total for v in vals]),
                           key=lambda kv: kv[1], reverse=True))
    return {n: 0.0 for n in feature_names}


def top_features(model, feature_names: list[str], k: int = 6) -> list[dict]:
    imp = importance_dict(model, feature_names)
    return [{"feature": n, "importance": s}
            for n, s in list(imp.items())[:k]]
