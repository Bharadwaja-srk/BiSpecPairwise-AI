"""Feature discretization: wraps the ML Table-5 bins (single source of truth).

Thresholds come from ml/explainability/feature_importance.py::DISCRETIZATION_BINS
(Zhang et al. 2024, Table 5). This module adds batch helpers + docs for the LLM.
Unknown features fall back to the raw value string — never arbitrary labels.
"""

from __future__ import annotations

try:
    from ml.explainability.feature_importance import DISCRETIZATION_BINS, discretize
except ImportError:  # standalone RAG container without ml on path
    DISCRETIZATION_BINS: dict = {
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

    def discretize(feature: str, value: float) -> str:  # type: ignore
        bins = DISCRETIZATION_BINS.get(feature)
        if not bins:
            return str(value)
        for upper, label in bins:
            if value <= upper:
                return label
        return bins[-1][1]


def discretize_all(features: dict[str, float]) -> dict[str, str]:
    """Map every feature to its natural-language bin label."""
    return {k: discretize(k, float(v)) for k, v in features.items()}


def describe_for_prompt(features: dict[str, float], importance: dict[str, float], top_k: int = 6) -> list[str]:
    """Top-k 'feature = value (label, importance x)' lines for the LLM prompt."""
    ranked = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    lines = []
    for name, imp in ranked:
        val = float(features.get(name, float("nan")))
        lines.append(f"- {name} = {val:.4g} ({discretize(name, val)}; importance {imp:.3f})")
    return lines
