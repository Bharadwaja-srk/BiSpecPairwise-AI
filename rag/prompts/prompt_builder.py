"""Prompt builder: separates ML prediction / retrieved evidence / interpretation / uncertainty."""

from __future__ import annotations

from typing import Any

from rag.discretization.feature_discretizer import describe_for_prompt


def build_prompt(
    target1: str,
    target2: str,
    prediction: dict[str, Any],
    features: dict[str, float],
    feature_importance: dict[str, float],
    evidence: list[dict[str, Any]],
    discretized: dict[str, str] | None = None,
) -> str:
    score = prediction.get("prediction_score")
    rank = prediction.get("rank", 0)
    model = prediction.get("model", "unknown")
    feat_lines = describe_for_prompt(features, feature_importance) if feature_importance else [
        f"- {k} = {v:.4g}" for k, v in features.items()
    ]
    if evidence:
        ev_lines = "\n".join(
            f"[{i}] {e.get('title', 'untitled')} ({e.get('source', 'unknown')}) — {str(e.get('text', ''))[:400]}"
            for i, e in enumerate(evidence, 1)
        )
    else:
        ev_lines = "(no retrieved biomedical evidence — state this explicitly; invent no citations)"

    return f"""You are a research-support assistant for bispecific antibody target selection. This is NOT clinical advice.

TARGETS: {target1} + {target2}

1. ML PREDICTION (do not treat as clinical probability):
- score = {score}, rank = {rank}, model = {model}
- A high score means the pair resembles historically successful bispecifics, not guaranteed efficacy.

2. BIOLOGICAL FEATURES (raw values):
{chr(10).join(f'- {k} = {float(v):.4g}' for k, v in features.items())}

3. IMPORTANT FEATURES (natural-language bins, Table-5 thresholds):
{chr(10).join(feat_lines)}

4. RETRIEVED BIOMEDICAL EVIDENCE (only facts below may be cited):
{ev_lines}

5. TASK — write a structured assessment with sections:
summary | biological_interpretation | mechanistic_rationale | safety_considerations | clinical_evidence | limitations | confidence (high/medium/low).
Rules: distinguish Prediction vs Evidence vs Interpretation vs Uncertainty. Do NOT claim clinical efficacy, regulatory approval, or trial outcomes without evidence. Do NOT fabricate citations, trials, or mechanisms. List limitations explicitly (incl. missing features, no-evidence cases, ML score limits). Do not blindly agree with the ML model — note disagreements.
"""
