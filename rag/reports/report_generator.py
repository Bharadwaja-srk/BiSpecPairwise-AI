"""Structured scientific report generation + validation."""

from __future__ import annotations

import re
from typing import Any

REQUIRED_KEYS = [
    "summary", "prediction", "biological_interpretation", "mechanistic_rationale",
    "safety_considerations", "clinical_evidence", "retrieved_evidence", "limitations", "confidence",
]


def _section(text: str, name: str) -> str:
    m = re.search(rf"##\s*{name}\s*\n(.*?)(?=\n##\s*|\Z)", text, re.S | re.I)
    return m.group(1).strip() if m else ""


def parse_llm_text(text: str) -> dict[str, str]:
    return {
        "summary": _section(text, "summary") or text[:500],
        "biological_interpretation": _section(text, "biological_interpretation"),
        "mechanistic_rationale": _section(text, "mechanistic_rationale"),
        "safety_considerations": _section(text, "safety_considerations"),
        "clinical_evidence": _section(text, "clinical_evidence"),
    }


def validate_report(report: dict[str, Any]) -> dict[str, Any]:
    missing = [k for k in REQUIRED_KEYS if k not in report]
    if missing:
        raise ValueError(f"report missing keys: {missing}")
    if report.get("confidence") not in ("high", "medium", "low"):
        raise ValueError("confidence must be high|medium|low")
    if not isinstance(report.get("limitations"), list) or not report["limitations"]:
        raise ValueError("limitations must be a non-empty list")
    if not isinstance(report.get("retrieved_evidence"), list):
        raise ValueError("retrieved_evidence must be a list")
    return report


def build_report(
    target1: str,
    target2: str,
    prediction: dict[str, Any],
    llm_text: str,
    evidence: list[dict[str, Any]],
    model_name: str,
) -> dict[str, Any]:
    sections = parse_llm_text(llm_text)
    ev = [
        {
            "source_title": e.get("title", "untitled"),
            "source_type": e.get("source", "unknown"),
            "source_url": e.get("url"),
            "source_id": e.get("source_id"),
            "snippet": str(e.get("text", ""))[:1000],
            "relevance_score": float(e.get("score", 0.0) or 0.0),
        }
        for e in evidence
    ]
    conf = (_section(llm_text, "confidence") or "low").strip().lower()
    if conf not in ("high", "medium", "low"):
        conf = "low" if not ev else "medium"
    report = {
        "summary": sections["summary"] or f"Assessment of {target1}+{target2}.",
        "prediction": {"score": prediction.get("prediction_score"), "rank": prediction.get("rank", 0)},
        "biological_interpretation": sections["biological_interpretation"],
        "mechanistic_rationale": sections["mechanistic_rationale"],
        "safety_considerations": sections["safety_considerations"],
        "clinical_evidence": sections["clinical_evidence"],
        "retrieved_evidence": ev,
        "limitations": [l for l in re.split(r"\n", _section(llm_text, "limitations")) if l.strip()]
        or ["LLM interpretation is not clinical evidence.", "ML score is not a clinical probability."],
        "confidence": conf,
        "model_name": model_name,
    }
    return validate_report(report)
