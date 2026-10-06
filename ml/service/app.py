"""ML HTTP service (FastAPI). Backend calls http://ml:8001 in compose."""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ml.config import get_logger
from ml.prediction.predict import load_bundle, predict_target_pair
from ml.prediction.ranking import rank_candidates

log = get_logger("ml-service")
app = FastAPI(title="BSPAI ML service")
_bundle: dict | None = None


class PredictRequest(BaseModel):
    target1: str
    target2: str
    features: dict


class RankRequest(BaseModel):
    candidates: list[PredictRequest]
    top_k: int | None = None


def bundle() -> dict:
    global _bundle
    if _bundle is None:
        path = os.getenv("MODEL_PATH", "models/xgb_pairwise_v1.pkl")
        if not os.path.exists(path):
            raise HTTPException(503, f"model not found: {path}. Train first.")
        _bundle = load_bundle(path)
        log.info("loaded model %s", path)
    return _bundle


@app.get("/health")
def health() -> dict:
    return {"status": "ok",
            "model_loaded": os.path.exists(
                os.getenv("MODEL_PATH", "models/xgb_pairwise_v1.pkl"))}


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    try:
        return predict_target_pair(req.target1, req.target2, req.features,
                                   bundle())
    except Exception as e:
        raise HTTPException(400, str(e)) from e


@app.post("/rank")
def rank(req: RankRequest) -> dict:
    try:
        cands = [c.model_dump() for c in req.candidates]
        return {"results": rank_candidates(cands, bundle(),
                                           top_k=req.top_k)}
    except Exception as e:
        raise HTTPException(400, str(e)) from e
