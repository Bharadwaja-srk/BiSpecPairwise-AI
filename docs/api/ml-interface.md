# BSPAI — ML Service Interface (for backend / RAG agents)

Backend calls the ML container as `http://ml:8001` (never `localhost` from
inside compose). Alternatively import `ml.prediction` directly (same schema).

## POST /predict
Request: `{"target1": "CD274", "target2": "CTLA4", "features": {...}}`
`features` keys follow `feature_engineering/feature_schema.py`
(`target_activity_1/2, single_ratio_max/min, double_positive, safety_score,
pathway_similarity, pathway_shared_count, gene_embedding_similarity,
gene_embedding_distance, pearson_correlation`). Missing → NaN (pathways → 0.0).

Response:
```json
{"target1":"CD274","target2":"CTLA4","prediction_score":0.93,"rank":0,
 "features":{...},"feature_importance":{"gene_embedding_similarity":0.4,...},
 "model":"xgboost_pairwise"}
```

## POST /rank
Request: `{"candidates":[{"target1":..,"target2":..,"features":{...}}],"top_k":10}`
Response: `{"results":[{...predict payload + rank 1..N...}]}` sorted desc.

## GET /health
`{"status":"ok","model_loaded":true}`; 503-style `model_loaded:false` when
`MODEL_PATH` missing (train via `ml/training/train_pairwise.py` first).

## Model loading
`MODEL_PATH` env (default `models/xgb_pairwise_v1.pkl`), joblib bundle
`{model, features, kind, reference?}`. Pairwise scoring: `P(x − reference)`.

## Errors
- `503 model not found` — train first; `400 <msg>` — bad shapes/unknown stage.
- Pairwise ties skipped; missing genes → NaN similarity; `dx+dy==0` → safety 0.0.

## RAG handoff
`feature_importance` (normalized, desc) + raw features + Table-5 bins via
`ml/explainability/feature_importance.py::discretize`.
