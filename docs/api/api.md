# BSPAI — Backend API Contract (for frontend)

Base URL (local): `http://localhost:8000`. Frontend talks **only** to the backend.
Backend → ML: `http://ml:8001` (compose) · Backend → RAG: `http://rag:8002` · DB: `postgres:5432`.
Never use `localhost` for inter-container calls.

## Endpoints

### GET /api/health
`200 {"status":"healthy"|"degraded","service","version","environment","database":"connected"|"disconnected"}`

### POST /api/analyze
Request: `{"target1":"CD274","target2":"CTLA4","features":{...optional 11-key dict...}}`
Symbols are uppercased; same-gene or empty → `422`. Optional `features` keys are the 11
`FEATURE_ORDER` names (see `GET /api/features`); omitted keys resolve via precomputed store
(`FEATURE_STORE_PATH`) or NaN-fill.

Response `200`:
```json
{
  "target1": "CD274", "target2": "CTLA4", "status": "completed",
  "prediction": {
    "prediction_score": 0.93, "rank": 0,
    "features": {"safety_score": 0.0012, "target_activity_1": null, "...": null},
    "discretized_features": {"safety_score": "safe", "...": "..."},
    "missing_features": ["target_activity_1", "..."],
    "feature_status": "incomplete",
    "feature_source": "nan-fill",
    "feature_importance": [{"feature": "safety_score", "importance": 0.6, "rank": 1}],
    "model": "xgboost_pairwise"
  },
  "retrieved_evidence": [],
  "report": {"summary": "...", "prediction": {"score": 0.93, "rank": 0},
    "biological_interpretation": "...", "mechanistic_rationale": "...",
    "safety_considerations": "...", "clinical_evidence": "...",
    "retrieved_evidence": [], "limitations": ["..."], "confidence": "low|medium|high"},
  "rag_error": null,
  "persistence": "persisted|skipped"
}
```
Notes: `rank: 0` = single-pair score (real ranks come from `/api/ranking`). `features` values are
`null` when unknown — never fabricated. `feature_status:"incomplete"` means the score is **not
biologically valid** on its own. `report:null + rag_error` set when RAG is down (ML result kept).
`model:"mock-*"` marks mock-ML output (dev/test only, never real biology). When DB persistence
succeeds the response additionally carries `analysis_id`; `created_at`/`completed_at` are always present.
Errors: `422` validation · `502/503/504` ML unreachable/model-missing/timeout.

### GET /api/ranking?top_k=10&candidate=T1:T2 (repeatable)
`200 {"items":[{"rank":1,"target1":"...","target2":"...","prediction_score":0.9,"model":"..."}],"total":N,"source":"caller-supplied"|"dev-candidates (NOT real biology)"}`.
No `candidate` params → small clearly-marked dev set (flagged via `source`). Maps to ML `POST /rank`.

### GET /api/targets → `{"items":[{"gene_symbol","gene_name"}],"total":N,"source":"database"|"dev-catalogue (NOT real biology)"}`
### GET /api/targets/{symbol} → one target or `404`
### GET /api/features → 11-name feature schema (`name/description/source/range/missing`)
### GET /api/reports/{analysis_id} → persisted analysis+prediction+report or `404` (`503` when DB down)

## RAG service (backend-internal, not for frontend)

`POST /generate-report {target1,target2,prediction,features,feature_importance,discretized_features}`
→ validated report JSON · `POST /retrieve {target1,target2,context,top_k}` → `{"results":[...]}` (or `[]`)
· `POST /ingest {documents:[{text,title,source,url,source_id,metadata}],chunk_size}` · `GET /health`.

## ML integration (existing service, unchanged)

`GET /health → {status,model_loaded}` · `POST /predict {target1,target2,features(11 keys)}` →
`{target1,target2,prediction_score,rank:0,features:echo,feature_importance:{name:score},model}` ·
`POST /rank {candidates,top_k} → {results:[...+rank 1..N]}`. `model_loaded:false` until
`ml/training/train_pairwise.py` produces `models/xgb_pairwise_v1.pkl`. Backend sends the full
11-key dict always; `feature_importance` dict is sorted into a ranked list for the frontend.

## Environment variables

`DATABASE_URL` · `ML_SERVICE_URL` (default `http://ml:8001`) · `ML_SERVICE_TIMEOUT=30` ·
`RAG_SERVICE_URL` (`http://rag:8002`) · `RAG_SERVICE_TIMEOUT=60` · `FEATURE_STORE_PATH` ·
`ENABLE_MOCK_ML/RAG/LLM` (dev/test) · `LLM_PROVIDER=mock|openai|openrouter` · `LLM_MODEL` ·
`OPENAI_API_KEY` / `OPENROUTER_API_KEY` (never committed, never sent to frontend) ·
`EMBEDDING_PROVIDER=hashing` · `EMBEDDING_DIM=128` · `PORT/HOST/SECRET_KEY`. See `.env.example`.
