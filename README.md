# BiSpecPairwise AI

> **Machine Learning & RAG Framework for Bispecific Antibody Target Pair Discovery and Prioritization**

BiSpecPairwise AI is an end-to-end computational and artificial intelligence platform designed to prioritize synergistic, tumor-specific antibody target pairs (e.g., Bispecific T-cell Engagers / BiTEs, dual CAR-T targets) from single-cell RNA sequencing (scRNA-seq) data, protein interactomes, and clinical knowledge graphs.

---

## 🔬 Core Capabilities

1. **Single-Cell Preprocessing & Quality Control (`feature_engineering/preprocessing/`)**
   - High-throughput streaming **10x Genomics MTX sparse matrix parser** with float32 in-memory accumulation.
   - Low-footprint, chunked sparse parsing that handles multi-gigabyte genomic matrices with memory caps and Stage-1/Stage-2 RSS tracking.
   - Atomic checkpointing and resumable batch runs (`run_10x_dataset.py`).

2. **Biological Feature Engineering (`feature_engineering/biological_features/`)**
   - **Co-expression & Specificity:** Spearman correlation, differential expression, double-positive cell frequency across cell types.
   - **Protein-Protein Interaction (PPI):** STRING-DB network shortest path, shared neighborhood connectivity.
   - **Pathway Co-membership:** Reactome and KEGG shared biological pathway enrichment.
   - **Target Activity Scoring:** Quantified cell-type target activation indices.
   - **Off-Target Safety Scoring:** Tumor vs normal tissue expression ratio and on-target/off-tumor risk assessment.
   - **Gene Embeddings:** 200-dimensional Gene2Vec vector space representation.

3. **Pairwise Ranking Machine Learning (`ml/`)**
   - Pairwise ranking and gradient-boosted trees (**XGBoost Ranking**, Logistic / Random Forest baselines).
   - Cross-validation and group-aware train/test splits.
   - Model interpretability and feature attribution via **SHAP**.

4. **Clinical Evidence RAG Service (`rag/`)**
   - Discretizes quantitative ML pair scores into biological semantics.
   - Vector-indexed literature and clinical trial evidence retrieval.
   - LLM-powered clinical report generation (supporting OpenAI, OpenRouter, and offline fallback).

5. **Production Microservices Backend (`backend/`)**
   - **FastAPI** REST API exposing endpoints for analysis orchestration, target querying, pairwise ranking, and report generation.
   - **PostgreSQL** relational schema with SQLAlchemy ORM and Alembic migrations.
   - **Docker Compose** containerization connecting API, ML, RAG, and Postgres services.

---

## 📅 Development Timeline & Git Milestones

| Day | Date & Time (+05:30) | Milestone | Key Components & Files |
|---|---|---|---|
| **Day 1** | **2026-10-06 11:57** | **Core ML Pipeline & Feature Engineering** | `ml/` (training, prediction, evaluation, explainability), `notebooks/` (01–06), biological feature modules (correlation, double positive, pathway, target activity, schema). |
| **Day 2** | **2026-10-08 22:37** | **Backend API & Microservices Infrastructure** | `backend/` (FastAPI 5 routes, 6 SQLAlchemy models, DB migrations, persistence, ML/RAG clients), Docker compose & multi-service setup. |
| **Day 3** | **2026-10-09 11:10** | **RAG Clinical Evidence Service & Safety Scoring** | `rag/` (loader, chunking, embeddings, vector store, retriever, LLM client, report generator, app), biological `safety.py` module, and first live scRNA-seq run. |
| **Day 4** | **2026-10-11 15:57** | **Gene2Vec & Clinical Target Universe Curation** | `docs/target_universe/` (30-row verified clinical target inventory across AML, CRC, HNSCC, PDAC), `gene_embedding.py` (Gene2Vec 200d), and repository configs. |
| **Day 4 (Phase 2)** | **2026-10-11 21:52** | **10x MTX Streaming Adapter & Checkpoints** | `mtx10x.py` (memory-controlled 10x MTX parser), atomic checkpoints, `run_10x_dataset.py`, `docs/dataset_inventory.md`, and test suite. |

---

## 🗂️ Project Directory Structure

```text
BiSpecPairwise AI/
├── backend/                  # FastAPI web application & persistence
│   ├── app/
│   │   ├── api/routes/       # REST API endpoints (analyze, targets, ranking, reports, features)
│   │   ├── models/           # SQLAlchemy ORM models (database.py)
│   │   ├── schemas/          # Pydantic request/response schemas
│   │   └── services/         # Analysis orchestration, ML client, RAG client
│   └── database/migrations/  # SQL schema migrations
├── rag/                      # Clinical Evidence RAG microservice
│   ├── ingestion/            # Target pair document loader & chunking
│   ├── embeddings/           # Vector embeddings generator
│   ├── retrieval/            # Vector store and semantic retriever
│   ├── llm/                  # OpenAI & OpenRouter LLM clients
│   └── reports/              # Automated clinical report synthesis
├── ml/                       # Machine learning model pipeline
│   ├── training/             # XGBoost pairwise ranker, baselines, trainer
│   ├── prediction/           # Inference and scoring engine
│   ├── evaluation/           # Metrics and cross-validation
│   └── explainability/       # SHAP feature importance
├── feature_engineering/      # Biological features & preprocessing
│   ├── biological_features/  # Co-expression, PPI, pathway, safety, gene2vec
│   └── preprocessing/        # 10x MTX streaming adapter, pipeline, QC
├── docs/                     # Documentation & dataset curation
│   ├── target_universe/      # 30-row verified clinical targets across 4 cancers
│   ├── api/                  # REST API & ML interface specifications
│   └── dataset_inventory.md  # GEO single-cell scRNA-seq inventory
├── notebooks/                # Exploratory data analysis & validation (01–06)
└── docker-compose.yml        # Orchestration for backend, ml, rag, and postgres
```

---

## 🎯 Target Universe & Indications

The platform includes a curated, clinically verified target universe (`docs/target_universe/clinical_four_indication_draft_v2.csv`) focusing on four high-unmet-need oncological indications:

1. **Acute Myeloid Leukemia (AML)**: CD33, CD123, CLEC12A, FLT3, WT1, CD70
2. **Colorectal Cancer (CRC)**: CEACAM5, EGFR, EPCAM, MSLN, GUCY2C
3. **Head and Neck Squamous Cell Carcinoma (HNSCC)**: EGFR, EPCAM, TROP2, MET
4. **Pancreatic Ductal Adenocarcinoma (PDAC)**: MSLN, CLDN18.2, CEACAM5, TROP2, MUC1

---

## 🚀 Quickstart

### Prerequisites
- Python 3.10+
- PostgreSQL (or Docker)

### Running with Docker Compose
```bash
docker-compose up --build
```
Services will be available at:
- **Backend API:** `http://localhost:8000/docs` (Swagger UI)
- **ML Service:** `http://localhost:8001`
- **RAG Service:** `http://localhost:8002`

### Running Tests
```bash
pytest tests/
```

---

## 📄 License
MIT License
