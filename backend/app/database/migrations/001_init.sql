-- BSPAI 001: initial schema (mirrors backend/app/models/database.py).
-- Run via psql or docker entrypoint. Uses pgcrypto for UUID defaults.
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS targets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    gene_symbol VARCHAR(50) UNIQUE NOT NULL,
    gene_name VARCHAR(200),
    uniprot_id VARCHAR(50),
    ensembl_id VARCHAR(50),
    description TEXT,
    function TEXT,
    pathway TEXT,
    subcellular_location VARCHAR(200),
    tissue_specificity TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_targets_symbol ON targets (gene_symbol);

CREATE TABLE IF NOT EXISTS target_pairs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    target1_id UUID NOT NULL REFERENCES targets(id),
    target2_id UUID NOT NULL REFERENCES targets(id),
    clinical_stage INTEGER,
    bsab_name VARCHAR(200),
    indication VARCHAR(200),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_target_pair UNIQUE (target1_id, target2_id)
);
CREATE INDEX IF NOT EXISTS ix_target_pair_targets ON target_pairs (target1_id, target2_id);

CREATE TABLE IF NOT EXISTS analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    target_pair_id UUID NOT NULL REFERENCES target_pairs(id),
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    request_data JSONB NOT NULL DEFAULT '{}',
    error_message TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_analyses_status ON analyses (status);

CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id UUID NOT NULL UNIQUE REFERENCES analyses(id) ON DELETE CASCADE,
    prediction_score DOUBLE PRECISION NOT NULL,
    rank INTEGER,
    total_candidates INTEGER,
    features JSONB NOT NULL DEFAULT '{}',
    feature_importance JSONB NOT NULL DEFAULT '{}',
    model_version VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id UUID NOT NULL UNIQUE REFERENCES analyses(id) ON DELETE CASCADE,
    summary TEXT NOT NULL,
    biological_interpretation TEXT NOT NULL,
    mechanistic_rationale TEXT NOT NULL,
    safety_considerations TEXT NOT NULL,
    clinical_evidence TEXT NOT NULL,
    limitations JSONB NOT NULL DEFAULT '[]',
    confidence VARCHAR(50) NOT NULL,
    retrieved_evidence JSONB NOT NULL DEFAULT '[]',
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    model_name VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS evidence (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id UUID NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
    source_title VARCHAR(500) NOT NULL,
    source_type VARCHAR(50) NOT NULL,
    source_url VARCHAR(1000),
    source_id VARCHAR(100),
    publication_date TIMESTAMPTZ,
    snippet TEXT NOT NULL,
    relevance_score DOUBLE PRECISION NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS ix_evidence_report ON evidence (report_id);
