# Docker + Data Runbook for Tomorrow (PREPARED, NOT EXECUTED — needs approval)

## 0. Preconditions (run first, stop if any fail)

```powershell
docker info --format '{{.ServerVersion}} mem={{.MemTotal}}'
# expect: version + MemTotal >= ~10 GB. If daemon unreachable: start Docker Desktop, wait, retry.
Get-PSDrive C | Select-Object Free   # expect >= ~40 GB free (archives ~4.5 GB + outputs ~15 GB + images)
docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}'  # expect bspai-ml:step3 present (~2.6 GB)
```

## 1. Pancreatic resume (GSE205013, 27 samples, 308,181 cells)

Valid inputs (verified, do NOT regenerate): `pancreatic_raw.batch{1,2,3}.h5ad`
(int64), `pancreatic_raw.h5ad`. Peak estimate float32 path: ~7-8 GB -> cap 10g.

```powershell
# Step 1a: recast remaining int64 batches to float32 (bounded, ~2.5 GB peak each)
docker run --rm -m 6g --cpus=8 -v "<repo>:/work" -w /work bspai-ml:step3 `
  python -m feature_engineering.preprocessing.run_10x_dataset `
  --tar data/raw/GSE205013_RAW.tar --name pancreatic `
  --out-raw data/processed/pancreatic_raw.h5ad --out data/processed/pancreatic.h5ad `
  --qc data/processed/qc_tmp.json --log data/processed/log_recast2.txt `
  --recast-batches "data/processed/pancreatic_raw.batch*.h5ad"
# verify: new *.f32.h5ad readable (h5py meta), then continue
# Step 1b: concat f32 batches + staged pipeline (single command, checkpoints resume)
docker run --rm -m 10g --cpus=10 -v "<repo>:/work" -w /work bspai-ml:step3 `
  python -m feature_engineering.preprocessing.run_10x_dataset `
  --tar data/raw/GSE205013_RAW.tar --name pancreatic `
  --out-raw data/processed/pancreatic_raw.h5ad --out data/processed/pancreatic.h5ad `
  --qc data/processed/qc_pancreatic.json --log data/processed/log_pancreatic_full.txt `
  --resume-batches "data/processed/pancreatic_raw.batch*.f32.h5ad" --overwrite
# expected: concat 308181 x 36601 -> QC/filter/normalize/HVG/PCA/cluster + qc_pancreatic.json
```
Rollback: stage checkpoints (`pancreatic.stage_*.h5ad`) resume automatically via
uns markers; `.bak` files and raw TAR never touched. On failure: inspect the
stage log tail + new checkpoint sizes before any retry. Do NOT repeat the
identical command without diagnosing (see Phase A PCA notes).

## 2. HCC (GSE156625, file is scanpy h5ad.gz, NOT raw counts)

```powershell
# inspect-only first: gunzip -c | h5ls equivalent via python h5py metadata read
# load with anndata, keep obs human-HCC-tumor + adjacent only; drop fetal/mouse
# (filter values TBD from obs columns at runtime - record used values in QC)
```
Gate: confirm obs columns contain species/sample-type fields before filtering.

## 3. NSCLC (GSE131907, double-gzipped R RDS matrix + annotation TSV)

Requires pyreadr (in bspai-ml:step3 image — verified present). Adapter needed:
RDS->matrix + GEO annotation join (208,506 rows verified). Draft command shape
TBD after adapter code review tomorrow. Tumor/normal split: tLung vs nLung.

## 4. Services (after data)

```powershell
docker compose build ml rag backend
docker compose up -d postgres ml rag backend
docker compose exec backend python -m pytest tests -q  # container-side suite
curl http://localhost:8000/api/health; curl http://localhost:8001/health; curl http://localhost:8002/health
```
Env needed: `OPENAI_API_KEY` for real LLM (else mock explicit);
`EMBEDDING_*` per model_manifest.json; model weights download (~876 MB) inside
RAG build. Postgres init via `backend/app/database/migrations/001_init.sql`.

## 5. RAG model verification (inside RAG container)

```powershell
# assert get_embedder() with EMBEDDING_PROVIDER=sentence-transformers loads
# NeuML/pubmedbert-base-embeddings@b79526d6, dim 768, fingerprint stable;
# ingest docs/rag_corpus_manifest.json sources; run docs/rag_retrieval_evalset.json queries manually.
```
