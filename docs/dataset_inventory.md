# Dataset Inventory — Phase 1 (verified 2026-10-11, no processing performed)

Scope: exactly four tumor datasets. No extraction beyond header/tar-listing reads.
No Scanpy runs, no feature generation, no training.

## 1. Verified identity table

| Accession | Tumor / subtype | Species | Design (per GEO) | Local archive | Size | Format | Status |
|---|---|---|---|---|---|---|---|
| GSE131907 | NSCLC as lung adenocarcinoma (LUAD); Kim et al. 2020 Nat Commun PMID 32385277 | Homo sapiens | 208,506 cells; 58 samples; 44 pts; primary tumor, LN, brain-mets, pleural effusion + normal lung/LN | `GSE131907_Lung_Cancer_cell_annotation.txt.gz` (1.9 MB) + `..._normalized_log2TPM_matrix.rds.gz` (866 MB) | 0.81 GB | gzipped annotation TSV (verified: 208,506 rows; origins nLung/tLung/tL/B/nLN/mLN/PE/mBrain) + double-gzipped R RDS matrix (magic verified; needs R/pyreadr double-gunzip) | NOT processed |
| GSE205013 | Pancreatic ductal adenocarcinoma (PDAC), primary + liver-metastatic, pre/post-chemo; Werba et al., PMID 36781852 | Homo sapiens | 27 samples (81 files = 27 x barcodes/features/matrix MTX triplets, P01–P27) | `GSE205013_RAW.tar` | 1.96 GB | TAR of 10x MTX (CellRanger-filtered despite "RAW" name, per GEO) | preprocessing FAILED (see §3) |
| GSE164690 | HNSCC, 6 HPV+ / 12 HPV- treatment-naive + matched PBL; Kurten/Ferris | Homo sapiens | 51 samples (153 files = 51 x triplets; HN01+ PBL/CD45p/CD45n fractions); 134,606 cells per GEO | `GSE164690_RAW.tar` | 0.70 GB | TAR of 10x MTX | PROCESSED, PASS |
| GSE156625 | HCC (hepatocellular carcinoma); Sharma et al. 2020 | Homo sapiens (+fetal/mouse in series — EXCLUDED) | 43 tumor + 14 adjacent (14 HCC pts) per secondary use; file is a scanpy object, not raw counts | `GSE156625_HCCscanpyobj.h5ad.gz` | 0.78 GB | gzipped HDF5 AnnData (inner magic verified) | NOT processed |

GEO sources: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE131907 (and GSE205013, GSE164690, GSE156625); HCA project page for GSE205013 (27 donors, ~139k cells est.).

## 2. Existing processed outputs (preserved, not rerun)

- `data/processed/hnscc_raw.h5ad` (1.45 GB): 151,680 cells x 33,545 genes, 51/51 samples kept.
- `data/processed/hnscc.h5ad` (2.23 GB): 143,441 cells x 2,000 HVG, 27 clusters, full `bspai_*` markers — VALID, keep.
- `log_hnscc.txt` PASS; `qc_hnscc.json` matches. `log_pancreatic.txt` FAIL only.

## 3. Known defect (root cause, from log evidence)

`process_10x` in ephemeral `/tmpjob/step3_preprocess.py` (NOT in repo):
`UnboundLocalError: 'batch'` at `batch.append(a)` after writing batch_0 (93,244
cells) — accumulator referenced before assignment on the multi-batch path.
Fix (Phase 2): repo-owned 10x adapter with initialized accumulator + guards,
plus regression test reproducing the shape first. No data discarded: batch_0
checkpoint exists only in lost /tmp; TAR source intact.

## 4. Exclusion rules (adopted for Phase 2)

- HCC: human HCC tumor + adjacent only; fetal-liver and mouse samples excluded.
- NSCLC: primary-tumor origins for features (tLung); nLung reserved for
  tumor-vs-normal safety; LN/brain/PE origins excluded from primary analysis
  unless a review-approved sensitivity includes them (brain: Oligodendrocytes
  present — non-lung TME).
- PDAC: record pre/post-chemo + primary/liver-met status per sample; default
  primary analysis on pre-treatment primary tumors unless approved otherwise.
- HNSCC: PBL fraction is peripheral blood, not TME — exclude from TME features
  unless approved; CD45 sorting noted (composition analyses biased).
- No adjacent-normal fabrication: datasets lacking normal tissue get safety
  marked unavailable (HNSCC/PDAC/HCC have no nLung equivalent in-hand).

## 5. Dependencies and resources

- Required: Python 3.11, scanpy/anndata/leidenalg, pyreadr (RDS), tar/gzip —
  all in `requirements-ml.txt` except system tar/gzip. Local Windows Python
  lacks scanpy (confirmed Phase 0); decided execution environment is
  **Docker Linux containers**.
- Host: 16 GB RAM, 267 GB free on C. Largest single archive 1.96 GB;
  HNSCC precedent peaks ~2.4 GB output + concat memory. Pancreatic full-concat
  (~140k+ cells) is the heaviest planned step — feasible but must run
  containerized with monitored memory.
- **BLOCKER: Docker Desktop daemon is DOWN** (`docker info` fails: missing
  `dockerDesktopLinuxEngine` pipe). Phase 2 containerized work cannot start
  until Docker is running. Alternative (not recommended): local Windows
  scanpy install — env drift risk, needs approval if Docker stays unavailable.

## 6. Coverage status and blockers summary

| Dataset | Identity | Coverage vs 63 genes | Blocker |
|---|---|---|---|
| HNSCC | verified, processed PASS | 63/63 joined (raw var index) | none (preserve outputs) |
| NSCLC | verified (LUAD) | pending (RDS unread pre-R mapping) | needs pyreadr path in container |
| Pancreatic | verified (PDAC) | pending | process_10x defect + Docker down |
| HCC | verified (HCC subset) | pending | Docker down; h5ad already AnnData (validate schema on load) |

Stop point: awaiting Phase 2 approval. Phase 2 entry requires Docker running
(or explicit approval for a fallback) plus the 10x regression test.
