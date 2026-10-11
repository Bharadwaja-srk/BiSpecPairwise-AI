# Clinical Quality Report — four-indication BsAb draft (row-by-row verification requested)

Date: 2026-10-11. Artifact: `clinical_four_indication_draft.csv` (26 rows x 15 cols).
Status: **draft for your verification. No feature generation, no training.**
Scope: NSCLC, pancreatic (PDAC), HNSCC, liver as HCC only. Out-of-scope records
from the earlier `clinical_pairs_draft.csv` are NOT merged here; §7 covers the
optional-extension question separately.

## 1. What was searched

ClinicalTrials.gov records (NCT05149326, NCT05899608, NCT06692738, NCT06627647,
NCT06129864, NCT06525220, NCT06496178, NCT07619911, NCT07644208, NCT05773105),
FDA/NCI pages (amivantamab; zenocutuzumab accelerated approval, NCI 2025-03-26),
sponsor pipeline appendix (AstraZeneca H1/Q2-2026: eVOLVE/ARTEMIDE/GEMINI),
peer-reviewed papers/abstracts (Ann Oncol 1330P KN046 NSCLC; Nat Commun 2025
KN046+HCC NCT04542837; Front Immunol COMPASSION-08; Springer real-world
cadonilimab HCC; iScience 2025 cadonilimab PDAC retrospective; ASCO 2026
eVOLVE-02 HNSCC interim; JCO 2025 petosemtamab+pembro HNSCC Ph2), and two
landscape reviews (Front Pharmacol 2026 NSCLC table; Frontiers GI review).
Each row carries its source URL; overall stage and indication-specific stage
are separate columns.

## 2. Stage coverage (eligible rows only, n=14)

- rank 0 Approved: 3 (amivantamab NSCLC; zenocutuzumab NSCLC + PDAC).
- rank 1 Phase 3: 8 (ivonescimab, KN046-NSCLC, rilvegostomig x2 histologies,
  volrustomig-NSCLC, acasunlimab, KN046-PDAC, volrustomig-HNSCC).
- rank 2 Phase 2/1-2: 3 (volrustomig-HNSCC-Ph2, KN046-HCC, cadonilimab-HCC).
- rank 3 preclinical / rank 4 discontinued: ZERO rows found in this pass
  (reported gap, not filled).

## 3. Indication coverage (eligible): NSCLC 8, pancreatic 2, HNSCC 2, HCC 2

All four indications present. Weakest: HNSCC/HCC/PDAC (2 rows each); HNSCC rows
share one pair (volrustomig PDCD1-CTLA4); HCC rows are both rank 2.

## 4. Duplicates, self-pairs, formats, CD3E bias

- Duplicate-pair rows (by design, indication/setting-specific): rilvegostomig
  (2 histologies), volrustomig HNSCC (Ph3 + Ph2), zenocutuzumab (NSCLC + PDAC).
  Caveat: same-pair rows at different stages yield zero feature-diffs in the
  pairwise transform — label policy must decide keep/exclude before training.
- Self-pairs: none in eligible set (zanidatamab excluded with prior draft).
- Incompatible formats: all eligible rows are IgG-like BsAbs. Excluded with
  reasons: SHR-1701 (TGF-beta trap), IBI363 + unnamed PD-1/IL-2 (cytokine
  fusion), tebentafusp-class absent here, cadonilimab+regorafenib row
  (phase unconfirmed).
- CD3E-arm bias: ZERO eligible rows contain CD3E (prior draft's 8/12
  CD3E dominance came from out-of-scope heme records). Eligible arms
  concentrate instead on PDCD1 (4 pairs), CTLA4/CD274 (2 each) — checkpoint
  bias remains, recorded.

## 5. Unique pairs (eligible): 7

EGFR-MET, ERBB2-ERBB3, PDCD1-VEGFA, CD274-CTLA4, PDCD1-CTLA4, PDCD1-TIGIT,
CD274-TNFRSF9. All 14 arms are inside the 63-gene universe. Excluded for
universe reasons: petosemtamab EGFR-LGR5 (LGR5 not in universe; expansion
candidate), SHR-1701/IBI363/IL-2 rows (ligand arms outside universe).

## 6. Valid pairwise comparisons under the label policy

`ml/data/pairwise.py` uses distinct-rank combos: 3x8 + 3x2 + 8x2 = **57
cross-rank combos** (mirrored to 114 training vectors). No negatives
manufactured; unreported pairs never labeled; ties skipped by construction.

## 7. Is this enough to train a meaningful model? Honestly: pilot-grade only

For: real multi-rank signal (3 ranks), all indications represented, zero CD3E
confounding, all arms in-universe, every row sourced.
Against: n=14 rows / 7 pairs is small; rank 0 rests on 2 pairs (both HER-family
bridges); no preclinical/discontinued contrast (rank 3/4 absent); checkpoint
bias (PDCD1 in 4/7 pairs); 2 eligible rows rest on single sources flagged in
CSV (acasunlimab: review-cited NCT ID, direct fetch pending; volrustomig-NSCLC:
sponsor appendix); HNSCC/HCC/PDAC thin.
Verdict: sufficient to validate the pipeline end-to-end and run a pilot with
leave-one-pair-out evaluation, with results labeled exploratory. NOT sufficient
for generalizable efficacy claims. Recommended additions (need approval to
curate): rank-4 discontinued + rank-3 records, more rank-2 rows, direct CT.gov
fetches for the 2 flagged rows, and petosemtamab universe decision (add LGR5?).

## 8. Optional methodological extension (NOT merged, scope unchanged)

The out-of-scope approved records (CD19/CD20/BCMA/GPRC5D/DLL3 x CD3 heme pairs
in `clinical_pairs_draft.csv`) could augment rank-0 contrast as a *separate
sensitivity analysis*, since BsAb developability shares format-level signal.
Adopting this would need your explicit approval and must be reported as
out-of-scope augmentation, never as supported indications. Default: excluded.

## v2 update (2026-10-11): `clinical_four_indication_draft_v2.csv` (28 rows, v1 preserved)

### Flagged-record resolution
- **Acasunlimab: RESOLVED.** Direct CT.gov API fetch of NCT06635824: Genmab
  sponsor, ACTIVE_NOT_RECRUITING, PHASE3, PD-L1+ LA/metastatic NSCLC
  (acasunlimab+pembrolizumab vs docetaxel). Verification upgraded to
  `registry`. Eligibility unchanged (yes).
- **Volrustomig eVOLVE-Lung02: RESOLVED with correction.** NCT05984277 located
  (D798AC00001): the NSCLC program was **discontinued 2026-08-17 for futility**
  (AZ press release + MPR; IDMC: PFS/OS unlikely met). Row re-ranked 1 -> 4
  with program outcome recorded; drug overall stays investigational
  (HNSCC/cervical/mesothelioma ongoing) — the overall-vs-indication split
  working as designed. This is the first verified rank-4 record. Note: CT.gov
  status field still reads ACTIVE_NOT_RECRUITING at fetch; press release
  governs the program status, both recorded.

### Verification catches (kept, not guessed)
- SHR-1701 review-claimed Phase 3 NCT05812066 is a **dental-prosthesis trial**;
  claim removed, row keeps verified Ph2 only. Still ineligible (trap format).
- Sabestomig row was **mistargeted**: RO7247669 is PD-1xLAG-3, now named
  **tobemstomig** (AACR/CCR 2026 first-in-human). Corrected to PDCD1-LAG3,
  NSCLC Ph1/2 (NCT04140500 direct fetch: PHASE1/2, conditions incl. NSCLC),
  eligible yes. HNSCC cohort exists only in the ASCO22 abstract, so scope is
  kept to verified NSCLC.
- New verified rows: **bintrafusp alfa** NSCLC INTR@PID Lung 037 discontinued
  2021-01-20 (Merck PR + 2021 annual report; rank 4, ineligible — trap format,
  TGFB arm outside universe) and **cadonilimab+anlotinib** NSCLC Ph1b/2
  NCT04646330 COMPLETED (direct fetch + PMC table ORR 62.5%; rank 2, eligible).

### Same-pair, multi-stage policy (adopted for curation; enforced at build time)
Longitudinal rows are preserved (rilvegostomig x2 histologies; volrustomig
HNSCC Ph3+Ph2; zenocutuzumab NSCLC+PDAC; cadonilimab-HCC + PDAC-retro
excluded-row). At dataset build: (a) duplicate-pair rows are NOT independent
examples — group key = pair for all splits/CV (`pair_id` already exists in
`ml/data/clinical_stages.py` + `group_split` in `ml/data/pairwise.py`);
(b) same-pair rows at different stages produce zero feature-diffs and are
EXCLUDED from the pairwise transform (they carry no ordering information,
only label noise); the most advanced verified stage per (pair, indication)
is the training representative; (c) histology/setting duplicates collapse to
one row per (pair, indication) before counting. Counts below follow (c).

### LGR5/petosemtamab assessment (separate; universe unchanged at 63)
Petosemtamab (EGFRxLGR5) has verified Ph3 HNSCC rows (LiGeR-HN1 NCT06525220,
HN2 NCT06496178) and Genmab confirms the EGFR+LGR5 mechanism, but LGR5 is
outside the provisional 63-gene universe (no HGNC/GO/Gene2Vec verification
done, no HNSCC-expression join). Options: (i) keep excluded (default);
(ii) verify LGR5 through the same ID/GO/expression pipeline and expand to 64
with your explicit approval. No genes added in v2.

### v2 recalculation (eligible n=16; counting rule (c) applied)
- Unique pairs: 8 (+PDCD1-LAG3). All arms in-universe.
- Ranks: 0:3 / 1:7 / 2:5 / 4:1. Rank 3 (preclinical) still ZERO — gap remains.
- Indications: NSCLC 10, pancreatic 2, HNSCC 2, HCC 2.
- CD3E-arm rows: 0. Arms: PDCD1 x5 pairs, CD274 x2, CTLA4 x2 — checkpoint bias.
- Valid cross-rank combos: **86** (21+15+3+35+7+5). Rank-4 contrast rests on ONE
  pair (PDCD1-CTLA4 volrustomig-NSCLC).
- Ineligible (12): format/universe (SHR-1701, IBI363, bintrafusp, PD-1/IL-2,
  cadonilimab+regorafenib-ambiguous), single-source (opamtistomig x2,
  rilve-HCC, cibisatamab), retrospective (cadonilimab-PDAC), universe
  (petosemtamab x2), abstract-only (now zero — tobemstomig resolved).

### Sufficiency v2: exploratory pilot defensible, generalization claim NOT
For: 4 ranks present (incl. verified discontinued), all indications present,
n=16 rows / 8 pairs, 86 combos, zero CD3E confounding, all arms in-universe,
two independent verification upgrades + two caught errors demonstrate the
process works. Pilot with leave-one-pair-out CV is now defensible as
pipeline validation with results labeled exploratory.
Against: rank 4 = single pair; rank 0 = 2 HER-family pairs; checkpoint bias;
HNSCC/HCC/PDAC thin (2 rows each); preclinical rank absent; no AUC promised —
with n=16, fold variance will dominate any point estimate.
Needed for stronger: rank-3 rows, 2+ more rank-4 pairs, more rank-2
(HNSCC/HCC/PDAC), LGR5 decision, re-verification pass at build time.
