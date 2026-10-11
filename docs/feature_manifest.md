# BSPAI Feature Manifest — 11-feature schema (source of truth: `feature_engineering/`)

 reference methodology: Zhang et al. 2024 (`paper.txt`, `BiSpec_Pairwise.pdf`).
 Expression inputs are log-normalized counts unless stated; threshold `T`
 defaults to `0.0` (`ml/config.py:DEFAULT_EXPR_THRESHOLD`; paper does not
 publish `T` — recorded deviation).

| # | Name | Definition | Units / range | Required inputs | Missing policy |
|---|---|---|---|---|---|
| 1 | `target_activity_1` | Mean expression of target 1 in top expressing cell pops (`target_activity.mean_expression`) | log-counts, `[0, inf)` | expr vector target 1 | NaN |
| 2 | `target_activity_2` | Same for target 2 | log-counts, `[0, inf)` | expr vector target 2 | NaN |
| 3 | `single_ratio_max` | Max of the two single-positive proportions `P_gene = mean(I(expr>T))` | fraction, `[0, 1]` | expr vectors, T | NaN |
| 4 | `single_ratio_min` | Min of the two single-positive proportions | fraction, `[0, 1]` | expr vectors, T | NaN |
| 5 | `double_positive` | `P_g1,g2 = mean(I(e1>T) & I(e2>T))` (`double_positive_proportion`) | fraction, `[0, 1]` | expr vectors, T | NaN |
| 6 | `safety_score` | `dx*dy/(dx+dy)`, `d = expr_tumor - expr_normal` per target (`safety.safety_score`, verbatim paper) | log-count diff, `(-inf, inf)` | tumor + adjacent-normal means per target | NaN; `0.0` if `dx+dy==0` |
| 7 | `pathway_similarity` | Jaccard of KEGG+GO pathway memberships (`pathway.pathway_features`) | `[0, 1]` | pathway membership map (GMT/CSV) | `0.0` (unknown genes) |
| 8 | `pathway_shared_count` | Count of shared pathways | count, `[0, inf)` | same map | `0.0` |
| 9 | `gene_embedding_similarity` | `1/(1+d)` over Gene2Vec vectors (Du et al. 2019, `dim_200_iter_9`, manifest + SHA pinned) | `(0, 1]` | embedding dict | NaN (missing gene; former-symbol aliases VSIR/NECTIN4/MS4A1 per manifest) |
| 10 | `gene_embedding_distance` | Euclidean `d(x,y)` | `[0, inf)` | same | NaN |
| 11 | `pearson_correlation` | Pearson `r` across single cells (`correlation.pearson_correlation`) | `[-1, 1]` | expr vectors (≥3 cells) | NaN (<3 cells); `0.0` if zero variance |

Grouped cell-type expression (paper method): activity/positive/double-positive
features are computed within the top-2 expressing populations
(`target_activity.top_expressing_populations`, `celltype_activity`), selected
per target pair from the dataset's cell-type obs column — not whole-matrix.
`feature_builder.build_features()` takes caller-sliced vectors; the offline
store script (`feature_engineering/build_feature_store.py`) performs the
grouped slicing with provenance recorded per row.

Store row (`data/precomputed_features.json`, versioned): pair IDs, tumor type,
all 11 values, source dataset ID, generator version, provenance
(cell populations, T, embedding/pathway versions), missing-feature status,
quality/compatibility flags. NaN stays NaN in the store; `nan_to_num(0.0)`
happens only inside training/inference (`ml/`) and is recorded, never silent.
Backends report `feature_status: incomplete` for missing keys (see
`docs/api/api.md`).
