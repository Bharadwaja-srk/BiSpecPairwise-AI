"""Exploratory leave-one-pair-out pilot (indication-matched, same-pair grouped).

Assembly rules (adopted policy):
  - Representative row = most advanced stage_rank per (pair, indication).
  - Features attach ONLY from the matching tumor store (HNSCC rows <- HNSCC
    store, HCC rows <- HCC store). Rows without a matching store are
    INCOMPLETE and excluded (never cross-tumor substitution).
  - Same-pair rows at different stages carry zero feature-diffs and are
    excluded from the pairwise transform by the representative rule.
  - Comparisons only within one tumor feature space (no cross-tumor diffs).
  - With zero valid comparisons the pilot reports NOT ESTIMABLE and fits
    nothing (never fabricates a metric).

Reusable: when NSCLC/PDAC stores land, the same assembly admits them.
"""
from __future__ import annotations

import csv
import json


def load_clinical_eligible(path: str) -> list[dict]:
    with open(path, encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r.get('eligible_training') == 'yes']
    for r in rows:
        r['pair'] = '__'.join(sorted([r['target1'].strip().upper(), r['target2'].strip().upper()]))
        r['stage_rank'] = int(r['stage_rank'])
    return rows


def load_store(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        s = json.load(f)
    out = {}
    for r in s['rows']:
        out['__'.join(sorted([r['target1'].upper(), r['target2'].upper()]))] = r
    return out


# indication_scope -> (tumor key, store path, feature-space tag)
SCOPE_STORES = {
    'in-scope:HNSCC': ('HNSCC', 'data/precomputed_features_hnscc.json'),
    'in-scope:liver-HCC': ('HCC', 'data/precomputed_features_hcc.json'),
    'in-scope:NSCLC': ('NSCLC', 'data/precomputed_features_nsclc.json'),
}

FEATURE_ORDER = [
    'target_activity_1', 'target_activity_2', 'single_ratio_max',
    'single_ratio_min', 'double_positive', 'safety_score',
    'pathway_similarity', 'pathway_shared_count',
    'gene_embedding_similarity', 'gene_embedding_distance',
    'pearson_correlation',
]


def assemble(rows: list[dict], stores: dict[str, dict]) -> dict:
    """Apply representative rule + indication-matched attach. Returns dataset report."""
    reps: dict[tuple, dict] = {}
    for r in rows:
        key = (r['pair'], r['indication_scope'])
        if key not in reps or r['stage_rank'] < reps[key]['stage_rank']:
            reps[key] = r
    attached, incomplete = [], []
    for (pair, scope), r in reps.items():
        tag = SCOPE_STORES.get(scope)
        srow = stores.get(tag[1], {}).get(pair) if tag else None
        if srow is None:
            incomplete.append({**r, 'reason': 'no indication-matched store'})
            continue
        feats = [srow['features'].get(k, float('nan')) for k in FEATURE_ORDER]
        attached.append({**r, 'features': feats, 'space': tag[0],
                         'missing': srow.get('missing_features', [])})
    # valid comparisons: distinct-rank representative pairs within one space
    by_space: dict[str, list] = {}
    for a in attached:
        by_space.setdefault(a['space'], []).append(a)
    comparisons = []
    for space, items in by_space.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if items[i]['stage_rank'] != items[j]['stage_rank']:
                    comparisons.append((space, items[i]['pair'], items[j]['pair']))
    return {'representatives': reps, 'attached': attached,
            'incomplete': incomplete, 'comparisons': comparisons}


def _concordant(pred_first_more_advanced: bool | None, actual_first: bool) -> float:
    """Score one cross-rank comparison. Ties (None prediction) score 0.5."""
    if pred_first_more_advanced is None:
        return 0.5
    return float(pred_first_more_advanced == actual_first)


def _heldout_pairs(te: list[dict], tr: list[dict]) -> list[tuple[dict, dict]]:
    """Cross-rank held-out comparisons, same feature space only."""
    out = []
    for t in te:
        for o in tr:
            if t['space'] == o['space'] and t['stage_rank'] != o['stage_rank']:
                out.append((t, o))
    return out


def run_lopo(attached: list[dict], comparisons: list, seed: int = 42) -> dict:
    """Leave-one-pair-out; held-out unit is the canonical pair (all indications).

    Arms (all transient, nothing saved):
      - pairwise: XGBClassifier on mirrored feature-diffs (existing method).
      - pointwise: XGBClassifier on features -> rank class (paper defaults).
      - dummy: constant predictor (train-mean rank for all rows); every
        comparison is a tie scored 0.5 by documented convention.
    Missing values are nan_to_num(0.0) strictly inside each fold.
    Folds/metrics without valid comparisons are NOT ESTIMABLE (None + status).
    """
    import numpy as np
    from ml.data.pairwise import build_pairwise
    from ml.training.train_xgboost import make_model
    from ml.config import seed_everything

    pairs = sorted(set(a['pair'] for a in attached))
    folds = []
    for held in pairs:
        tr = [a for a in attached if a['pair'] != held]
        te = [a for a in attached if a['pair'] == held]
        Xr = np.array([a['features'] for a in tr], dtype=float)
        yr = np.array([a['stage_rank'] for a in tr])
        cross = _heldout_pairs(te, tr)
        fold = {'held_out_pair': held, 'n_train_rows': len(tr),
                'n_heldout_comparisons': len(cross)}
        if not cross:
            fold['status'] = 'NOT ESTIMABLE (no held-out cross-rank comparisons)'
            fold['models'] = {}
            folds.append(fold)
            continue
        scores: dict[str, float | None] = {}
        # pairwise arm
        built = build_pairwise(Xr, yr, seed=seed)
        fold['n_train_pairs'] = built['n_pairs']
        if built['n_pairs'] == 0:
            scores['pairwise_xgb'] = None
        else:
            seed_everything(seed)
            model = make_model({'random_state': seed})
            model.fit(np.nan_to_num(built['X_diff'], nan=0.0), built['y'])
            s = 0.0
            for t, o in cross:
                d = np.nan_to_num(np.array(t['features']) - np.array(o['features']),
                                  nan=0.0)
                s += _concordant(float(model.predict(d.reshape(1, -1))[0]) > 0.5,
                                 t['stage_rank'] < o['stage_rank'])
            scores['pairwise_xgb'] = s / len(cross)
        # pointwise arm: features -> rank class (label-encoded, multiclass objective)
        with np.errstate(all='ignore'):
            Xt = np.nan_to_num(Xr, nan=0.0)
        classes = sorted(set(yr.tolist()))
        if len(classes) < 2:
            scores['pointwise_xgb'] = None
        else:
            enc = {c: k for k, c in enumerate(classes)}
            seed_everything(seed)
            pw = make_model({'random_state': seed, 'objective': 'multi:softprob',
                             'num_class': len(classes), 'eval_metric': 'mlogloss'})
            pw.fit(Xt, np.array([enc[v] for v in yr.tolist()]))
            s = 0.0
            for t, o in cross:
                ft = np.nan_to_num(np.array(t['features']), nan=0.0).reshape(1, -1)
                fo = np.nan_to_num(np.array(o['features']), nan=0.0).reshape(1, -1)
                pt, po = classes[int(pw.predict(ft).reshape(-1)[0])], classes[int(pw.predict(fo).reshape(-1)[0])]
                s += _concordant(None if pt == po else pt < po,
                                 t['stage_rank'] < o['stage_rank'])
            scores['pointwise_xgb'] = s / len(cross)
        # dummy arm: constant train-mean prediction -> all ties -> 0.5
        scores['dummy_constant'] = 0.5
        fold['status'] = 'scored'
        fold['models'] = scores
        folds.append(fold)

    def _mean(arm: str):
        vs = [f['models'][arm] for f in folds
              if f['models'].get(arm) is not None]
        return sum(vs) / len(vs) if vs else None, len(vs)

    summary = {}
    for arm in ('pairwise_xgb', 'pointwise_xgb', 'dummy_constant'):
        m, n = _mean(arm)
        summary[arm] = {'mean_concordance': m, 'n_scored_folds': n}
    summary['n_heldout_comparisons_total'] = sum(
        f['n_heldout_comparisons'] for f in folds)
    return {'folds': folds, 'summary': summary,
            'design': ('LOPO over canonical pairs (all indications held out together); '
                       'concordance over same-space cross-rank held-out comparisons; '
                       'ties score 0.5; denominator = n_heldout_comparisons_total')}


def main() -> dict:
    rows = load_clinical_eligible('docs/target_universe/clinical_four_indication_draft_v3.csv')
    stores = {p: load_store(p) for _, p in SCOPE_STORES.values()}
    asm = assemble(rows, stores)
    report = {
        'eligible_rows': len(rows),
        'representative_rows': len(asm['representatives']),
        'attached_rows': len(asm['attached']),
        'incomplete_rows': [{'pair': r['pair'], 'scope': r['indication_scope'],
                             'reason': r['reason']} for r in asm['incomplete']],
        'valid_comparisons': len(asm['comparisons']),
    }
    if not asm['comparisons']:
        report['verdict'] = ('NOT ESTIMABLE: zero valid indication-matched, '
                             'same-pair-grouped cross-rank comparisons; no model fitted')
        return report
    report['lopo'] = run_lopo(asm['attached'], asm['comparisons'])
    report['verdict'] = 'exploratory pilot (see fold table; no generalization claim)'
    return report


if __name__ == '__main__':
    print(json.dumps(main(), indent=2))
