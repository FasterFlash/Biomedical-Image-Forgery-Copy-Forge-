"""
decision.py  --  The learned decision layer, shared by train_final.py and score_biofors.py.

Same model family and settings as version5.5 cv_decision.py (gradient-boosted trees), so the numbers
carry over. The final model is the AVERAGE of 5 models, each trained on 4/5 of the image ids
("fold ensemble"); its cut-off comes from scores each image got from the model that never saw it.
Validated on Recod.ai development data only (held-out ids excluded): realised FPR 4.8% +- 0.7% for a
5% target, over 8 random splits.
"""

import csv

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

META = {"label", "image_id", "mask", "time_s", "mcc", "f1", "precision", "recall"}
GROUPS = {
    "image": ["h", "w", "log_mpx", "gray_mean", "gray_std", "sat_mean", "is_gray", "edge_density"],
    "kp": ["n_kp", "kp_per_mpx", "kp_capped"],
}


def make_gbm():
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=250, max_depth=3,
                                          min_samples_leaf=20, l2_regularization=1.0,
                                          early_stopping=False, random_state=0)


def load_recod(path, mask_mode, drop_groups):
    """version5.5 results\\features.csv -> X (rows), y, group id per row, feature names, image keys."""
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        names = [c for c in rd.fieldnames if c not in META]
        rows = [r for r in rd if r["mask"] == mask_mode]
    drop = {n for g in drop_groups for n in GROUPS[g]}
    names = [n for n in names if n not in drop]
    rows.sort(key=lambda r: (r["label"], r["image_id"]))
    X = np.array([[float(r[n]) for n in names] for r in rows])
    y = np.array([1 if r["label"] == "forged" else 0 for r in rows])
    ids = sorted({r["image_id"] for r in rows})
    idx = {i: k for k, i in enumerate(ids)}
    gid = np.array([idx[r["image_id"]] for r in rows])
    keys = [(r["label"], r["image_id"]) for r in rows]
    return X, y, gid, names, keys


def fold_of(gid, k, seed):
    n = int(gid.max()) + 1
    f = np.empty(n, dtype=int)
    f[np.random.default_rng(seed).permutation(n)] = np.arange(n) % k
    return f[gid]


def fit_fold_ensemble(X, y, gid, k=5, seed=0):
    """Returns (models, out_of_fold_scores). Folds are per image id, never per row."""
    fold = fold_of(gid, k, seed)
    models, oof = [], np.zeros(len(y))
    for j in range(k):
        m = make_gbm()
        m.fit(X[fold != j], y[fold != j])
        oof[fold == j] = m.predict_proba(X[fold == j])[:, 1]
        models.append(m)
    return models, oof


def predict(models, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def threshold_for_fpr(neg_scores, target, weights=None):
    """Smallest t such that 'score >= t' flags at most `target` (weighted share) of the negatives."""
    s = np.asarray(neg_scores, dtype=float)
    w = np.ones(len(s)) if weights is None else np.asarray(weights, dtype=float)
    order = np.argsort(-s)
    s, w = s[order], w[order]
    total = w.sum()
    cands = np.unique(s)[::-1]                      # high -> low
    best = cands[0] + 1e-12                         # flags nothing
    for t in cands:
        if w[s >= t].sum() / total <= target:
            best = t
        else:
            break
    return float(best)


def mcc(tp, fp, fn, tn):
    d = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    return float((float(tp) * tn - float(fp) * fn) / d) if d > 0 else 0.0
