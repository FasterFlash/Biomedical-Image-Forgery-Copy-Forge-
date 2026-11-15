"""
cv_decision.py  --  Does a learned decision layer beat "confirmed matches >= threshold"?   (v2)

Reads results\\features.csv (from extract_features.py) and compares, with cross-validation:

    baseline   score = number of confirmed matches   (what V5 does), mask OFF and mask ON
    GBM        gradient-boosted trees on the features (OFF, ON, ON+OFF)
    LogReg     logistic regression on ON+OFF features

THE LEAK THIS SCRIPT AVOIDS
  Authentic and forged folders share image ids (the same figure with and without the forgery).
  A random split would put one version in training and the other in test. Here ALL rows of an
  image id always sit in the same fold / the same side of the held-out split.

HELD-OUT TEST (new in v2)
  --holdout 0.2   sets aside 20% of the image ids BEFORE anything else. Every table in sections 1-4
                  uses the remaining 80% ("development") only. Model and operating point are chosen there.
  --final         evaluates the ONE chosen model on the held-out ids, once, with the threshold fixed from
                  development data. A lock file (results\\holdout_used.txt) refuses a second look unless you
                  pass --allow-repeat, so the test set stays a test set.

FEATURE GROUPS  (--drop image kp)
  image : size, grey level, colour saturation, edge density  -> properties of the PICTURE, not the forgery
  kp    : keypoint counts                                    -> depend on resolution and the 8000 cap
  Dropping them gives a model that cannot lean on dataset-specific picture statistics. Use it for the
  model you carry to another dataset (BioFors).

USAGE (from the version5.5 folder):
  python cv_decision.py                                    # everything, development numbers only
  python cv_decision.py --holdout 0.2                      # same, 20% of ids kept aside (untouched)
  python cv_decision.py --holdout 0.2 --drop image kp      # robust feature set
  python cv_decision.py --holdout 0.2 --drop image kp --final --final-model "GBM, mask OFF features" --final-fpr 0.10

OUTPUT: printed report + results\\cv_report.txt + results\\cv_oof_scores.csv (+ results\\holdout_final.txt)
"""

import os
import sys
import csv
import time
import argparse

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

META = {"label", "image_id", "mask", "time_s", "mcc", "f1", "precision", "recall"}
GROUPS = {
    "image": ["h", "w", "log_mpx", "gray_mean", "gray_std", "sat_mean", "is_gray", "edge_density"],
    "kp": ["n_kp", "kp_per_mpx", "kp_capped"],
}
BIOFORS_PREVALENCE = 102 / 2409          # manipulated share of the BioFors internal-duplication test set


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s)
        self.f.write(s + "\n")
        self.f.flush()


# ---------------------------------------------------------------------------------- data
def load(path):
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        names = [c for c in rd.fieldnames if c not in META]
        by, modes = {}, set()
        for r in rd:
            by.setdefault((r["label"], r["image_id"]), {})[r["mask"]] = r
            modes.add(r["mask"])
    return by, names, sorted(modes)


def build(by, names, modes):
    keys = sorted(k for k, v in by.items() if all(m in v for m in modes))
    skipped = len(by) - len(keys)
    X = {m: np.array([[float(by[k][m][n]) for n in names] for k in keys]) for m in modes}
    y = np.array([1 if k[0] == "forged" else 0 for k in keys])
    ids = sorted({k[1] for k in keys})
    idx = {i: n for n, i in enumerate(ids)}
    gid = np.array([idx[k[1]] for k in keys])
    mcc = {m: np.array([float(by[k][m]["mcc"]) if by[k][m]["mcc"] not in ("", None) else np.nan
                        for k in keys]) for m in modes}
    return keys, X, y, gid, mcc, skipped


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def make_gbm():
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=250, max_depth=3,
                                          min_samples_leaf=20, l2_regularization=1.0,
                                          early_stopping=False, random_state=0)


def make_logreg():
    return make_pipeline(FunctionTransformer(signed_log), StandardScaler(),
                         LogisticRegression(C=0.3, max_iter=5000))


def matrix(Xd, modes):
    return np.hstack([Xd[m] for m in modes])


def reindex(gid):
    return np.unique(gid, return_inverse=True)[1]


# ---------------------------------------------------------------------------------- cross-validation
def oof_scores(factory, X, y, gid, repeats, folds, seed):
    """Out-of-fold probability for every row, averaged over `repeats` fold assignments (folds per ID)."""
    n_ids = int(gid.max()) + 1
    folds = min(folds, n_ids)
    out = np.zeros(len(y))
    for r in range(repeats):
        rng = np.random.default_rng(seed + r)
        fold_of_id = np.empty(n_ids, dtype=int)
        fold_of_id[rng.permutation(n_ids)] = np.arange(n_ids) % folds
        fold = fold_of_id[gid]
        pr = np.zeros(len(y))
        for k in range(folds):
            tr, te = fold != k, fold == k
            if len(np.unique(y[tr])) < 2:
                pr[te] = 0.5
                continue
            m = factory()
            m.fit(X[tr], y[tr])
            pr[te] = m.predict_proba(X[te])[:, 1]
        out += pr
    return out / repeats


# ---------------------------------------------------------------------------------- metrics
def tpr_at_fpr(scores, y, w, fprs):
    fpr, tpr, _ = roc_curve(y, scores, sample_weight=w)
    return np.interp(fprs, fpr, tpr)


def auc(scores, y, w=None):
    return float(roc_auc_score(y, scores, sample_weight=w))


def mcc_binary(tp, fp, fn, tn):
    d = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / d) if d > 0 else 0.0


def mcc_at_prevalence(tpr, fpr, prev=BIOFORS_PREVALENCE):
    """Image-level MCC you would get if these TPR/FPR held on a set where `prev` of images are forged."""
    return mcc_binary(prev * tpr, (1 - prev) * fpr, prev * (1 - tpr), (1 - prev) * (1 - fpr))


def threshold_for_fpr(neg_scores, target):
    """Smallest t such that 'score >= t' flags at most `target` of the negatives."""
    cands = np.unique(neg_scores)
    ok = [t for t in cands if np.mean(neg_scores >= t) <= target]
    return ok[0] if ok else cands[-1] + 1e-12


def bootstrap(scores_by_model, y, gid, fprs, B, seed):
    rng = np.random.default_rng(seed)
    n_ids = int(gid.max()) + 1
    A = {m: np.zeros(B) for m in scores_by_model}
    T = {m: np.zeros((B, len(fprs))) for m in scores_by_model}
    for b in range(B):
        cnt = np.bincount(rng.integers(0, n_ids, n_ids), minlength=n_ids)
        w = cnt[gid].astype(float)
        for m, s in scores_by_model.items():
            A[m][b] = auc(s, y, w)
            T[m][b] = tpr_at_fpr(s, y, w, fprs)
    return A, T


def ci(a):
    return np.percentile(a, [2.5, 97.5])


def honest_point(scores, y, gid, mcc_modes, target_fpr, reps, seed):
    """Threshold chosen on one random half of the ids, applied to the other half (and back)."""
    rng = np.random.default_rng(seed)
    n_ids = int(gid.max()) + 1
    res = {"fpr": [], "tpr": [], **{f"mcc_{m}": [] for m in mcc_modes}}
    for _ in range(reps):
        half = np.zeros(n_ids, dtype=bool)
        half[rng.permutation(n_ids)[: n_ids // 2]] = True
        for a_side in (True, False):
            A = half[gid] == a_side
            B = ~A
            neg = scores[A & (y == 0)]
            if len(neg) == 0:
                continue
            t = threshold_for_fpr(neg, target_fpr)
            pred = scores >= t
            nb, pb = B & (y == 0), B & (y == 1)
            if nb.sum() == 0 or pb.sum() == 0:
                continue
            res["fpr"].append(pred[nb].mean())
            res["tpr"].append(pred[pb].mean())
            for m, arr in mcc_modes.items():
                res[f"mcc_{m}"].append(np.nanmean(np.where(pred[pb], arr[pb], 0.0)))
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in res.items() if v}


# ---------------------------------------------------------------------------------- held-out final test
def final_eval(P, args, names_ranked, specs, scores, Xdev, ydev, gdev, Xte, yte, gte, mcc_te, modes, ni, boot):
    lock = os.path.join(args.out, "holdout_used.txt")
    if os.path.exists(lock) and not args.allow_repeat:
        P(f"\nHELD-OUT TEST REFUSED: {lock} already exists ({open(lock).read().splitlines()[0]}).")
        P("The test set has been looked at. Looking again and re-tuning is how test sets stop being tests.")
        P("If you really must, pass --allow-repeat, and say so in the write-up.")
        return
    chosen = args.final_model
    if chosen not in specs:
        sys.exit(f"--final-model must be one of: {list(specs)}")
    best_base = max((k for k in specs if specs[k]["kind"] == "base"), key=lambda k: auc(scores[k], ydev))
    P("\n" + "=" * 100)
    P(f"5. HELD-OUT TEST  -- evaluated ONCE.  chosen model: {chosen}   baseline: {best_base}")
    P(f"   threshold = the one that gives {100 * args.final_fpr:.0f}% FPR on DEVELOPMENT out-of-fold scores; nothing is re-tuned on test")
    P("=" * 100)

    rows = {}
    for name in (best_base, chosen):
        sp = specs[name]
        thr = threshold_for_fpr(scores[name][ydev == 0], args.final_fpr)
        if sp["kind"] == "base":
            s_te = Xte[sp["modes"][0]][:, ni]
        else:
            m = sp["factory"]()
            m.fit(matrix(Xdev, sp["modes"]), ydev)
            s_te = m.predict_proba(matrix(Xte, sp["modes"]))[:, 1]
        rows[name] = (thr, s_te)

    rng = np.random.default_rng(args.seed + 99)
    n_ids = int(gte.max()) + 1
    B = boot
    stats = {n: {"fpr": np.zeros(B), "tpr": np.zeros(B), "mcc": np.zeros(B), "auc": np.zeros(B),
                 **{f"pix_{m}": np.zeros(B) for m in mcc_te}} for n in rows}
    cnts = [np.bincount(rng.integers(0, n_ids, n_ids), minlength=n_ids) for _ in range(B)]

    def point(s_te, thr, w):
        pred = s_te >= thr
        pos, neg = yte == 1, yte == 0
        tp = (w * (pred & pos)).sum(); fn = (w * (~pred & pos)).sum()
        fp = (w * (pred & neg)).sum(); tn = (w * (~pred & neg)).sum()
        out = {"fpr": fp / (fp + tn), "tpr": tp / (tp + fn), "mcc": mcc_binary(tp, fp, fn, tn),
               "auc": auc(s_te, yte, w)}
        for m, arr in mcc_te.items():
            out[f"pix_{m}"] = float((w[pos] * np.where(pred[pos], arr[pos], 0.0)).sum() / w[pos].sum())
        return out

    ones = np.ones(len(yte))
    pts = {}
    for name, (thr, s_te) in rows.items():
        pts[name] = point(s_te, thr, ones)
        for b in range(B):
            r = point(s_te, thr, cnts[b][gte].astype(float))
            for k, v in r.items():
                stats[name][k][b] = v

    P(f"test ids: {n_ids}   test images: {len(yte)} (forged {int(yte.sum())}, authentic {int((1 - yte).sum())})")
    P(f"{'model':34s} {'FPR [95% CI]':22s} {'detection [95% CI]':24s} {'image MCC [CI]':22s} {'test AUC [CI]':22s}")
    for name in rows:
        p, s = pts[name], stats[name]
        f = lambda k, sc=1.0, d=1: f"{sc * p[k]:.{d}f} [{sc * ci(s[k])[0]:.{d}f},{sc * ci(s[k])[1]:.{d}f}]"
        P(f"{name:34s} {f('fpr', 100)+'%':22s} {f('tpr', 100)+'%':24s} {f('mcc', 1, 3):22s} {f('auc', 1, 3):22s}")
    P(f"\nExpected image-level MCC if these FPR/TPR held on a BioFors-like set ({100 * BIOFORS_PREVALENCE:.1f}% forged):")
    for name in rows:
        P(f"   {name:34s} {mcc_at_prevalence(pts[name]['tpr'], pts[name]['fpr']):.3f}   "
          f"(this is an ASSUMPTION about transfer, not a measurement)")
    for m in mcc_te:
        P(f"\nPixel-level MCC (mean over forged test images; not flagged = 0), using the mask from mask-{m.upper()} runs:")
        for name in rows:
            P(f"   {name:34s} {pts[name]['pix_' + m]:.4f} [{ci(stats[name]['pix_' + m])[0]:.4f},{ci(stats[name]['pix_' + m])[1]:.4f}]")
    d = stats[chosen]["tpr"] - stats[best_base]["tpr"]
    P(f"\nDetection difference chosen - baseline: {100 * (pts[chosen]['tpr'] - pts[best_base]['tpr']):+.1f} pts "
      f"[{100 * ci(d)[0]:+.1f},{100 * ci(d)[1]:+.1f}]   (note: their realised FPRs differ slightly, see table)")

    with open(lock, "w") as f:
        f.write(f"held-out test evaluated {time.strftime('%Y-%m-%d %H:%M:%S')} args={vars(args)}\n")
    P(f"\nLock written: {lock}")


# ---------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join("results", "features.csv"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--fpr", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20, 0.30])
    ap.add_argument("--honest-fpr", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--drop", nargs="*", default=[], choices=list(GROUPS), help="feature groups to remove")
    ap.add_argument("--holdout", type=float, default=0.0, help="fraction of image ids set aside as a test set")
    ap.add_argument("--holdout-seed", type=int, default=12345)
    ap.add_argument("--final", action="store_true", help="evaluate the chosen model on the held-out ids (once)")
    ap.add_argument("--final-model", default="GBM, mask OFF features")
    ap.add_argument("--final-fpr", type=float, default=0.10)
    ap.add_argument("--allow-repeat", action="store_true")
    args = ap.parse_args()
    if args.final and args.holdout <= 0:
        sys.exit("--final needs --holdout (e.g. --holdout 0.2)")

    os.makedirs(args.out, exist_ok=True)
    P = Tee(os.path.join(args.out, "cv_report.txt"))

    by, names, modes = load(args.features)
    keys, X, y, gid, mcc, skipped = build(by, names, modes)
    drop = {n for g in args.drop for n in GROUPS[g]}
    keep = [i for i, n in enumerate(names) if n not in drop]
    names = [names[i] for i in keep]
    X = {m: X[m][:, keep] for m in modes}

    test = None
    if args.holdout > 0:
        n_all = int(gid.max()) + 1
        rng = np.random.default_rng(args.holdout_seed)
        test_ids = rng.permutation(n_all)[: int(round(args.holdout * n_all))]
        is_test = np.isin(gid, test_ids)
        test = dict(X={m: X[m][is_test] for m in modes}, y=y[is_test], gid=reindex(gid[is_test]),
                    mcc={m: mcc[m][is_test] for m in modes if not np.all(np.isnan(mcc[m][is_test]))})
        dev = ~is_test
        X = {m: X[m][dev] for m in modes}
        y, mcc = y[dev], {m: mcc[m][dev] for m in modes}
        keys = [k for k, d in zip(keys, dev) if d]
        gid = reindex(gid[dev])

    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    P(f"DEVELOPMENT rows: {len(keys)}  (forged {n_pos}, authentic {n_neg})   image ids: {int(gid.max()) + 1}   "
      f"mask modes: {modes}   features per mode: {len(names)}")
    if test is not None:
        P(f"HELD OUT (not used below): {len(test['y'])} rows, {int(test['gid'].max()) + 1} image ids  "
          f"[split seed {args.holdout_seed}]")
    if drop:
        P(f"dropped feature groups: {args.drop}  ({len(drop)} features)")
    if skipped:
        P(f"note: {skipped} (class, id) pairs lacked a mask mode and were left out")
    if n_pos < 10 or n_neg < 10:
        sys.exit("Too few images of one class to cross-validate.")
    both_ids = len({k[1] for k in keys if k[0] == "forged"} & {k[1] for k in keys if k[0] == "authentic"})
    P(f"ids present as BOTH forged and authentic: {both_ids}  (kept in the same fold)\n")

    ni = names.index("n_inl")
    specs = {}
    for m in modes:
        specs[f"baseline: matches, mask {m.upper()}"] = dict(kind="base", modes=[m])
    for m in modes:
        specs[f"GBM, mask {m.upper()} features"] = dict(kind="ml", modes=[m], factory=make_gbm)
    if len(modes) > 1:
        specs["GBM, ON+OFF features"] = dict(kind="ml", modes=list(modes), factory=make_gbm)
        specs["LogReg, ON+OFF features"] = dict(kind="ml", modes=list(modes), factory=make_logreg)
    else:
        specs["LogReg, same features"] = dict(kind="ml", modes=list(modes), factory=make_logreg)

    scores = {}
    for name, sp in specs.items():
        if sp["kind"] == "base":
            scores[name] = X[sp["modes"][0]][:, ni]
        else:
            scores[name] = oof_scores(sp["factory"], matrix(X, sp["modes"]), y, gid, args.repeats, args.folds, args.seed)

    fprs = np.array(args.fpr)
    A, T = bootstrap(scores, y, gid, fprs, args.boot, args.seed)
    ones = np.ones(len(y))
    best_base = max((k for k in specs if specs[k]["kind"] == "base"), key=lambda k: auc(scores[k], y))

    P("=" * 100)
    P("1. DISCRIMINATION  (development, out-of-fold; CIs bootstrap image ids)")
    P("=" * 100)
    P(f"{'model':34s} {'AUC [95% CI]':24s}" + "".join(f"det@{int(f * 100)}%FPR".rjust(13) for f in fprs))
    for k, s in scores.items():
        lo, hi = ci(A[k])
        t = tpr_at_fpr(s, y, ones, fprs)
        P(f"{k:34s} {auc(s, y):.4f} [{lo:.3f},{hi:.3f}]   " + "".join(f"{100 * v:12.1f}%" for v in t))

    P("\n" + "=" * 100)
    P(f"2. DIFFERENCE vs best baseline  ({best_base})   -- if the CI contains 0 it is NOT an improvement")
    P("=" * 100)
    P(f"{'model':34s} {'dAUC [95% CI]':26s}" + "".join(f"d det@{int(f * 100)}%".rjust(13) for f in fprs))
    for k in scores:
        if specs[k]["kind"] == "base":
            continue
        d = A[k] - A[best_base]
        lo, hi = ci(d)
        row = f"{k:34s} {auc(scores[k], y) - auc(scores[best_base], y):+.4f} [{lo:+.3f},{hi:+.3f}]  "
        t0 = tpr_at_fpr(scores[k], y, ones, fprs) - tpr_at_fpr(scores[best_base], y, ones, fprs)
        for j in range(len(fprs)):
            l, h = ci(T[k][:, j] - T[best_base][:, j])
            row += f"{100 * t0[j]:+9.1f}{'*' if (l > 0 or h < 0) else ' '}   "
        P(row)
    P("   (* = 95% CI of that difference excludes 0)")

    P("\n" + "=" * 100)
    P("3. HONEST OPERATING POINT  (threshold picked on one random half of the ids, applied to the other)")
    P("=" * 100)
    mcc_modes = {m: mcc[m] for m in modes if not np.all(np.isnan(mcc[m]))}
    for tf in args.honest_fpr:
        P(f"\ntarget FPR {100 * tf:.0f}%")
        P(f"{'model':34s} {'realised FPR':>14s} {'detection':>14s}  {'img MCC@4%':>10s}" +
          "".join(f"  pixMCC({m.upper()})".rjust(15) for m in mcc_modes))
        for k, s in scores.items():
            r = honest_point(s, y, gid, mcc_modes, tf, reps=20, seed=args.seed + 7)
            row = (f"{k:34s} {100 * r['fpr'][0]:8.1f}+-{100 * r['fpr'][1]:3.1f}% {100 * r['tpr'][0]:8.1f}+-{100 * r['tpr'][1]:3.1f}%"
                   f"  {mcc_at_prevalence(r['tpr'][0], r['fpr'][0]):10.3f}")
            for m in mcc_modes:
                v = r.get(f"mcc_{m}")
                row += f"{v[0]:12.4f}   " if v else "         n/a   "
            P(row)
    P("   img MCC@4% = image-level MCC if these rates held on a set with ~4% forged images (BioFors-like).")
    P("   It is an assumption about transfer, not a measurement.")

    P("\n" + "=" * 100)
    P("4. SINGLE-FEATURE AUC  (0.5 = useless; oriented so >= 0.5; top 15 per mask mode)")
    P("=" * 100)
    for m in modes:
        rows = []
        for j, n in enumerate(names):
            col = X[m][:, j]
            if np.ptp(col) == 0:
                continue
            a = auc(col, y)
            rows.append((max(a, 1 - a), "higher=forged" if a >= 0.5 else "lower=forged", n))
        rows.sort(reverse=True)
        P(f"\nmask {m.upper()}")
        for a, d, n in rows[:15]:
            P(f"  {n:18s} {a:.4f}   {d}")

    if args.final:
        final_eval(P, args, list(specs), specs, scores, X, y, gid, test["X"], test["y"], test["gid"],
                   test["mcc"], modes, ni, args.boot)
    elif test is not None:
        P("\n(held-out ids are untouched. When the model and threshold are chosen, run once more with --final.)")

    out_csv = os.path.join(args.out, "cv_oof_scores.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "image_id"] + list(scores))
        for i, k in enumerate(keys):
            w.writerow([k[0], k[1]] + [round(float(scores[n][i]), 6) for n in scores])
    P(f"\nSaved: {os.path.join(args.out, 'cv_report.txt')}   {out_csv}")
    P("Caveat: ONE dataset. The real test of generalisation is a different dataset (BioFors).")


if __name__ == "__main__":
    main()
