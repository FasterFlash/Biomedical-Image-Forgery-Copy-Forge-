"""
specialise.py  --  VERSION 7, last improvement: does a decision model SPECIALISED per image type do better than the
one global model? Recod.ai only (in-dataset). Nothing here touches BioFors.

Same detector output and the same 32 robust features as version5.5/6 (mask OFF). Compared per image type:
  count rule        number of confirmed matches
  global model      one model for all types (= version6)
  global + type     the global model, also told the image type
  specialised       a separate model per type, trained only on that type's figures
Every score is OUT-OF-FOLD: folds are by figure id, so neither version of a figure is ever scored by a model that saw
it (averaged over 3 different fold assignments).

Cut-offs are honest: chosen on a random half of the figures (for that type), measured on the other half, 20 repeats,
reported as mean +- sd. Two operating points:
  5% FPR  the version6 operating point
  0% FPR  cut-off just above the highest-scoring authentic image of that type in the choosing half
          ("0%" measured on the other half; with a few hundred authentic images it means "about 0-1%", not "never")

ceiling = share of forged images where the detector found >= 3 confirmed matches, i.e. drew any region at all.
No decision model can detect more than that: past it, only a better DETECTOR helps.

Image types: version6.7 reviewed labels. "Other" (15 figures) is left out (too small).
Pixel MCC: per-image mean over forged images (not flagged = 0), as in version5.5.

USAGE (PowerShell, from version7, venv active):   python specialise.py
OUTPUT (results\\): specialise_<time>.txt (+ _latest), specialise_metrics_<time>.json (+ _latest), run_log.jsonl
"""

import os
import csv
import json
import shutil
import argparse

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from runlog import Run, stamp

TYPES = ["Blot/Gel", "Microscopy", "Histology", "Macroscopy"]
META = {"label", "image_id", "mask", "time_s", "mcc", "f1", "precision", "recall"}
DROP = {"h", "w", "log_mpx", "gray_mean", "gray_std", "sat_mean", "is_gray", "edge_density", "n_kp", "kp_per_mpx", "kp_capped"}
PREV_BIOFORS = 102 / 2363
RULES = ["count rule", "global model", "global + type", "specialised"]


def gbm():   # identical settings to version5.5/6
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=250, max_depth=3, min_samples_leaf=20,
                                          l2_regularization=1.0, early_stopping=False, random_state=0)


def oof(X, y, g, seeds):
    g = np.unique(g, return_inverse=True)[1]
    n = int(g.max()) + 1
    out = np.zeros(len(y))
    for s in seeds:
        f = np.empty(n, int)
        f[np.random.default_rng(s).permutation(n)] = np.arange(n) % 5
        fold = f[g]
        for k in range(5):
            m = gbm().fit(X[fold != k], y[fold != k])
            out[fold == k] += m.predict_proba(X[fold == k])[:, 1]
    return out / len(seeds)


def cutoff(neg, target):
    c = np.unique(neg)[::-1]
    best = c[0] + 1e-9
    for v in c:
        if (neg >= v).mean() <= target:
            best = v
        else:
            break
    return best


def mcc(tp, fp, fn, tn):
    d = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    return float((float(tp) * tn - float(fp) * fn) / d) if d > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v55", default=os.path.join("..", "version5.5", "results"))
    ap.add_argument("--v67", default=os.path.join("..", "version6.7", "results"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    with Run("specialise", args, args.out) as run:
        P = run.p
        types = {r["image_id"]: r["type"] for r in csv.DictReader(open(os.path.join(args.v67, "recod_types.csv")))}
        rows = [r for r in csv.DictReader(open(os.path.join(args.v55, "features.csv")))
                if r["mask"] == "off" and types.get(r["image_id"]) in TYPES]
        rows.sort(key=lambda r: (r["label"], r["image_id"]))
        names = [c for c in rows[0] if c not in META and c not in DROP]
        X = np.array([[float(r[n]) for n in names] for r in rows])
        y = np.array([r["label"] == "forged" for r in rows]).astype(int)
        ids = np.array([r["image_id"] for r in rows])
        typ = np.array([types[r["image_id"]] for r in rows])
        n_inl = X[:, names.index("n_inl")]
        pix = np.array([float(r["mcc"]) if r["mcc"] else 0.0 for r in rows])
        region = n_inl >= 3
        P(f"Recod.ai images {len(rows)} (forged {y.sum()}, authentic {(1 - y).sum()}), types {TYPES}, features {len(names)}")

        seeds = list(range(args.seeds))
        onehot = np.array([[t == u for u in TYPES] for t in typ], float)
        S = {"count rule": n_inl, "global model": oof(X, y, ids, seeds), "global + type": oof(np.hstack([X, onehot]), y, ids, seeds),
             "specialised": np.zeros(len(y))}
        for t in TYPES:
            m = typ == t
            S["specialised"][m] = oof(X[m], y[m], ids[m], seeds)
            print(f"  specialised model for {t} done")

        res = {}
        rng = np.random.default_rng(1)
        for t in TYPES:
            m = typ == t
            uid, gi = np.unique(ids[m], return_inverse=True)
            yt, rt, pt = y[m], region[m], pix[m]
            ceiling = float(rt[yt == 1].mean())
            res[t] = {"n_forged": int(yt.sum()), "n_authentic": int((1 - yt).sum()), "ceiling": ceiling, "rules": {}}
            for r in RULES:
                s = S[r][m]
                a = float(roc_auc_score(yt, s))
                bs = []
                for _ in range(args.boot):
                    w = np.bincount(rng.integers(0, len(uid), len(uid)), minlength=len(uid))[gi]
                    if w[yt == 1].sum() and w[yt == 0].sum():
                        bs.append(roc_auc_score(yt, s, sample_weight=w))
                op = {}
                for target in (0.05, 0.0):
                    acc = []
                    r2 = np.random.default_rng(7)
                    for _ in range(args.repeats):
                        half = np.zeros(len(uid), bool)
                        half[r2.permutation(len(uid))[: len(uid) // 2]] = True
                        for side in (True, False):
                            A, B = half[gi] == side, half[gi] != side
                            c = cutoff(s[A & (yt == 0)], target)
                            fl = (s >= c) & rt
                            nb, pb = B & (yt == 0), B & (yt == 1)
                            fpr, det = fl[nb].mean(), fl[pb].mean()
                            tp, fp = (fl & pb).sum(), (fl & nb).sum()
                            acc.append((fpr, det, mcc(tp, fp, pb.sum() - tp, nb.sum() - fp),
                                        mcc(PREV_BIOFORS * det, (1 - PREV_BIOFORS) * fpr, PREV_BIOFORS * (1 - det), (1 - PREV_BIOFORS) * (1 - fpr)),
                                        np.where(fl[pb], pt[pb], 0).mean()))
                    a_ = np.array(acc)
                    op[str(target)] = {k: [float(a_[:, i].mean()), float(a_[:, i].std())]
                                       for i, k in enumerate(["fpr", "det", "img_mcc", "img_mcc_at_4pct", "pix_mcc"])}
                res[t]["rules"][r] = {"auc": a, "auc_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "op": op}

        for target, label in (("0.05", "5% FPR TARGET"), ("0.0", "0% FPR TARGET")):
            P("\n" + "=" * 120)
            P(f"{label}   (cut-off chosen on half of the figures, measured on the other half; mean +- sd over {2 * args.repeats})")
            P("=" * 120)
            for t in TYPES:
                d = res[t]
                P(f"\n[{t}]  forged {d['n_forged']}, authentic {d['n_authentic']}   ceiling (detector drew a region): {100 * d['ceiling']:.1f}%")
                P(f"  {'rule':15s} {'AUC [95% CI]':>21s} {'FPR':>13s} {'detection':>13s} {'image MCC':>14s} {'img MCC @4.3%':>14s} {'pixel MCC':>14s}")
                for r in RULES:
                    q = d["rules"][r]
                    o = q["op"][target]
                    f = lambda k, sc=1, dd=3: f"{sc * o[k][0]:.{dd}f}+-{sc * o[k][1]:.{dd}f}"
                    P(f"  {r:15s} {q['auc']:.3f} [{q['auc_ci'][0]:.3f},{q['auc_ci'][1]:.3f}] {f('fpr', 100, 1):>12s}% "
                      f"{f('det', 100, 1):>12s}% {f('img_mcc'):>14s} {f('img_mcc_at_4pct'):>14s} {f('pix_mcc'):>14s}")

        P("\n" + "=" * 120)
        P("SUMMARY at 5% FPR target: best decision rule per type vs the version6 global model")
        P("=" * 120)
        summ = {}
        for t in TYPES:
            g = res[t]["rules"]["global model"]["op"]["0.05"]
            best = max(RULES, key=lambda r: res[t]["rules"][r]["op"]["0.05"]["img_mcc"][0])
            b = res[t]["rules"][best]["op"]["0.05"]
            summ[t] = best
            P(f"  {t:11s} best = {best:14s} detection {100 * g['det'][0]:5.1f}% -> {100 * b['det'][0]:5.1f}%   "
              f"image MCC {g['img_mcc'][0]:.3f} -> {b['img_mcc'][0]:.3f}   pixel MCC {g['pix_mcc'][0]:.3f} -> {b['pix_mcc'][0]:.3f}   "
              f"(ceiling {100 * res[t]['ceiling']:.1f}%)")
        P("\nIn-dataset (Recod.ai) results only. These models were not, and will not be, tested on BioFors.")

        ts = stamp()
        pj = os.path.join(args.out, f"specialise_metrics_{ts}.json")
        json.dump({"results": res, "best_at_5pct": summ, "features": names}, open(pj, "w"), indent=1)
        shutil.copyfile(pj, os.path.join(args.out, "specialise_metrics_latest.json"))
        run.output(pj)
        run.note(best_at_5pct=summ)
        P(f"Saved: {pj}, specialise_metrics_latest.json")


if __name__ == "__main__":
    main()
