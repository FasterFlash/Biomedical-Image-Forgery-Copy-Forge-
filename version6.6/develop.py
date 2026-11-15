"""
develop.py  --  STEP 3 of V6.6. Every choice for experiment 2 is made HERE, without the BioFors test set.

Data it uses (no BioFors test image, no BioFors forgery label):
  Recod.ai features                 ..\\version5.5\\results\\features.csv        (mask OFF rows)
  synthetic BioFors pairs           results\\synth_features.csv                  (from make_synthetic + run_synth)
  genuine BioFors calibration set   ..\\version6\\results\\biofors_features.csv  (set == calib rows only)
  BioFors test image-TYPE counts    same file, set == test, column 'category' only (no labels read)

What it reports:
  1. Feature stability: does each feature separate forged/genuine on Recod.ai AND on BioFors-like images?
  2. Five candidates, all scored on synthetic BioFors with folds by BioFors DOCUMENT:
       C0 count     confirmed-match count (the rule that transferred in version6)
       C1 model     all 32 features, trained on Recod.ai only            (= the version6 model)
       C2 model     STABLE features only, trained on Recod.ai only       (fix a)
       C3 model     all features, trained on Recod.ai + synthetic BioFors (fix b)
       C4 model     stable features, Recod.ai + synthetic BioFors        (fix a + b)
     "Stable" is decided inside each fold from the training part only, so it never sees the scored images.
  3. For each candidate the false-positive target that maximises image MCC at the BioFors test prevalence
     (102 of 2363 forged, published), and that expected MCC with a 95% CI.
  4. Robustness: detection by copy variant and image type for C0 and for the chosen candidate.

SELECTION RULE (fixed before any result is seen):
  choose the candidate with the highest expected image MCC; if a simpler candidate is within 0.01, take the simpler
  one. Simplicity order: C0 < C2 < C1 < C4 < C3.
Then the chosen candidate is retrained on all its data, and its cut-off is set on the GENUINE BioFors calibration
images (real, untouched images like the test's) at the chosen false-positive target. C0 gets the same treatment as
a reference. Both are frozen in results\\final_v66.joblib for test_biofors.py.

USAGE:  python develop.py
OUTPUT (results\\): develop_<time>.txt, final_v66.joblib, final_v66.json, develop_metrics_<time>.json, synth_oof_scores.csv
"""

import os
import csv
import json
import argparse

import numpy as np
import joblib
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

import decision as D
from runlog import Run, stamp, now_iso, versions

STABLE_MIN = 0.10            # |AUC - 0.5| on synthetic BioFors needed to call a feature stable (same sign as Recod.ai)
FPR_GRID = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10]
SIMPLICITY = ["C0", "C2", "C1", "C4", "C3"]
NAMES = {"C0": "count (confirmed matches)", "C1": "model, all features, Recod.ai only",
         "C2": "model, stable features, Recod.ai only", "C3": "model, all features, Recod.ai + synthetic BioFors",
         "C4": "model, stable features, Recod.ai + synthetic BioFors"}


def safe_auc(y, x, w=None):
    if len(np.unique(y)) < 2 or np.ptp(x) == 0:
        return 0.5
    return float(roc_auc_score(y, x, sample_weight=w))


def stable_idx(auc_recod, Xs, ys):
    out = []
    for j, ar in enumerate(auc_recod):
        a_s = safe_auc(ys, Xs[:, j])
        if (ar - 0.5) * (a_s - 0.5) > 0 and abs(a_s - 0.5) >= STABLE_MIN:
            out.append(j)
    return out


def exp_mcc(tpr, fpr, prev):
    return D.mcc(prev * tpr, (1 - prev) * fpr, prev * (1 - tpr), (1 - prev) * (1 - fpr))


def operating(scores, ys, w, prev):
    """For each FPR target: threshold from genuine synthetic scores, TPR on forged, expected MCC at `prev`."""
    rows = []
    for t in FPR_GRID:
        thr = D.threshold_for_fpr(scores[ys == 0], t, w[ys == 0])
        tpr = np.average(scores[ys == 1] >= thr, weights=w[ys == 1])
        fpr = np.average(scores[ys == 0] >= thr, weights=w[ys == 0])
        rows.append({"target": t, "thr": float(thr), "tpr": float(tpr), "fpr": float(fpr), "mcc": exp_mcc(tpr, fpr, prev)})
    return max(rows, key=lambda r: r["mcc"]), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recod", default=os.path.join("..", "version5.5", "results", "features.csv"))
    ap.add_argument("--synth", default=os.path.join("results", "synth_features.csv"))
    ap.add_argument("--biofors", default=os.path.join("..", "version6", "results", "biofors_features.csv"))
    ap.add_argument("--prevalence", type=float, default=102 / 2363)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    with Run("develop", args, args.out) as run:
        P = run.p
        Xr, yr, gr, names, _ = D.load_recod(args.recod, "off", ["image", "kp"])
        S = list(csv.DictReader(open(args.synth, newline="")))
        Xs = np.array([[float(r[n]) for n in names] for r in S])
        ys = np.array([1 if r["set"] == "synth_forged" else 0 for r in S])
        docs = np.array([r["doc"] for r in S])
        gs = np.unique(docs, return_inverse=True)[1]
        var_s = np.array([r["variant"] for r in S])
        cat_s = np.array([r["category"] for r in S])
        has_region = np.array([float(r["px_pred_raw"]) > 0 for r in S])
        B = list(csv.DictReader(open(args.biofors, newline="")))
        C = [r for r in B if r["set"] == "calib"]
        Xc = np.array([[float(r[n]) for n in names] for r in C])
        cat_c = np.array([r["category"] for r in C])
        test_types = [r["category"] for r in B if r["set"] == "test"]          # types only, labels are not read
        cats = sorted(set(test_types))
        mix = {c: test_types.count(c) / len(test_types) for c in cats}
        ws = np.array([mix.get(c, 0) / np.mean(cat_s == c) for c in cat_s])     # synthetic -> test type mix
        wc = np.array([mix.get(c, 0) / np.mean(cat_c == c) for c in cat_c])     # calibration -> test type mix
        ni = names.index("n_inl")
        P(f"Recod.ai rows {len(yr)}   synthetic BioFors rows {len(ys)} ({ys.sum()} forged) from {gs.max() + 1} documents   "
          f"genuine calibration images {len(C)}")
        P(f"BioFors test type mix (labels not read): " + ", ".join(f"{c} {100 * v:.1f}%" for c, v in mix.items())
          + f"   prevalence used: {100 * args.prevalence:.2f}%")

        # ---------------------------------------------------------------- 1. feature stability
        auc_r = [safe_auc(yr, Xr[:, j]) for j in range(len(names))]
        pub = var_s == "publication"
        Xr_gen = Xr[yr == 0]
        P("\n" + "=" * 104)
        P("1. FEATURE STABILITY   (AUC: 0.5 = useless; same side of 0.5 on both = same meaning)")
        P("   genuine shift = Kolmogorov-Smirnov distance between genuine Recod.ai and genuine BioFors (0 = same, 1 = disjoint)")
        P("=" * 104)
        P(f"{'feature':18s} {'AUC Recod':>9s} {'AUC synth':>9s} {'synth publ.':>11s} {'genuine shift':>13s}   stable?")
        stab_all = stable_idx(auc_r, Xs, ys)
        stab_rows = []
        for j, n in enumerate(names):
            a_s, a_p = safe_auc(ys, Xs[:, j]), safe_auc(ys[pub], Xs[pub, j])
            ks = float(ks_2samp(Xr_gen[:, j], Xc[:, j]).statistic)
            stab_rows.append({"feature": n, "auc_recod": auc_r[j], "auc_synth": a_s, "auc_synth_publication": a_p,
                              "genuine_shift_ks": ks, "stable": j in stab_all})
            P(f"{n:18s} {auc_r[j]:9.3f} {a_s:9.3f} {a_p:11.3f} {ks:13.3f}   {'yes' if j in stab_all else ''}")
        P(f"stable features ({len(stab_all)}): {[names[j] for j in stab_all]}")

        # ---------------------------------------------------------------- 2. candidates, document-grouped folds
        fold = D.fold_of(gs, args.folds, args.seed)
        oof = {k: np.zeros(len(ys)) for k in NAMES}
        oof["C0"] = Xs[:, ni].copy()
        rec_models, _ = D.fit_fold_ensemble(Xr, yr, gr, args.folds, args.seed)
        oof["C1"] = D.predict(rec_models, Xs)
        n_rid = int(gr.max()) + 1
        for k in range(args.folds):
            tr, te = fold != k, fold == k
            sel = stable_idx(auc_r, Xs[tr], ys[tr])
            m2, _ = D.fit_fold_ensemble(Xr[:, sel], yr, gr, args.folds, args.seed)
            oof["C2"][te] = D.predict(m2, Xs[te][:, sel])
            Xa = np.vstack([Xr, Xs[tr]])
            ya = np.concatenate([yr, ys[tr]])
            ga = np.concatenate([gr, n_rid + gs[tr]])
            ga = np.unique(ga, return_inverse=True)[1]
            m3, _ = D.fit_fold_ensemble(Xa, ya, ga, args.folds, args.seed)
            oof["C3"][te] = D.predict(m3, Xs[te])
            m4, _ = D.fit_fold_ensemble(Xa[:, sel], ya, ga, args.folds, args.seed)
            oof["C4"][te] = D.predict(m4, Xs[te][:, sel])
            print(f"  fold {k + 1}/{args.folds} done ({len(sel)} stable features)")

        P("\n" + "=" * 104)
        P("2. CANDIDATES on synthetic BioFors (scores from models that never saw the scored documents)")
        P("=" * 104)
        P(f"{'':4s}{'candidate':52s} {'AUC':>6s} {'AUC publ.':>9s}" + "".join(f"{v[:6]:>8s}" for v in
          ["clean", "rotate", "scale", "flip", "blur", "jpeg"]))
        variants = ["clean", "rotate", "scale", "flip", "blur", "jpeg"]
        cand = {}
        for k in NAMES:
            s = oof[k]
            a = safe_auc(ys, s, ws)
            ap_ = safe_auc(ys[pub], s[pub], ws[pub])
            per_v = [safe_auc(ys[var_s == v], s[var_s == v], ws[var_s == v]) for v in variants]
            best, grid = operating(s, ys, ws, args.prevalence)
            cand[k] = {"name": NAMES[k], "auc": a, "auc_publication": ap_, "auc_by_variant": dict(zip(variants, per_v)),
                       "best": best, "grid": grid}
            P(f"{k:4s}{NAMES[k]:52s} {a:6.3f} {ap_:9.3f}" + "".join(f"{x:8.3f}" for x in per_v))

        P(f"\n3. OPERATING POINT: false-positive target that maximises image MCC at {100 * args.prevalence:.1f}% forged")
        P(f"{'':4s}{'candidate':52s} {'FPR target':>10s} {'detection':>9s} {'exp. image MCC [95% CI]':>26s}")
        rng = np.random.default_rng(args.seed)
        nd = int(gs.max()) + 1
        boots = {k: [] for k in NAMES}
        for _ in range(args.boot):
            cnt = np.bincount(rng.integers(0, nd, nd), minlength=nd)
            wb = ws * cnt[gs]
            for k in NAMES:
                thr = cand[k]["best"]["thr"]
                s = oof[k]
                if wb[ys == 1].sum() == 0 or wb[ys == 0].sum() == 0:
                    continue
                tpr = np.average(s[ys == 1] >= thr, weights=wb[ys == 1])
                fpr = np.average(s[ys == 0] >= thr, weights=wb[ys == 0])
                boots[k].append(exp_mcc(tpr, fpr, args.prevalence))
        for k in NAMES:
            b = cand[k]["best"]
            lo, hi = np.percentile(boots[k], [2.5, 97.5])
            cand[k]["exp_mcc_ci"] = [float(lo), float(hi)]
            P(f"{k:4s}{NAMES[k]:52s} {100 * b['target']:9.1f}% {100 * b['tpr']:8.1f}% {b['mcc']:10.3f} [{lo:.3f},{hi:.3f}]")
        diffs = {}
        for k in NAMES:
            if k == "C0":
                continue
            d = np.array(boots[k]) - np.array(boots["C0"])
            diffs[k] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
            P(f"    {k} - C0: {cand[k]['best']['mcc'] - cand['C0']['best']['mcc']:+.3f}  95% CI [{diffs[k][0]:+.3f},{diffs[k][1]:+.3f}]")

        top = max(NAMES, key=lambda k: cand[k]["best"]["mcc"])
        chosen = next(k for k in SIMPLICITY if cand[k]["best"]["mcc"] >= cand[top]["best"]["mcc"] - 0.01)
        P(f"\nSELECTION (rule fixed in advance): best = {top} ({cand[top]['best']['mcc']:.3f});  chosen = {chosen}  "
          f"[{NAMES[chosen]}]")

        # ---------------------------------------------------------------- 4. robustness
        P("\n" + "=" * 104)
        P("4. ROBUSTNESS on synthetic BioFors: share of forged images caught at the chosen operating point")
        P("   (rows = how the copy was made, columns = image type; last column = genuine images wrongly flagged)")
        P("=" * 104)
        rob = {}
        for k in dict.fromkeys(["C0", chosen]):
            thr = cand[k]["best"]["thr"]
            flag = (oof[k] >= thr) & has_region
            P(f"\n{k} {NAMES[k]}   (cut-off {thr:.4g}, image counted as flagged only if a region is drawn)")
            P(f"{'variant':14s}" + "".join(f"{c:>12s}" for c in cats) + f"{'all':>8s}{'FPR':>8s}")
            rob[k] = {}
            for v in variants + ["publication"]:
                vs = var_s == v
                share = lambda m_: float(flag[m_].mean()) if m_.any() else float("nan")
                cells = [share(vs & (cat_s == c) & (ys == 1)) for c in cats]
                allv = share(vs & (ys == 1))
                fp = share(vs & (ys == 0))
                rob[k][v] = {"by_type": dict(zip(cats, map(float, cells))), "all": float(allv), "fpr": float(fp)}
                P(f"{v:14s}" + "".join(f"{100 * x:11.1f}%" for x in cells) + f"{100 * allv:7.1f}%{100 * fp:7.1f}%")

        # ---------------------------------------------------------------- 5. freeze
        def final_scorer(k):
            if k == "C0":
                return None, [ni]
            if k == "C1":
                return rec_models, list(range(len(names)))
            sel = stab_all if k in ("C2", "C4") else list(range(len(names)))
            if k == "C2":
                m, _ = D.fit_fold_ensemble(Xr[:, sel], yr, gr, args.folds, args.seed)
                return m, sel
            Xa = np.vstack([Xr, Xs])[:, sel]
            ya = np.concatenate([yr, ys])
            ga = np.unique(np.concatenate([gr, n_rid + gs]), return_inverse=True)[1]
            m, _ = D.fit_fold_ensemble(Xa, ya, ga, args.folds, args.seed)
            return m, sel

        frozen = {}
        P("\n" + "=" * 104)
        P("5. FROZEN for experiment 2 (cut-off set on GENUINE BioFors calibration images, weighted to the test type mix)")
        P("=" * 104)
        for k in dict.fromkeys([chosen, "C0"]):
            models, sel = final_scorer(k)
            sc = Xc[:, ni] if k == "C0" else D.predict(models, Xc[:, sel])
            target = cand[k]["best"]["target"]
            thr = D.threshold_for_fpr(sc, target, wc)
            frozen[k] = {"name": NAMES[k], "models": models, "feature_idx": sel, "features": [names[j] for j in sel],
                         "fpr_target": target, "threshold": float(thr),
                         "calib_fpr_realised": float(np.average(sc >= thr, weights=wc)),
                         "synthetic_expected_mcc": cand[k]["best"]["mcc"], "synthetic_expected_mcc_ci": cand[k]["exp_mcc_ci"]}
            P(f"{k} {NAMES[k]}: FPR target {100 * target:.1f}%  ->  cut-off {thr:.4g} "
              f"(flags {100 * frozen[k]['calib_fpr_realised']:.1f}% of genuine calibration images)   features {len(sel)}")

        meta = {"created": now_iso(), "versions": versions(), "chosen": chosen, "selection_rule":
                "highest expected image MCC on synthetic BioFors; simpler within 0.01 wins; order " + " < ".join(SIMPLICITY),
                "prevalence": args.prevalence, "feature_names_all": names, "stable_features": [names[j] for j in stab_all],
                "candidates": {k: {kk: vv for kk, vv in v.items() if kk != "grid"} for k, v in cand.items()},
                "diff_vs_C0_ci": diffs, "sources": {"recod": os.path.abspath(args.recod), "synth": os.path.abspath(args.synth),
                                                    "biofors": os.path.abspath(args.biofors)}}
        joblib.dump({"frozen": frozen, "meta": meta}, os.path.join(args.out, "final_v66.joblib"))
        with open(os.path.join(args.out, "final_v66.json"), "w") as f:
            json.dump({"meta": meta, "frozen": {k: {kk: vv for kk, vv in v.items() if kk != "models"} for k, v in frozen.items()}},
                      f, indent=1, default=float)
        ts = stamp()
        with open(os.path.join(args.out, f"develop_metrics_{ts}.json"), "w") as f:
            json.dump({"stability": stab_rows, "candidates": cand, "diff_vs_C0_ci": diffs, "robustness": rob,
                       "chosen": chosen}, f, indent=1, default=float)
        with open(os.path.join(args.out, "synth_oof_scores.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["set", "sid", "doc", "category", "variant"] + list(NAMES))
            for i, r in enumerate(S):
                w.writerow([r["set"], r["sid"], r["doc"], r["category"], r["variant"]] + [round(float(oof[k][i]), 6) for k in NAMES])
        for p in ("final_v66.joblib", "final_v66.json", f"develop_metrics_{ts}.json", "synth_oof_scores.csv"):
            run.output(os.path.join(args.out, p))
        run.note(chosen=chosen, **{f"{k}_exp_mcc": round(cand[k]["best"]["mcc"], 4) for k in NAMES})
        P("\nNothing above used a BioFors test label. Next:  python test_biofors.py   (looks at the test set ONCE)")


if __name__ == "__main__":
    main()
