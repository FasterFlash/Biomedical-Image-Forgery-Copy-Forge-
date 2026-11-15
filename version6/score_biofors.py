"""
score_biofors.py  --  STEP 3 of V6. The generalisation numbers. Fast (seconds); rerun any time.

Scored EXACTLY like the BioFors authors' eval_IDD.py:
  - ground truth: the boxes in idd_gt.json filled in as a mask
  - image level : an image counts as "flagged" only if its predicted mask is non-empty
  - pixel level : TP/FP/FN/TN summed over ALL pixels of ALL test images, then one MCC / F1
                  (NOT an average of per-image scores, which is what version5.5 reported on Recod.ai)
  - image types : Microscopy, Blot/Gel, Macroscopy  (their default)

Settings compared (each one is a different cut-off on the SAME detector output):
  V4 rule                      flag if >= 3 confirmed matches (what V4/V5 did)
  count, Recod.ai cut-off      old rule, cut-off fixed on Recod.ai at 5% FPR
  count, recalibrated          old rule, cut-off re-set on GENUINE BioFors training images at 5% FPR
  model, zero-shot             the V5.5 model and its Recod.ai cut-off, unchanged      <- generalisation, no adaptation
  model, recalibrated          same model, cut-off re-set on GENUINE BioFors images      <- label-free adaptation
  model, recal. per type       same, one cut-off per image type (Blot/Gel, Microscopy, Macroscopy)
Recalibration uses NO forged BioFors image and NO test image.

USAGE:
  python score_biofors.py
  python score_biofors.py --boot 2000 --no-masks

OUTPUT (results\\):
  score_biofors_<time>.txt + score_biofors_latest.txt   the report
  biofors_metrics_<time>.json + biofors_metrics_latest.json   every number, with 95% CIs
  biofors_scores.csv            per image: score, matches, flag under every setting
  predictions\\<setting>\\<doc>\\<panel>   predicted masks in the authors' format (0/255), so their own
                                        eval_IDD.py can be run on them as an independent check
"""

import os
import csv
import json
import shutil
import argparse

import numpy as np
import joblib
from sklearn.metrics import roc_auc_score

import decision as D
from runlog import Run, stamp

# BioFors paper (Sabir et al., ICCV 2021), Table 5 "Results for internal duplication detection (IDD) task by image
# class and a combined result". (image MCC, pixel MCC). Checked against the ICCV open-access PDF on 2026-11-13.
# DF-*, DCT, DWT, Zernike are classical (not trained); the paper says deep baselines (BusterNet) were fine-tuned on
# synthetic manipulations of BioFors training images. Whether classical parameters were tuned is in their supplement.
PUBLISHED = {
    "DF-PCT":    {"ALL": (0.569, 0.364), "Microscopy": (0.764, 0.202), "Blot/Gel": (0.503, 0.466), "Macroscopy": (0.712, 0.487)},
    "DF-ZM":     {"ALL": (0.564, 0.353), "Microscopy": (0.764, 0.197), "Blot/Gel": (0.515, 0.449), "Macroscopy": (0.573, 0.478)},
    "DF-FMT":    {"ALL": (0.509, 0.316), "Microscopy": (0.638, 0.167), "Blot/Gel": (0.480, 0.400), "Macroscopy": (0.495, 0.458)},
    "DWT":       {"ALL": (0.341, 0.171), "Microscopy": (0.299, 0.067), "Blot/Gel": (0.384, 0.295), "Macroscopy": (0.591, 0.268)},
    "BusterNet": {"ALL": (0.269, 0.107), "Microscopy": (0.183, 0.178), "Blot/Gel": (0.226, 0.076), "Macroscopy": (0.021, 0.106)},
    "Zernike":   {"ALL": (0.257, 0.114), "Microscopy": (0.192, 0.032), "Blot/Gel": (0.336, 0.187), "Macroscopy": (0.493, 0.262)},
    "DCT":       {"ALL": (0.196, 0.095), "Microscopy": (0.187, 0.022), "Blot/Gel": (0.250, 0.168), "Macroscopy": (0.158, 0.143)},
}


def load_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def metrics(flag, r, w=None):
    """flag: bool per test row (already AND-ed with 'mask non-empty'). r: dict of arrays. w: bootstrap weights."""
    w = np.ones(len(flag)) if w is None else w
    g = r["gt"] == 1
    tp, fn = (w * (flag & g)).sum(), (w * (~flag & g)).sum()
    fp, tn = (w * (flag & ~g)).sum(), (w * (~flag & ~g)).sum()
    ptp = np.where(flag, r["tp_raw"], 0)
    pfp = np.where(flag, r["pred_raw"] - r["tp_raw"], 0)
    pfn = np.where(flag, r["gt_px"] - r["tp_raw"], r["gt_px"])
    ptn = r["total"] - ptp - pfp - pfn
    PTP, PFP, PFN, PTN = (w * ptp).sum(), (w * pfp).sum(), (w * pfn).sum(), (w * ptn).sum()
    f1 = lambda a, b, c: 2 * a / (2 * a + b + c) if (2 * a + b + c) > 0 else 0.0
    return {"img_tp": tp, "img_fp": fp, "img_fn": fn, "img_tn": tn,
            "fpr": fp / (fp + tn) if fp + tn else 0.0, "detection": tp / (tp + fn) if tp + fn else 0.0,
            "img_precision": tp / (tp + fp) if tp + fp else 0.0, "img_f1": f1(tp, fp, fn),
            "img_mcc": D.mcc(tp, fp, fn, tn),
            "px_precision": PTP / (PTP + PFP) if PTP + PFP else 0.0, "px_recall": PTP / (PTP + PFN) if PTP + PFN else 0.0,
            "px_f1": f1(PTP, PFP, PFN), "px_mcc": D.mcc(PTP, PFP, PFN, PTN)}


def boot(flag, r, docs, B, seed, keys=("fpr", "detection", "img_mcc", "px_mcc", "img_f1", "px_f1")):
    rng = np.random.default_rng(seed)
    ud, inv = np.unique(docs, return_inverse=True)
    out = {k: np.zeros(B) for k in keys}
    for b in range(B):
        cnt = np.bincount(rng.integers(0, len(ud), len(ud)), minlength=len(ud))
        m = metrics(flag, r, cnt[inv].astype(float))
        for k in keys:
            out[k][b] = m[k]
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}


def auc_ci(score, y, docs, B, seed):
    if len(np.unique(y)) < 2:
        return None, None
    a = float(roc_auc_score(y, score))
    rng = np.random.default_rng(seed)
    ud, inv = np.unique(docs, return_inverse=True)
    vals = []
    for _ in range(B):
        cnt = np.bincount(rng.integers(0, len(ud), len(ud)), minlength=len(ud))
        w = cnt[inv].astype(float)
        if w[y == 1].sum() == 0 or w[y == 0].sum() == 0:
            continue
        vals.append(roc_auc_score(y, score, sample_weight=w))
    return a, [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join("results", "biofors_features.csv"))
    ap.add_argument("--model", default=os.path.join("results", "final_model.joblib"))
    ap.add_argument("--fpr", type=float, default=None, help="operating FPR (default: the one saved with the model)")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-masks", action="store_true", help="do not write authors-format prediction masks")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    # environment guard: a model saved by one scikit-learn version cannot be loaded by another, and features
    # made with different OpenCV versions are not interchangeable. Stop with a clear message instead of a traceback.
    import sklearn
    meta_json = os.path.splitext(args.model)[0] + ".json"
    if os.path.exists(meta_json):
        saved = json.load(open(meta_json)).get("versions", {})
        if saved.get("sklearn") and saved["sklearn"] != sklearn.__version__:
            raise SystemExit(f"final_model was trained with scikit-learn {saved['sklearn']} but this Python has "
                             f"{sklearn.__version__}. Activate the same environment (..\\.venv\\Scripts\\Activate.ps1) "
                             f"and rerun train_final.py, then this script.")

    with Run("score_biofors", args, args.out) as run:
        bundle = joblib.load(args.model)
        models, meta = bundle["models"], bundle["meta"]
        op = args.fpr if args.fpr is not None else meta["operating_fpr"]
        names = meta["feature_names"]
        rows = load_rows(args.features)
        test = [x for x in rows if x["set"] == "test"]
        calib = [x for x in rows if x["set"] == "calib"]
        run.p(f"model: {meta['model']}   trained {meta['created']} on {meta['n_rows']} Recod.ai images, mask {meta['mask'].upper()}")
        run.p(f"BioFors rows: test {len(test)} ({sum(int(x['gt_forged']) for x in test)} forged), calib {len(calib)} genuine")
        if not test or not calib:
            run.p("Need both the test and the calib sets. Run run_biofors.py first.")
            return
        cats = sorted({x["category"] for x in test})

        def arr(rs, k, typ=float):
            return np.array([typ(x[k]) for x in rs])

        Xt = np.array([[float(x[n]) for n in names] for x in test])
        Xc = np.array([[float(x[n]) for n in names] for x in calib])
        st, sc = D.predict(models, Xt), D.predict(models, Xc)
        nt, nc = arr(test, "n_inl"), arr(calib, "n_inl")
        r = {"gt": arr(test, "gt_forged", int), "total": arr(test, "px_total"), "gt_px": arr(test, "px_gt"),
             "pred_raw": arr(test, "px_pred_raw"), "tp_raw": arr(test, "px_tp_raw")}
        has_mask = r["pred_raw"] > 0
        cat_t, cat_c = np.array([x["category"] for x in test]), np.array([x["category"] for x in calib])
        docs = np.array([x["doc"] for x in test])

        # calibration weights: make the genuine sample mirror the test set's image-type mix
        share_t = {c: np.mean(cat_t == c) for c in cats}
        share_c = {c: np.mean(cat_c == c) for c in cats}
        wc = np.array([share_t.get(c, 0) / share_c[c] if share_c.get(c) else 0.0 for c in cat_c])

        tm0, tc0 = meta["thresholds_model"][str(op)], meta["thresholds_count"][str(op)]
        tm_rec = D.threshold_for_fpr(sc, op, wc)
        tc_rec = D.threshold_for_fpr(nc, op, wc)
        tm_cat = {c: D.threshold_for_fpr(sc[cat_c == c], op) for c in cats}

        settings = {
            "V4 rule (>=3 matches)": nt >= 3,
            "count, Recod.ai cut-off": nt >= tc0,
            "count, recalibrated": nt >= tc_rec,
            "model, zero-shot": st >= tm0,
            "model, recalibrated": st >= tm_rec,
            "model, recal. per type": np.array([s >= tm_cat[c] for s, c in zip(st, cat_t)]),
        }
        lost = {k: int((v & ~has_mask).sum()) for k, v in settings.items()}
        flags = {k: v & has_mask for k, v in settings.items()}

        run.p(f"\noperating FPR {100 * op:.0f}%")
        run.p(f"cut-offs   model: Recod.ai {tm0:.4f}  ->  BioFors-recalibrated {tm_rec:.4f}   per type "
              + ", ".join(f"{c} {t:.4f}" for c, t in tm_cat.items()))
        run.p(f"           count: Recod.ai {tc0:.0f}  ->  BioFors-recalibrated {tc_rec:.0f} matches")

        # label-free drift check: what the Recod.ai cut-off does on GENUINE BioFors training images
        run.p("\n" + "=" * 104)
        run.p("A. DRIFT CHECK on genuine BioFors training images (no labels needed): share flagged at the Recod.ai cut-off")
        run.p("=" * 104)
        drift = {}
        for c in cats + ["ALL (test-mix weighted)"]:
            if c.startswith("ALL"):
                fm, fc = np.average(sc >= tm0, weights=wc), np.average(nc >= tc0, weights=wc)
            else:
                fm, fc = np.mean(sc[cat_c == c] >= tm0), np.mean(nc[cat_c == c] >= tc0)
            drift[c] = {"model": float(fm), "count": float(fc)}
            run.p(f"  {c:26s} model {100 * fm:5.1f}%    count rule {100 * fc:5.1f}%     (target {100 * op:.0f}%)")

        # threshold-free
        run.p("\n" + "=" * 104)
        run.p("B. RANKING (threshold-free AUC on the test set; 95% CI by resampling documents)")
        run.p("=" * 104)
        aucs = {}
        for c in ["ALL"] + cats:
            sel = np.ones(len(st), bool) if c == "ALL" else cat_t == c
            am, cim = auc_ci(st[sel], r["gt"][sel], docs[sel], args.boot, args.seed)
            ac, cic = auc_ci(nt[sel], r["gt"][sel], docs[sel], args.boot, args.seed)
            aucs[c] = {"model": am, "model_ci": cim, "count": ac, "count_ci": cic,
                       "n": int(sel.sum()), "n_forged": int(r["gt"][sel].sum())}
            if am is None:
                run.p(f"  {c:12s} (only one class present)")
            else:
                run.p(f"  {c:12s} n={sel.sum():5d} forged={r['gt'][sel].sum():3d}   model AUC {am:.3f} [{cim[0]:.3f},{cim[1]:.3f}]"
                      f"   match-count AUC {ac:.3f} [{cic[0]:.3f},{cic[1]:.3f}]")

        # main table
        results = {}
        for c in ["ALL"] + cats:
            sel = np.ones(len(st), bool) if c == "ALL" else cat_t == c
            rs = {k: v[sel] for k, v in r.items()}
            run.p("\n" + "=" * 104)
            run.p(f"C. {'ALL IMAGE TYPES' if c == 'ALL' else c.upper()}   ({sel.sum()} images, {rs['gt'].sum()} forged, "
                  f"{len(np.unique(docs[sel]))} documents)   scored the BioFors-authors' way")
            run.p("=" * 104)
            run.p(f"{'setting':26s} {'FPR':>16s} {'detection':>17s} {'image MCC':>18s} {'pixel MCC':>18s} {'img F1':>7s} {'px F1':>7s}")
            results[c] = {}
            for name, fl in flags.items():
                fs = fl[sel]
                m = metrics(fs, rs)
                ci = boot(fs, rs, docs[sel], args.boot, args.seed)
                results[c][name] = {k: float(v) for k, v in m.items()}
                results[c][name]["ci95"] = ci
                fmt = lambda k, sc_=100.0, d=1, suf="%": (f"{sc_ * m[k]:.{d}f}{suf} [{sc_ * ci[k][0]:.{d}f},{sc_ * ci[k][1]:.{d}f}]")
                run.p(f"{name:26s} {fmt('fpr'):>16s} {fmt('detection'):>17s} {fmt('img_mcc', 1, 3, ''):>18s} "
                      f"{fmt('px_mcc', 1, 3, ''):>18s} {m['img_f1']:7.3f} {m['px_f1']:7.3f}")
            run.p("\n  Published on the same test set (BioFors paper, Table 5; DF-*/DCT/DWT/Zernike classical, "
                  "BusterNet deep, fine-tuned on synthetic BioFors data):")
            for n_, d_ in PUBLISHED.items():
                if c in d_:
                    run.p(f"  {n_:24s} image MCC {d_[c][0]:.3f}   pixel MCC {d_[c][1]:.3f}")
        if any(lost.values()):
            run.p(f"\nnote: images a setting flagged but where the detector drew no region (count as NOT flagged, "
                  f"as in the authors' script): {lost}")

        # save everything
        ts = stamp()
        out_json = {"created": ts, "operating_fpr": op, "model_meta": meta,
                    "thresholds": {"model_recod": tm0, "model_recal": tm_rec, "model_per_type": tm_cat,
                                   "count_recod": tc0, "count_recal": tc_rec},
                    "calib_type_weights": {c: float(share_t.get(c, 0) / share_c[c]) for c in share_c if share_c[c]},
                    "drift_check": drift, "auc": aucs, "results": results, "published_reference": PUBLISHED,
                    "flagged_but_no_region": lost, "n_test": len(test), "n_calib": len(calib)}
        pj = os.path.join(args.out, f"biofors_metrics_{ts}.json")
        with open(pj, "w") as f:
            json.dump(out_json, f, indent=1, default=float)
        shutil.copyfile(pj, os.path.join(args.out, "biofors_metrics_latest.json"))
        run.output(pj)

        ps = os.path.join(args.out, "biofors_scores.csv")
        with open(ps, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["set", "doc", "panel", "category", "gt_forged", "n_inl", "model_score", "has_region"] + list(flags))
            for i, x in enumerate(test):
                w.writerow([x["set"], x["doc"], x["panel"], x["category"], x["gt_forged"], int(nt[i]),
                            round(float(st[i]), 6), int(has_mask[i])] + [int(flags[k][i]) for k in flags])
            for i, x in enumerate(calib):
                w.writerow([x["set"], x["doc"], x["panel"], x["category"], 0, int(nc[i]), round(float(sc[i]), 6),
                            int(x["px_pred_raw"] != "0"), "", "", "", "", "", ""])
        run.output(ps)

        if not args.no_masks:
            import cv2
            raw_root = os.path.join(args.out, "raw_masks")
            for name in ("model, zero-shot", "model, recalibrated"):
                slug = name.replace(", ", "_").replace(" ", "_").replace("-", "")
                root = os.path.join(args.out, "predictions", slug)
                for i, x in enumerate(test):
                    d = os.path.join(root, x["doc"])
                    os.makedirs(d, exist_ok=True)
                    m = None
                    if flags[name][i]:
                        m = cv2.imread(os.path.join(raw_root, x["doc"], x["panel"]), cv2.IMREAD_GRAYSCALE)
                    if m is None:
                        m = np.zeros((int(float(x["h"])), int(float(x["w"]))), np.uint8)
                    cv2.imwrite(os.path.join(d, x["panel"]), m)
                run.output(root)
            run.p("\nwrote authors-format prediction masks to results\\predictions\\<setting>\\")

        a = results["ALL"]
        run.note(**{f"{k}__img_mcc": round(v["img_mcc"], 4) for k, v in a.items()},
                 **{f"{k}__px_mcc": round(v["px_mcc"], 4) for k, v in a.items()})
        run.p(f"\nSaved: {pj}, biofors_metrics_latest.json, biofors_scores.csv")


if __name__ == "__main__":
    main()
