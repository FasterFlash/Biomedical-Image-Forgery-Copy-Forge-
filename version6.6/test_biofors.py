"""
test_biofors.py  --  STEP 4 of V6.6 = EXPERIMENT 2. Looks at the BioFors test set ONCE.

Applies what develop.py froze (results\\final_v66.joblib) to the BioFors test images. No detector rerun: the
version6 run already stored every test image's features and regions (..\\version6\\results\\).
Scored exactly like version6 / the BioFors authors. Shown next to the version6 (experiment 1) numbers.

A lock file (results\\biofors_test_used.txt) refuses a second look unless --allow-repeat. If you rerun it,
say so in the write-up.

USAGE:  python test_biofors.py
OUTPUT (results\\): test_biofors_<time>.txt, test_biofors_metrics_<time>.json, predictions_v66\\<doc>\\<panel>
"""

import os
import csv
import json
import time
import argparse

import numpy as np
import joblib

import decision as D
from runlog import Run, stamp
from score_biofors import metrics, boot, auc_ci, PUBLISHED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", default=os.path.join("results", "final_v66.joblib"))
    ap.add_argument("--v6", default=os.path.join("..", "version6", "results"))
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-repeat", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    lock = os.path.join(args.out, "biofors_test_used.txt")
    if os.path.exists(lock) and not args.allow_repeat:
        raise SystemExit(f"REFUSED: the BioFors test set was already used for experiment 2 ({open(lock).read().strip()}).\n"
                         "Rerunning after changes is how a test set stops being a test. Use --allow-repeat only if you "
                         "will report it.")

    with Run("test_biofors", args, args.out) as run:
        P = run.p
        bundle = joblib.load(args.frozen)
        frozen, meta = bundle["frozen"], bundle["meta"]
        names = meta["feature_names_all"]
        rows = [r for r in csv.DictReader(open(os.path.join(args.v6, "biofors_features.csv"), newline=""))
                if r["set"] == "test"]
        X = np.array([[float(r[n]) for n in names] for r in rows])
        r_ = {"gt": np.array([int(r["gt_forged"]) for r in rows]), "total": np.array([float(r["px_total"]) for r in rows]),
              "gt_px": np.array([float(r["px_gt"]) for r in rows]), "pred_raw": np.array([float(r["px_pred_raw"]) for r in rows]),
              "tp_raw": np.array([float(r["px_tp_raw"]) for r in rows])}
        has_region = r_["pred_raw"] > 0
        cat = np.array([r["category"] for r in rows])
        docs = np.array([r["doc"] for r in rows])
        cats = sorted(set(cat))
        P(f"EXPERIMENT 2   chosen in develop.py: {meta['chosen']} ({frozen[meta['chosen']]['name']})   frozen {meta['created']}")
        P(f"test images {len(rows)}, forged {r_['gt'].sum()}   (features from the version6 run, detector unchanged)")

        flags, scores = {}, {}
        for k, fz in frozen.items():
            sel = fz["feature_idx"]
            s = X[:, sel[0]] if fz["models"] is None else D.predict(fz["models"], X[:, sel])
            scores[k] = s
            flags[f"{k}: {fz['name']} @ {100 * fz['fpr_target']:.1f}% FPR target"] = (s >= fz["threshold"]) & has_region

        v6 = {}
        p6 = os.path.join(args.v6, "biofors_metrics_latest.json")
        if os.path.exists(p6):
            v6 = json.load(open(p6))["results"]

        results = {}
        for c in ["ALL"] + cats:
            sel = np.ones(len(rows), bool) if c == "ALL" else cat == c
            rs = {k: v[sel] for k, v in r_.items()}
            P("\n" + "=" * 104)
            P(f"{'ALL IMAGE TYPES' if c == 'ALL' else c.upper()}   ({sel.sum()} images, {rs['gt'].sum()} forged)")
            P("=" * 104)
            P(f"{'setting':58s} {'FPR':>7s} {'detect':>7s} {'image MCC [95% CI]':>21s} {'pixel MCC [95% CI]':>21s}")
            results[c] = {}
            for name, fl in flags.items():
                m = metrics(fl[sel], rs)
                ci = boot(fl[sel], rs, docs[sel], args.boot, args.seed)
                results[c][name] = {**{k: float(v) for k, v in m.items()}, "ci95": ci}
                P(f"{name:58s} {100 * m['fpr']:6.1f}% {100 * m['detection']:6.1f}% "
                  f"{m['img_mcc']:7.3f} [{ci['img_mcc'][0]:.3f},{ci['img_mcc'][1]:.3f}] "
                  f"{m['px_mcc']:7.3f} [{ci['px_mcc'][0]:.3f},{ci['px_mcc'][1]:.3f}]")
            for k6 in ("count, Recod.ai cut-off", "model, zero-shot", "model, recalibrated"):
                if c in v6 and k6 in v6[c]:
                    m = v6[c][k6]
                    P(f"{'version6: ' + k6:58s} {100 * m['fpr']:6.1f}% {100 * m['detection']:6.1f}% "
                      f"{m['img_mcc']:7.3f} {'':15s}{m['px_mcc']:7.3f}")
            for n_, d_ in PUBLISHED.items():
                if c in d_:
                    P(f"{'published: ' + n_:58s} {'':16s}{d_[c][0]:7.3f} {'':15s}{d_[c][1]:7.3f}")

        P("\nthreshold-free ranking on the test set (AUC, 95% CI by documents):")
        aucs = {}
        for k, s in scores.items():
            a, ci = auc_ci(s, r_["gt"], docs, args.boot, args.seed)
            aucs[k] = {"auc": a, "ci": ci}
            P(f"  {k}: {frozen[k]['name']:52s} AUC {a:.3f} [{ci[0]:.3f},{ci[1]:.3f}]")

        ts = stamp()
        pj = os.path.join(args.out, f"test_biofors_metrics_{ts}.json")
        with open(pj, "w") as f:
            json.dump({"chosen": meta["chosen"], "results": results, "auc": aucs, "frozen": {k: {kk: vv for kk, vv in v.items()
                       if kk != "models"} for k, v in frozen.items()}, "version6_reference": v6.get("ALL", {})}, f, indent=1, default=float)
        run.output(pj)

        import cv2
        ch = meta["chosen"]
        fl = next(v for k, v in flags.items() if k.startswith(ch + ":"))
        root = os.path.join(args.out, "predictions_v66")
        for i, r in enumerate(rows):
            d = os.path.join(root, r["doc"])
            os.makedirs(d, exist_ok=True)
            m = cv2.imread(os.path.join(args.v6, "raw_masks", r["doc"], r["panel"]), cv2.IMREAD_GRAYSCALE) if fl[i] else None
            if m is None:
                m = np.zeros((int(float(r["h"])), int(float(r["w"]))), np.uint8)
            cv2.imwrite(os.path.join(d, r["panel"]), m)
        run.output(root)
        with open(lock, "w") as f:
            f.write(f"used {time.strftime('%Y-%m-%d %H:%M:%S')} for {ch}\n")
        run.output(lock)
        a = results["ALL"]
        run.note(**{k.split(":")[0] + "__img_mcc": round(v["img_mcc"], 4) for k, v in a.items()},
                 **{k.split(":")[0] + "__px_mcc": round(v["px_mcc"], 4) for k, v in a.items()})
        P(f"\nSaved: {pj}, predictions_v66\\ (authors' format), lock {lock}")


if __name__ == "__main__":
    main()
