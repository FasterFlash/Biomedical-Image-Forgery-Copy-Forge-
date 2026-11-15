"""
train_final.py  --  STEP 1 of V6. Freeze the Recod.ai model before anything touches BioFors.

What it does:
  - reads the Recod.ai features from version5.5 (no detector rerun needed)
  - uses the model chosen in version5.5: gradient-boosted trees, mask OFF, robust features
    (picture statistics and keypoint counts removed)
  - trains on ALL Recod.ai images (the held-out test was already reported in version5.5) as an average
    of 5 models, and fixes the cut-offs from scores each image got from a model that never saw it
  - also fixes the cut-off for the old rule (count of confirmed matches) at the same false-positive rates,
    so BioFors can compare old vs new fairly

USAGE (PowerShell, from the version6 folder, venv active):
  python train_final.py

OUTPUT (results\\):
  final_model.joblib            the 5 models + feature list + cut-offs (used by score_biofors.py)
  final_model.json              the same information without the models, human-readable
  recod_oof_scores.csv          every Recod.ai image's out-of-fold score
  train_final_<time>.txt        this report;  run_log.jsonl  one line per run
"""

import os
import csv
import json
import argparse

import numpy as np
import joblib
from sklearn.metrics import roc_auc_score

import decision as D
from runlog import Run, now_iso, versions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join("..", "version5.5", "results", "features.csv"))
    ap.add_argument("--mask", default="off", choices=["off", "on"])
    ap.add_argument("--drop", nargs="*", default=["image", "kp"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--operating-fpr", type=float, default=0.05)
    ap.add_argument("--fpr-targets", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.10, 0.20])
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    with Run("train_final", args, args.out) as run:
        X, y, gid, names, keys = D.load_recod(args.features, args.mask, args.drop)
        run.p(f"Recod.ai rows: {len(y)}  (forged {int(y.sum())}, authentic {int((1 - y).sum())})   "
              f"image ids: {int(gid.max()) + 1}   features: {len(names)}   mask: {args.mask.upper()}")
        models, oof = D.fit_fold_ensemble(X, y, gid, args.folds, args.seed)
        n_inl = X[:, names.index("n_inl")]

        run.p(f"\nout-of-fold AUC   model {roc_auc_score(y, oof):.4f}   old rule (match count) {roc_auc_score(y, n_inl):.4f}")
        run.p("\ncut-offs fixed on Recod.ai (out-of-fold), and what they give on Recod.ai:")
        run.p(f"{'target FPR':>10s} {'model cut-off':>14s} {'FPR':>7s} {'detect':>7s}   {'count cut-off':>13s} {'FPR':>7s} {'detect':>7s}")
        thr_model, thr_count = {}, {}
        for t in args.fpr_targets:
            tm = D.threshold_for_fpr(oof[y == 0], t)
            tc = D.threshold_for_fpr(n_inl[y == 0], t)
            thr_model[str(t)], thr_count[str(t)] = tm, tc
            run.p(f"{100 * t:9.0f}% {tm:14.4f} {100 * (oof[y == 0] >= tm).mean():6.1f}% {100 * (oof[y == 1] >= tm).mean():6.1f}%"
                  f"   {tc:13.0f} {100 * (n_inl[y == 0] >= tc).mean():6.1f}% {100 * (n_inl[y == 1] >= tc).mean():6.1f}%")

        meta = {
            "created": now_iso(), "versions": versions(),
            "source_features": os.path.abspath(args.features), "mask": args.mask, "dropped_groups": args.drop,
            "feature_names": names, "n_rows": int(len(y)), "n_forged": int(y.sum()), "n_authentic": int((1 - y).sum()),
            "n_ids": int(gid.max()) + 1, "folds": args.folds, "seed": args.seed,
            "model": "average of 5 HistGradientBoostingClassifier (lr 0.05, 250 iter, depth 3, min leaf 20, l2 1.0)",
            "oof_auc_model": float(roc_auc_score(y, oof)), "oof_auc_count": float(roc_auc_score(y, n_inl)),
            "operating_fpr": args.operating_fpr,
            "thresholds_model": thr_model, "thresholds_count": thr_count,
        }
        joblib.dump({"models": models, "meta": meta}, os.path.join(args.out, "final_model.joblib"))
        with open(os.path.join(args.out, "final_model.json"), "w") as f:
            json.dump(meta, f, indent=2)
        with open(os.path.join(args.out, "recod_oof_scores.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["label", "image_id", "oof_score", "n_inl"])
            for k, s, c in zip(keys, oof, n_inl):
                w.writerow([k[0], k[1], round(float(s), 6), int(c)])
        for p in ("final_model.joblib", "final_model.json", "recod_oof_scores.csv"):
            run.output(os.path.join(args.out, p))
        run.note(oof_auc_model=meta["oof_auc_model"], oof_auc_count=meta["oof_auc_count"],
                 threshold_model_operating=thr_model[str(args.operating_fpr)],
                 threshold_count_operating=thr_count[str(args.operating_fpr)])
        run.p(f"\nOperating point: {100 * args.operating_fpr:.0f}% FPR -> model cut-off {thr_model[str(args.operating_fpr)]:.4f}, "
              f"count cut-off {thr_count[str(args.operating_fpr)]:.0f} matches")
        run.p("Saved: results\\final_model.joblib, final_model.json, recod_oof_scores.csv")
        run.p("Next:  python run_biofors.py --limit 10     (smoke test), then  python run_biofors.py --workers 4")


if __name__ == "__main__":
    main()
