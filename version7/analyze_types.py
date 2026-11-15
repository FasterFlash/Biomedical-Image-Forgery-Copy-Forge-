"""
analyze_types.py  --  VERSION 7 (final). Where does copy-move detection work, and where does it fail, by type of
biomedical image? On Recod.ai and on BioFors, side by side.

Nothing is rerun or retrained. It joins numbers that already exist:
  ..\\version5.5\\results\\features.csv          per-image detector measurements (mask OFF rows) + per-image pixel MCC
  ..\\version6\\results\\recod_oof_scores.csv    learned-model score for every Recod.ai image, OUT-OF-FOLD
                                               (each image scored by a model that never saw its figure)
  ..\\version6\\results\\final_model.json       the cut-offs fixed in version6 (5% false positives on Recod.ai)
  ..\\version6\\results\\biofors_scores.csv     BioFors test: model score, matches, per image
  ..\\version6\\results\\biofors_features.csv   BioFors test: pixel counts and measurements
  ..\\version6.7\\results\\recod_types.csv      the reviewed image type of every Recod.ai figure

Three decision rules, all with their version6 cut-offs (nothing re-tuned here):
  V4 rule        flag if >= 3 confirmed matches
  count rule     flag if >= the Recod.ai 5%-FPR cut-off of confirmed matches (48)
  learned model  flag if the version5.5/6 model score >= its Recod.ai 5%-FPR cut-off (0.544)
An image counts as flagged only if the detector actually drew a region (>= 3 confirmed matches), as in BioFors.

Sections of the report:
  A. Recod.ai by image type (all six types)
  B. BioFors test by image type (version6 numbers, recomputed in the same format)
  C. Recod.ai vs BioFors, type by type (Histology merged into Microscopy, as BioFors does)
  D. Why types differ: what the detector sees in each type (image size, keypoints, matches, match exactness)
  E. Exploratory: does a separate cut-off per image type help? (Recod.ai only, honest half-split; NOT applied to
     BioFors, whose test set has been used twice already)

Metrics: FPR (authentic flagged), detection (forged flagged), AUC (threshold-free), image MCC at the dataset's own
mix of forged/authentic AND at 4.3% forged (BioFors' mix, to compare like with like), pixel MCC as the per-image
average over forged images (not flagged = 0), and for BioFors also pooled over all pixels (the authors' metric).
95% CIs by resampling figures (Recod.ai) or documents (BioFors).

USAGE (PowerShell, from version7, venv active):   python analyze_types.py
OUTPUT (results\\): analyze_types_<time>.txt (+ _latest), types_metrics_<time>.json (+ _latest),
                   types_auc.png, types_detection.png, run_log.jsonl
"""

import os
import csv
import json
import shutil
import argparse

import numpy as np
from sklearn.metrics import roc_auc_score

from runlog import Run, stamp

RECOD_TYPES = ["Blot/Gel", "Microscopy", "Histology", "Macroscopy", "Other"]
BIO_TYPES = ["Blot/Gel", "Microscopy", "Macroscopy"]
RULES = ["V4 rule (>=3)", "count rule", "learned model"]
PREV_BIOFORS = 102 / 2363
# BioFors paper Table 5 (image MCC, pixel MCC); best classical method per type, for reference
PUBLISHED_BEST = {"ALL": ("DF-PCT", 0.569, 0.364), "Microscopy": ("DF-PCT/DF-ZM", 0.764, 0.202),
                  "Blot/Gel": ("DF-ZM / DF-PCT", 0.515, 0.466), "Macroscopy": ("DF-PCT", 0.712, 0.487)}


def mcc(tp, fp, fn, tn):
    d = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    return float((float(tp) * tn - float(fp) * fn) / d) if d > 0 else 0.0


def exp_mcc(tpr, fpr, prev):
    return mcc(prev * tpr, (1 - prev) * fpr, prev * (1 - tpr), (1 - prev) * (1 - fpr))


def safe_auc(y, s, w=None):
    if len(np.unique(y[w > 0] if w is not None else y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s, sample_weight=w))


def point(flag, y, pix, w, pooled=None):
    """flag/y/pix arrays; w weights. pix = per-image pixel MCC for forged images (0 when not flagged)."""
    pos, neg = y == 1, y == 0
    tp, fn = (w * (flag & pos)).sum(), (w * (~flag & pos)).sum()
    fp, tn = (w * (flag & neg)).sum(), (w * (~flag & neg)).sum()
    out = {"fpr": fp / (fp + tn) if fp + tn else float("nan"), "det": tp / (tp + fn) if tp + fn else float("nan"),
           "img_mcc": mcc(tp, fp, fn, tn)}
    out["img_mcc_at_4pct"] = exp_mcc(out["det"], out["fpr"], PREV_BIOFORS) if tp + fn and fp + tn else float("nan")
    out["pix_mcc"] = float((w[pos] * np.where(flag[pos], pix[pos], 0.0)).sum() / w[pos].sum()) if w[pos].sum() else float("nan")
    if pooled is not None:
        c = {k: np.where(flag, v, 0) for k, v in pooled.items() if k in ("tp", "fp")}
        PTP, PFP = (w * c["tp"]).sum(), (w * c["fp"]).sum()
        PFN = (w * (pooled["gt"] - np.where(flag, pooled["tp"], 0))).sum()
        PTN = (w * pooled["total"]).sum() - PTP - PFP - PFN
        out["pix_mcc_pooled"] = mcc(PTP, PFP, PFN, PTN)
    return out


def evaluate(sel, groups, y, flags, scores, pix, B, seed, pooled=None):
    """Metrics + bootstrap CIs (resampling `groups`) for every rule, on the rows in `sel`."""
    g = np.unique(groups[sel], return_inverse=True)[1]
    ng = int(g.max()) + 1 if sel.any() else 0
    res = {}
    for r in RULES:
        f, s_ = flags[r][sel], scores[r][sel]
        pl = {k: v[sel] for k, v in pooled.items()} if pooled else None
        one = np.ones(sel.sum())
        m = point(f, y[sel], pix[sel], one, pl)
        m["auc"] = safe_auc(y[sel], s_)
        rng = np.random.default_rng(seed)
        bs = {k: [] for k in m}
        for _ in range(B):
            w = np.bincount(rng.integers(0, ng, ng), minlength=ng)[g].astype(float)
            mb = point(f, y[sel], pix[sel], w, pl)
            mb["auc"] = safe_auc(y[sel], s_, w)
            for k, v in mb.items():
                bs[k].append(v)
        m["ci"] = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in bs.items()}
        res[r] = m
    return res


def table(P, title, res, n_f, n_a, pooled=False):
    P(f"\n{title}   (forged {n_f}, authentic {n_a})" + ("   ** small group: read the CIs **" if min(n_f, n_a) < 40 else ""))
    P(f"  {'rule':16s} {'AUC':>18s} {'FPR':>17s} {'detection':>18s} {'image MCC':>8s} {'@4.3%':>6s} {'pixel MCC':>19s}"
      + (f" {'pooled px':>9s}" if pooled else ""))
    for r, m in res.items():
        ci = m["ci"]
        f = lambda k, sc=1, d=3: f"{sc * m[k]:.{d}f} [{sc * ci[k][0]:.{d}f},{sc * ci[k][1]:.{d}f}]"
        P(f"  {r:16s} {f('auc'):>18s} {f('fpr', 100, 1):>17s} {f('det', 100, 1):>18s} {m['img_mcc']:8.3f} "
          f"{m['img_mcc_at_4pct']:6.3f} {f('pix_mcc'):>19s}" + (f" {m['pix_mcc_pooled']:9.3f}" if pooled else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v55", default=os.path.join("..", "version5.5", "results"))
    ap.add_argument("--v6", default=os.path.join("..", "version6", "results"))
    ap.add_argument("--v67", default=os.path.join("..", "version6.7", "results"))
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    with Run("analyze_types", args, args.out) as run:
        P = run.p
        fm = json.load(open(os.path.join(args.v6, "final_model.json")))
        t_model, t_count = fm["thresholds_model"]["0.05"], fm["thresholds_count"]["0.05"]
        P(f"cut-offs from version6 (5% FPR on Recod.ai): model {t_model:.4f}, count {t_count:.0f} matches")

        # ---------------------------------------------------------------- Recod.ai
        types = {r["image_id"]: (r["type"], r["type_biofors"]) for r in csv.DictReader(open(os.path.join(args.v67, "recod_types.csv")))}
        oof = {(r["label"], r["image_id"]): float(r["oof_score"]) for r in csv.DictReader(open(os.path.join(args.v6, "recod_oof_scores.csv")))}
        R = [r for r in csv.DictReader(open(os.path.join(args.v55, "features.csv"))) if r["mask"] == "off"
             and (r["label"], r["image_id"]) in oof and r["image_id"] in types]
        y = np.array([r["label"] == "forged" for r in R]).astype(int)
        ids = np.array([r["image_id"] for r in R])
        typ = np.array([types[r["image_id"]][0] for r in R])
        typ_b = np.array([types[r["image_id"]][1] for r in R])
        n_inl = np.array([float(r["n_inl"]) for r in R])
        score = np.array([oof[(r["label"], r["image_id"])] for r in R])
        pix = np.array([float(r["mcc"]) if r["mcc"] not in ("", None) else 0.0 for r in R])
        region = n_inl >= 3
        flags = {"V4 rule (>=3)": region, "count rule": n_inl >= t_count, "learned model": (score >= t_model) & region}
        scores = {"V4 rule (>=3)": n_inl, "count rule": n_inl, "learned model": score}
        P(f"Recod.ai images: {len(R)} (forged {y.sum()}, authentic {(1 - y).sum()}) from {len(set(ids))} figures")

        out = {"cutoffs": {"model": t_model, "count": t_count}, "recod": {}, "biofors": {}}
        P("\n" + "=" * 118)
        P("A. RECOD.AI BY IMAGE TYPE   (out-of-fold scores; version6 cut-offs; CIs by resampling figures)")
        P("   image MCC = at Recod.ai's own forged/authentic mix;  @4.3% = expected at BioFors' mix;  pixel MCC = per-image mean")
        P("=" * 118)
        for t in ["ALL"] + RECOD_TYPES:
            sel = np.ones(len(R), bool) if t == "ALL" else typ == t
            if not sel.any():
                continue
            res = evaluate(sel, ids, y, flags, scores, pix, args.boot, args.seed)
            out["recod"][t] = res
            table(P, f"[{t}]", res, int(y[sel].sum()), int((1 - y[sel]).sum()))

        # ---------------------------------------------------------------- BioFors
        S = {(r["doc"], r["panel"]): r for r in csv.DictReader(open(os.path.join(args.v6, "biofors_scores.csv"))) if r["set"] == "test"}
        Fb = [r for r in csv.DictReader(open(os.path.join(args.v6, "biofors_features.csv"))) if r["set"] == "test"]
        yb = np.array([int(r["gt_forged"]) for r in Fb])
        docs = np.array([r["doc"] for r in Fb])
        cat = np.array([r["category"] for r in Fb])
        nb = np.array([float(r["n_inl"]) for r in Fb])
        sb = np.array([float(S[(r["doc"], r["panel"])]["model_score"]) for r in Fb])
        tot = np.array([float(r["px_total"]) for r in Fb]); gt = np.array([float(r["px_gt"]) for r in Fb])
        pr = np.array([float(r["px_pred_raw"]) for r in Fb]); tp = np.array([float(r["px_tp_raw"]) for r in Fb])
        pixb = np.array([mcc(a, b - a, c - a, d - b - c + a) if g_ else 0.0 for a, b, c, d, g_ in zip(tp, pr, gt, tot, yb)])
        regb = pr > 0
        flb = {"V4 rule (>=3)": (nb >= 3) & regb, "count rule": (nb >= t_count) & regb, "learned model": (sb >= t_model) & regb}
        scb = {"V4 rule (>=3)": nb, "count rule": nb, "learned model": sb}
        pooled = {"tp": tp, "fp": pr - tp, "gt": gt, "total": tot}
        P("\n" + "=" * 118)
        P("B. BIOFORS TEST BY IMAGE TYPE   (version6 run, same cut-offs; CIs by resampling documents; no new look at labels)")
        P("   pooled px = pixel MCC over all pixels of all images, the BioFors authors' metric")
        P("=" * 118)
        for t in ["ALL"] + BIO_TYPES:
            sel = np.ones(len(Fb), bool) if t == "ALL" else cat == t
            res = evaluate(sel, docs, yb, flb, scb, pixb, args.boot, args.seed, pooled)
            out["biofors"][t] = res
            table(P, f"[{t}]", res, int(yb[sel].sum()), int((1 - yb[sel]).sum()), pooled=True)
            if t in PUBLISHED_BEST:
                n_, a, b = PUBLISHED_BEST[t]
                P(f"  published best ({n_}): image MCC {a:.3f}, pooled pixel MCC {b:.3f}")

        # ---------------------------------------------------------------- C. cross-dataset
        P("\n" + "=" * 118)
        P("C. RECOD.AI vs BIOFORS, TYPE BY TYPE   (Histology merged into Microscopy for this table)")
        P("   'image MCC @4.3%' puts both datasets on BioFors' forged/authentic mix, so the columns compare like with like")
        P("=" * 118)
        cross = {}
        for t in BIO_TYPES + ["ALL"]:
            sel = np.ones(len(R), bool) if t == "ALL" else typ_b == t
            rr = out["recod"]["ALL"] if t == "ALL" else evaluate(sel, ids, y, flags, scores, pix, args.boot, args.seed)
            bb = out["biofors"][t]
            cross[t] = {"recod": rr, "biofors": bb}
            P(f"\n[{t}]   Recod.ai forged {int(y[sel].sum())}  |  BioFors forged {int(yb[(cat == t) if t != 'ALL' else slice(None)].sum())}")
            P(f"  {'rule':16s} {'AUC Recod':>10s} {'AUC BioF':>9s} {'det Recod':>10s} {'det BioF':>9s} {'FPR Recod':>10s} "
              f"{'FPR BioF':>9s} {'imgMCC@4.3 R':>13s} {'imgMCC BioF':>12s} {'pixMCC R':>9s} {'pixMCC B':>9s}")
            for r in RULES:
                a, b = rr[r], bb[r]
                P(f"  {r:16s} {a['auc']:10.3f} {b['auc']:9.3f} {100 * a['det']:9.1f}% {100 * b['det']:8.1f}% "
                  f"{100 * a['fpr']:9.1f}% {100 * b['fpr']:8.1f}% {a['img_mcc_at_4pct']:13.3f} {b['img_mcc']:12.3f} "
                  f"{a['pix_mcc']:9.3f} {b['pix_mcc']:9.3f}")
        out["cross"] = {t: {"recod": {r: {k: v for k, v in d["recod"][r].items() if k != "ci"} for r in RULES},
                            "biofors": {r: {k: v for k, v in d["biofors"][r].items() if k != "ci"} for r in RULES}}
                        for t, d in cross.items()}

        # ---------------------------------------------------------------- D. profile
        P("\n" + "=" * 118)
        P("D. WHAT THE DETECTOR SEES IN EACH TYPE   (medians)")
        P("   inl_ratio = how exact the confirmed matches are (lower = more exact; real copies ~0.4, natural repeats ~0.65)")
        P("=" * 118)
        P(f"  {'dataset / type':26s} {'size (h x w)':>13s} {'keypoints':>10s} {'kp/Mpx':>8s} {'gen: matches':>13s} "
          f"{'gen: >=3':>9s} {'forg: matches':>14s} {'forg: inl_ratio':>16s}")
        prof = {}

        def row(name, h, w, kp, kpm, ni, ir, yy):
            g, f = yy == 0, yy == 1
            d = {"h": float(np.median(h)), "w": float(np.median(w)), "n_kp": float(np.median(kp)), "kp_per_mpx": float(np.median(kpm)),
                 "genuine_matches": float(np.median(ni[g])) if g.any() else None, "genuine_share_ge3": float((ni[g] >= 3).mean()) if g.any() else None,
                 "forged_matches": float(np.median(ni[f])) if f.any() else None,
                 "forged_inl_ratio": float(np.median(ir[f & (ni > 0)])) if (f & (ni > 0)).any() else None}
            prof[name] = d
            P(f"  {name:26s} {d['h']:6.0f} x{d['w']:5.0f} {d['n_kp']:10.0f} {d['kp_per_mpx']:8.0f} {d['genuine_matches']:13.0f} "
              f"{100 * d['genuine_share_ge3']:8.1f}% {d['forged_matches']:14.0f} {d['forged_inl_ratio']:16.3f}")
        col = lambda rows, k: np.array([float(r[k]) for r in rows])
        for t in RECOD_TYPES:
            sel = typ == t
            if sel.sum() and (y[sel] == 1).any():
                rows = [R[i] for i in np.where(sel)[0]]
                row(f"Recod.ai {t}", col(rows, "h"), col(rows, "w"), col(rows, "n_kp"), col(rows, "kp_per_mpx"),
                    col(rows, "n_inl"), col(rows, "inl_ratio_mean"), y[sel])
        for t in BIO_TYPES:
            sel = cat == t
            rows = [Fb[i] for i in np.where(sel)[0]]
            row(f"BioFors {t}", col(rows, "h"), col(rows, "w"), col(rows, "n_kp"), col(rows, "kp_per_mpx"),
                col(rows, "n_inl"), col(rows, "inl_ratio_mean"), yb[sel])
        out["profile"] = prof

        # ---------------------------------------------------------------- E. exploratory per-type cut-off
        P("\n" + "=" * 118)
        P("E. EXPLORATORY: one cut-off per image type vs one global cut-off (learned model, Recod.ai only, target 5% FPR)")
        P("   cut-offs chosen on a random half of the figures, measured on the other half, 20 repeats. Not applied to BioFors.")
        P("=" * 118)
        rng = np.random.default_rng(args.seed + 5)
        uid, gi = np.unique(ids, return_inverse=True)
        acc = {t: {"global": [], "per_type": []} for t in RECOD_TYPES}

        def thr(neg, target=0.05):
            c = np.unique(neg)[::-1]
            best = c[0] + 1e-9
            for v in c:
                if (neg >= v).mean() <= target:
                    best = v
                else:
                    break
            return best
        for _ in range(20):
            half = np.zeros(len(uid), bool)
            half[rng.permutation(len(uid))[: len(uid) // 2]] = True
            for side in (True, False):
                A, Bm = half[gi] == side, half[gi] != side
                tg = thr(score[A & (y == 0)])
                for t in RECOD_TYPES:
                    ta, tb = A & (typ == t), Bm & (typ == t)
                    if (ta & (y == 0)).sum() < 10 or (tb & (y == 0)).sum() == 0 or (tb & (y == 1)).sum() == 0:
                        continue
                    tt = thr(score[ta & (y == 0)])
                    for name, cut in (("global", tg), ("per_type", tt)):
                        fl = (score >= cut) & region
                        acc[t][name].append(((fl[tb & (y == 0)]).mean(), (fl[tb & (y == 1)]).mean()))
        ex = {}
        P(f"  {'type':12s} {'global: FPR':>12s} {'detection':>10s} {'MCC@4.3%':>9s}   {'per-type: FPR':>14s} {'detection':>10s} {'MCC@4.3%':>9s}")
        for t in RECOD_TYPES:
            if not acc[t]["global"]:
                continue
            g_ = np.array(acc[t]["global"]).mean(0); p_ = np.array(acc[t]["per_type"]).mean(0)
            ex[t] = {"global": {"fpr": g_[0], "det": g_[1], "mcc4": exp_mcc(g_[1], g_[0], PREV_BIOFORS)},
                     "per_type": {"fpr": p_[0], "det": p_[1], "mcc4": exp_mcc(p_[1], p_[0], PREV_BIOFORS)}}
            P(f"  {t:12s} {100 * g_[0]:11.1f}% {100 * g_[1]:9.1f}% {ex[t]['global']['mcc4']:9.3f}   "
              f"{100 * p_[0]:13.1f}% {100 * p_[1]:9.1f}% {ex[t]['per_type']['mcc4']:9.3f}")
        out["exploratory_per_type_cutoff"] = ex

        # ---------------------------------------------------------------- save
        ts = stamp()
        pj = os.path.join(args.out, f"types_metrics_{ts}.json")
        json.dump(out, open(pj, "w"), indent=1, default=float)
        shutil.copyfile(pj, os.path.join(args.out, "types_metrics_latest.json"))
        run.output(pj)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            for key, lab, fname in (("auc", "AUC (threshold-free)", "types_auc.png"), ("det", "detection at the Recod.ai 5%-FPR cut-off", "types_detection.png")):
                fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
                for ax, rule in zip(axs, ["count rule", "learned model"]):
                    x = np.arange(len(BIO_TYPES))
                    a = [cross[t]["recod"][rule][key] for t in BIO_TYPES]
                    b = [cross[t]["biofors"][rule][key] for t in BIO_TYPES]
                    ax.bar(x - 0.2, a, 0.38, label="Recod.ai", color="#4C72B0")
                    ax.bar(x + 0.2, b, 0.38, label="BioFors", color="#DD8452")
                    ax.set_xticks(x); ax.set_xticklabels(BIO_TYPES); ax.set_title(rule); ax.set_ylim(0, 1.08)
                    for xi, v in zip(x - 0.2, a): ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
                    for xi, v in zip(x + 0.2, b): ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
                axs[0].set_ylabel(lab)
                h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, loc="upper right", ncol=2, frameon=False)
                fig.suptitle(f"{lab} by image type (Histology counted as Microscopy)")
                fig.tight_layout(); fig.savefig(os.path.join(args.out, fname), dpi=140); plt.close(fig)
                run.output(os.path.join(args.out, fname))
            P("\nfigures: results\\types_auc.png, results\\types_detection.png")
        except ImportError:
            P("\n(matplotlib not installed: no figures. pip install matplotlib, then rerun)")
        a = out["recod"]["ALL"]["learned model"]; b = out["biofors"]["ALL"]["count rule"]
        run.note(recod_model_auc=round(a["auc"], 4), biofors_count_img_mcc=round(b["img_mcc"], 4))
        P(f"\nSaved: {pj}, types_metrics_latest.json")


if __name__ == "__main__":
    main()
