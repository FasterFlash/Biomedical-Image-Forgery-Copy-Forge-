"""
label_report.py  --  STEP 4 of V6.7. Summarises the labelling and freezes the final labels for version7.

Reports: BiomedCLIP's accuracy on BioFors (from validate_biomedclip.py), how often you accepted / corrected its
Recod.ai suggestions (overall and per suggested type), the suggested-vs-final confusion, and the final number of
figures per type (and how many of them have a forged / authentic version).

USAGE:  python label_report.py
OUTPUT (results\\): recod_types.csv  (image_id, type)  <- the file version7 uses;  label_report_<time>.txt
"""

import os
import csv
import json
import argparse

import config
import utils
import labels_common as L
from runlog import Run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=os.path.join("results", "recod_labels.csv"))
    ap.add_argument("--prelabels", default=os.path.join("results", "recod_prelabels.csv"))
    ap.add_argument("--validation", default=os.path.join("results", "biomedclip_validation.json"))
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--authentic-dir", default=config.AUTHENTIC_DIR)
    ap.add_argument("--forged-dir", default=config.FORGED_DIR)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    with Run("label_report", args, args.out) as run:
        P = run.p
        pre = list(csv.DictReader(open(args.prelabels, newline="")))
        lab = {r["image_id"]: r for r in csv.DictReader(open(args.labels, newline=""))} if os.path.exists(args.labels) else {}
        P(f"figures: {len(pre)}   reviewed: {len(lab)}")
        if len(lab) < len(pre) and not args.allow_incomplete:
            raise SystemExit(f"{len(pre) - len(lab)} figures are not reviewed yet. Finish review_labels.py "
                             "(or pass --allow-incomplete for a progress report; no recod_types.csv is written then).")
        if os.path.exists(args.validation):
            v = json.load(open(args.validation))
            P(f"\nBiomedCLIP on BioFors (known types, held-out check split of {v['n_check']} images): "
              f"balanced accuracy {100 * v['check_balanced_accuracy']:.1f}%, per type "
              + ", ".join(f"{c} {100 * a:.0f}%" for c, a in v["per_class"].items()))

        rev = list(lab.values())
        acc = sum(r["decision"] == "accepted" for r in rev)
        P(f"\nreview on Recod.ai: {acc}/{len(rev)} suggestions accepted ({100 * acc / max(1, len(rev)):.1f}%), "
          f"{len(rev) - acc} corrected")
        P("\nsuggested (rows) vs final (columns):")
        P(f"{'':12s}" + "".join(f"{c[:10]:>11s}" for c in L.CLASSES) + f"{'accepted':>10s}")
        for c in L.CLASSES:
            row = [sum(1 for r in rev if r["suggested"] == c and r["label"] == d) for d in L.CLASSES]
            tot = sum(row)
            P(f"{c:12s}" + "".join(f"{x:11d}" for x in row) + (f"{100 * row[L.CLASSES.index(c)] / tot:9.0f}%" if tot else f"{'-':>10s}"))

        auth = set(utils.list_image_ids(args.authentic_dir))
        forg = set(utils.list_image_ids(args.forged_dir))
        P("\nfinal labels (figures; and images in each class of the detection data):")
        P(f"{'type':12s}{'figures':>9s}{'forged imgs':>13s}{'authentic imgs':>16s}")
        for c in L.CLASSES:
            ids = [r["image_id"] for r in rev if r["label"] == c]
            P(f"{c:12s}{len(ids):9d}{sum(i in forg for i in ids):13d}{sum(i in auth for i in ids):16d}")

        P("\nfor BioFors comparisons (Histology merged into Microscopy, as BioFors does): "
          + ", ".join(f"{c} {sum(1 for r in rev if L.TO_BIOFORS.get(r['label'], r['label']) == c)}" for c in L.BIOFORS_CLASSES + ["Other"]))

        if len(lab) == len(pre):
            out = os.path.join(args.out, "recod_types.csv")
            with open(out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["image_id", "type", "type_biofors"])
                for r in pre:
                    t = lab[r["image_id"]]["label"]
                    w.writerow([r["image_id"], t, L.TO_BIOFORS.get(t, t)])
            run.output(out)
            P(f"\nSaved {out}  <- version7 uses this file")
        run.note(reviewed=len(lab), accepted=acc)


if __name__ == "__main__":
    main()
