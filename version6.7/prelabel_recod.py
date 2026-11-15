"""
prelabel_recod.py  --  STEP 2 of V6.7. BiomedCLIP suggests an image type for every Recod.ai FIGURE.

One label per figure id: the forged and authentic versions of an id are the same figure, so the AUTHENTIC version
is shown/scored when it exists (no forgery artefacts), otherwise the forged one. Uses the prompt wrapper chosen in
validate_biomedclip.py. Nothing here is final: review_labels.py is where labels become real.

USAGE:  python prelabel_recod.py
OUTPUT (results\\): recod_prelabels.csv (id, file shown, suggestion, confidence, probability per class), report, log
"""

import os
import csv
import json
import argparse

import numpy as np

import config
import utils
import labels_common as L
from runlog import Run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--authentic-dir", default=config.AUTHENTIC_DIR)
    ap.add_argument("--forged-dir", default=config.FORGED_DIR)
    ap.add_argument("--validation", default=os.path.join("results", "biomedclip_validation.json"))
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=256, help="images loaded at a time (memory)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    with Run("prelabel_recod", args, args.out) as run:
        P = run.p
        if not os.path.exists(args.validation):
            raise SystemExit("Run validate_biomedclip.py first (it chooses the prompt wrapper).")
        val = json.load(open(args.validation))
        auth = set(utils.list_image_ids(args.authentic_dir))
        forg = set(utils.list_image_ids(args.forged_dir))
        ids = sorted(auth | forg, key=lambda s: (0, int(s)) if s.isdigit() else (1, s))
        P(f"figures: {len(ids)}  (authentic version available for {len(auth)}, forged-only {len(forg - auth)})")
        P(f"prompt wrapper: {val['chosen_name']}   (BioFors check balanced accuracy {100 * val['check_balanced_accuracy']:.1f}%)")

        enc = L.Encoder(threads=args.threads)
        ce = L.class_embeddings(enc, val["chosen_templates"])
        rows = []
        for i in range(0, len(ids), args.chunk):
            part = ids[i:i + args.chunk]
            files = [os.path.join(args.authentic_dir if x in auth else args.forged_dir, f"{x}.png") for x in part]
            imgs = [L.read_bgr(f) for f in files]
            pr = L.probabilities(enc.encode_images(imgs), ce, enc.logit_scale)
            for x, f, p in zip(part, files, pr):
                k = int(p.argmax())
                rows.append({"image_id": x, "file": os.path.relpath(f), "version_shown": "authentic" if x in auth else "forged",
                             "suggested": L.CLASSES[k], "confidence": round(float(p[k]), 4),
                             **{f"p_{c}": round(float(v), 4) for c, v in zip(L.CLASSES, p)}})
            print(f"  {min(i + args.chunk, len(ids))}/{len(ids)}")

        out = os.path.join(args.out, "recod_prelabels.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        run.output(out)
        P("\nsuggested types (BEFORE review):")
        for c in L.CLASSES:
            sel = [r for r in rows if r["suggested"] == c]
            mc = np.mean([r["confidence"] for r in sel]) if sel else float("nan")
            P(f"  {c:11s} {len(sel):5d}   mean confidence {mc:.2f}")
        run.note(n=len(rows), **{c: sum(r["suggested"] == c for r in rows) for c in L.CLASSES})
        P("Next:  python review_labels.py")


if __name__ == "__main__":
    main()
