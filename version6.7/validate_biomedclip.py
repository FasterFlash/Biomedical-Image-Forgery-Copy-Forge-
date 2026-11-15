"""
validate_biomedclip.py  --  STEP 1 of V6.7. How accurate is BiomedCLIP at telling image types apart, measured on
BioFors images whose type is KNOWN (the authors' classification.json)? Decides whether its suggestions are worth
using as a starting point for the Recod.ai labels.

Fair measurement:
  - only BioFors TRAINING documents (test documents are never read)
  - per type up to 500 images (Blot/Gel, Microscopy, Macroscopy, FACS), fixed seed
  - images are split BY DOCUMENT into a small TUNE part (~20%) and a larger CHECK part (~80%)
  - the prompt wrapper (template) is chosen on TUNE only; the accuracy you report comes from CHECK only

Histology is not a BioFors class (BioFors' Microscopy contains stained tissue), so a Histology guess is scored as
Microscopy here and the Histology/Microscopy split itself is NOT validated by this script. The "Other" class exists only for Recod.ai (charts, text ...). BioFors has none, so an image predicted "Other"
counts as an error here.

USAGE:  python validate_biomedclip.py
OUTPUT (results\\): biomedclip_validation.json (chosen template, accuracies, confusion), validate_biomedclip_<time>.txt
"""

import os
import json
import pickle
import argparse

import numpy as np

import labels_common as L
from runlog import Run


class _ListOnly(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing to load {module}.{name} from a .pkl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=os.path.join("..", "biofors_images"))
    ap.add_argument("--annotations", default=os.path.join("..", "biofors_annotations"))
    ap.add_argument("--per-type", type=int, default=500)
    ap.add_argument("--tune-share", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--go-threshold", type=float, default=0.85, help="balanced accuracy needed to use the pre-labels")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    with Run("validate_biomedclip", args, args.out) as run:
        P = run.p
        cls = json.load(open(os.path.join(args.annotations, "classification.json")))
        train = _ListOnly(open(os.path.join(args.annotations, "train_pdfs.pkl"), "rb")).load()
        test = _ListOnly(open(os.path.join(args.annotations, "test_pdfs.pkl"), "rb")).load()
        if set(train) & set(test):
            raise SystemExit("train/test document lists overlap - stop.")
        rng = np.random.default_rng(args.seed)
        bio = L.BIOFORS_CLASSES
        items = []
        for c in bio:
            pool = [(d, p) for d in sorted(train, key=int) for p, cc in sorted(cls[d].items())
                    if cc == c and os.path.exists(os.path.join(args.images, d, p))]
            pick = rng.choice(len(pool), size=min(args.per_type, len(pool)), replace=False)
            items += [(pool[i][0], pool[i][1], c) for i in sorted(pick)]
            P(f"{c:11s} sampled {min(args.per_type, len(pool))} of {len(pool)} available training images")
        docs = sorted({d for d, _, _ in items}, key=int)
        tune_docs = set(np.array(docs)[rng.permutation(len(docs))[: int(round(args.tune_share * len(docs)))]].tolist())
        is_tune = np.array([d in tune_docs for d, _, _ in items])
        y = np.array([L.CLASSES.index(c) for _, _, c in items])
        P(f"tune images {is_tune.sum()}  /  check images {(~is_tune).sum()}   (split by document)")

        enc = L.Encoder(threads=args.threads)
        imgs = [L.read_bgr(os.path.join(args.images, d, p)) for d, p, _ in items]
        emb = enc.encode_images(imgs)

        to_bio = np.array([L.CLASSES.index(L.TO_BIOFORS.get(c, c)) for c in L.CLASSES])

        def evaluate(mask, cls_emb):
            pr = L.probabilities(emb[mask], cls_emb, enc.logit_scale)
            raw = pr.argmax(1)
            pred = to_bio[raw]                     # Histology counts as Microscopy, like BioFors
            yt = y[mask]
            per = {c: float((pred[yt == L.CLASSES.index(c)] == L.CLASSES.index(c)).mean())
                   for c in bio if (yt == L.CLASSES.index(c)).any()}
            return float(np.mean(list(per.values()))), float((pred == yt).mean()), per, raw, pr

        configs = [("template " + repr(L.TEMPLATES[t]), [t]) for t in range(len(L.TEMPLATES))]
        configs.append(("all templates averaged", list(range(len(L.TEMPLATES)))))
        P("\nTUNE split (choosing the prompt wrapper):")
        scored = []
        for name, tids in configs:
            ce = L.class_embeddings(enc, tids)
            bal, acc, _, _, _ = evaluate(is_tune, ce)
            scored.append((bal, name, tids))
            P(f"  {name:45s} balanced accuracy {100 * bal:5.1f}%   accuracy {100 * acc:5.1f}%")
        bal, name, tids = max(scored)
        P(f"chosen: {name}")

        ce = L.class_embeddings(enc, tids)
        bal, acc, per, raw, pr = evaluate(~is_tune, ce)
        pred = to_bio[raw]
        yt = y[~is_tune]
        P("\n" + "=" * 90)
        P(f"CHECK split (never used for choices): balanced accuracy {100 * bal:.1f}%   plain accuracy {100 * acc:.1f}%")
        P("=" * 90)
        for c, v in per.items():
            P(f"  {c:11s} {100 * v:5.1f}% correct")
        P("(Histology guesses count as Microscopy here: BioFors' Microscopy includes stained tissue)")
        P("\nconfusion (rows = true BioFors type, columns = BiomedCLIP's raw guess):")
        P(f"{'':12s}" + "".join(f"{c[:10]:>11s}" for c in L.CLASSES))
        conf = np.zeros((len(bio), len(L.CLASSES)), int)
        for t, p_ in zip(yt, raw):
            conf[bio.index(L.CLASSES[t]), p_] += 1
        for k, c in enumerate(bio):
            P(f"{c:12s}" + "".join(f"{v:11d}" for v in conf[k]))
        conf_max = pr.max(1)
        P("\nhow trustworthy is the confidence?  (share of CHECK images above a confidence, and their accuracy)")
        cov = []
        for t in (0.5, 0.7, 0.8, 0.9, 0.95):
            m = conf_max >= t
            a = float((pred[m] == yt[m]).mean()) if m.any() else float("nan")
            cov.append({"min_confidence": t, "share": float(m.mean()), "accuracy": a})
            P(f"  confidence >= {t:.2f}: {100 * m.mean():5.1f}% of images, {100 * a:5.1f}% of those correct")

        go = bal >= args.go_threshold
        P(f"\nVERDICT: {'USE as pre-labeller' if go else 'WEAK - review will mostly be manual'} "
          f"(balanced accuracy {100 * bal:.1f}% vs threshold {100 * args.go_threshold:.0f}%). "
          "Either way every Recod.ai label is reviewed by a person.")
        P("NOTE: the Histology-vs-Microscopy split cannot be validated on BioFors (no such labels there); "
          "it is checked only by the manual review, whose correction rate per class label_report.py prints.")
        out = {"chosen_templates": tids, "chosen_name": name, "tune_scores": [(n, b) for b, n, _ in scored],
               "check_balanced_accuracy": bal, "check_accuracy": acc, "per_class": per,
               "confusion_rows_true": bio, "confusion_cols_pred": L.CLASSES, "confusion": conf.tolist(),
               "confidence_coverage": cov, "n_tune": int(is_tune.sum()), "n_check": int((~is_tune).sum()),
               "verdict_use": bool(go), "model": L.MODEL_ID, "fake_encoder": enc.fake}
        p = os.path.join(args.out, "biomedclip_validation.json")
        json.dump(out, open(p, "w"), indent=1)
        run.output(p)
        run.note(balanced_accuracy=round(bal, 4), chosen=name, verdict_use=bool(go))
        P("Next:  python prelabel_recod.py")


if __name__ == "__main__":
    main()
