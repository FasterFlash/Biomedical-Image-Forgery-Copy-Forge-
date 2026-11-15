"""
make_synthetic.py  --  STEP 1 of V6.6. Make LABELLED BioFors-like copy-move forgeries from GENUINE BioFors images.

Why: the BioFors test set has been looked at once (version6). To choose features, model and cut-off for
BioFors-like images we need labelled BioFors-like data that is NOT the test set. We make it ourselves.

Source images: genuine images from BioFors TRAINING documents (pristine by construction), image types
Blot/Gel, Microscopy, Macroscopy. Images used for recalibration in version6 are EXCLUDED, test documents are
never touched.

Every synthetic item is a PAIR, like Recod.ai:
  forged   a rectangular patch copied to another, non-overlapping place in the same image
  genuine  the same original image, given the SAME whole-image processing (JPEG / resampling) as the forged one,
           so "was this image recompressed" can never give the answer away

Copy variants (spread evenly):
  clean        pasted exactly
  rotate       patch rotated 5-45 degrees
  scale        patch resized 0.8-1.2x
  flip         patch mirrored
  blur         patch softened (Gaussian, sigma 0.7-1.5)
  jpeg         whole image JPEG-compressed afterwards (quality 60-85)
  publication  whole image shrunk to 50-80%, enlarged back, JPEG 70-90: mimics what happens to a figure on its way
               into a published PDF (the likely reason real BioFors forgeries are less "exact" than Recod.ai's)
Ground truth = source rectangle + pasted region (both copies), like BioFors marks duplicated regions.
The copied patch must contain content (grey-level spread >= max(8, half the image's)): copying blank background is
not a meaningful forgery. Limitation: patches go to random places, so many forgeries are cruder than real ones
(visible seams). Treat results on this set as a development signal, not as BioFors performance.

USAGE (PowerShell, from version6.6, venv active):
  python make_synthetic.py                         # default: Blot/Gel 700, Microscopy 600, Macroscopy 150

OUTPUT (results\\):
  synth\\forged\\<id>.png, synth\\genuine\\<id>.png, synth\\masks\\<id>.png (0/255)
  synth_manifest.csv      one row per pair: id, document, panel, type, variant, all random parameters, boxes
  make_synthetic_<time>.txt, run_log.jsonl
"""

import os
import sys
import csv
import json
import pickle
import argparse

import cv2
import numpy as np

from runlog import Run

VARIANTS = ["clean", "rotate", "scale", "flip", "blur", "jpeg", "publication"]
CATEGORIES = ["Blot/Gel", "Microscopy", "Macroscopy"]
MIN_SIDE = 40          # px; smaller panels cannot hold two non-overlapping patches of a useful size
MIN_PATCH = 12         # px


class _ListOnly(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing to load {module}.{name} from a .pkl")


def load_pkl(path):
    with open(path, "rb") as f:
        obj = _ListOnly(f).load()
    if not isinstance(obj, list) or not all(isinstance(x, str) for x in obj):
        raise ValueError(f"{path}: expected a list of document ids")
    return obj


def _jpeg(img, q):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _global(img, variant, p):
    """Whole-image processing applied identically to the forged and the genuine image."""
    if variant == "jpeg":
        return _jpeg(img, p["jpeg_q"])
    if variant == "publication":
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(1, round(w * p["resample"])), max(1, round(h * p["resample"]))),
                           interpolation=cv2.INTER_AREA)
        return _jpeg(cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR), p["jpeg_q"])
    return img


def make_pair(img, variant, rng):
    """Returns (forged, genuine, gt_mask, params) or None if this image cannot host a copy-move."""
    h, w = img.shape[:2]
    if min(h, w) < MIN_SIDE:
        return None
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    need = max(8.0, 0.5 * float(g.std()))      # the copied patch must contain content, not blank background
    for _ in range(120):
        ph = max(MIN_PATCH, int(rng.uniform(0.15, 0.35) * h))
        pw = max(MIN_PATCH, int(rng.uniform(0.15, 0.35) * w))
        sy, sx = int(rng.integers(0, h - ph + 1)), int(rng.integers(0, w - pw + 1))
        if float(g[sy:sy + ph, sx:sx + pw].std()) < need:
            continue
        patch = img[sy:sy + ph, sx:sx + pw].copy()
        pm = np.ones((ph, pw), np.uint8)
        p = {"variant": variant, "src": [sx, sy, sx + pw, sy + ph]}

        if variant == "rotate":
            ang = float(rng.uniform(5, 45) * rng.choice([-1, 1]))
            c, s = abs(np.cos(np.radians(ang))), abs(np.sin(np.radians(ang)))
            nw, nh = int(np.ceil(pw * c + ph * s)), int(np.ceil(pw * s + ph * c))
            M = cv2.getRotationMatrix2D((pw / 2, ph / 2), ang, 1.0)
            M[0, 2] += (nw - pw) / 2
            M[1, 2] += (nh - ph) / 2
            patch = cv2.warpAffine(patch, M, (nw, nh), flags=cv2.INTER_LINEAR)
            pm = cv2.warpAffine(pm, M, (nw, nh), flags=cv2.INTER_NEAREST)
            p["angle"] = round(ang, 2)
        elif variant == "scale":
            f = float(rng.uniform(0.8, 1.2))
            if abs(f - 1) < 0.05:
                f = 1.1
            nw, nh = max(MIN_PATCH, round(pw * f)), max(MIN_PATCH, round(ph * f))
            patch = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_LINEAR)
            pm = np.ones((nh, nw), np.uint8)
            p["scale"] = round(f, 3)
        elif variant == "flip":
            patch = patch[:, ::-1].copy()
        elif variant == "blur":
            sig = float(rng.uniform(0.7, 1.5))
            patch = cv2.GaussianBlur(patch, (0, 0), sig)
            p["sigma"] = round(sig, 2)
        elif variant == "jpeg":
            p["jpeg_q"] = int(rng.integers(60, 86))
        elif variant == "publication":
            p["resample"] = round(float(rng.uniform(0.5, 0.8)), 3)
            p["jpeg_q"] = int(rng.integers(70, 91))

        th, tw = patch.shape[:2]
        if th > h or tw > w:
            continue
        dy, dx = int(rng.integers(0, h - th + 1)), int(rng.integers(0, w - tw + 1))
        # source and target rectangles must not overlap (with a 2 px gap)
        if not (dx >= sx + pw + 2 or dx + tw + 2 <= sx or dy >= sy + ph + 2 or dy + th + 2 <= sy):
            continue
        forged = img.copy()
        region = forged[dy:dy + th, dx:dx + tw]
        region[pm > 0] = patch[pm > 0]
        gt = np.zeros((h, w), np.uint8)
        gt[sy:sy + ph, sx:sx + pw] = 1
        gt[dy:dy + th, dx:dx + tw][pm > 0] = 1
        p["dst"] = [dx, dy, dx + tw, dy + th]
        return _global(forged, variant, p), _global(img.copy(), variant, p), gt, p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=os.path.join("..", "biofors_images"))
    ap.add_argument("--annotations", default=os.path.join("..", "biofors_annotations"))
    ap.add_argument("--exclude-manifest", default=os.path.join("..", "version6", "results", "biofors_calib_manifest.json"),
                    help="version6 calibration images; never reused here ('none' to skip)")
    ap.add_argument("--counts", nargs="+", default=["Blot/Gel:700", "Microscopy:600", "Macroscopy:150"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out = os.path.join(args.out, "synth")
    for sub in ("forged", "genuine", "masks"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    with Run("make_synthetic", args, args.out) as run:
        cls = json.load(open(os.path.join(args.annotations, "classification.json")))
        train = load_pkl(os.path.join(args.annotations, "train_pdfs.pkl"))
        for name in ("idd_pdfs.pkl", "test_pdfs.pkl"):
            pth = os.path.join(args.annotations, name)
            if os.path.exists(pth) and set(load_pkl(pth)) & set(train):
                sys.exit(f"{name} overlaps the training documents - stop.")
        excluded = set()
        if args.exclude_manifest != "none":
            if not os.path.exists(args.exclude_manifest):
                sys.exit(f"{args.exclude_manifest} not found. Run version6 first, or pass --exclude-manifest none.")
            man = json.load(open(args.exclude_manifest))
            excluded = {x for v in man["sampled"].values() for x in v}
        want = {c: int(n) for c, n in (s.rsplit(":", 1) for s in args.counts)}
        run.p(f"excluded (version6 calibration images): {len(excluded)}")

        rng = np.random.default_rng(args.seed)
        rows, made = [], {c: 0 for c in want}
        skipped = {"missing": 0, "too_small_or_no_room": 0}
        for c in want:
            pool = [(d, p) for d in sorted(train, key=int) for p, cc in sorted(cls[d].items())
                    if cc == c and f"{d}/{p}" not in excluded]
            order = rng.permutation(len(pool))
            run.p(f"{c}: {len(pool)} genuine training images available, want {want[c]}")
            for k in order:
                if made[c] >= want[c]:
                    break
                d, panel = pool[k]
                path = os.path.join(args.images, d, panel)
                img = cv2.imread(path, cv2.IMREAD_COLOR) if os.path.exists(path) else None
                if img is None:
                    skipped["missing"] += 1
                    continue
                variant = VARIANTS[made[c] % len(VARIANTS)]
                res = make_pair(img, variant, rng)
                if res is None:
                    skipped["too_small_or_no_room"] += 1
                    continue
                forged, genuine, gt, p = res
                sid = f"s{len(rows):05d}"
                cv2.imwrite(os.path.join(out, "forged", sid + ".png"), forged)
                cv2.imwrite(os.path.join(out, "genuine", sid + ".png"), genuine)
                cv2.imwrite(os.path.join(out, "masks", sid + ".png"), gt * 255)
                rows.append({"sid": sid, "doc": d, "panel": panel, "category": c, "variant": variant,
                             "h": img.shape[0], "w": img.shape[1], "gt_px": int(gt.sum()), "params": json.dumps(p)})
                made[c] += 1
            if made[c] < want[c]:
                run.p(f"  WARNING: only {made[c]} {c} pairs could be made")

        man_path = os.path.join(args.out, "synth_manifest.csv")
        with open(man_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["sid"])
            w.writeheader()
            w.writerows(rows)
        run.output(man_path)
        run.output(out)
        run.p(f"\npairs made: {len(rows)}   by type {made}   skipped {skipped}")
        for v in VARIANTS:
            run.p(f"  {v:12s} {sum(1 for r in rows if r['variant'] == v)}")
        run.note(pairs=len(rows), by_type=made, skipped=skipped)
        run.p("Next:  python run_synth.py --workers 4")


if __name__ == "__main__":
    main()
