"""
run_biofors.py  --  STEP 2 of V6. Run the frozen detector on BioFors and save everything per image.

Two sets of images:
  test   the BioFors internal-duplication (copy-move) test set: the 54 test documents in idd_pdfs.pkl,
         image types Microscopy, Blot/Gel, Macroscopy (the authors' default; FACS has no forgeries).
         2363 images, 102 forged.  Ground truth = boxes in idd_gt.json.
  calib  a fixed random sample of GENUINE images from the 696 training documents (training documents are
         pristine by construction). Used ONLY to re-set the cut-off ("recalibration"); no forgery labels.
         Default sample: Blot/Gel 1000, Microscopy 800, Macroscopy 400 (seed 0, saved to a manifest).

Nothing is decided here: no model, no threshold. The detector runs (mask OFF, same as the chosen model) and
per image we save the features, the detector's raw region mask, and pixel counts against the ground truth,
so score_biofors.py can evaluate ANY cut-off without rerunning detection.

USAGE (PowerShell, from the version6 folder, venv active):
  python run_biofors.py --limit 10            # smoke test: 10 test + 10 calib images; note s/image
  python run_biofors.py --workers 4           # the real run; resumable, Ctrl+C is safe

OUTPUT (results\\):
  biofors_features.csv          one row per image (set, document, panel, type, forged?, pixel counts, features)
  biofors_calib_manifest.json   exactly which calibration images were sampled
  raw_masks\\<doc>\\<panel>       the detector's region mask for test images where it found anything (0/255 png)
  biofors_errors.txt            only if an image failed
  run_biofors_<time>.txt, run_log.jsonl
"""

import os
import sys
import csv
import json
import time
import pickle
import argparse
import multiprocessing as mp

import numpy as np

import config
import utils
import features as F
from runlog import Run

CATEGORIES = ["Microscopy", "Blot/Gel", "Macroscopy"]
META = ["set", "doc", "panel", "category", "gt_forged", "time_s",
        "px_total", "px_gt", "px_pred_raw", "px_tp_raw", "gt_boxes"]
FIELDS = META + F.FEATURE_NAMES


class _ListOnly(pickle.Unpickler):
    """The BioFors .pkl files hold plain lists of strings. Refuse anything else (no code execution)."""
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing to load {module}.{name} from a .pkl")


def load_pkl(path):
    with open(path, "rb") as f:
        obj = _ListOnly(f).load()
    if not isinstance(obj, list) or not all(isinstance(x, str) for x in obj):
        raise ValueError(f"{path}: expected a list of document ids")
    return obj


def parse_calib(spec):
    out = {}
    for item in spec:
        cat, n = item.rsplit(":", 1)
        out[cat] = int(n)
    return out


def build_jobs(args):
    ann = args.annotations
    cls = json.load(open(os.path.join(ann, "classification.json")))
    gt = json.load(open(os.path.join(ann, "idd_gt.json")))
    idd_docs = load_pkl(os.path.join(ann, "idd_pdfs.pkl"))
    train_docs = load_pkl(os.path.join(ann, "train_pdfs.pkl"))
    if set(idd_docs) & set(train_docs):
        sys.exit("BioFors lists overlap between IDD test documents and training documents - stop.")

    test = []
    for d in idd_docs:
        for panel, c in sorted(cls[d].items()):
            if c in args.categories:
                boxes = gt.get(d, {}).get(panel)
                test.append(("test", d, panel, c, json.dumps(boxes) if boxes else ""))

    want = parse_calib(args.calib)
    pool = {c: [] for c in want}
    for d in sorted(train_docs, key=int):
        for panel, c in sorted(cls[d].items()):
            if c in pool:
                pool[c].append((d, panel))
    rng = np.random.default_rng(args.calib_seed)
    calib, manifest = [], {"seed": args.calib_seed, "requested": want, "available": {}, "sampled": {}}
    for c, items in pool.items():
        n = min(want[c], len(items))
        pick = sorted(rng.choice(len(items), size=n, replace=False).tolist())
        manifest["available"][c] = len(items)
        manifest["sampled"][c] = [items[i][0] + "/" + items[i][1] for i in pick]
        calib += [("calib", items[i][0], items[i][1], c, "") for i in pick]
    return test, calib, manifest


def _init_worker():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import cv2
    cv2.setNumThreads(1)


def _work(job):
    """job = (set, doc, panel, category, gt_boxes_json, image_root, raw_mask_root)"""
    import cv2
    from pipeline import detect
    st, doc, panel, cat, boxes_json, img_root, mask_root = job
    config.EXCLUDE_ENABLED = False
    path = os.path.join(img_root, doc, panel)
    t0 = time.time()
    try:
        gray = utils.load_image(path)
        color = utils.load_color(path)
        res = detect(gray, color)
        pred = (res["mask"] > 0).astype(np.uint8)
        gtm = np.zeros(gray.shape, np.uint8)
        if boxes_json:
            for x1, y1, x2, y2 in json.loads(boxes_json):     # exactly as the authors' eval_IDD.py
                gtm[y1:y2, x1:x2] = 1
        row = {"set": st, "doc": doc, "panel": panel, "category": cat, "gt_forged": int(bool(boxes_json)),
               "px_total": int(gray.size), "px_gt": int(gtm.sum()), "px_pred_raw": int(pred.sum()),
               "px_tp_raw": int((pred & gtm).sum()), "gt_boxes": boxes_json}
        row.update({k: round(float(v), 6) for k, v in F.compute(gray, color, res).items()})
        if st == "test" and pred.any():
            out = os.path.join(mask_root, doc)
            os.makedirs(out, exist_ok=True)
            cv2.imwrite(os.path.join(out, panel), pred * 255)
        row["time_s"] = round(time.time() - t0, 3)
        return job[:3], row, None
    except Exception as e:
        return job[:3], None, f"{type(e).__name__}: {e}"


def _repair(path):
    with open(path, newline="") as f:
        lines = f.read().split("\n")
    keep, dropped = [], 0
    for i, line in enumerate(lines):
        if line == "" and i == len(lines) - 1:
            continue
        ok = i < len(lines) - 1 and len(next(csv.reader([line.rstrip("\r")]))) == len(FIELDS)
        if ok:
            keep.append(line.rstrip("\r"))
        else:
            dropped += 1
    if dropped:
        with open(path, "w", newline="") as f:
            f.write("\n".join(keep) + "\n")
        print(f"Repaired {path}: dropped {dropped} incomplete line(s) from an interrupted run.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=os.path.join("..", "biofors_images"))
    ap.add_argument("--annotations", default=os.path.join("..", "biofors_annotations"))
    ap.add_argument("--categories", nargs="+", default=CATEGORIES)
    ap.add_argument("--calib", nargs="+", default=["Blot/Gel:1000", "Microscopy:800", "Macroscopy:400"])
    ap.add_argument("--calib-seed", type=int, default=0)
    ap.add_argument("--only", choices=["test", "calib", "both"], default="both")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="only the first N images of each set (smoke test)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    feats = os.path.join(args.out, "biofors_features.csv")
    mask_root = os.path.join(args.out, "raw_masks")

    with Run("run_biofors", args, args.out) as run:
        test, calib, manifest = build_jobs(args)
        with open(os.path.join(args.out, "biofors_calib_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1)
        run.output(os.path.join(args.out, "biofors_calib_manifest.json"))
        n_forged = sum(1 for j in test if j[4])
        run.p(f"test set : {len(test)} images, {n_forged} forged   types: "
              + ", ".join(f"{c} {sum(1 for j in test if j[3] == c)}" for c in args.categories))
        run.p(f"calib set: {len(calib)} genuine images from training documents   "
              + ", ".join(f"{c} {len(v)} of {manifest['available'][c]}" for c, v in manifest["sampled"].items()))

        sets = {"test": test, "calib": calib}
        chosen = [s for s in ("test", "calib") if args.only in (s, "both")]
        jobs = []
        for s in chosen:
            jobs += sets[s][: args.limit] if args.limit else sets[s]

        missing = [j for j in jobs if not os.path.exists(os.path.join(args.images, j[1], j[2]))]
        if missing:
            run.p(f"WARNING: {len(missing)} images not found under {args.images}, e.g. {missing[0][1]}/{missing[0][2]}")
            jobs = [j for j in jobs if j not in missing]

        done = set()
        new_file = not os.path.exists(feats)
        if not new_file:
            _repair(feats)
            with open(feats, newline="") as f:
                rd = csv.DictReader(f)
                if rd.fieldnames != FIELDS:
                    sys.exit(f"{feats} has different columns. Move it away and rerun.")
                done = {(r["set"], r["doc"], r["panel"]) for r in rd}
        jobs = [j + (args.images, mask_root) for j in jobs if j[:3] not in done]
        run.p(f"\njobs to run: {len(jobs)}   (already done: {len(done)})   workers: {args.workers}\n")

        t0, n_ok, n_err = time.time(), 0, 0
        pool = None
        if args.workers > 1 and jobs:
            pool = mp.get_context("spawn").Pool(args.workers, initializer=_init_worker)
            stream = pool.imap(_work, jobs, chunksize=2)
        else:
            stream = (_work(j) for j in jobs)
        times = []
        try:
            with open(feats, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if new_file:
                    w.writeheader()
                for k, (key, row, err) in enumerate(stream, 1):
                    if err:
                        n_err += 1
                        with open(os.path.join(args.out, "biofors_errors.txt"), "a") as ef:
                            ef.write(f"{key}\t{err}\n")
                        print(f"  ! {key}: {err}")
                    else:
                        w.writerow(row)
                        f.flush()
                        n_ok += 1
                        times.append(row["time_s"])
                    if k % 50 == 0 or k == len(jobs):
                        el = time.time() - t0
                        print(f"  [{k}/{len(jobs)}] {el / 60:.1f} min elapsed, ~{el / k * (len(jobs) - k) / 60:.0f} min left")
        finally:
            if pool is not None:
                pool.terminate()

        run.output(feats)
        run.output(mask_root)
        run.note(rows_written=n_ok, errors=n_err, mean_s_per_image=round(float(np.mean(times)), 3) if times else None)
        run.p(f"\nrows written this run: {n_ok}   errors: {n_err}"
              + (f"   mean detector time {np.mean(times):.2f} s/image (per worker)" if times else ""))
        run.p("Next:  python score_biofors.py")


if __name__ == "__main__":
    main()
