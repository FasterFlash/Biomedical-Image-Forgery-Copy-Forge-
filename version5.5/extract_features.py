"""
extract_features.py  --  Run the V5 detector on every image and save ONE ROW of features
per (image, mask mode).  Nothing is classified here; this is the data dump that cv_decision.py
learns from.

USAGE  (PowerShell, from the version5.5 folder, venv active):

  python extract_features.py --limit 20            # try it on 20 image ids first; look at s/image
  python extract_features.py --workers 4           # the real run, 4 processes
  python extract_features.py --mask both           # default: mask ON and OFF for every image
  python extract_features.py --mask on             # only one mask mode (half the time)
  python extract_features.py --only authentic      # only one class (cv_decision needs BOTH)

OUTPUT   results\\features.csv   one row per (label, image_id, mask)
         results\\features_errors.txt   only if some image failed

RESUMABLE: Ctrl+C any time and run the same command again; finished rows are skipped.
Rows are written one at a time and flushed, so nothing is lost.
Results do not depend on --workers or on processing order (every image is seeded in detect()).

TIME: roughly (seconds per image from the --limit run) x 4000 images x number of mask modes
/ number of workers that actually run in parallel.  Use --mask on first if you are short of time.
"""

import os
import sys
import csv
import time
import argparse
import multiprocessing as mp

import config
import utils
import features as F

META = ["label", "image_id", "mask", "time_s", "mcc", "f1", "precision", "recall"]
FIELDS = META + F.FEATURE_NAMES


def _init_worker():
    os.environ.setdefault("OMP_NUM_THREADS", "1")      # avoid every worker grabbing every core
    import cv2
    cv2.setNumThreads(1)


def _work(job):
    """job = (label, image_id, mask_mode, folder, mask_dir)  ->  (job, row_or_None, error_or_None)"""
    from pipeline import detect                         # imported here so spawned workers set up first
    label, image_id, mask, folder, mask_dir = job
    config.EXCLUDE_ENABLED = (mask == "on")
    path = os.path.join(folder, f"{image_id}.png")
    t0 = time.time()
    try:
        gray = utils.load_image(path)
        color = utils.load_color(path)
        res = detect(gray, color)
        row = {"label": label, "image_id": image_id, "mask": mask,
               "mcc": "", "f1": "", "precision": "", "recall": ""}
        row.update({k: round(float(v), 6) for k, v in F.compute(gray, color, res).items()})
        if label == "forged":
            truth = utils.load_mask(os.path.join(mask_dir, f"{image_id}.npy"), gray.shape)
            sc = utils.score_masks(res["mask"], truth)
            if sc:
                row.update(mcc=sc["MCC"], f1=sc["F1"], precision=sc["precision"], recall=sc["recall"])
        row["time_s"] = round(time.time() - t0, 2)
        return job, row, None
    except Exception as e:                              # one bad image must not kill a 3-hour run
        return job, None, f"{type(e).__name__}: {e}"


def _repair(path):
    """After a crash / power loss the last line can be cut in half. Keep only complete rows
    (right number of fields, newline-terminated) and rewrite the file if anything was dropped."""
    with open(path, newline="") as f:
        raw = f.read()
    lines = raw.split("\n")
    keep, dropped = [], 0
    for i, line in enumerate(lines):
        if line == "" and i == len(lines) - 1:
            continue                                   # normal trailing newline
        complete = (i < len(lines) - 1) and len(next(csv.reader([line.rstrip("\r")]))) == len(FIELDS)
        if complete:
            keep.append(line.rstrip("\r"))
        else:
            dropped += 1
    if dropped:
        with open(path, "w", newline="") as f:
            f.write("\n".join(keep) + "\n")
        print(f"Repaired {path}: dropped {dropped} incomplete line(s) left by an interrupted run.")


def _natural(s):
    return (0, int(s), s) if s.isdigit() else (1, 0, s)


def main():
    ap = argparse.ArgumentParser(description="Extract per-image features for the decision layer.")
    ap.add_argument("--mask", choices=["on", "off", "both"], default="both")
    ap.add_argument("--only", choices=["authentic", "forged", "both"], default="both")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes (default 1)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N image ids (smoke test)")
    ap.add_argument("--out", default=os.path.join(config.OUTPUT_DIR, "features.csv"))
    ap.add_argument("--forged-dir", default=config.FORGED_DIR)
    ap.add_argument("--authentic-dir", default=config.AUTHENTIC_DIR)
    ap.add_argument("--mask-dir", default=config.MASK_DIR, help="ground-truth .npy masks (forged only)")
    args = ap.parse_args()

    config.ensure_output_dir()
    modes = ["on", "off"] if args.mask == "both" else [args.mask]
    labels = ["authentic", "forged"] if args.only == "both" else [args.only]
    ids_by_label = {}
    for lab in labels:
        folder = args.forged_dir if lab == "forged" else args.authentic_dir
        ids_by_label[lab] = set(utils.list_image_ids(folder))
    all_ids = sorted(set().union(*ids_by_label.values()), key=_natural)
    if args.limit:
        all_ids = all_ids[:args.limit]

    # resume
    done = set()
    new_file = not os.path.exists(args.out)
    if not new_file:
        _repair(args.out)
        with open(args.out, newline="") as f:
            rd = csv.DictReader(f)
            if rd.fieldnames != FIELDS:
                sys.exit(f"{args.out} was written with different columns. Move or delete it, then rerun.")
            done = {(r["label"], r["image_id"], r["mask"]) for r in rd}

    # ids interleaved (both classes, both masks per id) so a partial run is still balanced and paired
    folder_of = {"forged": args.forged_dir, "authentic": args.authentic_dir}
    jobs = [(lab, i, m, folder_of[lab], args.mask_dir) for i in all_ids for lab in labels
            if i in ids_by_label[lab] for m in modes if (lab, i, m) not in done]

    print(f"Jobs to run: {len(jobs)}   (already done: {len(done)})")
    print(f"Mask modes: {modes}   classes: {labels}   workers: {args.workers}")
    print(f"Writing to: {args.out}\n")
    if not jobs:
        print("Nothing to do.")
        return

    err_path = os.path.join(config.OUTPUT_DIR, "features_errors.txt")
    t_start = time.time()
    n_ok = n_err = 0

    pool = None
    if args.workers > 1:
        pool = mp.get_context("spawn").Pool(args.workers, initializer=_init_worker)
        stream = pool.imap(_work, jobs, chunksize=1)
    else:
        stream = (_work(j) for j in jobs)

    try:
        with open(args.out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                w.writeheader()
            for k, (job, row, err) in enumerate(stream, 1):
                if err:
                    n_err += 1
                    with open(err_path, "a") as ef:
                        ef.write(f"{job[:3]}\t{err}\n")
                    print(f"  ! {job[:3]}: {err}")
                else:
                    w.writerow(row)
                    f.flush()
                    n_ok += 1
                if k % 25 == 0 or k == len(jobs):
                    el = time.time() - t_start
                    eta = el / k * (len(jobs) - k)
                    print(f"  [{k}/{len(jobs)}] {el/60:.1f} min elapsed, ~{eta/60:.0f} min left")
    finally:
        if pool is not None:
            pool.terminate()

    print(f"\nDone. {n_ok} rows written, {n_err} errors.  File: {args.out}")
    if n_err:
        print(f"Errors listed in {err_path}")
    print("Next:  python cv_decision.py")


if __name__ == "__main__":
    main()
