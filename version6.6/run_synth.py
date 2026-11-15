"""
run_synth.py  --  STEP 2 of V6.6. Run the unchanged detector (mask OFF) on every synthetic pair.

Per image it saves the features and pixel counts against the synthetic mask, exactly like version6's
run_biofors.py, so develop.py can try any model / cut-off without rerunning detection.

USAGE (PowerShell, from version6.6, venv active):
  python run_synth.py --limit 10        # smoke test
  python run_synth.py --workers 4       # the real run, resumable (Ctrl+C safe)

OUTPUT (results\\):  synth_features.csv, synth_errors.txt (only if something failed), run_synth_<time>.txt, run_log.jsonl
"""

import os
import sys
import csv
import time
import argparse
import multiprocessing as mp

import numpy as np

import config
import utils
import features as F
from runlog import Run

META = ["set", "sid", "doc", "panel", "category", "variant", "gt_forged", "time_s",
        "px_total", "px_gt", "px_pred_raw", "px_tp_raw"]
FIELDS = META + F.FEATURE_NAMES


def _init_worker():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import cv2
    cv2.setNumThreads(1)


def _work(job):
    """job = (set, sid, doc, panel, category, variant, image_path, mask_path_or_empty)"""
    import cv2
    from pipeline import detect
    st, sid, doc, panel, cat, variant, path, mpath = job
    config.EXCLUDE_ENABLED = False
    t0 = time.time()
    try:
        gray = utils.load_image(path)
        color = utils.load_color(path)
        res = detect(gray, color)
        pred = (res["mask"] > 0).astype(np.uint8)
        gtm = np.zeros(gray.shape, np.uint8)
        if mpath:
            m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
            gtm = (m > 127).astype(np.uint8)
        row = {"set": st, "sid": sid, "doc": doc, "panel": panel, "category": cat, "variant": variant,
               "gt_forged": int(bool(mpath)), "px_total": int(gray.size), "px_gt": int(gtm.sum()),
               "px_pred_raw": int(pred.sum()), "px_tp_raw": int((pred & gtm).sum())}
        row.update({k: round(float(v), 6) for k, v in F.compute(gray, color, res).items()})
        row["time_s"] = round(time.time() - t0, 3)
        return job[:2], row, None
    except Exception as e:
        return job[:2], None, f"{type(e).__name__}: {e}"


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
    ap.add_argument("--manifest", default=os.path.join("results", "synth_manifest.csv"))
    ap.add_argument("--synth-dir", default=os.path.join("results", "synth"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="only the first N pairs (smoke test)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    feats = os.path.join(args.out, "synth_features.csv")

    with Run("run_synth", args, args.out) as run:
        man = list(csv.DictReader(open(args.manifest, newline="")))
        if args.limit:
            man = man[: args.limit]
        jobs = []
        for r in man:
            base = (r["sid"], r["doc"], r["panel"], r["category"], r["variant"])
            jobs.append(("synth_forged",) + base + (os.path.join(args.synth_dir, "forged", r["sid"] + ".png"),
                                                     os.path.join(args.synth_dir, "masks", r["sid"] + ".png")))
            jobs.append(("synth_genuine",) + base + (os.path.join(args.synth_dir, "genuine", r["sid"] + ".png"), ""))

        done, new_file = set(), not os.path.exists(feats)
        if not new_file:
            _repair(feats)
            with open(feats, newline="") as f:
                rd = csv.DictReader(f)
                if rd.fieldnames != FIELDS:
                    sys.exit(f"{feats} has different columns. Move it away and rerun.")
                done = {(x["set"], x["sid"]) for x in rd}
        jobs = [j for j in jobs if j[:2] not in done]
        run.p(f"pairs in manifest: {len(man)}   jobs to run: {len(jobs)}   (already done: {len(done)})   workers: {args.workers}\n")

        t0, n_ok, n_err, times = time.time(), 0, 0, []
        pool = None
        if args.workers > 1 and jobs:
            pool = mp.get_context("spawn").Pool(args.workers, initializer=_init_worker)
            stream = pool.imap(_work, jobs, chunksize=2)
        else:
            stream = (_work(j) for j in jobs)
        try:
            with open(feats, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if new_file:
                    w.writeheader()
                for k, (key, row, err) in enumerate(stream, 1):
                    if err:
                        n_err += 1
                        with open(os.path.join(args.out, "synth_errors.txt"), "a") as ef:
                            ef.write(f"{key}\t{err}\n")
                        print(f"  ! {key}: {err}")
                    else:
                        w.writerow(row)
                        f.flush()
                        n_ok += 1
                        times.append(row["time_s"])
                    if k % 100 == 0 or k == len(jobs):
                        el = time.time() - t0
                        print(f"  [{k}/{len(jobs)}] {el / 60:.1f} min elapsed, ~{el / k * (len(jobs) - k) / 60:.0f} min left")
        finally:
            if pool is not None:
                pool.terminate()
        run.output(feats)
        run.note(rows_written=n_ok, errors=n_err, mean_s_per_image=round(float(np.mean(times)), 3) if times else None)
        run.p(f"\nrows written this run: {n_ok}   errors: {n_err}")
        run.p("Next:  python develop.py")


if __name__ == "__main__":
    main()
