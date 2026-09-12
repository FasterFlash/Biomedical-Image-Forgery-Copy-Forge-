"""
run_batch.py  —  Run the detector on a BATCH of images for a reality check.

This is where you find out where you ACTUALLY stand (not just cherry-picked
single images). It:
  - takes the first N forged images
  - runs the SIFT+RANSAC pipeline on each
  - scores each against its ground-truth mask
  - prints per-image results + an overall average
  - saves everything to results/batch_results.csv

USAGE:
  python run_batch.py           # first 50 forged images (default)
  python run_batch.py 100       # first 100

Notes:
  - This runs on FORGED images only (they have ground-truth masks to score against).
  - Timing per image is printed so you can spot slow ones.
  - The average MCC/F1 across the batch is your honest reality-check number.
"""

import os
import sys
import csv
import time

import config
import utils
from pipeline import detect


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50

    ids = utils.list_image_ids(config.FORGED_DIR)[:n]
    print(f"Running on {len(ids)} forged images...\n")

    rows = []
    totals = {"F1": 0.0, "IoU": 0.0, "precision": 0.0, "recall": 0.0, "MCC": 0.0}
    detected = 0
    t_start = time.time()

    for idx, image_id in enumerate(ids, 1):
        img_path  = os.path.join(config.FORGED_DIR, f"{image_id}.png")
        mask_path = os.path.join(config.MASK_DIR,   f"{image_id}.npy")

        try:
            img = utils.load_image(img_path)
        except FileNotFoundError:
            print(f"  [{idx}/{len(ids)}] {image_id}: image not found, skipping")
            continue

        truth = utils.load_mask(mask_path, img.shape)

        t0 = time.time()
        result = detect(img)
        dt = time.time() - t0

        scores = utils.score_masks(result["mask"], truth)
        if scores is None:
            # no ground-truth mask; skip scoring but note it
            print(f"  [{idx}/{len(ids)}] {image_id}: no mask, skipped scoring")
            continue

        if result["is_forged"]:
            detected += 1
        for k in totals:
            totals[k] += scores[k]

        rows.append({
            "image_id": image_id,
            "verdict": "forged" if result["is_forged"] else "authentic",
            "keypoints": len(result["keypoints"]),
            "inliers": len(result["inliers"]),
            "time_s": round(dt, 2),
            **scores,
        })

        print(f"  [{idx}/{len(ids)}] {image_id}: "
              f"MCC={scores['MCC']:.3f} F1={scores['F1']:.3f} "
              f"({len(result['keypoints'])} kp, {dt:.1f}s)")

    # ---- summary ----
    total_time = time.time() - t_start
    scored = len(rows)
    if scored == 0:
        print("\nNo images scored. Check paths in config.py.")
        return

    print("\n" + "=" * 50)
    print(f"BATCH SUMMARY  ({scored} images scored)")
    print("=" * 50)
    print(f"  Detected as forged: {detected}/{scored} "
          f"({100*detected/scored:.0f}%)")
    print(f"  Average scores across the batch:")
    for k in totals:
        print(f"     {k:10s}: {totals[k]/scored:.4f}")
    print(f"  Total time: {total_time:.1f}s "
          f"({total_time/scored:.1f}s per image)")
    print("=" * 50)
    print("  ^ THIS is your honest reality-check number (not single images).")

    # ---- save CSV ----
    out_dir = config.ensure_output_dir()
    csv_path = os.path.join(out_dir, "batch_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  saved per-image results -> {csv_path}")


if __name__ == "__main__":
    main()