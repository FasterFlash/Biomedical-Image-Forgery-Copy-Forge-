"""
run_authentic.py  —  The false-positive test.

Runs the detector on AUTHENTIC (genuine, un-tampered) images and counts how many
it WRONGLY flags as forged. Since these images are real, the correct answer for
EVERY one is "authentic". Any image flagged "forged" is a FALSE POSITIVE.

This is the most honest measure of whether your detector is trigger-happy.
A good triage tool must almost NEVER cry wolf on genuine images.

USAGE:
  python run_authentic.py            # first 200 authentic images
  python run_authentic.py 100        # first 100

Output:
  - false positive rate: what fraction of genuine images got flagged as forged
  - a CSV listing which authentic images were (wrongly) flagged, so you can
    open them and see WHY they fooled the detector.
"""

import os
import sys
import csv
import time

import config
import utils
from pipeline import detect


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200

    ids = utils.list_image_ids(config.AUTHENTIC_DIR)[:n]
    print(f"Running on {len(ids)} AUTHENTIC images (correct answer = 'authentic' for all)...\n")

    flagged = []          # images wrongly called 'forged' (false positives)
    correct = 0           # images correctly called 'authentic'
    processed = 0
    t_start = time.time()

    for idx, image_id in enumerate(ids, 1):
        img_path = os.path.join(config.AUTHENTIC_DIR, f"{image_id}.png")
        try:
            img = utils.load_image(img_path)
        except FileNotFoundError:
            continue

        result = detect(img)
        processed += 1

        if result["is_forged"]:
            # FALSE POSITIVE — flagged a genuine image as forged
            flagged.append({
                "image_id": image_id,
                "keypoints": len(result["keypoints"]),
                "inliers": len(result["inliers"]),
            })
            verdict = "FORGED (false positive!)"
        else:
            correct += 1
            verdict = "authentic (correct)"

        if idx % 20 == 0 or result["is_forged"]:
            print(f"  [{idx}/{len(ids)}] {image_id}: {verdict}")

    total_time = time.time() - t_start
    if processed == 0:
        print("\nNo images processed. Check AUTHENTIC_DIR in config.py.")
        return

    fpr = len(flagged) / processed
    print("\n" + "=" * 55)
    print(f"FALSE-POSITIVE TEST  ({processed} authentic images)")
    print("=" * 55)
    print(f"  Correctly called authentic: {correct}/{processed}")
    print(f"  WRONGLY flagged as forged:  {len(flagged)}/{processed}")
    print(f"  >>> FALSE POSITIVE RATE:    {100*fpr:.1f}%  <<<")
    print(f"  Total time: {total_time:.1f}s")
    print("=" * 55)
    if fpr > 0.5:
        print("  A high FPR means the detector cries 'forgery' on genuine images.")
        print("  This is the precision problem — your #1 thing to fix.")
    print("  (A real triage tool needs this number to be LOW.)")

    # save which authentic images got falsely flagged, so you can inspect them
    if flagged:
        out_dir = config.ensure_output_dir()
        csv_path = os.path.join(out_dir, "false_positives.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=flagged[0].keys())
            writer.writeheader()
            writer.writerows(flagged)
        print(f"\n  saved list of false positives -> {csv_path}")
        print("  ^ open these authentic images with run_one.py to see WHY they fooled it.")


if __name__ == "__main__":
    main()