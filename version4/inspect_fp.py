"""
inspect_fp.py  —  Look at WHY an authentic image was falsely flagged.

Shows the match lines the detector drew on a genuine (authentic) image, so you
can SEE what it's wrongly matching on:
  - text / scale bars / captions   -> fix is region exclusion
  - graph axes / gridlines         -> fix is region exclusion
  - genuinely-similar cells/bands  -> fix is the semantic (SGO) filter

USAGE:
  python inspect_fp.py 12345      # an authentic image ID (from false_positives.csv)

Open false_positives.csv, grab a few IDs, and run this on each. Tell what the
green lines land on — that decides the next fix.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

import config
import utils
from pipeline import detect


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_fp.py <authentic_image_id>")
        print("  (get IDs from results/false_positives.csv)")
        return

    image_id = sys.argv[1]
    img_path = os.path.join(config.AUTHENTIC_DIR, f"{image_id}.png")

    print(f"Authentic image: {image_id}  (correct answer = authentic)")
    img = utils.load_image(img_path)

    result = detect(img)
    kp = result["keypoints"]
    matches = result["clustered"]      # the matches that caused the false flag

    print(f"  keypoints: {len(kp)}")
    print(f"  matched pairs that triggered the false flag: {len(matches)}")
    print(f"  verdict: {'FORGED (false positive)' if result['is_forged'] else 'authentic (correct)'}")

    # draw the offending matches
    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i, j in matches:
        p1 = tuple(map(int, kp[i].pt))
        p2 = tuple(map(int, kp[j].pt))
        cv2.line(canvas, p1, p2, (0, 255, 0), 1)
        cv2.circle(canvas, p1, 4, (0, 0, 255), -1)
        cv2.circle(canvas, p2, 4, (255, 0, 0), -1)

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.title("Authentic image")
    plt.imshow(img, cmap="gray"); plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"What it WRONGLY matched ({len(matches)} pairs)")
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)); plt.axis("off")

    plt.tight_layout()
    out_dir = config.ensure_output_dir()
    out_path = os.path.join(out_dir, f"inspect_fp_{image_id}.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"  saved -> {out_path}")
    plt.show()


if __name__ == "__main__":
    main()