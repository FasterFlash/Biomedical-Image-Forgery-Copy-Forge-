"""
run_one.py  —  Run the detector on ONE image and SEE the result.

This is your day-to-day "does it work" script. It:
  - loads one forged image + its ground-truth mask
  - runs the SIFT+RANSAC pipeline
  - shows: original | matches before RANSAC | matches after RANSAC |
           true forged region | our predicted region
  - prints the scores (F1, IoU, MCC, etc.)

USAGE:
  python run_one.py            # uses the default image ID below
  python run_one.py 1027       # run on a specific image ID

Watch the "before RANSAC" vs "after RANSAC" panels — you should see the messy
matches get filtered down to a clean cluster on the actually-copied region.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

import config
import utils
from pipeline import detect


# default image to test if none given on command line
DEFAULT_ID = "414"


def draw_pairs(img, kp, pairs, color=(0, 255, 0)):
    """Draw lines connecting matched point pairs on a color copy of the image."""
    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i, j in pairs:
        p1 = tuple(map(int, kp[i].pt))
        p2 = tuple(map(int, kp[j].pt))
        cv2.line(canvas, p1, p2, color, 1)
        cv2.circle(canvas, p1, 3, (0, 0, 255), -1)
        cv2.circle(canvas, p2, 3, (255, 0, 0), -1)
    return canvas


def main():
    image_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ID

    img_path  = os.path.join(config.FORGED_DIR, f"{image_id}.png")
    mask_path = os.path.join(config.MASK_DIR,   f"{image_id}.npy")

    print(f"Image ID: {image_id}")
    print(f"  image: {img_path}")
    img = utils.load_image(img_path)
    truth = utils.load_mask(mask_path, img.shape)

    # ---- run the detector ----
    result = detect(img)
    print(f"  SIFT keypoints:        {len(result['keypoints'])}")
    print(f"  matches before RANSAC: {len(result['raw_pairs'])}")
    print(f"  matches after RANSAC:  {len(result['inliers'])}")
    print(f"  verdict: {'FORGED' if result['is_forged'] else 'authentic'}")

    # ---- score against ground truth ----
    scores = utils.score_masks(result["mask"], truth)
    if scores is not None:
        print("\n  === SCORES (predicted region vs true region) ===")
        for k, v in scores.items():
            print(f"     {k:10s}: {v}")
        print("  (low is expected for this crude baseline — it's the starting line)")
    else:
        print("  (no ground-truth mask found for this image)")

    # ---- visualize ----
    before = draw_pairs(img, result["keypoints"], result["raw_pairs"])
    after  = draw_pairs(img, result["keypoints"], result["inliers"])

    panels = 5 if truth is not None else 3
    plt.figure(figsize=(4 * panels, 4))

    plt.subplot(1, panels, 1); plt.title("Forged image")
    plt.imshow(img, cmap="gray"); plt.axis("off")

    plt.subplot(1, panels, 2); plt.title(f"Before RANSAC ({len(result['raw_pairs'])})")
    plt.imshow(cv2.cvtColor(before, cv2.COLOR_BGR2RGB)); plt.axis("off")

    plt.subplot(1, panels, 3); plt.title(f"After RANSAC ({len(result['inliers'])})")
    plt.imshow(cv2.cvtColor(after, cv2.COLOR_BGR2RGB)); plt.axis("off")

    if truth is not None:
        plt.subplot(1, panels, 4); plt.title("TRUE region")
        plt.imshow(img, cmap="gray"); plt.imshow(truth, cmap="Reds", alpha=0.5); plt.axis("off")

        plt.subplot(1, panels, 5); plt.title("OUR detection")
        plt.imshow(img, cmap="gray"); plt.imshow(result["mask"], cmap="Blues", alpha=0.5); plt.axis("off")

    plt.tight_layout()
    out_dir = config.ensure_output_dir()
    out_path = os.path.join(out_dir, f"run_one_{image_id}.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\n  saved visualization -> {out_path}")
    plt.show()


if __name__ == "__main__":
    main()