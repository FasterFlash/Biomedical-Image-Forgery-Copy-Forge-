"""
run_pipeline_viz.py  —  Visualize EVERY stage of the pipeline on ONE image.

Shows what happens at each step, so you can present the pipeline visually:
  Panel 1: Original image
  Panel 2: Raw SIFT matches (before any filtering)
  Panel 3: After clustering (DBSCAN — dense groups kept, scattered dropped)
  Panel 4: After per-cluster RANSAC (geometrically verified within each cluster)
  Panel 5: Ground-truth forged region
  Panel 6: Our final predicted mask

USAGE:
  python run_pipeline_viz.py 414        # any forged image ID
  python run_pipeline_viz.py 653        # corn cobs
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

import config
import utils

# We need to access internal pipeline functions, so import them directly
from pipeline import _extract_keypoints, _self_match_sift, _cluster_points, _paint_mask


DEFAULT_ID = "414"


def draw_pairs(img, kp, pairs, color=(0, 255, 0)):
    """Draw match lines on a color copy of the image."""
    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i, j in pairs:
        p1 = tuple(map(int, kp[i].pt))
        p2 = tuple(map(int, kp[j].pt))
        cv2.line(canvas, p1, p2, color, 1)
        cv2.circle(canvas, p1, 3, (0, 0, 255), -1)
        cv2.circle(canvas, p2, 3, (255, 0, 0), -1)
    return canvas


def draw_clusters(img, kp, pairs, clusters, verified_pairs):
    """
    Draw clusters in different colors, with verified (post-RANSAC) matches
    shown as solid lines and rejected ones as faint dotted.
    """
    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Generate distinct colors for each cluster
    colors = [
        (0, 255, 0), (255, 0, 0), (0, 255, 255),
        (255, 0, 255), (255, 255, 0), (0, 128, 255),
        (128, 255, 0), (255, 128, 0), (0, 255, 128),
    ]

    verified_set = set()
    for i, j in verified_pairs:
        verified_set.add((i, j))

    for lab, pair_indices in clusters.items():
        color = colors[lab % len(colors)]
        cluster_pairs = [pairs[k] for k in pair_indices]
        for i, j in cluster_pairs:
            p1 = tuple(map(int, kp[i].pt))
            p2 = tuple(map(int, kp[j].pt))
            if (i, j) in verified_set:
                cv2.line(canvas, p1, p2, color, 2)
            else:
                # faint line for rejected matches
                cv2.line(canvas, p1, p2, color, 1)
            cv2.circle(canvas, p1, 3, color, -1)
            cv2.circle(canvas, p2, 3, color, -1)

    return canvas


def main():
    image_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ID

    img_path  = os.path.join(config.FORGED_DIR, f"{image_id}.png")
    mask_path = os.path.join(config.MASK_DIR,   f"{image_id}.npy")

    print(f"Pipeline visualization for image: {image_id}")
    img = utils.load_image(img_path)
    truth = utils.load_mask(mask_path, img.shape)

    # ========== STAGE 1: Raw SIFT self-matching ==========
    print("  Stage 1: SIFT keypoint extraction + self-matching...")
    kp, raw_pairs = _self_match_sift(img)
    print(f"    keypoints: {len(kp)}")
    print(f"    raw matches: {len(raw_pairs)}")

    # ========== STAGE 2: DBSCAN clustering ==========
    print("  Stage 2: DBSCAN spatial clustering...")
    clusters = _cluster_points(kp, raw_pairs)
    clustered_pairs = []
    for lab, pair_indices in clusters.items():
        for k in pair_indices:
            clustered_pairs.append(raw_pairs[k])
    noise_removed = len(raw_pairs) - len(clustered_pairs)
    print(f"    clusters found: {len(clusters)}")
    print(f"    matches in clusters: {len(clustered_pairs)}")
    print(f"    noise (scattered) removed: {noise_removed}")

    # ========== STAGE 3: Per-cluster RANSAC verification ==========
    print("  Stage 3: RANSAC verification within each cluster...")
    verified_pairs = []
    for lab, pair_indices in clusters.items():
        if len(pair_indices) < config.MIN_INLIERS:
            continue
        cluster_pairs = [raw_pairs[k] for k in pair_indices]
        src = np.float32([kp[i].pt for i, _ in cluster_pairs]).reshape(-1, 1, 2)
        dst = np.float32([kp[j].pt for _, j in cluster_pairs]).reshape(-1, 1, 2)
        try:
            M, inlier_mask = cv2.estimateAffinePartial2D(
                src, dst,
                method=cv2.RANSAC,
                ransacReprojThreshold=config.RANSAC_THRESHOLD,
            )
        except cv2.error:
            continue
        if inlier_mask is None or M is None:
            continue
        scale = np.sqrt(abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]))
        if scale > config.MAX_PLAUSIBLE_SCALE or scale < (1.0 / config.MAX_PLAUSIBLE_SCALE):
            continue
        cluster_inliers = [cluster_pairs[k] for k in range(len(cluster_pairs))
                           if inlier_mask[k]]
        if len(cluster_inliers) >= config.MIN_INLIERS:
            verified_pairs.extend(cluster_inliers)
    print(f"    verified matches: {len(verified_pairs)}")

    # ========== STAGE 4: Final mask ==========
    print("  Stage 4: Generating final mask...")
    is_forged = len(verified_pairs) >= config.MIN_INLIERS
    pred_mask = _paint_mask(kp, verified_pairs, img.shape) if is_forged else np.zeros(img.shape, np.uint8)
    print(f"    verdict: {'FORGED' if is_forged else 'authentic'}")

    scores = utils.score_masks(pred_mask, truth)
    if scores:
        print(f"    MCC: {scores['MCC']}  F1: {scores['F1']}")

    # ========== BUILD THE VISUALIZATION ==========
    has_truth = truth is not None
    n_panels = 6 if has_truth else 4

    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4.5))
    fig.suptitle(f"Pipeline Stages — Image {image_id}", fontsize=14, fontweight='bold')

    # Panel 1: Original
    axes[0].set_title("1. Original Image", fontsize=10)
    axes[0].imshow(img, cmap="gray")
    axes[0].axis("off")

    # Panel 2: Raw SIFT matches
    raw_vis = draw_pairs(img, kp, raw_pairs)
    axes[1].set_title(f"2. Raw SIFT Matches ({len(raw_pairs)})", fontsize=10)
    axes[1].imshow(cv2.cvtColor(raw_vis, cv2.COLOR_BGR2RGB))
    axes[1].axis("off")

    # Panel 3: After clustering (color-coded clusters)
    cluster_vis = draw_clusters(img, kp, raw_pairs, clusters, verified_pairs)
    axes[2].set_title(f"3. After Clustering ({len(clusters)} clusters)", fontsize=10)
    axes[2].imshow(cv2.cvtColor(cluster_vis, cv2.COLOR_BGR2RGB))
    axes[2].axis("off")

    # Panel 4: After per-cluster RANSAC
    ransac_vis = draw_pairs(img, kp, verified_pairs, color=(0, 255, 0))
    axes[3].set_title(f"4. After RANSAC ({len(verified_pairs)} verified)", fontsize=10)
    axes[3].imshow(cv2.cvtColor(ransac_vis, cv2.COLOR_BGR2RGB))
    axes[3].axis("off")

    if has_truth:
        # Panel 5: Ground truth
        axes[4].set_title("5. TRUE Forged Region", fontsize=10)
        axes[4].imshow(img, cmap="gray")
        axes[4].imshow(truth, cmap="Reds", alpha=0.5)
        axes[4].axis("off")

        # Panel 6: Our prediction
        axes[5].set_title("6. OUR Detection", fontsize=10)
        axes[5].imshow(img, cmap="gray")
        axes[5].imshow(pred_mask, cmap="Blues", alpha=0.5)
        axes[5].axis("off")

    plt.tight_layout()
    out_dir = config.ensure_output_dir()
    out_path = os.path.join(out_dir, f"pipeline_viz_{image_id}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  saved -> {out_path}")
    plt.show()


if __name__ == "__main__":
    main()