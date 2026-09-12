"""
crawl1.py  —  First step: find a copy-move forgery in ONE image using SIFT.

WHAT THIS DOES (in plain words):
  1. Loads one FORGED blot image.
  2. Uses SIFT to drop "pins" on distinctive spots and describe each one.
  3. Matches pins to OTHER pins in the SAME image that look nearly identical.
     -> Those matches are the copy<->paste fingerprint of a forgery.
  4. Draws lines connecting the matched (duplicated) spots.
  5. Loads the ground-truth mask (.npy) and shows how much our detection
     overlapped the real forged region.

RUN IT:
  python crawl1.py

If your folders are laid out like:
  forgery-detection/
    dataset/
      train_images/forged/414.png
      train_masks/414.npy
    craw1/crawl1.py   <-- this file lives here

...then the default paths below should just work. If not, edit the 3 lines
under "----- EDIT THESE IF NEEDED -----".
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ----- EDIT THESE IF NEEDED -----
# Path from THIS script's folder up to the dataset. ".." means "go up one level".
IMAGE_NUMBER = "3678"                       # which image to test
FORGED_DIR   = os.path.join("..", "dataset", "train_images", "forged")
MASK_DIR     = os.path.join("..", "dataset", "train_masks")
# --------------------------------


def load_image(path):
    """Load an image in grayscale. SIFT works on intensity, not color."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(
            f"Could not read image at: {path}\n"
            f"   -> Check the filename/extension and that the path is right."
        )
    return img


def load_mask(path, target_shape):
    """
    Load the .npy ground-truth mask. Shape in the dataset is (1, H, W) with
    values {0,1}. We squeeze it to (H, W) and, if needed, resize to match
    the image so we can overlay them.
    """
    if not os.path.exists(path):
        print(f"[note] No mask found at {path} (that's expected for authentic images).")
        return None
    mask = np.load(path, allow_pickle=True)
    mask = np.squeeze(mask)                 # (1,H,W) -> (H,W)
    mask = (mask > 0).astype(np.uint8)      # force to clean 0/1
    if mask.shape != target_shape:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    return mask


def find_copymove_matches(img, ratio=0.75, min_pixel_dist=10):
    """
    The heart of it.
      - SIFT finds keypoints + descriptors.
      - We match each descriptor to its 2 nearest neighbors AMONG THE SAME
        image's descriptors (self-matching).
      - Lowe's ratio test keeps only confident matches.
      - We throw away matches between a point and itself / its neighbor
        (min_pixel_dist), so we only keep pairs that are FAR APART in space
        but LOOK the same  ->  the signature of copy-move.
    Returns the keypoints and the list of good (i, j) index pairs.
    """
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)

    if des is None or len(kp) < 2:
        print("[warn] SIFT found too few keypoints. This image may be too smooth.")
        return kp, []

    # Match descriptors against themselves. k=3 because the closest match to
    # any point is itself; we want the next-closest OTHER points.
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des, des, k=3)

    good = []
    for m_group in matches:
        if len(m_group) < 3:
            continue
        # m_group[0] is the point matching itself (distance ~0). Skip it.
        # Compare the 2nd and 3rd nearest (the real "other" candidates).
        first, second = m_group[1], m_group[2]
        # Lowe's ratio test: keep only if the best "other" match is clearly
        # better than the runner-up (a confident, distinctive match).
        if first.distance < ratio * second.distance:
            pi = kp[first.queryIdx].pt
            pj = kp[first.trainIdx].pt
            spatial_dist = np.hypot(pi[0] - pj[0], pi[1] - pj[1])
            if spatial_dist > min_pixel_dist:
                good.append((first.queryIdx, first.trainIdx))

    return kp, good


def draw_matches(img, kp, pairs):
    """Draw lines connecting each duplicated spot to its twin."""
    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i, j in pairs:
        p1 = tuple(map(int, kp[i].pt))
        p2 = tuple(map(int, kp[j].pt))
        cv2.line(canvas, p1, p2, (0, 255, 0), 1)
        cv2.circle(canvas, p1, 3, (0, 0, 255), -1)
        cv2.circle(canvas, p2, 3, (255, 0, 0), -1)
    return canvas


def predicted_mask_from_pairs(kp, pairs, shape, radius=12):
    """
    Turn the matched points into a rough predicted forgery mask by painting
    filled circles around every matched keypoint. Crude on purpose — this is
    the baseline we'll improve later.
    """
    pred = np.zeros(shape, dtype=np.uint8)
    for i, j in pairs:
        for idx in (i, j):
            x, y = map(int, kp[idx].pt)
            cv2.circle(pred, (x, y), radius, 1, -1)
    return pred


def overlap_score(pred, truth):
    """
    Intersection-over-Union (IoU) and a simple F1 between our predicted
    region and the true mask. These are the 'how right were we' numbers.
    """
    if truth is None:
        return None
    pred_b = pred.astype(bool)
    true_b = truth.astype(bool)
    inter = np.logical_and(pred_b, true_b).sum()
    union = np.logical_or(pred_b, true_b).sum()
    iou = inter / union if union > 0 else 0.0
    tp = inter
    fp = np.logical_and(pred_b, ~true_b).sum()
    fn = np.logical_and(~pred_b, true_b).sum()
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return {"IoU": iou, "F1": f1}


def main():
    img_path  = os.path.join(FORGED_DIR, f"{IMAGE_NUMBER}.png")
    mask_path = os.path.join(MASK_DIR,   f"{IMAGE_NUMBER}.npy")

    print(f"Loading forged image: {img_path}")
    img = load_image(img_path)
    print(f"   image shape: {img.shape}")

    print("Running SIFT self-matching to find duplicated regions...")
    kp, pairs = find_copymove_matches(img)
    print(f"   SIFT keypoints: {len(kp)}")
    print(f"   suspicious duplicate matches found: {len(pairs)}")

    truth = load_mask(mask_path, img.shape)
    pred  = predicted_mask_from_pairs(kp, pairs, img.shape)

    scores = overlap_score(pred, truth)
    if scores is not None:
        print(f"\n=== HOW CLOSE DID WE GET (vs ground-truth mask) ===")
        print(f"   IoU: {scores['IoU']:.3f}   F1: {scores['F1']:.3f}")
        print(f"   (0 = missed it entirely, 1 = perfect. Expect low for now — that's fine.)")

    # ---- Show everything visually ----
    match_vis = draw_matches(img, kp, pairs)

    n_panels = 4 if truth is not None else 2
    plt.figure(figsize=(4 * n_panels, 4))

    plt.subplot(1, n_panels, 1)
    plt.title("Forged image")
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, n_panels, 2)
    plt.title(f"SIFT matches ({len(pairs)})")
    plt.imshow(cv2.cvtColor(match_vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    if truth is not None:
        plt.subplot(1, n_panels, 3)
        plt.title("TRUE forged region")
        plt.imshow(img, cmap="gray")
        plt.imshow(truth, cmap="Reds", alpha=0.5)
        plt.axis("off")

        plt.subplot(1, n_panels, 4)
        plt.title("OUR detection")
        plt.imshow(img, cmap="gray")
        plt.imshow(pred, cmap="Blues", alpha=0.5)
        plt.axis("off")

    plt.tight_layout()
    out_path = f"crawl1_result_{IMAGE_NUMBER}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved visualization to: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()