"""
pipeline.py  —  The core copy-move detector: SIFT + RANSAC.

This is the heart of the prototype. Given one image, it:
  1. Finds keypoints (SIFT) and self-matches them  -> candidate duplicated points
  2. Filters those matches with RANSAC              -> keeps only geometrically
                                                       consistent ones (real copies)
  3. Paints a predicted forgery mask from the surviving points

RANSAC is the key upgrade over raw SIFT: a real copied region moves as a rigid
block, so its matched points all agree on ONE geometric transform. RANSAC finds
that agreement and throws out random false matches (e.g. two similar-but-genuine
bands that matched by coincidence). This is what cleans up the messy scatter.

Import this and call detect(image) from other scripts, or run it directly on
one image to see it work.
"""

import cv2
import numpy as np

import config


def _extract_keypoints(img):
    """
    Extract SIFT keypoints, using the 2nd-place tricks for low-texture bio images:
      - very low contrast_threshold  -> flood the image with keypoints
      - upscale small images 4x      -> more keypoints, then map coords back
    Returns keypoints (in ORIGINAL image coordinates) and descriptors.
    """
    h, w = img.shape[:2]
    scale = 1.0
    work = img

    # Upscale small images so SIFT has more to grab
    if min(h, w) < config.UPSCALE_IF_SMALLER_THAN:
        scale = config.UPSCALE_FACTOR
        work = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    sift = cv2.SIFT_create(contrastThreshold=config.SIFT_CONTRAST_THRESHOLD)
    kp, des = sift.detectAndCompute(work, None)

    # Map keypoint coordinates back to the ORIGINAL image scale
    if scale != 1.0 and kp:
        for k in kp:
            k.pt = (k.pt[0] / scale, k.pt[1] / scale)

    return kp, des


def _self_match_sift(img):
    """
    Run SIFT and self-match descriptors to find duplicated regions.
    Returns keypoints and a list of matched (src_idx, dst_idx) index pairs.
    """
    kp, des = _extract_keypoints(img)

    if des is None or len(kp) < 2:
        return kp, []

    # Match descriptors against themselves. k=3: nearest is the point itself,
    # so we look at the 2nd/3rd nearest to find OTHER matching points.
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(des, des, k=3)

    pairs = []
    for group in knn:
        if len(group) < 3:
            continue
        first, second = group[1], group[2]      # skip self-match at [0]
        # Lowe's ratio test: keep only confident, distinctive matches
        if first.distance < config.LOWE_RATIO * second.distance:
            p_src = kp[first.queryIdx].pt
            p_dst = kp[first.trainIdx].pt
            dist = np.hypot(p_src[0] - p_dst[0], p_src[1] - p_dst[1])
            if dist > config.MIN_MATCH_DISTANCE:
                pairs.append((first.queryIdx, first.trainIdx))

    return kp, pairs


def _ransac_filter(kp, pairs):
    """
    Filter matches with RANSAC geometric verification.
    Keeps only matches consistent with a single affine transform — i.e. the
    matches that correspond to a genuinely copied block, not random coincidences.
    Returns the filtered list of (src_idx, dst_idx) pairs (the 'inliers').
    """
    if len(pairs) < config.MIN_INLIERS:
        return []

    src_pts = np.float32([kp[i].pt for i, _ in pairs]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp[j].pt for _, j in pairs]).reshape(-1, 1, 2)

    # Estimate an affine transform (handles translation, rotation, scale) and
    # get the inlier mask — which matches actually fit that transform.
    try:
        _, inlier_mask = cv2.estimateAffinePartial2D(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=config.RANSAC_THRESHOLD,
        )
    except cv2.error:
        return []

    if inlier_mask is None:
        return []

    inliers = [pairs[k] for k in range(len(pairs)) if inlier_mask[k]]
    return inliers if len(inliers) >= config.MIN_INLIERS else []


def _paint_mask(kp, pairs, shape):
    """Paint filled circles around matched points to build a predicted mask."""
    mask = np.zeros(shape, dtype=np.uint8)
    for i, j in pairs:
        for idx in (i, j):
            x, y = map(int, kp[idx].pt)
            cv2.circle(mask, (x, y), config.MASK_RADIUS, 1, -1)
    return mask


def detect(img):
    """
    Full detection on one grayscale image.
    Returns a dict:
      - 'is_forged': bool  (did we find a consistent duplicated region?)
      - 'mask':      predicted forgery mask (H,W) of 0/1
      - 'keypoints': SIFT keypoints (for visualization)
      - 'raw_pairs': matches before RANSAC (for visualization)
      - 'inliers':   matches after RANSAC (the confirmed ones)
    """
    kp, raw_pairs = _self_match_sift(img)
    inliers = _ransac_filter(kp, raw_pairs)
    is_forged = len(inliers) >= config.MIN_INLIERS
    mask = _paint_mask(kp, inliers, img.shape) if is_forged else np.zeros(img.shape, np.uint8)

    return {
        "is_forged": is_forged,
        "mask": mask,
        "keypoints": kp,
        "raw_pairs": raw_pairs,
        "inliers": inliers,
    }