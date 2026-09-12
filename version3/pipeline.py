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

    sift = cv2.SIFT_create(
        contrastThreshold=config.SIFT_CONTRAST_THRESHOLD,
        nfeatures=config.MAX_KEYPOINTS,   # keep only the strongest N keypoints
    )
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

    # Match descriptors against themselves using FLANN (KDTree) approximate
    # nearest-neighbor search. This is the 2nd-place speed trick: brute-force
    # matching is O(N^2) and chokes on the huge keypoint counts that the
    # sensitive contrast_threshold produces. FLANN drops it to ~O(N log N).
    # k=3: the nearest match to any descriptor is ITSELF (distance ~0), so we
    # look at the 2nd/3rd nearest to find genuinely OTHER matching points.
    index_params = dict(algorithm=1, trees=5)   # algorithm=1 -> KDTree
    search_params = dict(checks=50)             # higher = more accurate, slower
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    des = np.asarray(des, dtype=np.float32)     # FLANN needs float32
    knn = flann.knnMatch(des, des, k=3)

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


def _cluster_filter(kp, pairs):
    """
    THE KEY PRECISION FIX. After RANSAC, keep only matches whose points form
    DENSE spatial clusters — real copied regions. Scattered/isolated matches
    (the main cause of false positives on authentic images) get discarded.

    Uses DBSCAN density clustering on the matched-point locations. Points that
    DBSCAN labels as noise (not in any dense cluster) are thrown away. A cluster
    must have at least MIN_CLUSTER_POINTS members to count.

    Returns the filtered pairs (only those in dense clusters).
    """
    if len(pairs) < config.MIN_CLUSTER_POINTS:
        return []

    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        # sklearn not installed — skip clustering (falls back to RANSAC-only)
        return pairs

    # Collect all matched-point coordinates (both ends of each pair)
    pts = []
    owner = []   # which pair each point belongs to
    for idx, (i, j) in enumerate(pairs):
        pts.append(kp[i].pt); owner.append(idx)
        pts.append(kp[j].pt); owner.append(idx)
    pts = np.array(pts)

    labels = DBSCAN(eps=config.DBSCAN_EPS,
                    min_samples=config.DBSCAN_MIN_SAMPLES).fit_predict(pts)

    # Count how many points each cluster has; keep only big-enough clusters
    from collections import Counter
    counts = Counter(labels)
    good_clusters = {lab for lab, c in counts.items()
                     if lab != -1 and c >= config.MIN_CLUSTER_POINTS}

    if not good_clusters:
        return []

    # Keep a pair if at least one of its points is in a good (dense) cluster
    keep = set()
    for pt_idx, lab in enumerate(labels):
        if lab in good_clusters:
            keep.add(owner[pt_idx])

    return [pairs[k] for k in sorted(keep)]


def _paint_mask(kp, pairs, shape):
    """
    Build a TIGHT predicted mask. Instead of one greedy convex hull around ALL
    points (which over-paints the gap between the source and target regions),
    we cluster the points and paint a tight hull around EACH dense cluster
    separately. This keeps the source-blob and target-blob masks tight and
    separate, which is what preserves precision.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    if not pairs:
        return mask

    pts = []
    for i, j in pairs:
        pts.append(kp[i].pt)
        pts.append(kp[j].pt)
    pts = np.array(pts, dtype=np.float32)

    # Re-cluster the points spatially so source and target regions are separate
    try:
        from sklearn.cluster import DBSCAN
        labels = DBSCAN(eps=config.DBSCAN_EPS,
                        min_samples=config.DBSCAN_MIN_SAMPLES).fit_predict(pts)
    except ImportError:
        labels = np.zeros(len(pts), dtype=int)   # fallback: one group

    # Paint a tight hull around each cluster separately
    for lab in set(labels):
        if lab == -1:
            continue                              # skip noise points
        cluster_pts = pts[labels == lab].astype(np.int32)
        if len(cluster_pts) >= 3:
            hull = cv2.convexHull(cluster_pts)
            cv2.fillConvexPoly(mask, hull, 1)
        else:
            for x, y in cluster_pts:
                cv2.circle(mask, (int(x), int(y)), config.MASK_RADIUS, 1, -1)

    # light dilation for coverage of the region (not just the point skeleton)
    kernel = np.ones((config.MASK_RADIUS, config.MASK_RADIUS), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
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

    # NEW: clustering filter — keep only dense match clusters (kills false positives)
    clustered = _cluster_filter(kp, inliers)

    is_forged = len(clustered) >= config.MIN_CLUSTER_POINTS
    mask = _paint_mask(kp, clustered, img.shape) if is_forged else np.zeros(img.shape, np.uint8)

    return {
        "is_forged": is_forged,
        "mask": mask,
        "keypoints": kp,
        "raw_pairs": raw_pairs,
        "inliers": inliers,       # after RANSAC (for visualization)
        "clustered": clustered,   # after clustering (the final confirmed matches)
    }