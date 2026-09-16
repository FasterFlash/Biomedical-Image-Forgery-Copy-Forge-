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


def _cluster_points(kp, pairs):
    """
    Cluster the matched points spatially (DBSCAN). Returns a dict mapping
    cluster_label -> list of pair-indices whose points fall in that cluster.
    This is step 1 of the Amerini-standard approach: group matches into spatial
    clusters FIRST, before geometric verification.
    """
    if len(pairs) < config.MIN_CLUSTER_POINTS:
        return {}

    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        return {0: list(range(len(pairs)))}   # fallback: one big cluster

    # Use the SOURCE point of each pair as its location for clustering.
    pts = np.array([kp[i].pt for i, _ in pairs])
    labels = DBSCAN(eps=config.DBSCAN_EPS,
                    min_samples=config.DBSCAN_MIN_SAMPLES).fit_predict(pts)

    clusters = {}
    for pair_idx, lab in enumerate(labels):
        if lab == -1:
            continue                            # noise — drop scattered matches
        clusters.setdefault(lab, []).append(pair_idx)
    return clusters


def _ransac_within_clusters(kp, pairs):
    """
    THE AMERINI-STANDARD APPROACH (cluster-first, RANSAC-per-cluster).

    Step 1: cluster the raw matches spatially.
    Step 2: run RANSAC SEPARATELY inside each cluster.

    Why this beats global RANSAC: a forgery has a source region and a target
    region, possibly with different local geometry. One global RANSAC tries to
    fit a single transform to everything and gets confused by multiple regions,
    letting false matches through. Verifying each cluster on its own gives a
    clean per-region geometric check -> higher precision.

    Reference: Amerini et al. (2011); recent DBSCAN-based CMFD post-processing.

    Returns the list of geometrically-verified pairs (across all clusters).
    """
    clusters = _cluster_points(kp, pairs)
    if not clusters:
        return []

    verified = []
    for _, pair_indices in clusters.items():
        if len(pair_indices) < config.MIN_INLIERS:
            continue

        cluster_pairs = [pairs[k] for k in pair_indices]
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

        # Reject geometrically implausible transforms (absurd scale = false match)
        scale = np.sqrt(abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]))
        if scale > config.MAX_PLAUSIBLE_SCALE or scale < (1.0 / config.MAX_PLAUSIBLE_SCALE):
            continue

        cluster_inliers = [cluster_pairs[k] for k in range(len(cluster_pairs))
                           if inlier_mask[k]]
        if len(cluster_inliers) >= config.MIN_INLIERS:
            verified.extend(cluster_inliers)

    return verified


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

    # AMERINI-STANDARD ORDER: cluster first, then RANSAC within each cluster.
    # (Replaces the old global-RANSAC-then-cluster order, which over-flagged
    #  because one global transform can't handle separate source/target regions.)
    verified = _ransac_within_clusters(kp, raw_pairs)

    is_forged = len(verified) >= config.MIN_INLIERS
    mask = _paint_mask(kp, verified, img.shape) if is_forged else np.zeros(img.shape, np.uint8)

    return {
        "is_forged": is_forged,
        "mask": mask,
        "keypoints": kp,
        "raw_pairs": raw_pairs,
        "inliers": verified,      # geometrically-verified matches (for visualization)
        "clustered": verified,    # same thing now; kept for run_one.py compatibility
    }