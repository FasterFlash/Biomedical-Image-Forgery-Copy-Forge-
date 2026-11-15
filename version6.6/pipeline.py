"""
pipeline.py  --  The core copy-move detector: SIFT + cluster-first RANSAC  (V5.5).

V5.5 vs V5: the detection logic is IDENTICAL (same keypoints, same matches, same
verdict, same mask). The only additions are bookkeeping, so features.py can see
HOW a verdict was reached and not just how many matches survived:

  - _self_match_sift(..., ratios=dict)        records the Lowe ratio of every raw match
  - _ransac_within_clusters(..., stats=dict)  records, per spatial cluster, how many matches
                                              went in, how many survived, and the fitted
                                              scale / rotation
  - detect() returns them under result["info"]

Nothing here changes a decision. run_check-style comparison against V5 (same inlier counts,
mask on and off) is in test_equivalence.py.
"""

import cv2
import numpy as np

import config
from exclude import build_exclusion_mask


def _extract_keypoints(img, exclude=None):
    """
    Extract SIFT keypoints for low-texture bio images:
      - very low contrast_threshold  -> flood the image with keypoints
      - upscale small images 4x      -> more keypoints, then map coords back
    Returns keypoints (in ORIGINAL image coordinates) and descriptors.
    """
    h, w = img.shape[:2]
    scale = 1.0
    work = img

    if min(h, w) < config.UPSCALE_IF_SMALLER_THAN:
        scale = config.UPSCALE_FACTOR
        work = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    sift = cv2.SIFT_create(
        contrastThreshold=config.SIFT_CONTRAST_THRESHOLD,
        nfeatures=config.MAX_KEYPOINTS,
    )

    # Mask BEFORE detection so the MAX_KEYPOINTS budget is spent on real content.
    allowed = None
    if exclude is not None:
        allowed = (exclude == 0).astype(np.uint8) * 255
        if scale != 1.0:
            allowed = cv2.resize(allowed, (work.shape[1], work.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
    kp, des = sift.detectAndCompute(work, allowed)

    if scale != 1.0 and kp:
        for k in kp:
            k.pt = (k.pt[0] / scale, k.pt[1] / scale)

    return kp, des


def _self_match_sift(img, exclude=None, ratios=None):
    """
    Run SIFT and self-match descriptors to find duplicated regions.
    Returns keypoints and a list of matched (src_idx, dst_idx) index pairs.
    If `ratios` (a dict) is given, it is filled with {(src, dst): lowe_ratio} (V5.5).
    """
    kp, des = _extract_keypoints(img, exclude)

    if des is None or len(kp) < 2:
        return kp, []

    index_params = dict(algorithm=1, trees=5)   # KDTree
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    des = np.asarray(des, dtype=np.float32)
    knn = flann.knnMatch(des, des, k=3)

    pairs = []
    for group in knn:
        if len(group) < 3:
            continue
        first, second = group[1], group[2]      # skip self-match at [0]
        if first.distance < config.LOWE_RATIO * second.distance:
            p_src = kp[first.queryIdx].pt
            p_dst = kp[first.trainIdx].pt
            dist = np.hypot(p_src[0] - p_dst[0], p_src[1] - p_dst[1])
            if dist > config.MIN_MATCH_DISTANCE:
                pairs.append((first.queryIdx, first.trainIdx))
                if ratios is not None:
                    ratios[(first.queryIdx, first.trainIdx)] = float(
                        first.distance / max(second.distance, 1e-9))

    return kp, pairs


def _cluster_points(kp, pairs):
    """Cluster matched points spatially (DBSCAN); {label: [pair indices]}."""
    if len(pairs) < config.MIN_CLUSTER_POINTS:
        return {}

    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        return {0: list(range(len(pairs)))}

    pts = np.array([kp[i].pt for i, _ in pairs])
    labels = DBSCAN(eps=config.DBSCAN_EPS,
                    min_samples=config.DBSCAN_MIN_SAMPLES).fit_predict(pts)

    clusters = {}
    for pair_idx, lab in enumerate(labels):
        if lab == -1:
            continue
        clusters.setdefault(lab, []).append(pair_idx)
    return clusters


def _ransac_within_clusters(kp, pairs, stats=None):
    """
    Cluster-first, RANSAC-per-cluster (Amerini et al. 2011 style).
    Returns the list of geometrically verified pairs (across all clusters).

    If `stats` (a dict) is given it is filled with
      stats["n_dbscan"]  number of DBSCAN clusters found
      stats["clusters"]  one dict per cluster that was big enough to be tested:
                         n_pairs, n_inl, passed, scale, rot_deg, pairs
    """
    clusters = _cluster_points(kp, pairs)
    if stats is not None:
        stats["n_dbscan"] = len(clusters)
        stats["clusters"] = []
    if not clusters:
        return []

    verified = []
    for _, pair_indices in clusters.items():
        if len(pair_indices) < config.MIN_INLIERS:
            continue

        cluster_pairs = [pairs[k] for k in pair_indices]
        src = np.float32([kp[i].pt for i, _ in cluster_pairs]).reshape(-1, 1, 2)
        dst = np.float32([kp[j].pt for _, j in cluster_pairs]).reshape(-1, 1, 2)
        rec = {"n_pairs": len(cluster_pairs), "n_inl": 0, "passed": False,
               "scale": float("nan"), "rot_deg": float("nan"), "pairs": []}

        try:
            M, inlier_mask = cv2.estimateAffinePartial2D(
                src, dst,
                method=cv2.RANSAC,
                ransacReprojThreshold=config.RANSAC_THRESHOLD,
            )
        except cv2.error:
            if stats is not None:
                stats["clusters"].append(rec)
            continue
        if inlier_mask is None or M is None:
            if stats is not None:
                stats["clusters"].append(rec)
            continue

        scale = np.sqrt(abs(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]))
        rec["scale"] = float(scale)
        rec["rot_deg"] = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))

        if scale > config.MAX_PLAUSIBLE_SCALE or scale < (1.0 / config.MAX_PLAUSIBLE_SCALE):
            if stats is not None:
                stats["clusters"].append(rec)
            continue

        cluster_inliers = [cluster_pairs[k] for k in range(len(cluster_pairs))
                           if inlier_mask[k]]
        rec["n_inl"] = len(cluster_inliers)
        if len(cluster_inliers) >= config.MIN_INLIERS:
            rec["passed"] = True
            rec["pairs"] = cluster_inliers
            verified.extend(cluster_inliers)
        if stats is not None:
            stats["clusters"].append(rec)

    return verified


def _paint_mask(kp, pairs, shape):
    """Tight predicted mask: one convex hull per spatial cluster of matched points."""
    mask = np.zeros(shape, dtype=np.uint8)
    if not pairs:
        return mask

    pts = []
    for i, j in pairs:
        pts.append(kp[i].pt)
        pts.append(kp[j].pt)
    pts = np.array(pts, dtype=np.float32)

    try:
        from sklearn.cluster import DBSCAN
        labels = DBSCAN(eps=config.DBSCAN_EPS,
                        min_samples=config.DBSCAN_MIN_SAMPLES).fit_predict(pts)
    except ImportError:
        labels = np.zeros(len(pts), dtype=int)

    for lab in set(labels):
        if lab == -1:
            continue
        cluster_pts = pts[labels == lab].astype(np.int32)
        if len(cluster_pts) >= 3:
            hull = cv2.convexHull(cluster_pts)
            cv2.fillConvexPoly(mask, hull, 1)
        else:
            for x, y in cluster_pts:
                cv2.circle(mask, (int(x), int(y)), config.MASK_RADIUS, 1, -1)

    kernel = np.ones((config.MASK_RADIUS, config.MASK_RADIUS), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def detect(img, color=None):
    """
    Full detection on one grayscale image (`color` optional BGR, same size, for the OCR rule).
    Returns a dict:
      is_forged, mask, keypoints, raw_pairs, inliers, clustered, exclude   (as in V5)
      info   V5.5: {"ratios": {(i, j): lowe_ratio}, "n_dbscan": int, "clusters": [...]}
    """
    # Per-image seed: result is a pure function of the image (see V5 notes).
    cv2.setRNGSeed(config.RANDOM_SEED)

    exclude = None
    if config.EXCLUDE_ENABLED:
        exclude, _ = build_exclusion_mask(img, config.EXCLUDE_RULES, color)

    ratios = {}
    kp, raw_pairs = _self_match_sift(img, exclude, ratios)

    cstats = {}
    verified = _ransac_within_clusters(kp, raw_pairs, cstats)

    is_forged = len(verified) >= config.MIN_INLIERS
    mask = _paint_mask(kp, verified, img.shape) if is_forged else np.zeros(img.shape, np.uint8)

    return {
        "is_forged": is_forged,
        "mask": mask,
        "keypoints": kp,
        "raw_pairs": raw_pairs,
        "inliers": verified,
        "clustered": verified,
        "exclude": exclude,
        "info": {"ratios": ratios,
                 "n_dbscan": cstats.get("n_dbscan", 0),
                 "clusters": cstats.get("clusters", [])},
    }
