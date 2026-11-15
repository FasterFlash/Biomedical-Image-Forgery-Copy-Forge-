"""
features.py  --  Turn ONE detector run into a flat row of numbers (V5.5).

Why: V5 decides "forged" from a single number (confirmed matches >= 3), and that number alone
separates forged from authentic only at AUC ~0.83. Authentic false positives and real copy-moves
differ in other ways the count ignores -- how repetitive the texture is, whether matches are
mutual, whether they all share ONE displacement, how far apart source and target are, how
confident the matches were. These features expose that, so a small classifier can use it.

Every feature is a plain number. Missing/undefined values use a fixed sentinel (documented per
feature) instead of NaN, so any model can read them.

Names ending in nothing special are per-image; the extractor adds the mask mode (on/off) as a
separate column, and cv_decision.py suffixes the feature names with _on / _off.
"""

import math

import cv2
import numpy as np

import config

# ---------------------------------------------------------------------------------------------
def _bbox(pts):
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return float(x0), float(y0), float(x1), float(y1)


def _bbox_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _bbox_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def _vec_spread(d, lengths):
    """Spread of displacement vectors after removing direction ambiguity (A->B vs B->A).
    ~0  : every match moves by the same vector (a true translated copy)
    large: many different offsets (repeating texture / lattice)."""
    if len(d) < 2:
        return 0.0
    sign = np.where((d[:, 0] < 0) | ((d[:, 0] == 0) & (d[:, 1] < 0)), -1.0, 1.0)[:, None]
    o = d * sign
    return float(math.sqrt(float(np.var(o[:, 0]) + np.var(o[:, 1]))) / (np.median(lengths) + 1e-9))


# ---------------------------------------------------------------------------------------------
def empty_features():
    """All feature names with their 'nothing found' values. compute() overwrites these, so the
    key set is identical for every image."""
    return {
        # --- image-level (independent of the detector) ---
        "h": 0.0, "w": 0.0,
        "log_mpx": 0.0,            # log10 of pixel count
        "gray_mean": 0.0, "gray_std": 0.0,
        "sat_mean": 0.0,           # mean HSV saturation 0..1  (colour photo vs grey microscopy)
        "is_gray": 0.0,            # 1 if the three colour channels are (almost) identical
        "edge_density": 0.0,       # fraction of Canny edge pixels (low-res)
        # --- keypoints / mask ---
        "n_kp": 0.0, "kp_per_mpx": 0.0,
        "kp_capped": 0.0,          # 1 if SIFT hit the MAX_KEYPOINTS cap
        "excl_frac": 0.0,          # fraction of the image removed by the V5 mask (0 when mask off)
        # --- raw matches (before RANSAC) ---
        "n_raw": 0.0,
        "raw_per_kp": 0.0,         # share of keypoints with a confident duplicate: repetitiveness
        "raw_ratio_mean": 1.0,     # mean Lowe ratio of raw matches (sentinel 1.0 = none; lower = sharper)
        "raw_ratio_q25": 1.0,
        # --- confirmed matches (after cluster + RANSAC) ---
        "n_inl": 0.0,
        "inl_per_raw": 0.0,        # how much of the raw evidence survives geometric checking
        "inl_ratio_mean": 1.0, "inl_ratio_q75": 1.0,
        "mutual_frac": 0.0,        # fraction of confirmed matches whose reverse (j->i) is also a raw match
        "uniq_pts_frac": 0.0,      # unique keypoints / (2 * confirmed): low = many-to-one chains
        "both_roles_frac": 0.0,    # keypoints that are both a source and a target (lattice chains)
        # --- clusters ---
        "n_dbscan": 0.0, "n_cl_tested": 0.0, "n_cl_passed": 0.0,
        "largest_inl": 0.0, "second_inl": 0.0,
        "top_share": 0.0,          # largest cluster / all confirmed matches
        # --- displacement of confirmed matches (normalised by image diagonal) ---
        "disp_med": 0.0, "disp_min": 0.0, "disp_max": 0.0, "disp_cv": 0.0,
        "vec_spread": 0.0,         # see _vec_spread, over ALL confirmed matches
        "vec_spread_lg": 0.0,      # same, inside the largest cluster
        # --- geometry of the largest verified cluster ---
        "lg_scale_dev": 0.0,       # |ln scale|  (0 = pure translation/rotation)
        "lg_rot": 0.0,             # |rotation| in degrees
        "lg_src_bbox": 0.0,        # source-region bbox / image area
        "lg_dst_bbox": 0.0,        # target-region bbox / image area
        "lg_sd_dist": 0.0,         # source-target centroid distance / diagonal
        "lg_sd_iou": 0.0,          # IoU of source and target bboxes (high = neighbouring repeats)
        "lg_density": 0.0,         # log10(1 + matches per pixel^2 * 1e4) in the source bbox
        "all_bbox": 0.0,           # bbox of every confirmed point / image area
    }


def _image_stats(gray, color, f):
    h, w = gray.shape[:2]
    f["h"], f["w"] = float(h), float(w)
    f["log_mpx"] = math.log10(max(1.0, h * w))
    s = 256.0 / max(h, w)
    small = cv2.resize(gray, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA) \
        if s < 1.0 else gray
    f["gray_mean"] = float(small.mean())
    f["gray_std"] = float(small.std())
    f["edge_density"] = float((cv2.Canny(small, 80, 160) > 0).mean())
    if color is not None and color.ndim == 3:
        cs = cv2.resize(color, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
        f["sat_mean"] = float(cv2.cvtColor(cs, cv2.COLOR_BGR2HSV)[..., 1].mean() / 255.0)
        spread = np.abs(cs[..., 0].astype(np.int16) - cs[..., 2].astype(np.int16)).max()
        f["is_gray"] = float(spread < 8)
    else:
        f["is_gray"] = 1.0


def compute(gray, color, result):
    """gray/color: the loaded image; result: the dict returned by pipeline.detect()."""
    f = empty_features()
    _image_stats(gray, color, f)

    kp = result["keypoints"]
    raw = result["raw_pairs"]
    inl = result["inliers"]
    info = result["info"]
    ratios = info["ratios"]
    h, w = gray.shape[:2]
    diag = math.hypot(h, w)
    area = float(h * w)

    n_kp, n_raw, n_inl = len(kp), len(raw), len(inl)
    f["n_kp"] = float(n_kp)
    f["kp_per_mpx"] = n_kp / (area / 1e6)
    f["kp_capped"] = float(n_kp >= 0.98 * config.MAX_KEYPOINTS)
    excl = result.get("exclude")
    f["excl_frac"] = float((excl > 0).mean()) if excl is not None else 0.0

    f["n_raw"] = float(n_raw)
    f["raw_per_kp"] = n_raw / n_kp if n_kp else 0.0
    if n_raw:
        rr = np.array([ratios[p] for p in raw if p in ratios], dtype=np.float64)
        if len(rr):
            f["raw_ratio_mean"] = float(rr.mean())
            f["raw_ratio_q25"] = float(np.quantile(rr, 0.25))

    f["n_inl"] = float(n_inl)
    f["n_dbscan"] = float(info["n_dbscan"])
    tested = info["clusters"]
    passed = sorted([c for c in tested if c["passed"]], key=lambda c: -c["n_inl"])
    f["n_cl_tested"] = float(len(tested))
    f["n_cl_passed"] = float(len(passed))

    if n_inl == 0:
        return f

    f["inl_per_raw"] = n_inl / n_raw if n_raw else 0.0
    ir = np.array([ratios[p] for p in inl if p in ratios], dtype=np.float64)
    if len(ir):
        f["inl_ratio_mean"] = float(ir.mean())
        f["inl_ratio_q75"] = float(np.quantile(ir, 0.75))

    raw_set = set(raw)
    f["mutual_frac"] = sum((j, i) in raw_set for i, j in inl) / n_inl
    srcs = {i for i, _ in inl}
    dsts = {j for _, j in inl}
    pts_all = srcs | dsts
    f["uniq_pts_frac"] = len(pts_all) / (2.0 * n_inl)
    f["both_roles_frac"] = len(srcs & dsts) / len(pts_all)

    P = np.array([kp[i].pt for i, _ in inl], dtype=np.float64)
    Q = np.array([kp[j].pt for _, j in inl], dtype=np.float64)
    d = Q - P
    L = np.hypot(d[:, 0], d[:, 1])
    f["disp_med"] = float(np.median(L) / diag)
    f["disp_min"] = float(L.min() / diag)
    f["disp_max"] = float(L.max() / diag)
    f["disp_cv"] = float(L.std() / (L.mean() + 1e-9))
    f["vec_spread"] = _vec_spread(d, L)
    allpts = np.vstack([P, Q])
    f["all_bbox"] = _bbox_area(_bbox(allpts)) / area

    if passed:
        lg = passed[0]
        f["largest_inl"] = float(lg["n_inl"])
        f["second_inl"] = float(passed[1]["n_inl"]) if len(passed) > 1 else 0.0
        f["top_share"] = lg["n_inl"] / n_inl
        if lg["scale"] > 0:
            f["lg_scale_dev"] = abs(math.log(lg["scale"]))
        f["lg_rot"] = abs(lg["rot_deg"])
        cp = lg["pairs"]
        cP = np.array([kp[i].pt for i, _ in cp], dtype=np.float64)
        cQ = np.array([kp[j].pt for _, j in cp], dtype=np.float64)
        cd = cQ - cP
        cL = np.hypot(cd[:, 0], cd[:, 1])
        f["vec_spread_lg"] = _vec_spread(cd, cL)
        bs, bd = _bbox(cP), _bbox(cQ)
        f["lg_src_bbox"] = _bbox_area(bs) / area
        f["lg_dst_bbox"] = _bbox_area(bd) / area
        f["lg_sd_dist"] = float(np.hypot(*(cQ.mean(axis=0) - cP.mean(axis=0))) / diag)
        f["lg_sd_iou"] = _bbox_iou(bs, bd)
        f["lg_density"] = math.log10(1.0 + 1e4 * len(cp) / (_bbox_area(bs) + 100.0))

    return f


FEATURE_NAMES = list(empty_features().keys())
