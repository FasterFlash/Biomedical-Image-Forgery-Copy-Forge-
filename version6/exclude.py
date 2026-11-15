"""
exclude.py  --  V5: find NON-BIOLOGICAL regions (text, scale bars, rulers, label
cards) so SIFT never spends keypoints on them.

Classical OpenCV + SciPy only (no OCR model, no downloads). Every rule is a
simple, explainable heuristic so each one can be ablated and defended:

  ocr       a pre-trained text detector (RapidOCR) run on the COLOUR image
  text      glyph-sized, high-contrast blobs that line up in a row/column
  corner    isolated high-contrast glyphs in an image corner (panel letters "b", "(c)")
  scalebar  long thin solid bars (+ the caption sitting next to them)
  ruler     chains of similar, aligned rectangles (checkerboard rulers, colour chips)
  card      bright, solid, rectangular label cards (the white tag with "CAJ 28")

Public API:
    build_exclusion_mask(gray, rules=ALL_RULES) -> (mask01, parts)
        mask01 : uint8 (H, W), 1 = EXCLUDE this pixel
        parts  : dict rule -> uint8 (H, W) mask of what that rule found (for debugging
                 and ablation)

All thresholds live in config.py (EXC_*), nothing is hard-coded here.
"""

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import config

ALL_RULES = ("ocr", "text", "corner", "scalebar", "ruler", "card")


# --------------------------------------------------------------------------
#  helpers
# --------------------------------------------------------------------------
def _work_copy(gray):
    """Downscale big images so thresholds mean the same thing on 650px and 3900px
    images. Never upscale. Returns (work_image, scale = work/orig)."""
    h, w = gray.shape[:2]
    long_side = max(h, w)
    if long_side <= config.EXC_WORK_LONG_SIDE:
        return gray, 1.0
    s = config.EXC_WORK_LONG_SIDE / float(long_side)
    work = cv2.resize(gray, (max(1, int(w * s)), max(1, int(h * s))),
                      interpolation=cv2.INTER_AREA)
    return work, s


def _contrast_blobs(work, bright, k=None):
    """Binary map of blobs that differ STRONGLY from their local mean.
    bright=True  -> pixels much brighter than neighbourhood (white text, white bars)
    bright=False -> pixels much darker than neighbourhood (black text, dark bars)
    The strong threshold is what keeps textured biology (kernels, cells) out."""
    blur = cv2.GaussianBlur(work, (3, 3), 0)
    k = config.EXC_CONTRAST if k is None else k
    if bright:
        return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, config.EXC_BLOCK, -k)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, config.EXC_BLOCK, k)


def _components(binary):
    """connected components -> array of [x, y, w, h, area] (background removed)."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return stats[1:] if n > 1 else np.zeros((0, 5), dtype=np.int32)


def _group(boxes, axis, max_gap_factor, min_members, size_ratio=(0.5, 2.0),
           align_tol=0.6):
    """
    Link boxes that sit in one ROW (axis='x') or COLUMN (axis='y') with similar size
    and a small gap; return list of member-index arrays for groups >= min_members.
    axis='x' -> boxes share a horizontal baseline (a text line / horizontal ruler)
    axis='y' -> boxes share a vertical line     (a vertical ruler / rotated text)
    """
    n = len(boxes)
    if n < min_members:
        return []
    x, y, w, h = (boxes[:, 0].astype(float), boxes[:, 1].astype(float),
                  boxes[:, 2].astype(float), boxes[:, 3].astype(float))
    cx, cy = x + w / 2.0, y + h / 2.0
    size = np.maximum(w, h)
    tree = cKDTree(np.c_[cx, cy])
    reach = float(np.max(size)) * (max_gap_factor + 1.0)
    pairs = tree.query_pairs(r=reach, output_type="ndarray")
    if len(pairs) == 0:
        return []
    i, j = pairs[:, 0], pairs[:, 1]
    sz = np.maximum(size[i], size[j])
    ratio = size[i] / size[j]
    ok = (ratio >= size_ratio[0]) & (ratio <= size_ratio[1])
    if axis == "x":
        perp = np.abs(cy[i] - cy[j])
        along = np.abs(cx[i] - cx[j]) - (w[i] + w[j]) / 2.0     # gap between boxes
    else:
        perp = np.abs(cx[i] - cx[j])
        along = np.abs(cy[i] - cy[j]) - (h[i] + h[j]) / 2.0
    ok &= perp <= align_tol * sz
    ok &= along <= max_gap_factor * sz
    i, j = i[ok], j[ok]
    if len(i) == 0:
        return []
    g = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    groups = []
    for l in np.unique(lab):
        idx = np.where(lab == l)[0]
        if len(idx) >= min_members:
            groups.append(idx)
    return groups


def _paint_boxes(shape, boxes, pad=0):
    m = np.zeros(shape, np.uint8)
    for b in boxes:
        bx, by, bw, bh = b[0], b[1], b[2], b[3]          # boxes may carry an area column
        x0, y0 = max(0, int(bx - pad)), max(0, int(by - pad))
        x1, y1 = min(shape[1], int(bx + bw + pad)), min(shape[0], int(by + bh + pad))
        m[y0:y1, x0:x1] = 1
    return m


def _union_bbox(boxes):
    x0 = boxes[:, 0].min()
    y0 = boxes[:, 1].min()
    x1 = (boxes[:, 0] + boxes[:, 2]).max()
    y1 = (boxes[:, 1] + boxes[:, 3]).max()
    return np.array([[x0, y0, x1 - x0, y1 - y0]])


# --------------------------------------------------------------------------
#  rules
# --------------------------------------------------------------------------
def _rule_text(work, blobs):
    """Glyph-like blobs that line up -> text lines."""
    H, W = work.shape
    hi = int(config.EXC_GLYPH_MAX_FRAC * max(H, W))
    lo = config.EXC_GLYPH_MIN
    out = np.zeros(work.shape, np.uint8)
    glyph_any = (blobs[0] > 0) | (blobs[1] > 0)          # all strong-contrast pixels
    for binary in blobs:
        c = _components(binary)
        if len(c) == 0:
            continue
        bw, bh, area = c[:, 2], c[:, 3], c[:, 4]
        fill = area / (bw * bh + 1e-6)
        keep = ((bh >= lo) & (bh <= hi) & (bw >= 2) & (bw <= hi * 1.5) &
                (bw / np.maximum(bh, 1) <= 3.0) & (bw / np.maximum(bh, 1) >= 0.08) &
                (fill >= 0.12) & (fill <= 0.92))
        c = c[keep]
        if len(c) > config.EXC_MAX_CANDIDATES:        # pathological texture -> skip
            continue
        for axis in ("x", "y"):
            for idx in _group(c, axis, config.EXC_TEXT_GAP, config.EXC_TEXT_MIN_GLYPHS):
                bb = _union_bbox(c[idx])
                gw, gh = bb[0, 2], bb[0, 3]
                # A text LINE is one glyph thick and several glyphs long. A textured
                # blob (corn kernels, cell outlines) chains into a fat 2-D region -> reject.
                if axis == "x":
                    thick, length, glyph = gh, gw, np.median(c[idx][:, 3])
                else:
                    thick, length, glyph = gw, gh, np.median(c[idx][:, 2])
                if thick > config.EXC_TEXT_LINE_THICK * glyph:
                    continue
                if length < config.EXC_TEXT_LINE_ASPECT * thick:
                    continue
                # Printed text sits on a FLAT background (card, SEM black, plain slide).
                # Kernels / cells sit inside texture. Measure the texture of everything
                # in the box that is NOT a glyph; textured -> it is biology, not text.
                x0, y0 = int(bb[0, 0]), int(bb[0, 1])
                x1, y1 = x0 + int(gw), y0 + int(gh)
                patch = work[y0:y1, x0:x1]
                bg = patch[~glyph_any[y0:y1, x0:x1]]
                gy = y0 + gh / 2.0
                band = config.EXC_TEXT_EDGE_BAND * H
                limit = (config.EXC_TEXT_BG_STD_EDGE if (gy < band or gy > H - band)
                         else config.EXC_TEXT_BG_STD)
                if bg.size >= 20 and bg.std() > limit:
                    continue
                out |= _paint_boxes(work.shape, bb)
    return out


def _rule_corner(work, blobs):
    """Panel letters / stray labels: isolated strong glyphs in a corner zone."""
    H, W = work.shape
    zh, zw = int(config.EXC_CORNER_FRAC * H), int(config.EXC_CORNER_FRAC * W)
    zones = [(0, 0), (W - zw, 0), (0, H - zh), (W - zw, H - zh)]
    hi = int(config.EXC_CORNER_GLYPH_MAX_FRAC * max(H, W))
    out = np.zeros(work.shape, np.uint8)
    for binary in blobs:
        c = _components(binary)
        if len(c) == 0:
            continue
        for zx, zy in zones:
            inz = ((c[:, 0] >= zx) & (c[:, 1] >= zy) &
                   (c[:, 0] + c[:, 2] <= zx + zw) & (c[:, 1] + c[:, 3] <= zy + zh))
            z = c[inz]
            z = z[(z[:, 3] >= config.EXC_CORNER_GLYPH_MIN) & (z[:, 3] <= hi) &
                  (z[:, 2] <= hi * 2)]
            # a real label is a handful of blobs, a textured corner is hundreds
            if 0 < len(z) <= config.EXC_CORNER_MAX_BLOBS:
                out |= _paint_boxes(work.shape, z)
    return out


def _rule_scalebar(work, blobs):
    """Long thin solid bars, plus the caption area above/below them."""
    H, W = work.shape
    out = np.zeros(work.shape, np.uint8)
    for binary in blobs:
        c = _components(binary)
        if len(c) == 0:
            continue
        bw, bh, area = c[:, 2], c[:, 3], c[:, 4]
        fill = area / (bw * bh + 1e-6)
        horiz = (bw >= config.EXC_BAR_MIN_LEN) & (bw / np.maximum(bh, 1) >= config.EXC_BAR_ASPECT) & \
                (bh >= 2) & (bh <= config.EXC_BAR_MAX_THICK) & (bw <= 0.45 * W)
        vert = (bh >= config.EXC_BAR_MIN_LEN) & (bh / np.maximum(bw, 1) >= config.EXC_BAR_ASPECT) & \
               (bw >= 2) & (bw <= config.EXC_BAR_MAX_THICK) & (bh <= 0.45 * H)
        sel = c[(horiz | vert) & (fill >= config.EXC_BAR_FILL)]
        for bx, by, bbw, bbh, _a in sel:
            # grow to cover the caption ("200 nm", "2 um") that sits beside a bar
            grow_x = max(10, int(0.3 * bbw))
            grow_y = max(config.EXC_CAPTION_PAD, int(0.6 * bbw)) if bbw >= bbh else max(10, int(0.3 * bbh))
            x0, x1 = max(0, bx - grow_x), min(W, bx + bbw + grow_x)
            if bbw >= bbh:
                y0, y1 = max(0, by - grow_y), min(H, by + bbh + config.EXC_CAPTION_PAD // 2)
            else:
                y0, y1 = max(0, by - grow_x), min(H, by + bbh + grow_x)
                x0, x1 = max(0, bx - grow_y), min(W, bx + bbw + grow_y)
            out[y0:y1, x0:x1] = 1
    return out


def _rule_ruler(work, blobs):
    """Chains of similar aligned rectangles = checkerboard ruler / colour chips."""
    out = np.zeros(work.shape, np.uint8)
    for binary in blobs:
        c = _components(binary)
        if len(c) == 0:
            continue
        bw, bh, area = c[:, 2], c[:, 3], c[:, 4]
        fill = area / (bw * bh + 1e-6)
        asp = bw / np.maximum(bh, 1)
        keep = ((bw >= config.EXC_RULER_MIN) & (bh >= config.EXC_RULER_MIN) &
                (bw <= config.EXC_RULER_MAX) & (bh <= config.EXC_RULER_MAX) &
                (fill >= config.EXC_RULER_FILL) & (asp >= 0.4) & (asp <= 2.5))
        c = c[keep]
        if len(c) > config.EXC_MAX_CANDIDATES or len(c) < config.EXC_RULER_MIN_CHAIN:
            continue
        for axis in ("x", "y"):
            for idx in _group(c, axis, config.EXC_RULER_GAP, config.EXC_RULER_MIN_CHAIN,
                              size_ratio=(0.6, 1.7), align_tol=0.35):
                m = c[idx]
                # ruler squares are machine-printed: near-identical size. Kernels vary.
                cv_w = m[:, 2].std() / m[:, 2].mean()
                cv_h = m[:, 3].std() / m[:, 3].mean()
                if cv_w > config.EXC_RULER_SIZE_CV or cv_h > config.EXC_RULER_SIZE_CV:
                    continue
                bb = _union_bbox(m)
                out |= _paint_boxes(work.shape, bb, pad=2)
    # bridge small breaks inside one ruler strip (a square whose contrast was too low)
    k = config.EXC_RULER_CLOSE
    horiz = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((5, k), np.uint8))   # along x
    vert = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((k, 5), np.uint8))    # along y
    return horiz | vert


def _rule_card(work):
    """Bright, solid, rectangular label cards (white tag with a code on it)."""
    H, W = work.shape
    _, bw_img = cv2.threshold(work, config.EXC_CARD_BRIGHT, 255, cv2.THRESH_BINARY)
    # close the dark text printed ON the card so the card is one solid shape
    bw_img = cv2.morphologyEx(bw_img, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    cnts, _ = cv2.findContours(bw_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(work.shape, np.uint8)
    for cnt in cnts:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 15 or bh < 10:
            continue
        frac = (bw * bh) / float(H * W)
        if not (config.EXC_CARD_MIN_FRAC <= frac <= config.EXC_CARD_MAX_FRAC):
            continue
        fill = cv2.contourArea(cnt) / float(bw * bh)       # filled area, holes ignored
        if fill >= config.EXC_CARD_FILL and max(bw / bh, bh / bw) <= 12:
            out[y:y + bh, x:x + bw] = 1
    return out


_OCR = None


def _ocr_engine():
    global _OCR
    if _OCR is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as e:
            raise ImportError(
                "Rule 'ocr' could not load the text detector.\n"
                f"  Real error: {type(e).__name__}: {e}\n"
                "  If the real error says the module is missing:  pip install rapidocr-onnxruntime\n"
                "  If it says 'DLL load failed' / 'initialization routine failed': memory or a missing\n"
                "  Visual C++ runtime, not a missing package.\n"
                "  Or remove 'ocr' from EXCLUDE_RULES in config.py") from e
        _OCR = RapidOCR()
    return _OCR


def _rule_ocr(color, work_shape, s_work):
    """Text boxes from a pre-trained detector. Runs on the colour image: orange label
    tags on a green board are nearly invisible in grayscale but obvious in colour.
    Junk boxes (on cells / kernels) come back with empty text and score ~0 -> dropped."""
    h, w = color.shape[:2]
    s = min(1.0, config.EXC_OCR_LONG_SIDE / float(max(h, w)))
    img = color if s == 1.0 else cv2.resize(color, (max(1, int(w * s)), max(1, int(h * s))),
                                            interpolation=cv2.INTER_AREA)
    res, _ = _ocr_engine()(img, use_det=True, use_cls=False, use_rec=True, text_score=0.0,
                           box_thresh=config.EXC_OCR_BOX_THRESH,
                           unclip_ratio=config.EXC_OCR_UNCLIP)
    out = np.zeros(work_shape, np.uint8)
    f = s_work / s                                   # detector coords -> working coords
    for box, text, score in (res or []):
        if float(score) < config.EXC_OCR_MIN_SCORE or not str(text).strip():
            continue
        pts = (np.array(box, np.float32) * f).astype(np.int32)
        cv2.fillPoly(out, [pts], 1)
    return out


# --------------------------------------------------------------------------
#  public entry point
# --------------------------------------------------------------------------
def build_exclusion_mask(gray, rules=ALL_RULES, color=None):
    """
    gray  : uint8 grayscale image (H, W)
    color : optional BGR image of the same size (used by rule "ocr"; if missing, the
            gray image is used, which finds less)
    Returns (mask01, parts):
      mask01 : uint8 (H, W), 1 = exclude (text/scale/ruler/card + safety margin)
      parts  : {rule: uint8 (H, W)} what each rule found, in ORIGINAL resolution
    """
    h, w = gray.shape[:2]
    work, s = _work_copy(gray)
    blobs = (_contrast_blobs(work, True), _contrast_blobs(work, False))

    found = {}
    if "ocr" in rules:
        if color is None:
            color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        found["ocr"] = _rule_ocr(color, work.shape, s)
    if "text" in rules:
        found["text"] = _rule_text(work, blobs)
    if "corner" in rules:
        found["corner"] = _rule_corner(work, blobs)
    if "scalebar" in rules:
        found["scalebar"] = _rule_scalebar(work, blobs)
    if "ruler" in rules:
        # rulers are picked out by SHAPE (chains of squares), so a lower contrast bar is fine
        rblobs = (_contrast_blobs(work, True, config.EXC_RULER_CONTRAST),
                  _contrast_blobs(work, False, config.EXC_RULER_CONTRAST))
        found["ruler"] = _rule_ruler(work, rblobs)
    if "card" in rules:
        found["card"] = _rule_card(work)

    margin = max(config.EXC_MARGIN_MIN, int(config.EXC_MARGIN_FRAC * max(work.shape)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
    parts, total = {}, np.zeros(work.shape, np.uint8)
    for name, m in found.items():
        m = cv2.dilate(m, k)
        total |= m
        parts[name] = (m if s == 1.0 else
                       cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST))
    mask01 = total if s == 1.0 else cv2.resize(total, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask01, parts
