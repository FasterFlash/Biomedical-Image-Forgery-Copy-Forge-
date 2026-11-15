"""
config.py  —  All paths and tunable parameters live here, nowhere else.

Why: when your dataset moves or you tweak a threshold, you change it ONCE here,
not in five different scripts. This is the single source of truth.

EDIT the DATASET_ROOT below to point at your actual dataset folder.
"""

import os

# ============================================================
#  PATHS  —  edit DATASET_ROOT to match your machine
# ============================================================
# Point this at the folder that contains train_images/, train_masks/, etc.
DATASET_ROOT = os.path.join("..", "dataset")

# These are derived from DATASET_ROOT — usually no need to change.
FORGED_DIR      = os.path.join(DATASET_ROOT, "train_images", "forged")
AUTHENTIC_DIR   = os.path.join(DATASET_ROOT, "train_images", "authentic")
MASK_DIR        = os.path.join(DATASET_ROOT, "train_masks")
TEST_DIR        = os.path.join(DATASET_ROOT, "test_images")

# Where to save outputs (visualizations, results)
OUTPUT_DIR = "results"


# ============================================================
#  SIFT + MATCHING PARAMETERS  —  tune these later
# --- SIFT sensitivity (the 2nd-place trick for low-texture bio images) ---
# Default OpenCV SIFT contrast_threshold is 0.04. Biological images (blots,
# smooth regions) have very little texture, so the default finds almost no
# keypoints. The 2nd-place solution dropped this ~40x to flood the image with
# keypoints. Lower = MANY more keypoints (helps smooth images, but noisier).
SIFT_CONTRAST_THRESHOLD = 0.001

# Upscale small images before feature extraction (also from 2nd place).
# Small images yield too few keypoints; upscaling 4x gives SIFT more to grab,
# then we map coordinates back to original scale. Images with the smaller
# dimension below this size get upscaled.
UPSCALE_IF_SMALLER_THAN = 1024
UPSCALE_FACTOR = 4

# Cap the number of keypoints per image. The sensitive contrast_threshold can
# produce 200k+ keypoints on textured images (corn), making matching crawl.
# SIFT returns keypoints by response strength, so keeping the top-N keeps the
# strongest ones. This bounds runtime for the batch. Raise if you want more
# thoroughness and can afford the time.
MAX_KEYPOINTS = 8000

# Lowe's ratio test: lower = stricter matching (fewer, more confident matches)
LOWE_RATIO = 0.75

# Minimum pixel distance between two matched points to count as copy-move.
# (Points too close together are probably the same feature, not a real copy.)
MIN_MATCH_DISTANCE = 10

# RANSAC: max pixel error allowed when fitting the geometric transform.
# Lower = stricter (matches must agree more tightly on the transform).
RANSAC_THRESHOLD = 8.0

# Minimum number of consistent matches needed to declare a real forgery.
# (A real copied region produces MANY points sharing one transform;
#  random false matches don't cluster like this.)
# Lowered to 3 so sparse-keypoint blots can still register a detection.
MIN_INLIERS = 3

# Radius (pixels) painted around each matched point when building the mask.
MASK_RADIUS = 15

# --- False-positive reduction: clustering (the key precision fix) ---
# After RANSAC, matched points must form DENSE clusters to be believed. A real
# copied region packs many points close together; false matches are scattered
# loners. We use DBSCAN density clustering to keep dense groups and drop the
# scattered ones — this is what kills false positives on authentic images.
#
# DBSCAN_EPS: max pixel distance for two points to be "neighbors" (cluster radius)
# DBSCAN_MIN_SAMPLES: min points needed to form a cluster (below this = noise)
# MIN_CLUSTER_POINTS: a surviving cluster must have at least this many points
#                     to count as a real forgery region (higher = stricter)
DBSCAN_EPS = 30
DBSCAN_MIN_SAMPLES = 5
MIN_CLUSTER_POINTS = 8

# Geometric plausibility: reject matches implying absurd transforms.
# A real copy-paste has a sane scale; a 5x+ scale is almost always a false match.
MAX_PLAUSIBLE_SCALE = 4.0


# Fixed seed applied at the start of every detect() call, so results do not depend on
# which images were processed earlier (RANSAC / FLANN are randomised otherwise).
RANDOM_SEED = 0


# ============================================================
#  V5  --  EXCLUSION OF NON-BIOLOGICAL REGIONS (text, scale bars, rulers)
# ============================================================
# Master switch. False = V4 behaviour exactly (no mask is built, SIFT runs as before).
EXCLUDE_ENABLED = False   # V6: the chosen model uses mask OFF (see version5.5 results)

# Which rules to run. Remove names to ablate one at a time.
# Options: "ocr", "text", "corner", "scalebar", "ruler", "card"
#   "ocr" = a small pre-trained text detector (RapidOCR / ONNX). Needs:  pip install rapidocr-onnxruntime
#           It sees the COLOUR image, so it also finds the orange label tags that are
#           invisible in grayscale.
EXCLUDE_RULES = ("ocr", "text", "corner", "scalebar", "ruler", "card")

# Detection runs on a copy whose long side is at most this many px, so the same
# thresholds work for 650px SEM figures and 3900px photos.
EXC_WORK_LONG_SIDE = 1000

# A pixel only counts as "text-like / bar-like" if it differs from its local mean
# by MORE than this (0-255). High on purpose: biology (kernels, cells) is low
# contrast, printed text and scale bars are not.
EXC_CONTRAST = 45
EXC_BLOCK = 51                 # local-mean window (odd). Must exceed glyph size.
EXC_MAX_CANDIDATES = 4000      # skip a rule on pathological, texture-flooded images

# --- text lines ---
EXC_GLYPH_MIN = 6              # min glyph height (px, working resolution)
EXC_GLYPH_MAX_FRAC = 0.12      # max glyph height as fraction of the long image side
EXC_TEXT_GAP = 1.3             # max gap between glyphs, in glyph heights
EXC_TEXT_MIN_GLYPHS = 3        # need this many aligned glyphs to call it text
EXC_TEXT_LINE_THICK = 1.8      # a text line is at most this many glyph-heights thick
EXC_TEXT_LINE_ASPECT = 1.8     # ...and at least this many times longer than thick
EXC_TEXT_BG_STD = 20           # text sits on a flat background: non-glyph pixel std must be below this
EXC_TEXT_BG_STD_EDGE = 40      # ...but SEM info bars sit on noisy dark image: be lenient inside the edge band
EXC_TEXT_EDGE_BAND = 0.15      # top / bottom fraction of the image where captions and info bars live

# --- OCR text detector (rule "ocr") ---
EXC_OCR_LONG_SIDE = 1200       # image is shrunk to this before detection (never enlarged)
EXC_OCR_BOX_THRESH = 0.3
EXC_OCR_UNCLIP = 1.8           # grows each text box a little so it covers the whole word
EXC_OCR_MIN_SCORE = 0.3        # junk boxes (on cells, kernels) come back with score ~0 and no text

# --- corner panel labels ("b", "(c)", "C") ---
EXC_CORNER_FRAC = 0.22         # corner zone size (fraction of width / height)
EXC_CORNER_GLYPH_MIN = 10
EXC_CORNER_GLYPH_MAX_FRAC = 0.18
EXC_CORNER_MAX_BLOBS = 6       # more blobs than this in a corner = texture, not a label

# --- scale bars ---
EXC_BAR_MIN_LEN = 20
EXC_BAR_ASPECT = 4.0           # length / thickness
EXC_BAR_MAX_THICK = 30
EXC_BAR_FILL = 0.85
EXC_CAPTION_PAD = 40           # px of caption area added around a bar

# --- rulers / checkerboards / colour chips ---
EXC_RULER_CONTRAST = 25        # lower than EXC_CONTRAST: chains of squares are identified by shape
EXC_RULER_MIN = 4
EXC_RULER_MAX = 60
EXC_RULER_FILL = 0.82
EXC_RULER_SIZE_CV = 0.22       # max spread of square sizes inside one chain (std/mean)
EXC_RULER_GAP = 2.5            # max gap between squares, in square sizes
EXC_RULER_MIN_CHAIN = 5
EXC_RULER_CLOSE = 61           # px, joins broken pieces of ONE ruler strip (thin kernel, along the strip only)

# --- bright label cards ---
EXC_CARD_BRIGHT = 215
EXC_CARD_MIN_FRAC = 0.0015
EXC_CARD_MAX_FRAC = 0.06
EXC_CARD_FILL = 0.88

# Safety margin dilated around every excluded box. SIFT descriptors look at a
# patch around each keypoint, so a keypoint just outside a letter still "sees" it.
EXC_MARGIN_MIN = 6
EXC_MARGIN_FRAC = 0.012


# ============================================================
#  helper: make sure output dir exists
# ============================================================
def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR