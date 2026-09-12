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


# ============================================================
#  helper: make sure output dir exists
# ============================================================
def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR