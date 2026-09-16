"""
utils.py  —  Reusable helpers used across the whole project.

Everything that loads data or computes a score lives here, so every script
uses the SAME logic. If you fix a bug here, it's fixed everywhere.

Functions:
  load_image(path)            -> grayscale image
  load_mask(path, shape)      -> binary {0,1} mask, resized to match image
  score_masks(pred, truth)    -> dict with F1, IoU, precision, recall, MCC
  list_image_ids(folder)      -> list of image numbers available in a folder
"""

import os
import cv2
import numpy as np


def load_image(path):
    """Load an image as grayscale. Raises a clear error if not found."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(
            f"Could not read image: {path}\n"
            f"  -> check the filename, extension (.png/.jpg?), and path in config.py"
        )
    return img


def load_mask(path, target_shape=None):
    """
    Load a .npy ground-truth mask.
    Dataset masks are shape (1, H, W), values {0,1}. We squeeze to (H, W).
    If target_shape is given and differs, resize (nearest-neighbor to keep 0/1).
    Returns None if the mask file doesn't exist (e.g. authentic images).
    """
    if not os.path.exists(path):
        return None
    mask = np.load(path, allow_pickle=True)
    mask = np.asarray(mask)

    # Force the mask down to a clean 2D (H, W) array, no matter how the .npy
    # stored it. Some masks are (1,H,W), some (H,W,1), some already (H,W).
    # np.squeeze removes ALL singleton dims; if anything odd remains, take the
    # first 2D slice we can find.
    mask = np.squeeze(mask)
    while mask.ndim > 2:
        mask = mask[0]                          # peel leading dims until 2D
    if mask.ndim < 2:
        return None                             # malformed; skip gracefully

    mask = (mask > 0).astype(np.uint8)          # force clean binary
    if target_shape is not None and mask.shape != target_shape:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    return mask


def score_masks(pred, truth):
    """
    Compare predicted forgery mask vs ground-truth mask.
    Returns F1, IoU, precision, recall, and MCC (Matthews Correlation Coefficient).

    MCC is included because it's what the SOTA (Integscan) reports, so your
    numbers are directly comparable to theirs. It's robust to the fact that
    forged pixels are a tiny fraction of the image.
    """
    if truth is None:
        return None

    pred_b = pred.astype(bool)
    true_b = truth.astype(bool)

    tp = int(np.logical_and(pred_b,  true_b).sum())
    fp = int(np.logical_and(pred_b, ~true_b).sum())
    fn = int(np.logical_and(~pred_b, true_b).sum())
    tn = int(np.logical_and(~pred_b, ~true_b).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1  = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    # MCC — guard against divide-by-zero using float math
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0

    return {
        "F1": round(f1, 4),
        "IoU": round(iou, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "MCC": round(mcc, 4),
    }


def list_image_ids(folder, extension=".png"):
    """
    Return the list of image ID strings (filenames without extension) in a folder.
    e.g. ['414', '1027', ...]. Lets you loop over the dataset easily.
    """
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"Folder not found: {folder} (check config.py)")
    ids = []
    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith(extension):
            ids.append(os.path.splitext(fname)[0])
    return ids