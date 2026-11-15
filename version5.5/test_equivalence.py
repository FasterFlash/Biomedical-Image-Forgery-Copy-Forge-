"""
test_equivalence.py  --  Prove that V5.5's detector gives the SAME answers as V5's.

Runs the V5 pipeline.py and the V5.5 pipeline.py on the images in a folder, mask ON and OFF, and
compares verdict, number of confirmed matches and the confirmed pair list. Any difference means
the bookkeeping changed behaviour, which would invalidate every V5 number carried over.

USAGE (from the version5.5 folder):
  python test_equivalence.py --v5 ..\\version5 --images ..\\v5_check\\forged ..\\v5_check\\authentic
"""

import os
import sys
import argparse
import importlib.util

import cv2


def load_pipeline(folder, tag):
    """Import <folder>\\pipeline.py with <folder> first on sys.path, under a private module name."""
    folder = os.path.abspath(folder)
    for name in ("config", "exclude", "utils", "pipeline"):
        sys.modules.pop(name, None)
    sys.path.insert(0, folder)
    try:
        spec = importlib.util.spec_from_file_location(f"pipeline_{tag}", os.path.join(folder, "pipeline.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod, sys.modules["config"]
    finally:
        sys.path.remove(folder)


def run(mod, cfg, path, mask_on):
    cfg.EXCLUDE_ENABLED = mask_on
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    color = cv2.imread(path, cv2.IMREAD_COLOR)
    r = mod.detect(gray, color)
    return r["is_forged"], len(r["inliers"]), sorted(r["inliers"]), int(r["mask"].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5", required=True, help="the version5 folder")
    ap.add_argument("--images", nargs="+", required=True, help="folder(s) of .png images")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    old, old_cfg = load_pipeline(args.v5, "v5")
    new, new_cfg = load_pipeline(here, "v55")

    paths = []
    for d in args.images:
        paths += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.lower().endswith(".png")]
    if args.limit:
        paths = paths[:args.limit]

    bad = 0
    for p in paths:
        for mask_on in (True, False):
            a = run(old, old_cfg, p, mask_on)
            b = run(new, new_cfg, p, mask_on)
            same = a == b
            bad += (not same)
            print(f"{'OK ' if same else 'DIFF'} {os.path.basename(os.path.dirname(p))}/{os.path.basename(p)} "
                  f"mask={'on ' if mask_on else 'off'}  inliers v5={a[1]} v5.5={b[1]}")
    print(f"\n{len(paths) * 2 - bad}/{len(paths) * 2} identical.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
