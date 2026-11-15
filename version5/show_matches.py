"""
show_matches.py  --  See WHICH keypoints match on authentic images, for eye review.

For every authentic image the detector wrongly flags, it saves ONE picture:

    [ left: the plain image  |  right: same image with the matches drawn on it ]

    green line   a match the detector confirmed (this is what triggered "forged")
    red dot      one end of a match      yellow dot   the other end
    blue tint    area the V5 exclusion mask removed from SIFT (only when mask is ON)

The picture is in COLOUR on purpose (orange label tags are invisible in grayscale).
Images are shrunk to 900px on the long side so 2000 of them stay a sane size.

USAGE  (run from the version5 folder, venv active):

  python show_matches.py                       # all authentic images, mask ON, saves only the flagged ones
  python show_matches.py --mask off            # same, mask OFF (V4 behaviour) -> separate folder
  python show_matches.py --from-csv results\\false_positives_excl.csv
                                               # skip detection on the rest; render only the listed ids (fast)
  python show_matches.py --limit 50            # first 50 authentic images, to try it out
  python show_matches.py --all                 # also save images with NO matches (usually pointless)
  python show_matches.py --ids 10799 12244     # just these ids

OUTPUT  (results\\matches_on\\  or  results\\matches_off\\):
  <id>_<n>in.jpg        the picture; <n> = number of confirmed matches
  matches_index.csv     one row per saved image, with EMPTY columns  modality / cause / notes
                        for you to fill in while you review (open it in Excel).

It is RESUMABLE: stop it any time, run it again, and it skips images already saved.
It never overwrites your notes in the CSV (rows are only appended).
"""

import os
import sys
import csv
import argparse

import cv2
import numpy as np

import config
import utils
from pipeline import detect

LONG_SIDE = 900          # px, size of each half of the picture
BLUE = np.array([255, 120, 0], np.float32)     # BGR tint for excluded areas
FIELDS = ["image_id", "inliers", "raw_pairs", "keypoints", "excluded_pct",
          "file", "modality", "cause", "notes"]


def render(color, result, image_id):
    """Build the side-by-side picture for one image."""
    h, w = color.shape[:2]
    s = min(1.0, LONG_SIDE / float(max(h, w)))
    size = (max(1, int(w * s)), max(1, int(h * s)))
    plain = cv2.resize(color, size, interpolation=cv2.INTER_AREA)
    over = plain.copy()

    excl = result.get("exclude")
    if excl is not None:
        m = cv2.resize(excl, size, interpolation=cv2.INTER_NEAREST) > 0
        over[m] = (0.6 * over[m] + 0.4 * BLUE).astype(np.uint8)

    kp = result["keypoints"]
    pairs = result["inliers"]
    if len(pairs) > 1500:                                   # keep huge match sets readable
        pairs = [pairs[i] for i in np.linspace(0, len(pairs) - 1, 1500).astype(int)]
    for a, b in pairs:
        p = (int(kp[a].pt[0] * s), int(kp[a].pt[1] * s))
        q = (int(kp[b].pt[0] * s), int(kp[b].pt[1] * s))
        cv2.line(over, p, q, (0, 255, 0), 1, cv2.LINE_AA)
    for a, b in pairs:
        cv2.circle(over, (int(kp[a].pt[0] * s), int(kp[a].pt[1] * s)), 3, (0, 0, 255), -1)
        cv2.circle(over, (int(kp[b].pt[0] * s), int(kp[b].pt[1] * s)), 3, (0, 255, 255), -1)

    body = np.hstack([plain, over])
    pct = 100.0 * float((excl > 0).mean()) if excl is not None else 0.0
    banner = np.full((30, body.shape[1], 3), 255, np.uint8)
    text = (f"{image_id}   confirmed matches: {len(result['inliers'])}   "
            f"raw matches: {len(result['raw_pairs'])}   keypoints: {len(kp)}   "
            f"mask: {'ON, %.1f%% excluded' % pct if excl is not None else 'OFF'}")
    cv2.putText(banner, text, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 160), 1, cv2.LINE_AA)
    return np.vstack([banner, body]), pct


def main():
    ap = argparse.ArgumentParser(description="Draw matched keypoints on authentic images.")
    ap.add_argument("--mask", choices=["on", "off"], default="on",
                    help="V5 exclusion mask on (default) or off (V4 behaviour)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N authentic images")
    ap.add_argument("--all", action="store_true", help="also save images that have no matches")
    ap.add_argument("--from-csv", default=None,
                    help="only render image ids listed in this csv (needs an image_id column)")
    ap.add_argument("--ids", nargs="*", default=None, help="only these image ids")
    ap.add_argument("--src", default=config.AUTHENTIC_DIR, help="authentic image folder")
    ap.add_argument("--out", default=None, help="output folder (default results\\matches_on|off)")
    args = ap.parse_args()

    config.EXCLUDE_ENABLED = (args.mask == "on")
    out_dir = args.out or os.path.join(config.OUTPUT_DIR, f"matches_{args.mask}")
    os.makedirs(out_dir, exist_ok=True)

    ids = utils.list_image_ids(args.src)
    if args.ids:
        ids = [i for i in ids if i in set(args.ids)]
    if args.from_csv:
        with open(args.from_csv, newline="") as f:
            wanted = {r["image_id"] for r in csv.DictReader(f)}
        ids = [i for i in ids if i in wanted]
    if args.limit:
        ids = ids[:args.limit]

    # resume: remember what is already in the index so we never redo or overwrite it
    index_path = os.path.join(out_dir, "matches_index.csv")
    done = set()
    new_file = not os.path.exists(index_path)
    if not new_file:
        with open(index_path, newline="") as f:
            done = {r["image_id"] for r in csv.DictReader(f)}

    print(f"Authentic images to process: {len(ids)}   (already saved: {len(done)})")
    print(f"Mask: {'ON ' + str(config.EXCLUDE_RULES) if config.EXCLUDE_ENABLED else 'OFF (V4 behaviour)'}")
    print(f"Saving pictures to: {out_dir}\n")

    saved = flagged = 0
    with open(index_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for n, image_id in enumerate(ids, 1):
            if image_id in done:
                continue
            path = os.path.join(args.src, f"{image_id}.png")
            gray = utils.load_image(path)
            color = cv2.imread(path, cv2.IMREAD_COLOR)
            if color is None:
                color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            result = detect(gray, color)
            if result["is_forged"]:
                flagged += 1
            elif not args.all:
                if n % 50 == 0:
                    print(f"  [{n}/{len(ids)}] scanned, {saved} saved so far")
                continue

            picture, pct = render(color, result, image_id)
            name = f"{image_id}_{len(result['inliers'])}in.jpg"
            cv2.imwrite(os.path.join(out_dir, name), picture, [cv2.IMWRITE_JPEG_QUALITY, 88])
            w.writerow({"image_id": image_id, "inliers": len(result["inliers"]),
                        "raw_pairs": len(result["raw_pairs"]), "keypoints": len(result["keypoints"]),
                        "excluded_pct": round(pct, 1), "file": name,
                        "modality": "", "cause": "", "notes": ""})
            f.flush()                                       # safe to Ctrl+C at any moment
            saved += 1
            if saved % 20 == 0 or n % 50 == 0:
                print(f"  [{n}/{len(ids)}] scanned, {saved} saved so far")

    print(f"\nDone. Saved {saved} pictures this run  ({flagged} flagged as forged).")
    print(f"Pictures:  {out_dir}")
    print(f"Index:     {index_path}   <- fill the modality / cause / notes columns as you review")


if __name__ == "__main__":
    main()
