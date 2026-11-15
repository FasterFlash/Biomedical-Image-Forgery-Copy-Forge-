"""
run_check.py  --  V4 vs V5 on a small hand-picked folder (the sanity check).

Expects:  <check_dir>/forged/<id>.png   and   <check_dir>/authentic/<id>.png
Runs every image TWICE with the same code: exclusion OFF (= V4 behaviour) and ON (= V5),
so the ONLY difference between the two columns is the mask.

USAGE:
  python run_check.py ..\\v5_check
  python run_check.py ..\\v5_check  <mask_dir>      # default mask dir = config.MASK_DIR

NOTE: this is a sanity check on ~60 hand-picked images. It is NOT a result.
      The real number comes from run_batch.py / run_authentic.py on the full set.
"""
import os, sys, csv, time
import config, utils
from pipeline import detect


def run(img, color, enabled):
    config.EXCLUDE_ENABLED = enabled
    t0 = time.time()
    r = detect(img, color)
    return r, time.time() - t0


def main():
    check_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join("..", "v5_check")
    mask_dir = sys.argv[2] if len(sys.argv) > 2 else config.MASK_DIR
    rows = []

    for kind in ("forged", "authentic"):
        d = os.path.join(check_dir, kind)
        if not os.path.isdir(d):
            print(f"(no {kind} folder, skipping)")
            continue
        for image_id in utils.list_image_ids(d):
            img = utils.load_image(os.path.join(d, image_id + ".png"))
            col = utils.load_color(os.path.join(d, image_id + ".png"))
            truth = utils.load_mask(os.path.join(mask_dir, image_id + ".npy"), img.shape) if kind == "forged" else None
            row = {"set": kind, "image_id": image_id}
            for tag, enabled in (("v4", False), ("v5", True)):
                r, dt = run(img, col, enabled)
                row[f"{tag}_verdict"] = "forged" if r["is_forged"] else "authentic"
                row[f"{tag}_kp"] = len(r["keypoints"])
                row[f"{tag}_inliers"] = len(r["inliers"])
                if truth is not None:
                    sc = utils.score_masks(r["mask"], truth)
                    row[f"{tag}_MCC"] = sc["MCC"]; row[f"{tag}_recall"] = sc["recall"]
                else:
                    row[f"{tag}_MCC"] = row[f"{tag}_recall"] = ""
                if enabled:
                    row["excluded_pct"] = round(100.0 * float((r["exclude"] > 0).mean()), 1)
            rows.append(row)
            print(f"  {kind:9s} {image_id:>6s}  V4:{row['v4_verdict']:9s}({row['v4_inliers']:>4}in)  "
                  f"V5:{row['v5_verdict']:9s}({row['v5_inliers']:>4}in)  excl={row['excluded_pct']}%")

    config.EXCLUDE_ENABLED = True
    if not rows:
        print("nothing to do"); return

    print("\n" + "=" * 62)
    for kind in ("authentic", "forged"):
        rs = [r for r in rows if r["set"] == kind]
        if not rs: continue
        n = len(rs)
        if kind == "authentic":
            a = sum(r["v4_verdict"] == "forged" for r in rs)
            b = sum(r["v5_verdict"] == "forged" for r in rs)
            flipped = sum(r["v4_verdict"] == "forged" and r["v5_verdict"] == "authentic" for r in rs)
            print(f"AUTHENTIC ({n}): falsely flagged  V4 {a}/{n}   V5 {b}/{n}   fixed by V5: {flipped}")
        else:
            a = sum(r["v4_verdict"] == "forged" for r in rs)
            b = sum(r["v5_verdict"] == "forged" for r in rs)
            lost = sum(r["v4_verdict"] == "forged" and r["v5_verdict"] == "authentic" for r in rs)
            m4 = sum(float(r["v4_MCC"]) for r in rs) / n
            m5 = sum(float(r["v5_MCC"]) for r in rs) / n
            print(f"FORGED    ({n}): detected       V4 {a}/{n}   V5 {b}/{n}   LOST by V5: {lost}")
            print(f"                 mean MCC       V4 {m4:.3f}   V5 {m5:.3f}")
    print("=" * 62)
    print("  (sanity check only -- 60 hand-picked images, not a result)")

    out = config.ensure_output_dir()
    path = os.path.join(out, "v5_check_results.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"  saved -> {path}")


if __name__ == "__main__":
    main()
