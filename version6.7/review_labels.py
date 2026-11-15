"""
review_labels.py  --  STEP 3 of V6.7. YOU check every suggested image type. This is what makes the labels count.

A window shows one figure at a time with BiomedCLIP's suggestion. Least-confident suggestions come first.
  Enter / Space    accept the suggestion
  1 2 3 4 5 6      set the type: 1 Blot/Gel  2 Microscopy  3 Macroscopy  4 FACS  5 Other (charts, text, unclear)
                   6 Histology (stained tissue sections: H&E pink/purple, IHC brown)
  B / Backspace    go back one image (to change your answer)
  Q / Esc          save and quit; run again to continue where you stopped

Rules of thumb:  stained tissue sections = 6 Histology;  cells (fluorescent, cultured, electron microscopy) = 2 Microscopy;  photos of animals, plants, organs,
specimens, gels-on-a-bench photos of objects = Macroscopy;  a figure that mixes types: pick the dominant one;
genuinely unsure = 5 Other.

Saved after EVERY key press to results\\recod_labels.csv (safe to close any time).

USAGE:  python review_labels.py --fresh      # start over: old labels are MOVED to a timestamped backup, not deleted
        python review_labels.py              # continue where you stopped
        python review_labels.py --size 1000  # bigger window
"""

import os
import csv
import time
import base64
import argparse

import cv2

import labels_common as L

FIELDS = ["image_id", "file", "suggested", "confidence", "label", "decision", "reviewed_at"]


class Session:
    """Everything except the window: order, saving, undo. Testable without a screen."""

    def __init__(self, pre_path, out_path):
        self.out = out_path
        self.pre = list(csv.DictReader(open(pre_path, newline="")))
        self.pre.sort(key=lambda r: (float(r["confidence"]), r["image_id"]))
        self.done = {}
        if os.path.exists(out_path):
            for r in csv.DictReader(open(out_path, newline="")):
                self.done[r["image_id"]] = r
        self.history = []
        self.i = self._next(0)

    def _next(self, start):
        k = start
        while k < len(self.pre) and self.pre[k]["image_id"] in self.done:
            k += 1
        return k

    def current(self):
        return self.pre[self.i] if self.i < len(self.pre) else None

    def decide(self, label=None):
        r = self.current()
        if r is None:
            return
        lab = label or r["suggested"]
        self.done[r["image_id"]] = {"image_id": r["image_id"], "file": r["file"], "suggested": r["suggested"],
                                     "confidence": r["confidence"], "label": lab,
                                     "decision": "accepted" if lab == r["suggested"] else "corrected",
                                     "reviewed_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.history.append(self.i)
        self.save()
        self.i = self._next(self.i + 1)

    def back(self):
        if not self.history:
            return
        k = self.history.pop()
        self.done.pop(self.pre[k]["image_id"], None)
        self.save()
        self.i = k

    def save(self):
        tmp = self.out + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            order = {r["image_id"]: n for n, r in enumerate(self.pre)}
            for r in sorted(self.done.values(), key=lambda r: order.get(r["image_id"], 1e9)):
                w.writerow(r)
        os.replace(tmp, self.out)

    def progress(self):
        return len(self.done), len(self.pre)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prelabels", default=os.path.join("results", "recod_prelabels.csv"))
    ap.add_argument("--out", default=os.path.join("results", "recod_labels.csv"))
    ap.add_argument("--size", type=int, default=800, help="longest side of the shown image, px")
    ap.add_argument("--fresh", action="store_true", help="start a new review; old labels are backed up, not deleted")
    args = ap.parse_args()
    if os.path.exists(args.out):
        if args.fresh:
            bak = args.out.replace(".csv", time.strftime("_backup_%Y%m%d_%H%M%S.csv"))
            os.replace(args.out, bak)
            print(f"old labels moved to {bak}")
        elif os.path.getmtime(args.out) < os.path.getmtime(args.prelabels):
            raise SystemExit(f"{args.out} was made before the current pre-labels (old classes). "
                             "Start over with:  python review_labels.py --fresh   (old labels are kept as a backup)")
    try:
        import tkinter as tk
    except ImportError:
        raise SystemExit("tkinter is missing: see check_setup.py")

    s = Session(args.prelabels, args.out)
    root = tk.Tk()
    root.title("Recod.ai image types - review")
    root.configure(bg="#202020")
    head = tk.Label(root, font=("Segoe UI", 14), fg="white", bg="#202020", justify="left")
    head.pack(anchor="w", padx=10, pady=(8, 2))
    sub = tk.Label(root, font=("Segoe UI", 11), fg="#bbbbbb", bg="#202020", justify="left")
    sub.pack(anchor="w", padx=10)
    pic = tk.Label(root, bg="#202020")
    pic.pack(padx=10, pady=8)
    help_ = tk.Label(root, font=("Segoe UI", 10), fg="#999999", bg="#202020",
                     text="Enter = accept   1 Blot/Gel   2 Microscopy   3 Macroscopy   4 FACS   5 Other   6 Histology   B = back   Q = save & quit")
    help_.pack(pady=(0, 8))
    state = {"img": None}

    def show():
        r = s.current()
        n, tot = s.progress()
        if r is None:
            head.config(text=f"All {tot} figures reviewed. Close the window, then run:  python label_report.py")
            sub.config(text="")
            pic.config(image="")
            return
        im = cv2.imread(r["file"], cv2.IMREAD_COLOR)
        if im is None:
            head.config(text=f"cannot read {r['file']} - press 5 (Other) or check the path")
            return
        sc = args.size / max(im.shape[:2])
        im = cv2.resize(im, (max(1, int(im.shape[1] * sc)), max(1, int(im.shape[0] * sc))),
                        interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_NEAREST)
        ok, buf = cv2.imencode(".png", im)
        state["img"] = tk.PhotoImage(data=base64.b64encode(buf.tobytes()))
        pic.config(image=state["img"])
        probs = sorted(((float(r[f"p_{c}"]), c) for c in L.CLASSES), reverse=True)[:3] if "p_Other" in r else []
        head.config(text=f"[{n + 1}/{tot}]   figure {r['image_id']}    suggestion:  {r['suggested']}  "
                         f"({100 * float(r['confidence']):.0f}%)")
        sub.config(text="   ".join(f"{c} {100 * p:.0f}%" for p, c in probs))

    def key(e):
        k = (e.keysym or "").lower()
        if k in ("return", "space"):
            s.decide()
        elif e.char in L.KEYS:
            s.decide(L.KEYS[e.char])
        elif k in ("b", "backspace"):
            s.back()
        elif k in ("q", "escape"):
            root.destroy()
            return
        show()

    # the shown dict rows from prelabels carry probabilities; Session.current() returns them directly
    root.bind("<Key>", key)
    show()
    root.mainloop()
    n, tot = s.progress()
    print(f"saved {n}/{tot} reviewed labels to {args.out}")


if __name__ == "__main__":
    main()
