"""
check_setup.py  --  STEP 0 of V6.7. Checks that everything version6.7 needs is installed, and prints the exact
install commands for whatever is missing. Run it first; run it again after installing.

USAGE (PowerShell, from version6.7, venv active):   python check_setup.py
             add  --load-model  to also download/load BiomedCLIP once (needs internet the first time)
"""

import sys
import argparse

INSTALL = [
    ("torch", "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"),
    ("torchvision", "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"),
    ("open_clip", "pip install open_clip_torch"),
    ("transformers", "pip install transformers"),
    ("PIL", "pip install pillow"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-model", action="store_true")
    args = ap.parse_args()
    print(f"python {sys.version.split()[0]}   ({sys.executable})")
    if ".venv" not in sys.executable:
        print("WARNING: this does not look like the project venv. Activate it first:  ..\\.venv\\Scripts\\Activate.ps1")

    missing = []
    for mod, cmd in INSTALL:
        try:
            m = __import__(mod)
            print(f"  ok       {mod:13s} {getattr(m, '__version__', '')}")
        except Exception as e:
            print(f"  MISSING  {mod:13s} ({type(e).__name__})")
            missing.append(cmd)
    try:
        import tkinter
        print(f"  ok       tkinter       {tkinter.TkVersion}   (needed by review_labels.py)")
    except Exception:
        print("  MISSING  tkinter  -> reinstall Python from python.org with 'tcl/tk and IDLE' ticked")
    for mod in ("numpy", "cv2", "sklearn"):
        try:
            print(f"  ok       {mod:13s} {__import__(mod).__version__}")
        except Exception:
            print(f"  MISSING  {mod}")

    if missing:
        print("\nInstall the missing packages (one-time, needs internet, ~1 GB in total):")
        for c in dict.fromkeys(missing):
            print("   " + c)
        return
    if args.load_model:
        from labels_common import Encoder, CLASSES
        enc = Encoder()
        v = enc.encode_texts(["a western blot"])
        print(f"\nBiomedCLIP loaded. text embedding size {v.shape[1]}, temperature {enc.logit_scale:.1f}. Classes: {CLASSES}")
    print("\nAll set. Next:  python validate_biomedclip.py")


if __name__ == "__main__":
    main()
