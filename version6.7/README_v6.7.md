# version6.7 - image-type labels for Recod.ai (so results can be reported per type, like BioFors)

Recod.ai has no image-type labels. BioFors has four: Blot/Gel, Microscopy, Macroscopy, FACS. This version gives every
Recod.ai figure one of those (plus **Other** for charts, text, unclear) in a way a reviewer can trust:

1. **Validate** BiomedCLIP (Microsoft, used as-is, no training) on BioFors training images whose type is known.
   The prompt wrapper is chosen on a small tuning split; accuracy is reported on a separate check split (by document).
2. **Pre-label** all 2,751 Recod.ai figures (one per figure id; the authentic version is shown where it exists).
3. **Review**: you check every suggestion in a small window (least confident first). Saved after every key.
4. **Report**: BiomedCLIP's BioFors accuracy, how often you corrected it, final counts -> `results\recod_types.csv`.

Write-up sentence this supports: "Image types were pre-labelled by BiomedCLIP (X% balanced accuracy on BioFors images of
known type) and every label was manually verified (Y% of suggestions corrected)."

## One-time install (needs internet, ~1 GB)
From `version6.7`, with the venv active (prompt shows `(.venv)`):
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install open_clip_torch transformers
python check_setup.py --load-model        # also downloads BiomedCLIP (~800 MB) once from Hugging Face
```
`check_setup.py` without `--load-model` just lists what is missing.

## Run
```
python validate_biomedclip.py     # ~2,000 BioFors images, a few minutes on CPU
python prelabel_recod.py          # ~2,751 Recod.ai figures, ~5-10 minutes on CPU
python review_labels.py --fresh   # YOU (--fresh only the first time: old labels go to a backup): ~1-1.5 h, stop/resume any time
python label_report.py            # summary + results\recod_types.csv for version7
```

## Review keys
Enter/Space accept - 1 Blot/Gel - 2 Microscopy - 3 Macroscopy - 4 FACS - 5 Other - 6 Histology - B back - Q save & quit.
Stained tissue sections (H&E pink/purple, IHC brown) = Histology; cells = Microscopy; photos of animals, plants, organs, specimens = Macroscopy;
mixed figure = dominant type; genuinely unsure = Other.

## Saved in results\
`biomedclip_validation.json` (chosen prompts, accuracy, confusion, confidence vs accuracy), `recod_prelabels.csv`
(suggestion + probability per class), `recod_labels.csv` (your decisions, time-stamped), `recod_types.csv` (final),
a timestamped report per script, `run_log.jsonl`.

## Notes
- No detector, model or test set is touched here. BioFors **test** documents are never read.
- If BiomedCLIP's validation is weak, the review still works; you will simply correct more.
- Classes: BioFors' four + Histology + Other. Histology is merged back into Microscopy for every BioFors comparison
  (`recod_types.csv` has both columns). The Histology/Microscopy split is checked only by your review, not by BioFors.
