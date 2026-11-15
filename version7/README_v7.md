# version7 - final: results by type of biomedical image, on Recod.ai and BioFors

**Question:** where does copy-move detection work and where does it fail, by image type, and does the pattern hold
on both datasets? Nothing is rerun, retrained or re-tuned: it joins the numbers from version5.5, version6 and the
reviewed image types from version6.7. The BioFors test labels are only re-sliced by type (no new choice is made on them).

## Run (PowerShell, from version7, venv active)
```
python analyze_types.py
```
~1 minute. Needs ..\version5.5\results, ..\version6\results, ..\version6.7\results.

## What it reports (results\analyze_types_latest.txt)
- **A** Recod.ai by type (Blot/Gel, Microscopy, Histology, Macroscopy, Other) for three rules: V4 (>=3 matches),
  count rule (>=48 matches) and learned model (score >= 0.544); cut-offs are the version6 ones (5% FPR on Recod.ai).
  Recod.ai model scores are out-of-fold (no image is scored by a model that saw its figure).
- **B** BioFors test by type, same rules, same format (adds pooled pixel MCC = the BioFors authors' metric and the
  best published method per type).
- **C** Recod.ai vs BioFors type by type (Histology merged into Microscopy, as BioFors does). "image MCC @4.3%" puts
  Recod.ai on BioFors' forged/authentic mix so the two compare like with like.
- **D** what the detector sees per type: image size, keypoints, matches, how exact forged matches are.
- **E** exploratory: one cut-off per type vs one global cut-off (Recod.ai only, half-split; not applied to BioFors).

## Saved (results\)
`analyze_types_<time>.txt` + `_latest`, `types_metrics_<time>.json` + `_latest` (every number with 95% CIs),
`types_auc.png`, `types_detection.png`, `run_log.jsonl`.

## Notes for the write-up
- Recod.ai pixel MCC is the per-image average (pixel counts were not stored there); BioFors gives both.
- Small groups: BioFors Microscopy 17 and Macroscopy 11 forged images; Recod.ai Other 15 figures. Read the CIs.
- Recod.ai has no FACS figures. Its Macroscopy is almost entirely one corn-ear collection.
- Image types: BiomedCLIP pre-labels (83.7% balanced accuracy on BioFors), the 1,000 least-confident reviewed
  individually (98 corrected), the remaining 1,751 (confidence >= 99.3%) accepted after spot checks.

## Last improvement: specialise.py (in-dataset only)
```
python specialise.py        # ~1 minute
```
Compares, per image type on Recod.ai: count rule, the version6 global model, the global model told the type, and a
separate model per type. Out-of-fold scores (folds by figure), honest cut-offs (chosen on half the figures, measured
on the other half, 40 times), at 5% and at ~0% false positives, plus the detector "ceiling" (share of forged images
where any region is drawn). Not tested on BioFors, on purpose. Saved: specialise_<time>.txt, specialise_metrics_<time>.json.
