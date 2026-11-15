# version6 - first generalisation test: Recod.ai -> BioFors

**Question:** the decision model built and tested on Recod.ai in version5.5 - does it still work on a dataset it
has never seen (BioFors internal-duplication test set, the copy-move task), and if not, can it be fixed
without any forged examples from the new dataset?

Nothing about the detector or the model changes in this version. Detector = version5.5 (mask OFF);
model = the one chosen in version5.5 (gradient-boosted trees, robust features), retrained on all Recod.ai images.

## Run it (PowerShell, from `version6`, venv active) - in this order

```
python train_final.py                       # ~10 s. Freezes the Recod.ai model and its cut-offs.
python run_biofors.py --limit 10            # smoke test (20 images); note seconds per image
python run_biofors.py --workers 4           # the real run, ~4,560 images, resumable (Ctrl+C safe)
python score_biofors.py                     # ~1 min. All the numbers.
```

Optional independent check with the BioFors authors' own scoring script (needs `pip install tqdm`):
```
python authors_eval_IDD.py --pdf_list ..\biofors_annotations\idd_pdfs.pkl --out_dir results\predictions\model_zeroshot\ --idd_gt ..\biofors_annotations\idd_gt.json --idd_classification ..\biofors_annotations\classification.json --img_path ..\biofors_images\ --categories Microscopy Blot/Gel Macroscopy
```
Its image TP/FP/TN/FN, image MCC and pixel MCC must equal the "model, zero-shot / ALL" row of score_biofors.
(Verified on a 34-image subset before delivery: identical to 4 decimals.)

## What is tested

| set | what | used for |
|---|---|---|
| test | BioFors IDD test: 54 documents, 2,363 images (Microscopy 987, Blot/Gel 1,299, Macroscopy 77), **102 forged** | the numbers |
| calib | 2,200 **genuine** images sampled from BioFors *training* documents (seed 0, list saved) | re-setting the cut-off only. No forged image, no test image |

Scored exactly like the BioFors authors: boxes as ground truth, image counted as flagged only if a region is drawn,
pixel MCC pooled over all pixels of all images (different from the per-image average used in version5.5).

## Settings in the report

| setting | meaning |
|---|---|
| V4 rule | >= 3 confirmed matches (V4/V5 behaviour) |
| count, Recod.ai cut-off / recalibrated | old rule with a 5% FPR cut-off from Recod.ai / from genuine BioFors images |
| **model, zero-shot** | V5.5 model, Recod.ai cut-off, unchanged - **pure generalisation** |
| **model, recalibrated** | same model, cut-off re-set on genuine BioFors images - **label-free adaptation** |
| model, recal. per type | one cut-off per image type |

Report sections: A = drift check (what the Recod.ai cut-off does on genuine BioFors images; no labels needed),
B = threshold-free AUC, C = full table (all types, then per type) with 95% CIs from resampling documents,
plus the published BioFors baselines per image type (Table 5 of the paper, checked against the PDF).

## Everything is saved (results\)

| file | content |
|---|---|
| `run_log.jsonl` | one line per run of any script: start, end, runtime, settings, versions, outputs, key numbers |
| `<script>_<YYYYmmdd_HHMMSS>.txt` + `<script>_latest.txt` | full printed report of every run |
| `final_model.joblib` / `final_model.json` | frozen model, feature list, cut-offs, training info |
| `recod_oof_scores.csv` | each Recod.ai image's out-of-fold score |
| `biofors_features.csv` | per BioFors image: set, document, type, forged?, pixel counts, all features, detector time |
| `biofors_calib_manifest.json` | exactly which genuine images were used for recalibration |
| `raw_masks\` | detector's region for every test image where it drew one |
| `biofors_metrics_<time>.json` + `_latest.json` | every number in the report, with CIs and cut-offs |
| `biofors_scores.csv` | per image: model score, matches, flagged or not under every setting |
| `predictions\<setting>\` | predicted masks in the authors' format |

## Honest limits
- 102 forged images in total (74 Blot/Gel, 17 Microscopy, 11 Macroscopy): CIs will be wide, per-type ones very wide.
- Blots dominate the forgeries and are low-texture; the point-matching detector is weakest there. Expect a drop.
- Published baselines: the dense-field (DF-*), DCT, DWT and Zernike methods are classical (no training); the paper says
  deep baselines (BusterNet) were fine-tuned on synthetic BioFors data. Ours never saw BioFors (zero-shot) or saw only
  genuine BioFors images (recalibrated).
- Version5.5 note, corrected: the held-out FPR of 7.5% (target 5%) is consistent with sampling noise; on development
  data all threshold methods held 4.7-4.8% (+-0.7-1.0) over 8 splits. Do not present it as systematic drift.
