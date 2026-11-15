# version5.5 - learned decision layer on top of the V5 detector

**Question this folder answers:** V5 says "forged" when confirmed matches >= 3. That single number separates
forged from authentic at AUC ~0.83 (V5 ROC analysis), so no threshold or mask change can get past it. Does a
classifier that also sees *how* the matches look (repetitiveness, mutual matches, one shared displacement,
source-target distance, match confidence, ...) separate them better?

**Not a promise.** This is an experiment. The report tells you whether the gain is real (CI excludes 0) or not.

## What changed vs version5
- `pipeline.py`: same detector, plus bookkeeping (Lowe ratios, per-cluster stats). `test_equivalence.py` proves
  verdicts, matches and masks are identical to V5.
- NEW `features.py`: one detector run -> ~45 numbers.
- NEW `extract_features.py`: runs every image (both classes, mask ON and OFF) -> `results\features.csv`.
- NEW `cv_decision.py`: grouped cross-validation, baseline vs learned model, CIs, honest operating point.
- Everything else (`exclude.py`, `config.py`, `utils.py`, diagnostics) is copied unchanged from version5.

## Run it (PowerShell, from `version5.5`, venv active)
```
pip install scikit-learn            # already there from V5
python test_equivalence.py --v5 ..\version5 --images ..\v5_check\forged ..\v5_check\authentic --limit 5
python extract_features.py --limit 20       # smoke test; note the seconds per job
python extract_features.py --workers 4      # the real run; resumable, Ctrl+C is safe
python cv_decision.py
```
Rough time: (seconds per job from the smoke test) x 8000 jobs (4000 images x 2 mask modes) / workers.
Use `--mask on` to halve it. Do not use more workers than physical cores.

## Reading the report
1. **AUC / detection at fixed FPR** - out-of-fold, each image scored by a model that never saw its id.
2. **Difference vs best baseline** - the only section that says "improved". A `*` means the 95% CI excludes 0.
3. **Honest operating point** - threshold chosen on one half of the ids, measured on the other half.
   Quote these numbers, not the ones from section 1, if you report a specific FPR / detection pair.
4. **Single-feature AUC** - which feature carries the signal.

## Why the folds are grouped by image id
Authentic and forged folders share ~1700 ids (same figure with / without the forgery). A random split leaks the
figure between train and test. `cv_decision.py` keeps every row of an id in one fold.

## Limits to state in any write-up
- One dataset, development numbers. The real test is training here and evaluating on BioFors (Phase 3).
- The features include simple image statistics (size, colour saturation, edge density). They let the model
  adapt to colour photographs vs grey microscopy, which is useful, but report it: it is part of the model.
- Final deployment needs a `train_final.py` (fit on all data, call from the pipeline). Not written yet; only
  worth doing if section 2 shows a real gain.

## v2 of cv_decision.py: held-out test and robust features
```
# development numbers only; 20% of image ids are set aside and NOT used. Pick your own seed and keep it.
python cv_decision.py --holdout 0.2 --holdout-seed 2027 --drop image kp

# once the model and FPR are chosen (decide from the development tables!), look at the held-out ids ONCE:
python cv_decision.py --holdout 0.2 --holdout-seed 2027 --drop image kp --final --final-model "GBM, ON+OFF features" --final-fpr 0.10
```
- `--drop image kp` removes picture statistics (size, grey level, saturation, edge density) and keypoint counts.
  A leak test showed those alone give AUC 0.54 (near chance), so they do not leak the label, but they are
  properties of this dataset's pictures and may not carry over to BioFors. Use the robust set for the model you
  take to BioFors.
- `--final` refuses a second look (results\holdout_used.txt) unless `--allow-repeat`. Say so in the write-up if used.
- `img MCC@4%` columns are image-level MCC *if* the measured TPR/FPR held on a set with ~4% forged images
  (the BioFors internal-duplication test set). That is an assumption about transfer, not a measurement.
