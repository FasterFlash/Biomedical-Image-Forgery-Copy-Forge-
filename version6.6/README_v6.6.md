# version6.6 - experiment 2: a principled fix, chosen without the BioFors test set

**Background (version6, experiment 1):** the learned decision model did not transfer to BioFors (AUC 0.79 vs 0.90
for the plain confirmed-match count). Diagnosis: on BioFors, forgeries are less exact and genuine repeats sit far
apart, so two cue families the model leaned on changed meaning. The BioFors test set has now been looked at once.

**Question:** can a fix chosen ONLY on BioFors-like data that we label ourselves do better on the real test?
Two fixes are compared against the plain count:
- (a) keep only features that behave the same on Recod.ai and on BioFors-like images ("stable" features)
- (b) train on Recod.ai **plus** synthetic BioFors copy-moves (the BioFors authors fine-tuned their deep baseline on
  synthetic BioFors manipulations in the same spirit)

Detector unchanged (mask OFF). No BioFors forgery label and no BioFors test image is used for any choice.

## Run it (PowerShell, from `version6.6`, venv active: prompt must show `(.venv)`)
```
python make_synthetic.py              # ~1-2 min. ~1,450 labelled pairs from genuine BioFors TRAINING images
python run_synth.py --limit 10        # smoke test
python run_synth.py --workers 4       # ~2,900 images, a few minutes, resumable
python develop.py                     # ~3-5 min. every choice is made here, by a rule written in advance
python test_biofors.py                # seconds. looks at the BioFors test ONCE (lock file afterwards)
```
Needs version6 to have run (it reuses `..\version6\results\biofors_features.csv`, `raw_masks\`, the calibration
manifest and `biofors_metrics_latest.json`).

## The synthetic set (`make_synthetic.py`)
- source: genuine images of BioFors **training** documents; version6 calibration images excluded; test documents never read
- per type: Blot/Gel 700, Microscopy 600, Macroscopy 150 (all that is left after calibration)
- a content-bearing patch is copied to a non-overlapping place; variants spread evenly:
  clean, rotate (5-45 deg), scale (0.8-1.2), flip, blur, jpeg, publication (shrink 50-80% + enlarge + JPEG; mimics a
  figure's way into a PDF)
- the genuine twin gets the SAME whole-image processing, so recompression can never reveal the answer
- ground truth = both copies. **Limitation:** patches land at random places, so many fakes are cruder than real ones.
  Use results on this set as a development signal, not as BioFors performance.

## The decision (`develop.py`) - rule fixed before any result
| candidate | what |
|---|---|
| C0 | confirmed-match count |
| C1 | model, all features, Recod.ai only (= version6) |
| C2 | model, stable features, Recod.ai only (fix a) |
| C3 | model, all features, Recod.ai + synthetic BioFors (fix b) |
| C4 | model, stable features, Recod.ai + synthetic BioFors (a + b) |

All scored on synthetic BioFors with folds by **document**; "stable" is decided inside each fold from the training part.
For each candidate the false-positive target (0.5-10%) that maximises image MCC at the BioFors test prevalence
(102 / 2363, published) is found. **Rule:** highest expected MCC wins; a simpler candidate within 0.01 wins instead
(C0 < C2 < C1 < C4 < C3). The winner is retrained on all its data and its cut-off is set on the **genuine BioFors
calibration images** (real, untouched) at that target. C0 is frozen the same way as a reference.

## Everything is saved (results\)
`run_log.jsonl` (every run: time, runtime, settings, versions, outputs), a timestamped report per run + `_latest`,
`synth_manifest.csv` (every pair and its random parameters), `synth\` (images and masks), `synth_features.csv`,
`synth_oof_scores.csv`, `develop_metrics_<time>.json` (stability, candidates, CIs, robustness),
`final_v66.joblib/.json` (what was frozen and why), `test_biofors_metrics_<time>.json`, `predictions_v66\`
(authors' format), `biofors_test_used.txt` (lock).

## Record of test-set contact (for the write-up)
- version6: the test set was scored once with six pre-declared settings.
- While testing the v6.6 code on 2026-11-13 the assistant ran `test_biofors.py` once on the real test features with a
  configuration chosen by a 15-pair toy synthetic set (count rule, cut-off 59 matches: image MCC 0.466). Nothing in
  v6.6 was changed or tuned after that; the selection rule and all code paths were already fixed.
- Your `python test_biofors.py` run is experiment 2. If it is ever rerun with `--allow-repeat`, report that.
