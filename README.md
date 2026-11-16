# **Copy-Move Forgery Detection in Biomedical Images**
Nov 15, 2026 · @Tarun
## **1. Project at a glance**
- **Task:** detect copy-move (internal duplication) forgery in biomedical research figures, at image and pixel level.
- **Objective 1:** high MCC and low false-positive rate on the Recod.ai dataset.
- **Objective 2:** test generalisation on a second, unseen dataset (BioFors), the main research question.
- **Constraint:** no neural-network training; the detector is classical (SIFT keypoints); only a small decision layer is learned.
- **Status:** all experiments complete (V4 to V7); literature comparison done; write-up next.

**Datasets**

|Dataset|Images|Forged|Authentic|Ground truth|Role|
| :- | :- | :- | :- | :- | :- |
|Recod.ai|**5,128** (**2,751** figures)|**2,751**|**2,377**|pixel masks|development + held-out test|
|BioFors copy-move test|**2,363** from **54** papers|**102**|**2,261**|boxes|generalisation test|
|BioFors training documents|**696** documents|**0**|all|image type|label-free recalibration, synthetic data|

- **2,377** Recod.ai figures exist in both an authentic and a forged version, so every split keeps both versions together.

**Headline results**

|Version|Dataset|Decision rule|AUC|False positives|Detection|Image MCC|Pixel MCC|
| :- | :- | :- | :- | :- | :- | :- | :- |
|V4|Recod.ai|3 or more matches||**64.0%**|**91.6%**||**0.546**|
|V5.5|Recod.ai held-out test|learned model|**0.916**|**7.5%**|**75.6%**|**0.683**|**0.499**|
|V6|BioFors|match count, Recod.ai cut-off|**0.901**|**0.8%**|**39.2%**|**0.500**|**0.316**|
|V6|BioFors|learned model, zero-shot|**0.790**|**23.7%**|**70.6%**|**0.218**|**0.235**|
|V6.6|BioFors|model + synthetic BioFors training|**0.818**|**0.5%**|**18.6%**|**0.329**|**0.237**|
|Published|BioFors|DF-PCT (classical, best in BioFors paper)||||**0.569**|**0.364**|
## **2. Terms used**
- **False positives (FPR):** share of authentic images wrongly flagged as forged.
- **Detection:** share of forged images correctly flagged.
- **AUC:** how well a score separates forged from authentic images, independent of any cut-off (0.5 = chance, 1.0 = perfect).
- **Image MCC:** quality of the forged/authentic decision per image (0 = no skill, 1 = perfect).
- **Pixel MCC:** overlap of the marked region with the true forged region. Recod.ai: average per forged image (not flagged = 0). BioFors: pooled over all pixels, as the BioFors authors score it.
- **Image MCC at 4.3%:** image MCC expected if only 4.3% of images were forged, as in the BioFors test, so the two datasets compare like with like.
- **Confirmed matches:** pairs of near-identical keypoints that move together as one copied block.
- **Out-of-fold / held-out:** every reported score comes from a model that never saw that figure.
## **3. V4: baseline detector**
**What it does**

- Finds thousands of SIFT keypoints (very low contrast threshold; images under 1024 px enlarged 4x so smooth and small panels still get points).
- Matches every keypoint against the same image; keeps a pair only if it passes the Lowe ratio test (0.75).
- Groups matches spatially (DBSCAN), then checks each group for one consistent movement (RANSAC).
- Flags the image as forged if **3 or more** confirmed matches survive; outlines the region with convex hulls.

**Assumption:** copied regions produce near-identical, consistently moving keypoint pairs; genuine images do not.

**Results (Recod.ai, 2,000 forged + 2,000 authentic)**

|Measure|Result|
| :- | :- |
|Forged images detected|**91.6%** (**1,832** of **2,000**)|
|Pixel MCC (forged images)|**0.546**|
|Pixel F1|**0.539**|
|Pixel precision / recall|**0.504** / **0.657**|
|Authentic images wrongly flagged|**64.0%** (**1,281** of **2,000**)|

**What the numbers told us**

- The problem is false alarms, not missed forgeries: two-thirds of genuine figures are flagged.
- Forged-only metrics hide this; from here on the authentic false-positive rate is always reported.
- Many false alarms were macroscopic photos with rulers, labels and text, which gave the V5 hypothesis.
- Later found: runs were not exactly repeatable (randomness in matching). Fixed in V5; the repeatable rerun gave **64.0%** false positives and **91.6%** detection, so V4's conclusions stand.

**Decision:** test whether non-biological content (text, scale bars, rulers) causes the false positives.
## **4. V5: excluding non-biological content**
**Hypothesis:** false alarms come from things that repeat by design (text, scale bars, rulers, label tags), mostly in macroscopic photos.

**What we did**

- Hand-checked set: **30** forged images with text or bars that V4 got wrong, plus their **30** authentic twins.
- Exclusion mask built before keypoint detection: rules for text lines, corner labels, scale bars, rulers and label cards, plus an off-the-shelf text detector (OCR) on the colour image.
- Fixed non-determinism: randomness seeded per image, so mask ON and OFF compare fairly.
- Built a viewer that draws the matches on every falsely flagged authentic image for visual inspection.

**Results (Recod.ai, 2,000 + 2,000, repeatable runs)**

|Measure|Mask OFF|Mask ON|
| :- | :- | :- |
|Authentic wrongly flagged|**64.0%**|**54.0%** (drop **9.95** pts, 95% CI **8.3** to **11.6**)|
|Authentic with 20 or more false matches|**18.3%**|**8.4%**|
|Forged detected|**91.6%**|**87.1%**|
|Pixel MCC|**0.545**|**0.505**|
|Pixel precision / recall|**0.502** / **0.655**|**0.481** / **0.595**|
|AUC|**0.832**|**0.837** (difference **+0.005**, CI **-0.002** to **+0.012**)|

**Fair comparison at equal false-positive rates**

|Allowed FPR|Detection, mask OFF|Detection, mask ON|Gain|
| :- | :- | :- | :- |
|5%|**56.4%**|**61.4%**|**+5.0**|
|10%|**62.8%**|**66.7%**|**+3.9** (CI **+2.2** to **+5.1**)|
|15%|**66.8%**|**69.8%**|**+3.0**|
|20%|**71.7%**|**72.4%**|**+0.7**|
|30%|**78.6%**|**78.0%**|**-0.6**|

**What the numbers told us**

- The 10-point drop was mostly an illusion: the mask removed matches from forged images too, so both kinds crossed the 3-match line less often.
- Real gain only at strict settings; AUC did not change significantly.
- The viewer showed the main cause: natural biological repetition (cells, corn kernels, gel bands), which no mask can remove.
- About half the remaining false alarms had only **3** to **9** matches.

**Decision:** the bottleneck is the decision rule (one match count, AUC ceiling **0.83**), not the clutter. Keep the detector, make the decision smarter.
## **5. V5.5: learned decision layer**
**Hypothesis:** real copies and natural repetition differ in how their matches look, not only how many: a copy-paste is near pixel-perfect and moves as one block; natural repeats only look similar.

**What we did**

- Detector unchanged (verified: identical matches and regions to V5).
- Ran it on all **5,128** Recod.ai images, mask ON and OFF.
- Described each image by **43** measurements of its matches (exactness, survival after geometric check, shared displacement, source-target distance, cluster geometry).
- Trained a small gradient-boosted tree model (seconds on a laptop; not a neural network).

**Evaluation protocol**

- Both versions of a figure always in the same fold or split (no figure leaks between training and test).
- **20%** of figures (**550** figures, **1,014** images) locked away and opened once; all choices made on the other **80%**.
- Leak test: a model given only picture properties (size, brightness, colour) scored AUC **0.54** (near chance); those properties were then removed (**32** features kept).

**Development results (80%, out-of-fold)**

|Model|AUC|Detection at 5% FPR|Pixel MCC at 5% FPR|
| :- | :- | :- | :- |
|Match count, mask OFF|**0.837**|**57.3%**|**0.431**|
|Match count, mask ON|**0.841**|**61.6%**|**0.448**|
|Learned model, mask OFF|**0.928**|**80.1%**|**0.538**|
|Learned model, mask ON|**0.924**|**75.3%**|**0.502**|
|Learned model, ON + OFF|**0.945**|**80.3%**|**0.538**|
|Logistic regression, ON + OFF|**0.916**|**77.6%**|**0.531**|

**Strongest single measurements**

|Measurement|AUC alone|
| :- | :- |
|Exactness of confirmed matches (Lowe ratio)|**0.871**|
|Exactness of all raw matches|**0.867**|
|Share of raw matches surviving geometry|**0.858**|
|Number of confirmed matches (V4 rule)|**0.837**|

**Locked test (opened once)**

|Rule|AUC|False positives (target 5%)|Detection|Image MCC|Pixel MCC|
| :- | :- | :- | :- | :- | :- |
|Match count|**0.826**|**3.7%**|**58.9%**|**0.583**|**0.426**|
|Learned model|**0.916** [0.899, 0.932]|**7.5%** [5.3, 10.1]|**75.6%** [72.0, 79.3]|**0.683** [0.640, 0.725]|**0.499** [0.469, 0.531]|

- Detection gain: **+16.7** points (95% CI **+13.5** to **+20.2**).
- Locked-test AUC (**0.916**) matched development (**0.928**): no overfitting within the dataset.
- The **7.5%** false positives vs the 5% target: rechecked on development data over **8** splits, every cut-off method held **4.7** to **4.8%**, so most likely sampling noise on **464** authentic test images.

**Decisions**

- Mask dropped: equal performance at strict settings (**80.1%** vs **80.3%**), its rules were tuned by eye on corn photos, and it doubles runtime.
- Operating point fixed at **5%** false positives, because BioFors is about **96%** genuine images.
- Match exactness explains V5: once it is used, text and scale bars barely matter.
## **6. V6: generalisation test 1, Recod.ai to BioFors**
**Hypothesis:** the V5.5 model learned what a copy-paste looks like and will work on an unseen dataset; if its cut-off drifts, resetting it on genuine BioFors images (no forgery labels) will fix it.

**What we did**

- Obtained the official BioFors annotations; test set = **2,363** images, **102** forged (Blot/Gel **74**, Microscopy **17**, Macroscopy **11**).
- Froze the model on all Recod.ai data before touching BioFors; cut-offs at 5% false positives on Recod.ai: model **0.544**, match count **48**.
- Declared **6** settings in advance; recalibration used **2,200** genuine BioFors training images only.
- Scored exactly as the BioFors authors; on a sample our scores matched their own script exactly.
- Every run logged with time, settings, versions and all numbers.

**Results (BioFors test, opened once)**

|Setting|False positives|Detection|Image MCC|Pixel MCC|
| :- | :- | :- | :- | :- |
|V4 rule (3 or more matches)|**73.5%**|**97.1%**|**0.110**|**0.160**|
|Match count, Recod.ai cut-off|**0.8%**|**39.2%**|**0.500** [0.383, 0.613]|**0.316** [0.195, 0.470]|
|Match count, recalibrated|**6.4%**|**63.7%**|**0.409**|**0.284**|
|Learned model, zero-shot (main test)|**23.7%**|**70.6%**|**0.218**|**0.235**|
|Learned model, recalibrated|**4.7%**|**37.3%**|**0.277**|**0.271**|
|Learned model, recalibrated per type|**4.6%**|**40.2%**|**0.303**|**0.272**|
|Published DF-PCT (classical)|||**0.569**|**0.364**|
|Published DF-FMT (classical)|||**0.509**|**0.316**|

**Ranking quality (AUC, no cut-off)**

|Rule|Recod.ai|BioFors|
| :- | :- | :- |
|Learned model|**0.925**|**0.790** [0.733, 0.836]|
|Match count|**0.832**|**0.901** [0.855, 0.935]|

**Drift check: genuine BioFors images flagged at the Recod.ai cut-off (target 5%)**

|Image type|Learned model|Match count|
| :- | :- | :- |
|Blot/Gel|**27.5%**|**0.3%**|
|Microscopy|**26.2%**|**1.4%**|
|Macroscopy|**10.2%**|**1.8%**|
|All (test mix)|**26.4%**|**0.8%**|

**Why the learned model failed (measured)**

|Measurement|Recod.ai|BioFors|
| :- | :- | :- |
|Exactness of forged matches (lower = more exact)|**0.42**|**0.57**|
|Distance between genuine repeats (share of image diagonal)|**3%**|**19%**|
|Median image size|**512 x 696** px|**94 x 247** px|

- Copies are less exact on BioFors while genuine repeats look the same, so the strongest cue weakens.
- Genuine repeats sit far apart on BioFors (blot lanes, repeated panels), so the learned rule "far apart = forged" fires on genuine images.
- The confirmed-match count stays stable, so the simple rule transfers.

**By image type, match count with Recod.ai cut-off vs best published**

|Type|Ours, image MCC|DF-PCT, image MCC|
| :- | :- | :- |
|Microscopy|**0.620**|**0.764**|
|Blot/Gel|**0.431**|**0.503**|
|Macroscopy|**0.660**|**0.712**|

**Decisions and honesty notes**

- The pre-declared main test (learned model, zero-shot) failed; the match-count result is one of the **6** declared settings, and its strict **0.8%** false-positive rate on BioFors was partly luck.
- Cut-off drift between datasets is real (**26.4%** instead of 5%); recalibration fixes false positives but cannot fix weak ranking.
- Next: try a principled fix chosen without touching the BioFors test set again.
## **7. V6.6: generalisation test 2, fix chosen on synthetic BioFors data**
**Hypothesis:** two possible fixes for V6: (a) use only measurements that behave the same on both datasets, (b) also train on BioFors-like forgeries. The choice must not use the BioFors test set.

**What we did**

- Built **1,450** labelled copy-move pairs from genuine BioFors training images of **517** documents (Blot/Gel **700**, Microscopy **600**, Macroscopy **150**); test documents and calibration images never used.
- **7** copy variants, spread evenly: clean, rotated (5 to 45 degrees), scaled (0.8 to 1.2x), mirrored, blurred, JPEG, publication-like (shrink, enlarge, recompress).
- Each forged image has a genuine twin with the same whole-image processing, so compression cannot reveal the answer.
- Compared **5** candidates with folds by document; selection rule fixed in code before any result: highest expected image MCC wins, a simpler candidate within **0.01** wins instead.
- **18** of **32** measurements behaved the same on both datasets ("stable").

**Development results (synthetic BioFors)**

|Candidate|AUC|Expected image MCC|
| :- | :- | :- |
|C0 match count|**0.698**|**0.390**|
|C1 V6 model|**0.717**|**0.382**|
|C2 model, stable features only|**0.719**|**0.380**|
|C3 model, Recod.ai + synthetic BioFors|**0.743**|**0.454** (vs C0 **+0.064**, CI **-0.002** to **+0.122**)|
|C4 both fixes|**0.745**|**0.436**|

- Rule chose **C3**.

**BioFors test (opened once more, then locked)**

|Rule|AUC|False positives|Detection|Image MCC|Pixel MCC|
| :- | :- | :- | :- | :- | :- |
|C3 (chosen)|**0.818**|**0.5%**|**18.6%**|**0.329** [0.216, 0.440]|**0.237**|
|C0 match count|**0.901**|**0.7%**|**33.3%**|**0.466** [0.359, 0.578]|**0.295**|
|V6 model, zero-shot|**0.790**|**23.7%**|**70.6%**|**0.218**|**0.235**|

**Robustness map (synthetic BioFors, match count at about 1% false positives)**

|How the copy was made|All types|Blot/Gel|Macroscopy|Microscopy|
| :- | :- | :- | :- | :- |
|Clean|**80.3%**|**69.0%**|**90.9%**|**90.7%**|
|Rotated|**33.7%**|**6.0%**|**50.0%**|**61.6%**|
|Scaled|**28.8%**|**8.0%**|**54.5%**|**46.5%**|
|JPEG|**16.5%**|**2.0%**|**23.8%**|**31.8%**|
|Publication-like|**9.2%**|**0.0%**|**9.5%**|**20.0%**|
|Blurred|**4.8%**|**2.0%**|**4.8%**|**8.1%**|
|Mirrored|**1.0%**|**0.0%**|**9.5%**|**0.0%**|

**What the numbers told us**

- Synthetic BioFors training helped the learned model (AUC **0.790** to **0.818**, image MCC **0.218** to **0.329**) but it still lost to the match count.
- C3's lead on synthetic data was never statistically clear (CI touched 0) and reversed on real data: a selection rule should require a lead whose CI excludes 0.
- Our synthetic forgeries were harder than real ones (match-count AUC **0.70** vs **0.90**): **6** of **7** variants transform the patch, real BioFors forgeries mostly behave like clean copies.
- Mirrored copies are invisible to the detector (descriptors are not mirror-invariant); blots fail on anything but a clean copy.

**Decision:** stop tuning on BioFors (test set used twice). Final conclusion: the simple geometric evidence transfers; every learned layer was better in-dataset and worse cross-dataset.

- Honesty note: while testing the code, one configuration identical to C0 was scored once in advance (image MCC **0.466**); nothing was tuned after it.
## **8. V6.7: image-type labels for Recod.ai**
**Why:** Recod.ai has no image-type labels; BioFors has four. Per-type results need both datasets labelled the same way.

**What we did**

- Types: BioFors' four (Blot/Gel, Microscopy, Macroscopy, FACS) plus **Histology** (stained tissue sections) and **Other** (charts, text, unclear).
- Pre-labelled with BiomedCLIP (Microsoft, used as-is, no training).
- Validated BiomedCLIP first on BioFors training images of known type: prompt wording chosen on a small tuning split, accuracy measured on a separate check split (by document).
- Every Recod.ai figure (**2,751**) went through a review tool, least-confident suggestions first.

**BiomedCLIP accuracy on BioFors (check split, 1,558 images; histology counted as microscopy)**

|Type|Correct|
| :- | :- |
|Blot/Gel|**97.5%**|
|Microscopy|**97.5%**|
|Macroscopy|**39.9%**|
|FACS|**100%**|
|Balanced accuracy|**83.7%**|

**Review on Recod.ai**

|Review order (least confident first)|Model confidence|How reviewed|Corrected|
| :- | :- | :- | :- |
|1 to 1,000|**28%** to **99%**|each figure individually|**98**|
|1,001 to 2,000|at least **99.3%**|accepted after spot checks|**5**|
|2,001 to 2,751|at least **99.9%**|accepted after spot checks|**0**|

**Final labels**

|Type|Figures|Forged images|Authentic images|
| :- | :- | :- | :- |
|Microscopy|**1,356**|**1,356**|**1,138**|
|Blot/Gel|**521**|**521**|**411**|
|Macroscopy|**445**|**445**|**445**|
|Histology|**414**|**414**|**373**|
|Other|**15**|**15**|**10**|
|FACS|**0**|**0**|**0**|

- For BioFors comparisons histology is merged into microscopy (BioFors' microscopy includes tissue): Microscopy **1,770** figures.
- Recod.ai has no FACS figures; its macroscopy is almost entirely one corn-ear collection.
- Limit: the histology/microscopy split cannot be validated on BioFors (no such labels there); it rests on the review.
## **9. V7: results by image type (final version)**
**What we did**

- Joined existing scores with the V6.7 labels; no detector rerun, no retraining, no new look at BioFors decisions.
- Recod.ai scores are out-of-fold; cut-offs are the V6 ones (5% false positives on Recod.ai).
- Added one last in-dataset improvement: a decision model specialised per image type.

**Recod.ai by type (learned model, cut-off 0.544)**

|Type|AUC, match count|AUC, learned model|False positives|Detection|Image MCC|Pixel MCC|
| :- | :- | :- | :- | :- | :- | :- |
|Blot/Gel|**0.965**|**0.975**|**5.6%**|**94.0%**|**0.883**|**0.753**|
|Microscopy|**0.874**|**0.901**|**6.8%**|**75.8%**|**0.692**|**0.432**|
|Histology|**0.696**|**0.772**|**2.9%**|**43.5%**|**0.472**|**0.233**|
|Macroscopy|**0.999**|**1.000**|**1.1%**|**100%**|**0.989**|**0.826**|
|Other (15 figures)|**0.823**|**0.933**|**10.0%**|**80.0%**|**0.686**|**0.638**|
|All|**0.832**|**0.925**|**4.9%**|**78.3%**|**0.737**|**0.528**|

**Recod.ai vs BioFors, AUC by type (histology merged into microscopy)**

|Type|Match count: Recod.ai|Match count: BioFors|Learned model: Recod.ai|Learned model: BioFors|
| :- | :- | :- | :- | :- |
|Blot/Gel|**0.965**|**0.890**|**0.975**|**0.753**|
|Microscopy|**0.830**|**0.927**|**0.870**|**0.843**|
|Macroscopy|**0.999**|**0.879**|**1.000**|**0.872**|
|All|**0.832**|**0.901**|**0.925**|**0.790**|

**What the detector sees per type (medians)**

|Dataset and type|Image size (px)|Keypoints|Matches, forged|Exactness, forged (lower = more exact)|Genuine with 3 or more matches|
| :- | :- | :- | :- | :- | :- |
|Recod.ai Blot/Gel|**157 x 660**|**8,000** (cap)|**172**|**0.362**|**81.3%**|
|Recod.ai Microscopy|**512 x 687**|**8,000**|**40**|**0.448**|**49.9%**|
|Recod.ai Histology|**447 x 512**|**8,000**|**8**|**0.521**|**46.1%**|
|Recod.ai Macroscopy|**666 x 1000**|**8,000**|**1,063**|**0.102**|**96.4%**|
|BioFors Blot/Gel|**49 x 295**|**1,827**|**28**|**0.582**|**89.0%**|
|BioFors Microscopy|**171 x 193**|**5,278**|**78**|**0.548**|**84.9%**|
|BioFors Macroscopy|**237 x 225**|**6,100**|**54**|**0.559**|**87.9%**|

**Specialised models (Recod.ai only; cut-off chosen on half the figures, measured on the other half, 40 repeats)**

|Type|Ceiling: forged with any region drawn|Detection at 5% FPR: global to specialised|Image MCC: global to specialised|Detection at about 0% FPR: global to specialised|False positives measured at "0%"|
| :- | :- | :- | :- | :- | :- |
|Blot/Gel|**98.8%**|**94.1%** to **95.0%**|**0.892** to **0.906**|**84.9%** to **82.1%**|**0.5%**|
|Microscopy|**92.7%**|**74.9%** to **75.6%**|**0.705** to **0.711**|**39.3%** to **50.7%**|**0.2%**|
|Histology|**69.1%**|**48.4%** to **50.1%**|**0.486** to **0.508**|**31.5%** to **36.1%**|**0.5%**|
|Macroscopy|**100%**|**100%** to **100%**|**0.953** to **0.953**|**100%** to **99.7%**|**0.4%**|

**What the numbers told us**

- Histology is the hardest type (AUC **0.77**, detection **43.5%**): copies of tissue get few, loose matches (median **8**, exactness **0.521**). BioFors cannot show this because it merges histology into microscopy.
- The detector caps histology: only **69.1%** of forgeries produce any region, so no decision model can exceed that.
- Recod.ai macroscopy is too easy to be representative (AUC **1.000**, exactness **0.102**): large, clean copies on one corn collection; on BioFors it drops to **0.879**.
- Blots explain the transfer failure: BioFors blots are tiny (**49 x 295** px, **1,827** keypoints) with loose copies (**0.582**); learned-model AUC drops **0.975** to **0.753**, match count only **0.965** to **0.890**.
- Specialisation gives **+1** to **+2** points at 5% false positives and up to **+11** points at strict settings (microscopy); pixel MCC barely moves because region drawing, not the decision, limits it.
- Macroscopy: **0.4%** false positives with **100%** detection is achievable; a true 0% cannot be claimed with **445** authentic images.
- Separate per-type cut-offs (instead of specialised models) gave no clear gain.
- Specialised models are in-dataset results only; they were deliberately not tested on BioFors.
## **10. Comparison with related work**
- **19** papers read; **7** selected by relevance (biomedical copy-move, same datasets, or the detector family we build on). Natural-image, CT-tampering and AI-generated-image papers were excluded as a different problem.
- Datasets cited separately: BioFors (Sabir et al., ICCV 2021) and Recod.ai/LUC (Cardenuto et al., 2025).

|Paper|Year, venue|Relation to our work|Their result|Ours|
| :- | :- | :- | :- | :- |
|Bertojo, Neraud, Puech: fast keypoint copy-move detection|2022, Frontiers in Signal Processing|Detector family we build on: keypoint matching with ratio test, DBSCAN clustering, convex-hull masks, low threshold for weak texture|natural-image benchmarks|same family, applied to biomedical figures|
|Nandi, Natarajan: BioTamperNet|2026, ICLR|State of the art on BioFors copy-move; **trained on BioFors**|image MCC **0.701**, pixel MCC about **0.53**|**0.500** / **0.316** with **no** BioFors training|
|Nandi, Natarajan: Rescind / IntegScan|2026, AAAI|Per-modality splits, leave-one-modality-out; argues BioFors-style synthetic forgeries are unrealistic|per-modality MCC, trained in-domain|consistent with our V6.6 finding on synthetic vs real forgeries|
|Tabib et al.: GraphSpecForge|2026, arXiv|Same Recod.ai dataset; training-free; calibrated on authentic images only; tested across datasets|image AUC **0.606**|match count (also training-free) **0.832**; learned model **0.925**|
|Wu: DINOv2 segmentation|2026, AICI|Same Recod.ai dataset; deep model; notes SIFT gives many false positives on biological patterns (our V4 finding)|AUC **0.89**; **55%** detection at **15%** FPR|AUC **0.925**; about **85%** detection at **15%** FPR|
|Qasmi: frozen DINOv2 features|2026, preprint (not peer-reviewed)|Same Recod.ai dataset; deep pixel-level model|pixel F1 **0.563** (validation)|pixel MCC **0.528** (different metric)|
|Sun, Li, Kalluri: VrySure|2026, bioRxiv|Screening platform trained partly on Recod.ai|copy-move F1 **0.605**; performance "moderate" for all tools|supports that the task remains hard|

**Positioning**

- Strongest image-level separation on Recod.ai among these papers (AUC **0.925**); splits differ between papers, so this is indicative, not a controlled comparison.
- On BioFors, without any BioFors training, image MCC (**0.500**) is in the range of published classical methods (**0.509** to **0.569**); BioTamperNet (**0.701**) is better but trained on BioFors.
- Not claimed: state of the art on BioFors; being first to test across datasets (GraphSpecForge and Rescind also do, in other forms).
- Contributions specific to this work: a learned decision layer that improves in-dataset but fails to transfer, with the measured cause; per-type results with histology separated; a robustness map; evaluation that keeps both versions of a figure together.
## **11. Conclusions, limits and next steps**
**Decisions by version**

|Version|Question|Answer from the numbers|Action taken|
| :- | :- | :- | :- |
|V4|Is the baseline usable?|**91.6%** detection but **64.0%** false positives|Look for the cause of false alarms|
|V5|Do text and scale bars cause them?|Mostly no: AUC **0.832** to **0.837** (not significant)|Change the decision rule, keep the detector|
|V5.5|Does a learned decision help?|Yes in-dataset: AUC **0.826** to **0.916** on the locked test|Freeze model, mask OFF, 5% FPR|
|V6|Does it transfer to BioFors?|No: AUC **0.790** vs match count **0.901**|Try a fix without the test set|
|V6.6|Does synthetic BioFors training fix it?|Partly: AUC **0.818**, still below **0.901**|Stop tuning on BioFors|
|V6.7|Can Recod.ai be typed like BioFors?|Yes: **2,751** figures labelled, **103** corrections|Per-type analysis|
|V7|Where does detection fail by type?|Histology hardest (detection ceiling **69.1%**)|Report; specialisation adds **+1** to **+2** points|

**Main findings**

- Natural biological repetition, not text or scale bars, causes most false alarms.
- Match exactness separates real copies from natural repeats within one dataset (AUC **0.87** alone).
- A learned decision layer overfits to its source dataset; the simple count of verified matches transfers (BioFors AUC **0.901**, image MCC **0.500**).
- Cut-off drift between datasets is real and detectable without labels (**26.4%** of genuine BioFors images flagged at a 5% cut-off).
- Detection breaks on mirrored, blurred and compressed copies, on blots, and on histology.

**Limits**

- BioFors has only **102** forgeries (**17** microscopy, **11** macroscopy): wide confidence intervals.
- The BioFors test set was used twice (V6 and V6.6), each with pre-declared settings.
- Recod.ai macroscopy is one corn collection; Recod.ai has no FACS.
- Image types: **1,000** least-confident labels reviewed individually, **1,751** high-confidence ones accepted after spot checks.
- Pixel MCC is limited by the classical detector's region drawing (convex hulls), not by the decision layer.
- Comparisons with other papers use different splits and, for some, different metrics.
