# Methods summary

Draft material for the 4-page technical report. Numbers are from `EXPERIMENTS.md` and
`docs/01_diagnosis.md`; each can be traced to a file in `experiments/results/`.

## 1. Data and validation protocol

- Training data: the 707 competition training images and their annotation file. 296 images carry two or
  three independent annotator records (1,154 records, 8,199 filaments in total). One record per image
  is the training target; all records are used for scoring, as the leaderboard does.
- Validation: 5 folds over the 707 images. Images taken within 7 days of each other share a fold,
  because the same filament is visible on consecutive days and a random split would leak it. Each fold
  holds out about 141 images for testing and about 92 of the rest for checkpoint selection; every image is predicted once by a
  model that never saw it. All reported numbers pool these 707 out-of-fold predictions.
- Metric: the organisers' self-evaluation code (IoU > 0.5 is a match, every annotator record scored
  separately, counts pooled): PQ, SQ, RQ, Dice, and the one-to-many / many-to-one counts.
- Everything fitted after training (the keep rule and its cut-off) is nested-cross-fitted, and every
  kept change is tested with a paired bootstrap over images. The public leaderboard was not used.

## 2. Preprocessing (`pipeline/preprocess.py`)

1. Fit the limb (disk centre and radius) on the image.
2. Remove limb darkening by dividing by the median brightness at each radius.
3. z-score over the analysis disk, radius 0.95 R; the outer ring is not analysed.

Only the image is used. Training tiles are 512 px with 25 % overlap; augmentation is flips, rotation by
any angle about the disk centre, mild blur, intensity scaling and scale jitter.

## 3. Model (`pipeline/model.py`, `TRAINING.md`)

- Encoder: SAM 2.1 Hiera-B+ image encoder (68.6 M weights, frozen), four feature scales.
- Adaptation: LoRA rank 4 on the attention `qkv` and `proj` layers, trainable LayerNorms.
- Decoder: FPN, 256 channels, one output channel. 5.45 M of 74.1 M weights are trained.
- Stage A: Dice + 0.5 x centre-line Dice (clDice), AdamW, cosine schedule.
- Stage B: fine-tune with an added per-component term (missed and spurious connected components are
  penalised), checkpoint chosen by full-disk validation PQ instead of the loss.
- Five fold models; at test time their probability maps are averaged.

Why this model: it was the best of the architectures in our earlier benchmark on MAGFiLO, but by a small
margin; a U-Net reached nearly the same recognition quality there. The architecture is not what limits
the score (see section 5).

## 4. Post-processing

### 4.1 Diagnosis first (`docs/01_diagnosis.md`)

Baseline (single global rule: seed at p >= 0.50 with area >= 400 px and mean p >= 0.85, grown to
p >= 0.45): PQ 0.3943 = SQ 0.665 x RQ 0.593. The loss is in RQ. SQ is already at the level two human
annotators reach on each other (0.634 on their matched pairs). Of 3,341 unmatched annotated filaments, 53 %
were not found at all and 47 % were found but drawn at an extent that misses the IoU 0.5 bar (too short,
split, or merged); each of the latter costs a false negative and a false positive. 36 % of the false
positives are confident dark blobs with no annotation in that record.

### 4.2 Learned keep rule (`pipeline/keep_rule.py`)

Candidates: 8-connected components of {p >= 0.45} on the analysis disk. For each candidate, 22 features:
area, perimeter, length, width, elongation, bounding-box fill; mean, spread and peak of the probability;
mean image value and darkness against a 61 px neighbourhood; distance from disk centre; relations to the
components at the other probability levels (size relative to the component it joins at the next lower
level, number and size of the pieces it splits into at the next higher level, highest level at which it
survives); and two image-level counts. A gradient-boosted regressor (scikit-learn
`HistGradientBoostingRegressor`, 400 trees, learning rate 0.05, leaves of at least 40 candidates) predicts `q`, the IoU the candidate would
earn as a true positive averaged over the annotator records (0 if it matches none).

Decision rule: keep the candidate if `q > tau`. This follows from the metric: PQ = sum IoU /
(TP + FP/2 + FN/2); one more prediction adds its IoU to the numerator if it matches and 1/2 to the
denominator in either case, so it raises PQ exactly when its expected IoU exceeds PQ/2. The fitted
tau is 0.19, close to PQ/2.

What the regressor uses (permutation importance on held-out folds, `experiments/results/keep_rule_importance_k4.json`;
held-out R^2 of `q` 0.42): by far the most important feature is the ratio of the candidate's area to the area
of the region it belongs to at p >= 0.10, i.e. how sharply the network bounds the object. A candidate with a
large low-probability halo is one whose extent the network is unsure of, and those are the ones that miss the
IoU 0.5 bar. Next come the distance from disk centre, the peak probability, the mean probability one level
below, and the perimeter. Area and mean probability, the two quantities the old rule thresholded, rank low
once the others are available.

### 4.3 Test-time augmentation (`pipeline/tta.py`)

Each fold model is run on the image and its three mirror images; the four maps are averaged. Filaments
have no preferred orientation on the disk.

### 4.4 What was tried and rejected

- Choosing the probability level per filament (a component tree over ten levels with the same regressor
  and a dynamic programme): +0.001 to +0.002, not separable from zero; merges more filaments.
- Eight views instead of four: +0.002, not separable from zero, twice the cost.
- Before this work, on the same maps: 4,992 global threshold / area / confidence cells, per-image adaptive
  thresholds, a peak-probability rule, morphological gap closing. None gave more than +0.003.

## 5. Results

| pipeline (input: competition JPEGs) | PQ | SQ | RQ | TP / FP / FN | Dice, matched pairs | one-to-many | many-to-one |
|---|---|---|---|---|---|---|---|
| baseline: one global decode rule, 1 view (our earlier submission, 0.34 on the public board; 0.3943 on its original image cache) | 0.3983 | 0.665 | 0.599 | 4988 / 3470 / 3211 | 0.796 | 528 | 127 |
| + learned keep rule | 0.4103 | 0.667 | 0.616 | 4884 / 2784 / 3315 | 0.797 | 334 | 126 |
| + 4 flip views (**this repository**) | **0.4183** | 0.669 | 0.625 | 4925 / 2634 / 3274 | 0.799 | 291 | 112 |

Gain of the final pipeline over the earlier submission's local score (0.3943): +0.0240, paired bootstrap over
images 95 % interval +0.019 to +0.029.

Context for these numbers: two human annotators scored against each other with the same rule reach PQ 0.340
(SQ 0.634, RQ 0.536) on the 296 multi-annotated training images. The model's agreement with an annotator is
above the agreement between annotators, so further gains mostly measure how well a method imitates the
average annotation style, and should be read with the qualitative results.

Local validation is about 0.05 above the public leaderboard for our earlier submission (0.394 locally,
0.34 public); other teams report the same offset. We did not tune anything to close it.

## 6. Limits

- Outer 5 % of the disk radius is not predicted.
- The keep rule was fitted on single-model maps and is applied to the five-model average at test time;
  this step has no local validation.
- Small or faint filaments that the network gives low probability remain missed; the keep rule can only
  choose among candidates the network proposes.
