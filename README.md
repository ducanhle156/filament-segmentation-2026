# Solar filament instance segmentation (GONG H-alpha, MAGFiLO)

Entry for the Solar Filament Segmentation Challenge 2026. From a 2048 x 2048 H-alpha image the pipeline
returns one mask per filament. Everything needed to reproduce the submitted predictions is in this
repository, including the weights; nothing is retrained.

## Our approach

The method has two parts.

1. **A segmentation network built from existing components.** The image encoder of SAM 2.1 (Hiera-B+, pretrained
   by Meta) is kept frozen and adapted with LoRA (rank 4) and trainable LayerNorms; an FPN decoder turns its four
   feature scales into one filament-probability map. Only 5.45 M of 74.1 M weights are trained, which suits a
   training set of 707 images. Five fold models and four mirrored views are averaged.
2. **A keep rule derived from the metric.** Connected components of the probability map (p >= 0.45) are the
   candidates. Panoptic Quality is `sum IoU / (TP + FP/2 + FN/2)`, so reporting one more candidate raises the
   score only if its expected IoU is above PQ/2. A gradient-boosted regressor predicts that IoU from 22 features
   of the candidate (shape, probability, image contrast, position, how sharply the network bounds it), and the
   candidate is kept if the prediction exceeds a single cut-off (0.19, close to PQ/2).

Why the second part: an error analysis of our first submission showed that the score was lost in recognition
(RQ 0.59), not in mask quality (SQ 0.67, already above the agreement between two human annotators). Global area
and confidence thresholds could not separate good candidates from bad ones; a per-candidate decision could.

```
test image ─► preprocess ─► probability map ─► candidates ─► keep rule ─► CSV
              limb fit,      SAM 2.1 Hiera-B+   connected     learned,     filament_id,
              flattening,    + LoRA + FPN,      components    per          segmentation_rle
              z-score        5 folds x 4 flips  of p >= 0.45  candidate    (COCO RLE)
```

Details: `METHODS.md`. Every experiment with its numbers: `EXPERIMENTS.md`.

## What we tried

All of the following were trained on the 707 competition training images only and evaluated the same way
(see the table below).

| approach | idea | outcome |
|---|---|---|
| U-Net | a light convolutional model trained from scratch with the same data, folds and Dice + clDice loss | clearly below SAM 2 (many more false positives) |
| SAM 2.1 + LoRA + FPN, global decode | area and confidence thresholds on the probability map, settings chosen on validation folds | our first submission |
| Multi-annotator fine-tuning | continue training with every annotator record as an extra target | no gain |
| Self-training with pseudo-labels | a teacher labels unlabeled GONG frames, a student retrains on labeled + pseudo-labeled data (two variants: frames chosen by DINOv2 nearest neighbours; pseudo-labels filtered by agreement under flips) | +0.002 and +0.006, within fold-to-fold noise; not adopted |
| Decode search | 4,992 global threshold / area / confidence settings, per-image adaptive thresholds, a peak-probability rule, gap closing | none above +0.003 |
| **Learned keep rule** | per-candidate decision from the PQ break-even point | **+0.012 to +0.014, adopted** |
| **Four mirrored views** | average the model over flips | **+0.006 to +0.008, adopted** |
| Per-filament threshold (component tree) | choose the probability level per object | +0.001 to +0.002, not separable from zero |
| Eight views; hole filling; averaging with the U-Net | | +0.002; +0.0003; +0.001 with more merged filaments; not adopted |

## Comparison

707 training images, each predicted by a fold model that never saw it; organisers' self-evaluation rule; every
annotator record scored (1,154 records). One-to-many = annotated filaments overlapped by several predictions;
many-to-one = predictions overlapping several annotated filaments.

| approach | PQ | SQ | RQ | TP / FP / FN | one-to-many | many-to-one | public board |
|---|---|---|---|---|---|---|---|
| Two human annotators against each other (296 multi-annotated images) | 0.340 | 0.634 | 0.536 | 4292 / 3709 / 3709 | | | |
| U-Net, global decode | 0.3534 | 0.655 | 0.540 | 5027 / 5406 / 3172 | 610 | 183 | |
| SAM 2.1 + LoRA + FPN, multi-annotator fine-tuning, global decode | 0.3895 | 0.662 | 0.588 | 4859 / 3469 / 3340 | 521 | 139 | |
| SAM 2.1 + LoRA + FPN, global decode (first submission) | 0.3943 | 0.665 | 0.593 | 4858 / 3322 / 3341 | 527 | 118 | 0.34 |
| + self-training, flip-agreement filter | 0.4004 | 0.664 | 0.603 | 4884 / 3123 / 3315 | 466 | 132 | 0.34 |
| **SAM 2.1 + LoRA + FPN + keep rule + 4 views (this repository)** | **0.4183** | **0.669** | **0.625** | 4925 / 2634 / 3274 | **291** | **112** | **0.36** |

The submitted pipeline has the highest PQ, SQ and RQ and the fewest fragmented and merged filaments. Against
the first submission: +0.0240 PQ (paired bootstrap over images, 95 % interval +0.019 to +0.029), 21 % fewer
false positives and 45 % fewer one-to-many cases. Rows 2 to 5 read our own cached copies of the same frames (a different JPEG encoding), the last row
the competition JPEGs, as the notebook does (the first-submission decode scores 0.3983 on the JPEGs).

## Run it

```bash
git clone https://github.com/ducanhle156/filament-segmentation-2026.git && cd filament-segmentation-2026
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126   # or the build for your CUDA
pip install -r requirements.txt

export FILAMENT_TEST_DIR=/path/to/test/test_images          # the competition's test JPEGs
# optional, only for the metric demo in the notebook:
export FILAMENT_TRAIN_DIR=/path/to/train/train_images
export FILAMENT_TRAIN_JSON=/path/to/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json

jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 filament_pipeline.ipynb
# -> outputs/submission.csv
```

Or without the notebook: `python -m pipeline.run --test-dir $FILAMENT_TEST_DIR --out outputs/submission.csv`.
On Kaggle the notebook finds `/kaggle/input/filament-segmentation-2026` by itself.
`FILAMENT_LIMIT=4` runs four images as a quick check.

Hardware: one GPU with 8 GB is enough. Measured: 20.9 s per image on an NVIDIA A40 (63 min for the 180 test images, peak GPU memory 4.7 GiB), 46 s per image on an NVIDIA L4. Results on two GPUs differ in the last bits of the probability map (mixed precision): the same filaments are found and more than 99.8 % of the mask pixels agree.
The first run joins the three encoder parts in `weights/sam2/` into one file (275 MB).

## What is where

| path | content |
|---|---|
| `filament_pipeline.ipynb` | the whole pipeline with figures: preprocessing, model, keep rule, CSV, evaluation, gallery |
| `submitted/2026-11-13_keep_tta4.csv` | the exact file uploaded to Kaggle (public leaderboard 0.36); the notebook writes the same predictions to `outputs/submission.csv` |
| `pipeline/` | the code the notebook calls, one module per step (see `pipeline/__init__.py`) |
| `weights/` | encoder trunk (3 parts), five fold files (22 MB each), the keep-rule regressor, `MANIFEST.json` with checksums |
| `configs/`, `splits/` | the two training-stage configs and the grouped 5-fold split of the 707 training images |
| `common/`, `models/sam2/src/` | the training / inference engine (dataset, augmentation, losses, tiling, metrics, model registry) |
| `third_party/sam2/` | Hiera backbone source from facebookresearch/sam2 (Apache-2.0), three files |
| `training/`, `TRAINING.md` | how the weights were produced, step by step |
| `experiments/` | scripts and result files behind every number in `EXPERIMENTS.md` |
| `METHODS.md` | methods summary: preprocessing, model, post-processing, validation |
| `EXPERIMENTS.md` | experiment log, before / after metrics for every change tried |
| `docs/` | error diagnosis of the baseline, the ranked list of ideas with sources |

## Results (local validation)

707 training images, each predicted by a fold model that never saw it (folds grouped by observation week),
scored with the organisers' self-evaluation rule against every annotator record.

| pipeline (input: competition JPEGs) | PQ | SQ | RQ | TP / FP / FN | Dice, matched pairs | one-to-many | many-to-one |
|---|---|---|---|---|---|---|---|
| baseline: one global decode rule, 1 view (our earlier submission, 0.34 on the public board; 0.3943 on its original image cache) | 0.3983 | 0.665 | 0.599 | 4988 / 3470 / 3211 | 0.796 | 528 | 127 |
| + learned keep rule | 0.4103 | 0.667 | 0.616 | 4884 / 2784 / 3315 | 0.797 | 334 | 126 |
| + 4 flip views (**this repository**) | **0.4183** | 0.669 | 0.625 | 4925 / 2634 / 3274 | 0.799 | 291 | 112 |

Gain of the final pipeline over the earlier submission's local score (0.3943): +0.0240, paired bootstrap over
images 95 % interval +0.019 to +0.029.

Details, confidence intervals and the changes that were tried and rejected: `EXPERIMENTS.md`.

## Rules followed

- Training used the 707 competition training images and their annotation file only. No test image, no test
  label, no external annotation (the public MAGFiLO release was not used for the stored weights).
- Pretrained weights: SAM 2.1 Hiera-B+ image encoder (Meta, Apache-2.0).
- Inference reads the test image and nothing else: no neighbouring frame, no observation time or station.
- The public leaderboard was not used to choose any setting; every choice was made on the local validation above.
- Predicted masks of one image never overlap (they are connected components of one map); the notebook checks it.

## Known limits

- Nothing is predicted in the outer 5 % of the disk radius.
- The keep-rule regressor was fitted on maps of single fold models; at test time it is applied to the average
  of five. This step could not be validated locally (an out-of-fold image has only one model that never saw it).
- Long, thin, faint filaments are the typical miss: the network gives them low probability, and the keep rule
  can only choose among candidates the network proposes.
- `scikit-learn` must stay at the pinned version to unpickle `weights/keep_rule.pkl`.
- Training was run in the research repository this package was exported from and was not repeated from here.
