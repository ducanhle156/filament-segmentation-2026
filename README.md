# Solar filament instance segmentation (GONG H-alpha, MAGFiLO)

Entry for the Solar Filament Segmentation Challenge 2026. From a 2048 x 2048 H-alpha image the pipeline
returns one mask per filament. Everything needed to reproduce the submitted predictions is in this
repository, including the weights; nothing is retrained.

```
test image ─► preprocess ─► probability map ─► candidates ─► keep rule ─► CSV
              limb fit,      SAM 2.1 Hiera-B+   connected     learned,     filament_id,
              flattening,    + LoRA + FPN,      components    per          segmentation_rle
              z-score        5 folds x 4 flips  of p >= 0.45  candidate    (COCO RLE)
```

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
