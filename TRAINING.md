# How the stored weights were trained

Inference needs none of this: `filament_pipeline.ipynb` loads the stored weights. This file records
the training so it can be repeated. It was run inside the research repository this package was
exported from (same `common/` engine, same configs); the commands below are the package-relative
equivalents. **The training path was not re-executed from this standalone package** (about 48 GPU
hours); the inference path was.

Data: the 707 competition training images and
`MAGFiLO_1.0_Annotations_kaggle2026_train.json` only. No test image, no test label, no external
annotation. One annotator record per image is used as the training target (296 images have two
or three records; the record was drawn once with seed 42); all records are used for scoring.

| step | what | command | cost |
|---|---|---|---|
| 1 | Preprocess every training image (limb fit, limb-darkening removal, z-score on the 0.95 R disk), rasterise its polygons, cut 512 px tiles with 25 % overlap | `python training/preprocess.py --config common/configs/preprocess_v1.yaml --steps 6 7 9 10` | 20 min, 32 CPU |
| 2 | Five folds over the 707 images, grouped so that images within 7 days of each other share a fold (the Sun rotates slowly: the same filament appears on consecutive days), 15 % of each training part held out for validation | `python training/build_splits_k707.py` -> `splits/folds_k707_v1.json` (stored) | seconds |
| 3 | Stage A, per fold: SAM 2.1 Hiera-B+ encoder (frozen) + LoRA rank 4 on attention qkv/proj + trainable LayerNorms + FPN decoder; loss Dice + 0.5 clDice; AdamW, lr 1e-4 (decoder) / 1e-5 (encoder side), cosine schedule, flips + full rotation + blur + intensity/scale jitter; early stopping on validation | `python training/run_fold.py --config configs/kag09_sam2_lora.yaml --fold F --seed 0` | 2 to 10 h per fold (A100 or L4, 6.9 GiB peak memory); 31.5 GPU-hours for the five folds |
| 4 | Stage B, per fold, warm-started from stage A: adds a component-level loss term (penalises missed and spurious connected components), lr 3e-5 / 3e-6, 40 epochs; the checkpoint is selected by full-disk validation PQ, not by the loss | `python training/run_fold.py --config configs/kag13_sam2_pqsel.yaml --fold F --seed 0` | 2 to 5 h per fold; 16.6 GPU-hours for the five folds |
| 5 | Out-of-fold probability maps of all 707 images (each image by the fold model that never saw it) | `python experiments/infer.py --source kaggle --folds 0 1 2 3 4` | 2 h GPU |
| 6 | Keep rule: candidate features and nested cross-fit, then the deployment fit on all 707 | `python experiments/tree_experiment.py --stage extract ...` then `--stage fit --final flat@0.45` | 20 min CPU |
| 7 | Export the weights into `weights/` | `python experiments/export_weights.py` | 1 min |

`experiments/` holds the scripts exactly as they were run on the cluster (they read the research
repository's caches through `experiments/lib.py`; paths there are cluster paths). Their outputs
are in `experiments/results/` and summarised in `EXPERIMENTS.md`.
