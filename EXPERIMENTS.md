# Experiment log — try_error (2026-11-13)

Local validation for every row: the 707 competition train disks, each predicted by the fold model that never
saw it (`kaggle/splits/folds_k707_v1.json`: 5 folds, grouped by 7-day observation window), scored with the
official self-evaluation rule against all 1,154 annotator records, counts pooled. Anything fitted on labels
(the keep-rule regressor, its cut-off tau) is nested-cross-fitted: for held-out fold F it is fitted on the
other four folds only. The public leaderboard was not used to choose anything.

Rule for keeping a change (set before the runs): validation PQ improves and neither match count gets worse.
Stop after two consecutive ideas under +0.003.

| # | change (one at a time) | PQ | SQ | RQ | TP / FP / FN | Dice, matched | one-to-many | many-to-one | Δ PQ vs previous kept | kept? |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | baseline: submitted pipeline (kag13 fold model, hysteresis 0.50 -> 0.45, area >= 400, mean p >= 0.85) | 0.3943 | 0.665 | 0.593 | 4858 / 3322 / 3341 | 0.796 | 527 | 118 | — | — |
| 1 | learned keep rule on the components of p >= 0.45 (`flat@0.45`) | **0.4087** | 0.665 | 0.614 | 4827 / 2691 / 3372 | 0.796 | 336 | 115 | **+0.0144** (bootstrap SD 0.0020, 95 % CI +0.0105 to +0.0185) | **yes** |
| 2 | per-object threshold, component tree over all levels (`tree`) | 0.4100 | 0.667 | 0.615 | 4813 / 2649 / 3386 | 0.797 | 320 | 164 | +0.0013 | no (many-to-one 115 -> 164; gain under 0.003) |
| 3 | test-time augmentation, 4 flip views, then the keep rule (re-fitted on the new maps) | **0.4143** | 0.668 | 0.620 | 4779 / 2437 / 3420 | 0.798 | 281 | 101 | **+0.0055** (bootstrap SD 0.0020, 95 % CI +0.0016 to +0.0095) | **yes** |
| 4 | 8 views (flips + 90-degree rotation) instead of 4 | 0.4160 | 0.668 | 0.622 | 4778 / 2377 / 3421 | 0.798 | 246 | 108 | +0.0017 (95 % CI -0.0011 to +0.0046) | no (many-to-one 101 -> 108; gain under 0.003; twice the inference time) |

Per-fold PQ: row 0 = 0.385 / 0.382 / 0.393 / 0.398 / 0.414; row 1 = 0.398 / 0.405 / 0.407 / 0.412 / 0.425
(all five folds improve); row 2 = 0.400 / 0.406 / 0.406 / 0.421 / 0.419; row 3 = 0.396 / 0.412 / 0.413 /
0.422 / 0.430; row 4 = 0.406 / 0.413 / 0.411 / 0.422 / 0.429.

## The pipeline as shipped: input = the competition's own JPEGs

Rows 0-4 ran on the project's image cache (the same frames after an RGB re-encode), which is what the fold
models were trained on. The submission notebook must start from the competition JPEGs, so the kept changes
were re-measured on the 707 training JPEGs taken from the competition folder and preprocessed by the same
code the notebook uses (`prepare_kaggle_train.py`; mean absolute difference to the cache 0.27 z-score units).
The keep rule was re-fitted on these maps with the same nested cross-fit.

| # | pipeline on competition JPEGs | PQ | SQ | RQ | TP / FP / FN | Dice, matched | one-to-many | many-to-one | Δ PQ | note |
|---|---|---|---|---|---|---|---|---|---|---|
| J0 | submitted decode, 1 view | 0.3983 | 0.665 | 0.599 | 4988 / 3470 / 3211 | 0.796 | 528 | 127 | +0.0040 vs row 0 (95 % CI +0.0003 to +0.0079) | the input change itself |
| J1 | + learned keep rule | 0.4103 | 0.667 | 0.616 | 4884 / 2784 / 3315 | 0.797 | 334 | 126 | +0.0120 vs J0 (95 % CI +0.0081 to +0.0159) | |
| J2 | + 4 flip views (**final pipeline**) | **0.4183** | 0.669 | 0.625 | 4925 / 2634 / 3274 | 0.799 | 291 | 112 | +0.0079 vs J1 (95 % CI +0.0040 to +0.0119) | |
| J3 | + fill holes inside kept filaments | 0.4186 | 0.669 | 0.625 | 4927 / 2632 / 3272 | 0.799 | 291 | 112 | +0.0003 vs J2 | not adopted: 2 more matches out of 4,925, 0.08 % of the mask pixels change; not separable from zero, so the pipeline stays one step shorter |

Per-fold PQ: J1 = 0.398 / 0.407 / 0.410 / 0.409 / 0.429; J2 = 0.402 / 0.417 / 0.416 / 0.422 / 0.435.

Stop: two consecutive ideas under +0.003 (8 views, hole filling), as set at the start. The remaining entries
of the ranked list (`docs/02_ideas_ranked.md`) need retraining or other models.

Check of the scoring path: `validate_masks.py` renders the final selection to masks, as the submission does, and
scores the masks; it reproduces J2 exactly (4925 / 2634 / 3274; `results/validate_masks_k4.json`).

**Final local validation PQ: 0.4183, against 0.3943 for the pipeline behind the 0.34 public score:
+0.0240 (paired bootstrap over the 707 images: SD 0.0025, 95 % CI +0.0189 to +0.0289).** SQ 0.665 -> 0.669,
RQ 0.593 -> 0.625; one-to-many 527 -> 291, many-to-one 118 -> 112.

This is a local-validation number. The earlier submission scored 0.394 locally and 0.34 on the public board;
the same offset would put the new file near 0.36 (see "After the pipeline was frozen" for the measured value).

Ablation (competition JPEGs, same folds): old decode with 4 views 0.4067 (4896 / 3000 / 3303, one-to-many 462,
many-to-one 116); keep rule with
1 view 0.4103; both 0.4183. The two changes add up almost independently.

Files: `results/fit_k1.json`, `results/fit_k4.json`, `results/bootstrap_pairs.json`,
logs `logs/805987.bright04.OU`, `logs/805988.bright04.OU`.

## After the pipeline was frozen

**Test CSV.** `filament_pipeline.ipynb` was executed on the 180 test JPEGs (job 805990): 1,190 filaments on 176
images, no overlapping masks; copy in `kaggle/try_error/submissions/2026-11-13_keep_tta4.csv`.

**Public leaderboard.** That one file was uploaded once, after everything above was fixed (2026-11-13 12:04 UTC):
public PQ **0.36**, against 0.34 for the earlier submission. The board shows two decimals. No setting was
changed afterwards and no other file was uploaded.

**Read-only check against the public MAGFiLO labels of the test images** (`kaggle/src/evaluate.py --stage
local180`, the same check the earlier submission got). These labels were never used for training or for any
choice; the file was scored once, after it was written. It is the only measurement here that includes the
five-model average, which local validation cannot test.

| CSV | PQ | SQ | RQ | TP / FP / FN | Dice, overlapping pairs | one-to-many | many-to-one |
|---|---|---|---|---|---|---|---|
| earlier submission `2026-10-20_ens5.csv` (public board 0.34) | 0.3869 | 0.665 | 0.582 | 1136 / 726 / 909 | 0.688 | 110 | 42 |
| this pipeline `2026-11-13_keep_tta4.csv` | 0.4141 | 0.668 | 0.619 | 1189 / 605 / 856 | 0.712 | 58 | 38 |

**Cross-model averaging (idea 5), measured after the stop because the U-Net maps arrived:** mean of the SAM2
4-view map and the U-Net map (U-Net alone: 0.3534 with its own decode), keep rule re-fitted: PQ 0.4000
(4670 / 2606 / 3529, one-to-many 302, many-to-one 148) against 0.4143 for SAM2 alone (row 3). With weights
0.75 / 0.25: 0.4151 (4818 / 2481 / 3381, one-to-many 286, many-to-one 121), i.e. +0.0008 with more merges
(101 -> 121). Rejected in both forms (`results/fit_u50.json`, `results/fit_u25.json`).

## Side measurements (not separate ideas)

| what | PQ | TP / FP / FN | one-to-many | many-to-one | note |
|---|---|---|---|---|---|
| keep rule on p >= 0.30 components | 0.3997 | 4715 / 2673 / 3484 | 305 | 133 | lower level merges neighbours |
| keep rule on p >= 0.40 components | 0.4065 | 4670 / 2408 / 3529 | 287 | 117 | |
| keep rule on p >= 0.50 components | 0.4004 | 4551 / 2400 / 3648 | 279 | 96 | higher level shortens filaments |
| tree, levels 0.40-0.60 | 0.4102 | 4742 / 2484 / 3457 | 311 | 119 | +0.0015 vs row 1, bootstrap 95 % CI -0.0008 to +0.0037 |
| tree, levels 0.45-0.90 | 0.4105 | 4736 / 2509 / 3463 | 302 | 113 | +0.0018 vs row 1, bootstrap 95 % CI -0.0010 to +0.0045 |
| tree, levels 0.30-0.90 | 0.4100 | 4762 / 2565 / 3437 | 308 | 129 | +0.0012 vs row 1, bootstrap 95 % CI -0.0016 to +0.0041 |

Reading: the keep rule is a real gain (the bootstrap interval excludes zero, every fold improves, and
fragmentation drops by a third because small pieces next to a matched filament are now removed). Choosing the
level per object is not: three tree variants give +0.001 to +0.002, none separable from zero, and picking the
best of them after the fact would be selection on noise. The simpler single-level rule is kept.

Test-time augmentation is a real gain as well (interval excludes zero) and lowers both match counts: averaging
mirrored views smooths the probability map, so fewer filaments break into pieces. Going from 4 to 8 views is
not separable from zero and is not adopted.

Files: `results/fit_tta4.json`, `results/fit_tta8.json`, `results/bootstrap_pairs.json`, `results/fit_kag13.json`, `results/fit_kag13_variants.json`, `results/bootstrap_kag13.json`;
logs `logs/805970.bright04.OU`, `logs/805971.bright04.OU`.
