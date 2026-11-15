# Steps 1 and 3 — ideas from the discussion board and the diagnosis, ranked

Source: all 52 threads of the competition discussion board with comments, pulled 2026-11-13
(`docs/kaggle/forum_digest_2026-11-13.md`, fetched with `docs/kaggle/fetch_forum.py`), and the ten public
notebooks saved in `docs/kaggle/public_notebooks_2026-11-12/`. Thread links are
`https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/<id>`, abbreviated `d/<id>`.

Most threads are rulings and leaderboard complaints; few contain a method. Gains in the "claimed" column are
the authors' numbers on their own splits and were **not** verified here. "Expected" is my estimate for this
pipeline before measuring, from the diagnosis (`01_diagnosis.md`); the measured value is in `EXPERIMENTS.md`.

## Ranked table (expected PQ gain per hour of work, best first)

| # | idea | source | claimed gain | expected here | cost | gain / hour | status |
|---|---|---|---|---|---|---|---|
| 1 | **Learned keep rule**: score each candidate component from its shape, confidence, darkness and position instead of three global thresholds | notebook "Solar Filament U-Net: OOF 0.45 vs Public 0.39" (D. Afshar); ensemble + "small learned post-processing filter" also in [d/753665](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/753665) | +0.0058 (bootstrap SD 0.0014) | +0.003 to +0.008 (1,210 pure FP, 28 % of misses under the area floor) | 2 h, CPU only | high | **kept: +0.0144** on the cache maps, +0.0120 on competition JPEGs (rows 1, J1) |
| 2 | **Per-object threshold (component tree)**: choose every filament at its own level of the probability map; covers threshold tuning, island removal, splitting and merging | diagnosis: 47 % of FN are found but at the wrong extent; per-image oracle 0.4724 vs 0.3943 | — | +0.005 to +0.02 | +1 h on top of #1, CPU only | high | rejected: +0.0013 to +0.0018, interval includes zero (row 2) |
| 3 | **Flip / rotation test-time augmentation** (D4, 8 views) | same notebook (TTA x8); "four-view TTA with topology guards" (Fususu handoff notebook) | not isolated by the authors | +0.002 to +0.008 | 1 h code + 2 h GPU | medium | **kept with 4 flips: +0.0055** / +0.0079 on competition JPEGs (rows 3, J2); 8 views rejected (+0.0017, row 4) |
| 4 | **Multi-annotator training** (train on every annotator record, not one per image) | [d/753816](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/753816), [d/753043](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/753043); YOLO26 notebook (union of annotators, LB 0.36) | none stated | +0.000 to +0.010 | already running (jobs 805948-805959, ~6 h GPU) | medium | still training when this log closed; not part of the shipped pipeline |
| 5 | **Cross-model averaging** (SAM2 + ImageNet U-Net; folds x resolutions) | [d/753665](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/753665) (10-network ensemble); 0.45-OOF notebook | not isolated | +0.000 to +0.008 (our U-Net and SAM2 make the same errors, Phase 6) | U-Net already training (805889-805899) + 1 h | medium | rejected: equal weights 0.4000, 0.75 / 0.25 weights 0.4151 vs 0.4143 for SAM2 alone, with more merges |
| 6 | Gap bridging along the filament axis (principal-axis endpoint linking) | ASTH-Net notebook ("APAG-Bridge"); rubric's one-to-many count | none measured | <= +0.003 (242 fragmented GT; closing gave +0.0034 on the 958 set, `post_processing/gap_linking`) | 3 h | low | not run; subsumed by #2 (a lower level joins fragments when the score says so) |
| 7 | Hysteresis (strict seed, permissive grow) | ASTH-Net notebook ("DTH-BR") | none measured | 0 (already in the submitted decode: +0.0135) | — | — | already adopted |
| 8 | Threshold / min-area / min-probability tuning, peak rule, per-image adaptive threshold | 0.45-OOF notebook keep rule; [d/735256](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/735256) (host: clean tiny detections in post-processing) | — | 0 (4,992 + 3,920 cells searched 2026-11-12: best cross-fitted +0.0003) | — | — | closed (`kaggle/docs/04*.md`) |
| 9 | Focal + Dice loss, class re-weighting | [d/744765](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/744765) | author later reports no success | < +0.005 | 18 h retrain | very low | not run |
| 10 | Spine / centre-line auxiliary head (allowed: provided metadata) | [d/755241](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/755241), [d/736543](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/736543) | none | unknown; clDice in our loss already rewards the centre line | 20 h+ retrain | very low | not run |
| 11 | Decoupled detect-then-refine pipeline (detector + crop refiner) | [d/744765](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/744765) (0.48 on a random, ungrouped 106-image split) | 0.44 -> 0.48, unverifiable split | unknown | days | very low | not run |
| 12 | Label transfer to neighbouring GONG frames (training only) | [d/748045](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/748045), [d/749225](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/749225) | none | ~0 (three uses of unlabeled frames gave nothing here; archive rate-limits) | days | very low | not run |
| 13 | Fine structures / barbs (host's advice) | [d/744765](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/744765) comment by the host | qualitative | SQ only; SQ is at the annotator level | — | — | judged in the gallery, not a PQ lever |

Added from the notebook figures after the table was first written: filling pinholes inside kept filaments
(no thread; morphology): +0.0003, not adopted (row J3).

## Ignored on purpose

| item | thread | why |
|---|---|---|
| Duplicate masks inflate the self-evaluation score | [d/755906](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/755906) | metric exploit; the leaderboard rejects overlapping masks; our masks are disjoint by construction |
| Public MAGFiLO labels of the test images, the 0.55 CSV | [d/728062](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/728062), [d/729396](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/729396), [d/743958](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/743958) | banned ground truth |
| Neighbouring frames, time or station from the file name at inference | [d/743385](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/743385), [d/744047](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/744047), [d/753665](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/753665) | ruled out by the host: inference uses the test image only |
| SWEFil labels | [d/746318](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/746318) | external ground truth, ruled out |
| Fitting anything to the public score (including the "sparser test labels" correction PQ') | 0.45-OOF notebook | leaderboard probing; we report it as an explanation of the 0.055 gap only |

## Rulings that shape the package (Step 5)

- Weights must be in the repository and loaded by the demo notebook; only the final model, not training checkpoints ([d/755907](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/755907), "Organizers' Remarks" [d/724503](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/724503)).
- The notebook shows preprocessing -> training (optional switch) -> post-processing -> evaluation, and must run on another machine ([d/755907](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/755907), [d/753665](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/753665)).
- An ensemble may be "the best model" ([d/753665](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/753665)); speed is a filter, not a ranking factor ([d/749526](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/749526)).
- No overlapping masks ([d/753816](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/753816)).
- Technical reports that read as unreviewed AI output count against the team ([d/724503](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/724503)).
