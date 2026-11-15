"""The few numbers that define the submitted pipeline (chosen on local validation, see EXPERIMENTS.md)."""
FOLDS = (0, 1, 2, 3, 4)      # fold models whose probability maps are averaged
N_VIEWS = 4                  # test-time augmentation views (1 = none, 4 = flips, 8 = flips + 90-degree rotation)
OVERLAP = 0.50               # tile overlap when stitching 512 px tiles into the full disk
KEEP_LEVEL = 0.45            # candidates are the 8-connected components of {p >= KEEP_LEVEL}
KEEP_FILE = "weights/keep_rule.pkl"
H = W = 2048
