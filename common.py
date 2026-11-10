"""
Shared helpers for Part 1 (ML) and Part 2 (DL).

Both parts use the SAME images in the SAME order and the SAME 5 cross-validation folds,
so their accuracies are directly comparable.

Input: the processed images made by mob_pipeline.py (phase 1):
    mc_mob_dataset/processed/color/<mob>/*.png   (224x224, enhanced, colour)
    mc_mob_dataset/processed/gray/<mob>/*.png    (224x224, enhanced, grayscale)

NOTE: do NOT use processed/aug/ for cross-validation. Those files are flipped/rotated copies of
the same photo; if one copy lands in training and another in testing, the test is contaminated
(the model has effectively seen the test image) and accuracy looks better than it really is.
Augmentation is done inside each training fold instead (see part2_dl.py).
"""
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

CLASSES = ["cow", "creeper", "pig", "zombie"]
ROOT = Path("mc_mob_dataset")
RESULTS = Path("results")
SEED = 42
N_FOLDS = 5
EXTS = {".png", ".jpg", ".jpeg"}


def list_images(subdir="color"):
    """Return (paths, labels). subdir is 'color' or 'gray'. Order = class, then filename."""
    base = ROOT / "processed" / subdir
    paths, labels = [], []
    for idx, name in enumerate(CLASSES):
        files = sorted(p for p in (base / name).glob("*") if p.suffix.lower() in EXTS)
        if not files:
            raise SystemExit(f"No images found in {base / name}. Run mob_pipeline.py (meta, crop, process) first.")
        paths += files
        labels += [idx] * len(files)
    y = np.array(labels)
    counts = np.bincount(y)
    print("Images per class:", dict(zip(CLASSES, counts.tolist())))
    if counts.min() < N_FOLDS:
        raise SystemExit(f"Need at least {N_FOLDS} images per class for {N_FOLDS}-fold cross-validation.")
    return paths, y


def get_folds(y):
    """5 stratified folds (each fold keeps the class balance). Same folds for every method."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros(len(y)), y))
