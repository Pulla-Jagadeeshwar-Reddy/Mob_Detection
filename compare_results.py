"""
Combine Part 1 and Part 2 results into one comparison table + bar chart.
Run AFTER part1_ml.py and part2_dl.py:   python compare_results.py
Output: results/final_comparison.csv, results/final_comparison.png
"""
import csv
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import RESULTS  # noqa: E402

rows = []
p1 = RESULTS / "part1_results.json"
if p1.exists():
    configs = json.load(open(p1))["configs"]
    for det in ("sift", "harris"):
        best = max((c for c in configs if c["detector"] == det), key=lambda c: c["mean_acc"])
        rows.append((f"{det.upper()} + BoVW (k={best['k']}) + SVM", best["mean_acc"], best["std_acc"]))
p2 = RESULTS / "part2_results.json"
if p2.exists():
    for name, r in json.load(open(p2))["models"].items():
        rows.append((f"{name} (frozen) + softmax", r["mean_acc"], r["std_acc"]))
if not rows:
    raise SystemExit("No results found. Run part1_ml.py and/or part2_dl.py first.")

print(f"{'Method':<42}{'CV accuracy':>12}{'std':>8}")
for m, a, s in rows:
    print(f"{m:<42}{a:>12.3f}{s:>8.3f}")

with open(RESULTS / "final_comparison.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "cv_accuracy", "std_over_folds"])
    w.writerows(rows)

fig, ax = plt.subplots(figsize=(8, 0.9 * len(rows) + 1.5))
ax.barh([r[0] for r in rows][::-1], [r[1] for r in rows][::-1], xerr=[r[2] for r in rows][::-1], color="#4C78A8")
ax.set_xlim(0, 1)
ax.set_xlabel("5-fold cross-validated accuracy (error bar = std over folds)")
for i, r in enumerate(rows[::-1]):
    ax.text(min(r[1] + r[2] + 0.01, 0.9), i, f"{r[1]:.2f}", va="center")
fig.tight_layout()
fig.savefig(RESULTS / "final_comparison.png", dpi=130)
print("Saved results/final_comparison.csv and .png")
