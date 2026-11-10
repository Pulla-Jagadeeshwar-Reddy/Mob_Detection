"""
PART 1 (classical machine learning): SIFT and Harris corners -> Bag of Visual Words -> SVM + cosine similarity

Pipeline (done separately for each detector, inside every cross-validation fold):
  1. Detect keypoints         SIFT detector   OR   Harris corner detector
  2. Describe each keypoint   128-number SIFT descriptor (Harris only finds corners, so we describe
                              the Harris corners with SIFT descriptors; this isolates the detector
                              as the only difference between the two pipelines)
  3. Build a vocabulary       k-means on the descriptors of the TRAINING images -> k "visual words"
  4. Image -> histogram       count how many of the image's descriptors fall in each visual word
  5. Classify                 SVM on the L2-normalised histograms (settings chosen by an inner grid search)
  6. Cosine similarity        between histograms: class-vs-class similarity matrix and a
                              1-nearest-neighbour (by cosine similarity) classifier

Run:   python part1_ml.py                       (default k = 25, 50, 100)
       python part1_ml.py --k_values 50
Needs: mc_mob_dataset/processed/gray/<cow|creeper|pig|zombie>/*.png   (from mob_pipeline.py)
Output: results/part1_results.json, part1_summary.csv, part1_keypoints.png,
        part1_confusion.png, part1_similarity.png
"""
import argparse
import csv
import json

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.cluster import MiniBatchKMeans  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402
from sklearn.preprocessing import normalize  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from common import CLASSES, RESULTS, SEED, get_folds, list_images  # noqa: E402

PARAM_GRID = [
    {"kernel": ["linear"], "C": [0.1, 1, 10, 100]},
    {"kernel": ["rbf"], "C": [0.1, 1, 10, 100], "gamma": ["scale", 0.1, 1, 10]},
]
MAX_KMEANS_DESCRIPTORS = 50000   # subsample if there are more (keeps k-means fast)


# ---------------------------------------------------------------- keypoints and descriptors
def harris_keypoints(gray, max_points=300, block=2, ksize=3, k=0.04, rel_thresh=0.01, size=16.0):
    """Harris corner response -> keep local maxima above a threshold -> strongest `max_points`."""
    resp = cv2.cornerHarris(np.float32(gray), block, ksize, k)
    if resp.max() <= 0:
        return []
    is_peak = resp == cv2.dilate(resp, np.ones((5, 5), np.uint8))      # non-maximum suppression
    ys, xs = np.where(is_peak & (resp > rel_thresh * resp.max()))
    order = np.argsort(-resp[ys, xs])[:max_points]
    return [cv2.KeyPoint(float(xs[i]), float(ys[i]), size) for i in order]


def extract_all(paths, detector, max_harris):
    """List of (n_keypoints x 128) SIFT descriptor arrays, one per image."""
    sift = cv2.SIFT_create()
    out = []
    for p in paths:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if detector == "sift":
            _, d = sift.detectAndCompute(g, None)
        else:
            kps = harris_keypoints(g, max_harris)
            _, d = sift.compute(g, kps) if kps else (None, None)
        out.append(d if d is not None else np.zeros((0, 128), np.float32))
    return out


# ---------------------------------------------------------------- bag of visual words
def build_vocab(desc_list, k):
    data = np.vstack([d for d in desc_list if len(d)])
    if len(data) > MAX_KMEANS_DESCRIPTORS:
        data = data[np.random.RandomState(SEED).choice(len(data), MAX_KMEANS_DESCRIPTORS, replace=False)]
    return MiniBatchKMeans(n_clusters=k, random_state=SEED, n_init=3, batch_size=1024, max_iter=100).fit(data)


def word_counts(desc, km):
    if len(desc) == 0:
        return np.zeros(km.n_clusters, np.float32)
    return np.bincount(km.predict(desc.astype(np.float32)), minlength=km.n_clusters).astype(np.float32)


# ---------------------------------------------------------------- one (detector, k) experiment
def run_config(descs, y, folds, k):
    C = len(CLASSES)
    fold_acc, fold_nn, best_params = [], [], []
    oof_pred = np.zeros(len(y), int)
    sim_sum, sim_cnt = np.zeros((C, C)), np.zeros(C)

    for tr, te in folds:
        km = build_vocab([descs[i] for i in tr], k)                       # vocabulary from TRAIN images only
        H = normalize(np.vstack([word_counts(d, km) for d in descs]))     # L2-normalised histograms

        inner = StratifiedKFold(3, shuffle=True, random_state=SEED)
        grid = GridSearchCV(SVC(), PARAM_GRID, cv=inner, scoring="accuracy").fit(H[tr], y[tr])
        pred = grid.predict(H[te])
        oof_pred[te] = pred
        fold_acc.append(float((pred == y[te]).mean()))
        best_params.append(grid.best_params_)

        S = cosine_similarity(H[te], H[tr])                               # (test images) x (train images)
        fold_nn.append(float((y[tr][S.argmax(1)] == y[te]).mean()))       # 1-NN by cosine similarity
        for c in range(C):                                                # mean similarity to each class
            np.add.at(sim_sum[:, c], y[te], S[:, y[tr] == c].mean(axis=1))
        np.add.at(sim_cnt, y[te], 1)

    sim = sim_sum / sim_cnt[:, None]          # row = true class, column = class compared against
    res = dict(
        k=k, fold_acc=fold_acc, mean_acc=float(np.mean(fold_acc)), std_acc=float(np.std(fold_acc)),
        nn_cos_acc=float(np.mean(fold_nn)),
        intra_sim=float(np.trace(sim) / C),                               # same-class similarity
        inter_sim=float((sim.sum() - np.trace(sim)) / (C * (C - 1))),     # different-class similarity
        best_params=best_params,
    )
    return res, dict(oof_pred=oof_pred, sim=sim)


# ---------------------------------------------------------------- plots
def heat(ax, M, title, fmt, xlabel, ylabel):
    ax.imshow(M, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45)
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, format(M[i, j], fmt), ha="center", va="center",
                    color="white" if M[i, j] > M.max() / 2 else "black")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def save_keypoint_figure(paths, y, max_harris):
    sift = cv2.SIFT_create()
    fig, ax = plt.subplots(2, len(CLASSES), figsize=(3.2 * len(CLASSES), 6.8))
    for c, name in enumerate(CLASSES):
        g = cv2.imread(str(paths[int(np.where(y == c)[0][0])]), cv2.IMREAD_GRAYSCALE)
        kp_s = sift.detect(g, None)
        kp_h = harris_keypoints(g, max_harris)
        a = cv2.drawKeypoints(g, kp_s, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        b = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        for kp in kp_h:
            cv2.circle(b, (int(kp.pt[0]), int(kp.pt[1])), 2, (0, 0, 255), -1)
        ax[0, c].imshow(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))
        ax[0, c].set_title(f"{name}: SIFT ({len(kp_s)} keypoints)", fontsize=9)
        ax[1, c].imshow(cv2.cvtColor(b, cv2.COLOR_BGR2RGB))
        ax[1, c].set_title(f"{name}: Harris ({len(kp_h)} corners)", fontsize=9)
        ax[0, c].axis("off")
        ax[1, c].axis("off")
    plt.tight_layout()
    plt.savefig(RESULTS / "part1_keypoints.png", dpi=130)
    plt.close()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k_values", type=int, nargs="+", default=[25, 50, 100], help="vocabulary sizes to try")
    ap.add_argument("--max_harris", type=int, default=300, help="max Harris corners kept per image")
    ap.add_argument("--no_viz", action="store_true", help="skip the keypoint picture")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    paths, y = list_images("gray")
    folds = get_folds(y)
    if not args.no_viz:
        save_keypoint_figure(paths, y, args.max_harris)

    all_res, extras = [], {}
    for det in ("sift", "harris"):
        descs = extract_all(paths, det, args.max_harris)
        n_kp = [len(d) for d in descs]
        print(f"\n[{det}] keypoints/image: mean {np.mean(n_kp):.0f}, min {min(n_kp)}, max {max(n_kp)}, "
              f"images with none: {sum(n == 0 for n in n_kp)}")
        for k in args.k_values:
            res, ex = run_config(descs, y, folds, k)
            res.update(detector=det, mean_keypoints=float(np.mean(n_kp)), images_without_keypoints=int(sum(n == 0 for n in n_kp)))
            all_res.append(res)
            extras[(det, k)] = ex
            print(f"  k={k:<4} SVM acc {res['mean_acc']:.3f} +/- {res['std_acc']:.3f} | "
                  f"cosine 1-NN acc {res['nn_cos_acc']:.3f} | same-class sim {res['intra_sim']:.3f} vs "
                  f"different-class sim {res['inter_sim']:.3f}")

    # best k per detector -> confusion matrices and similarity heat-maps
    best = {d: max((r for r in all_res if r["detector"] == d), key=lambda r: r["mean_acc"]) for d in ("sift", "harris")}
    fig1, ax1 = plt.subplots(1, 2, figsize=(10, 4.2))
    fig2, ax2 = plt.subplots(1, 2, figsize=(10, 4.2))
    for i, d in enumerate(("sift", "harris")):
        ex = extras[(d, best[d]["k"])]
        heat(ax1[i], confusion_matrix(y, ex["oof_pred"]), f"{d.upper()} + SVM (k={best[d]['k']}), "
             f"acc {best[d]['mean_acc']:.2f}", "d", "predicted", "true")
        heat(ax2[i], ex["sim"], f"{d.upper()} (k={best[d]['k']}): mean cosine similarity", ".2f",
             "compared against class", "true class of test image")
    fig1.tight_layout()
    fig1.savefig(RESULTS / "part1_confusion.png", dpi=130)
    fig2.tight_layout()
    fig2.savefig(RESULTS / "part1_similarity.png", dpi=130)

    with open(RESULTS / "part1_results.json", "w") as f:
        json.dump({"n_folds": len(folds), "configs": all_res}, f, indent=2)
    cols = ["detector", "k", "mean_acc", "std_acc", "nn_cos_acc", "intra_sim", "inter_sim",
            "mean_keypoints", "images_without_keypoints"]
    with open(RESULTS / "part1_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_res)
    print("\nSaved everything in", RESULTS)


if __name__ == "__main__":
    main()
