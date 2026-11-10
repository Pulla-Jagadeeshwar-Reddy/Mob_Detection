"""
PART 2 (deep learning): pretrained networks used as FROZEN feature extractors + a trainable softmax classifier

  ResNet : ImageNet-pretrained. Everything is frozen except a NEW last layer (Linear -> 4 mobs).
  MViT   : ImageNet-pretrained MViTv2 (via the `timm` library). Frozen feature extractor, with the
           same kind of new softmax classifier on top. (Any other timm model name also works,
           e.g. --models resnet50 vit_base_patch16_224)

Why we compute the features once instead of re-running the network every epoch:
  A frozen network always gives the same output for the same image, so "train only the last layer" is
  mathematically identical to (1) running the frozen network once to get a feature vector per image and
  (2) training the last layer on those vectors. It is just much faster (works on a laptop CPU).

Softmax: the last layer outputs 4 raw scores (logits). Softmax turns them into 4 probabilities that sum to 1.
  nn.CrossEntropyLoss applies softmax internally during training; at test time we call torch.softmax
  ourselves to get the probabilities (saved per image in results/part2_predictions_<model>.csv).

Augmentation: each TRAINING image is used twice (original + horizontally flipped). Test images are never
  augmented, and the flipped copy of a test image is never in the training set (no leakage).

Run:   python part2_dl.py
       python part2_dl.py --models resnet18 mvitv2_tiny --epochs 150
Needs: mc_mob_dataset/processed/color/<cow|creeper|pig|zombie>/*.png   (from mob_pipeline.py)
Output: results/part2_results.json, part2_predictions_<model>.csv, part2_confusion.png, part2_curves.png
        (extracted features are cached in features_cache/)
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import models

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import CLASSES, RESULTS, SEED, get_folds, list_images  # noqa: E402

CACHE = Path("features_cache")
TRANSFORM = T.Compose([
    T.Resize((224, 224)),                      # images are already 224x224; this is a safety net
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),   # ImageNet statistics (what the nets were trained with)
])


# ---------------------------------------------------------------- frozen backbones
def get_backbone(name, device):
    if name.startswith("resnet"):
        weights = {"resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
                   "resnet50": models.ResNet50_Weights.IMAGENET1K_V2}[name]
        net = getattr(models, name)(weights=weights)
        dim = net.fc.in_features
        net.fc = nn.Identity()                 # drop the original 1000-class layer; we train our own 4-class layer
    else:
        import timm                            # MViT and other transformers
        net = timm.create_model(name, pretrained=True, num_classes=0)   # num_classes=0 -> returns pooled features
        dim = net.num_features
    for p in net.parameters():
        p.requires_grad = False                # FROZEN
    return net.eval().to(device), dim


@torch.no_grad()
def extract(net, paths, device, flip, bs=16):
    feats = []
    for i in range(0, len(paths), bs):
        batch = torch.stack([TRANSFORM(Image.open(p).convert("RGB")) for p in paths[i:i + bs]])
        if flip:
            batch = torch.flip(batch, dims=[3])
        feats.append(net(batch.to(device)).cpu())
    return torch.cat(feats).numpy()


def get_features(name, paths, device, recompute):
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{name}.npz"
    names = np.array([f"{p.parent.name}/{p.name}" for p in paths])
    if f.exists() and not recompute:
        d = np.load(f)
        if np.array_equal(d["names"], names):
            print(f"[{name}] loaded cached features")
            return d["X"], d["Xflip"]
        print(f"[{name}] cache does not match current images; recomputing")
    print(f"[{name}] extracting features ({len(paths)} images x 2) ...")
    net, dim = get_backbone(name, device)
    X, Xf = extract(net, paths, device, False), extract(net, paths, device, True)
    np.savez_compressed(f, X=X, Xflip=Xf, names=names)
    return X, Xf


# ---------------------------------------------------------------- trainable softmax classifier
def train_head(Xtr, ytr, Xte, yte, args):
    torch.manual_seed(SEED)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6      # standardise using TRAIN stats only
    Xtr, Xte = [torch.tensor((a - mu) / sd, dtype=torch.float32) for a in (Xtr, Xte)]
    ytr, yte = torch.tensor(ytr), torch.tensor(yte)

    head = nn.Linear(Xtr.shape[1], len(CLASSES))                               # the new last layer
    opt = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()                                            # softmax + negative log-likelihood
    hist = {"train_loss": [], "test_loss": [], "test_acc": []}

    for _ in range(args.epochs):
        head.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad()
            loss_fn(head(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        head.eval()
        with torch.no_grad():                  # fixed number of epochs, so these test curves are diagnostics only
            logits = head(Xte)
            hist["train_loss"].append(loss_fn(head(Xtr), ytr).item())
            hist["test_loss"].append(loss_fn(logits, yte).item())
            hist["test_acc"].append((logits.argmax(1) == yte).float().mean().item())
    return torch.softmax(logits, dim=1).numpy(), hist                          # probabilities that sum to 1


# ---------------------------------------------------------------- cross-validation for one model
def run_model(name, paths, y, folds, args, device):
    X, Xf = get_features(name, paths, device, args.recompute)
    oof_prob = np.zeros((len(y), len(CLASSES)))
    fold_acc, curves = [], []
    for i, (tr, te) in enumerate(folds, 1):
        Xtr, ytr = np.concatenate([X[tr], Xf[tr]]), np.concatenate([y[tr], y[tr]])   # original + flipped
        probs, hist = train_head(Xtr, ytr, X[te], y[te], args)
        oof_prob[te] = probs
        fold_acc.append(float((probs.argmax(1) == y[te]).mean()))
        curves.append(hist)
        print(f"  [{name}] fold {i}/{len(folds)} accuracy {fold_acc[-1]:.3f}")
    pred = oof_prob.argmax(1)
    res = dict(feature_dim=int(X.shape[1]), fold_acc=fold_acc, mean_acc=float(np.mean(fold_acc)),
               std_acc=float(np.std(fold_acc)),
               report=classification_report(y, pred, target_names=CLASSES, output_dict=True, zero_division=0))
    return res, dict(prob=oof_prob, pred=pred, curves=curves)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["resnet50", "mvitv2_tiny"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--recompute", action="store_true", help="ignore cached features")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths, y = list_images("color")
    folds = get_folds(y)
    print("device:", device)

    results, extras = {}, {}
    for name in args.models:
        results[name], extras[name] = run_model(name, paths, y, folds, args, device)
        r = results[name]
        print(f"[{name}] {len(folds)}-fold accuracy: {r['mean_acc']:.3f} +/- {r['std_acc']:.3f}\n")
        with open(RESULTS / f"part2_predictions_{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "true", "predicted"] + [f"softmax_{c}" for c in CLASSES])
            for p, t, pr, pb in zip(paths, y, extras[name]["pred"], extras[name]["prob"]):
                w.writerow([f"{p.parent.name}/{p.name}", CLASSES[t], CLASSES[pr]] + [f"{v:.4f}" for v in pb])

    # confusion matrices
    fig, ax = plt.subplots(1, len(args.models), figsize=(4.8 * len(args.models), 4.2), squeeze=False)
    for a, name in zip(ax[0], args.models):
        M = confusion_matrix(y, extras[name]["pred"])
        a.imshow(M, cmap="Blues")
        a.set_xticks(range(len(CLASSES)))
        a.set_xticklabels(CLASSES, rotation=45)
        a.set_yticks(range(len(CLASSES)))
        a.set_yticklabels(CLASSES)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                a.text(j, i, M[i, j], ha="center", va="center", color="white" if M[i, j] > M.max() / 2 else "black")
        a.set_title(f"{name}  (acc {results[name]['mean_acc']:.2f})", fontsize=10)
        a.set_xlabel("predicted")
        a.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(RESULTS / "part2_confusion.png", dpi=130)

    # learning curves (average over folds)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for name in args.models:
        cv = extras[name]["curves"]
        ax[0].plot(np.mean([c["train_loss"] for c in cv], 0), label=f"{name} train")
        ax[0].plot(np.mean([c["test_loss"] for c in cv], 0), "--", label=f"{name} held-out")
        ax[1].plot(np.mean([c["test_acc"] for c in cv], 0), label=name)
    ax[0].set_title("Cross-entropy loss (mean of folds)")
    ax[0].set_xlabel("epoch")
    ax[0].legend(fontsize=8)
    ax[1].set_title("Held-out accuracy (mean of folds)")
    ax[1].set_xlabel("epoch")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "part2_curves.png", dpi=130)

    with open(RESULTS / "part2_results.json", "w") as f:
        json.dump({"n_folds": len(folds), "epochs": args.epochs, "models": results}, f, indent=2)
    print("Saved everything in", RESULTS)


if __name__ == "__main__":
    main()
