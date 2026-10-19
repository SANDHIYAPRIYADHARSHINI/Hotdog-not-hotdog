"""
src/evaluate.py  --  Step 5: Test-Set Evaluation
==================================================
After training, we evaluate the BEST saved model on the test set —
images the model has NEVER seen during training or validation.

Metrics reported:
  Accuracy   - overall % correct
  Precision  - of all "hot dog" predictions, how many were right?
  Recall     - of all actual hot dogs, how many did we catch?
  F1         - harmonic mean of Precision and Recall (balances both)

WHY multiple metrics and not just accuracy?
  Accuracy can be misleading on imbalanced datasets.
  Example: if 90% of images are "not hot dog", a model that always
  predicts "not hot dog" gets 90% accuracy but is completely useless.
  Precision + Recall + F1 expose this.
  (Our dataset IS balanced, but it's good practice to always report all four.)
"""

import os
import sys
import pathlib

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)

# Make project root importable when run as a script
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.data  import get_dataloaders, get_transforms
from src.model import build_model

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "weights", "best_model.pt")
PLOTS_DIR    = os.path.join(PROJECT_ROOT, "plots")
MISCLF_DIR   = os.path.join(PLOTS_DIR, "misclassified")
N_MISCLF     = 8   # how many misclassified images to save


def load_model():
    """Load the trained model from disk."""
    model = build_model(freeze_backbone=False)   # unfreeze all -- we're not training

    # load_state_dict loads the saved weight values into the model structure.
    # map_location=DEVICE handles the case where training was on GPU but
    # evaluation is on CPU (or vice versa).
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()   # inference mode: disables Dropout, fixes BatchNorm
    return model


def get_all_predictions(model, loader):
    """
    Runs the model over the full DataLoader.
    Returns two flat numpy arrays: (all_labels, all_preds).
    WHY collect everything first?
      sklearn metrics expect flat arrays, not batches.
    """
    all_labels = []
    all_preds  = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            logits = model(images)                       # [B, 1]
            preds  = (logits.sigmoid() > 0.5).long()     # [B, 1]  0 or 1
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.squeeze(1).cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def print_metrics(labels, preds, class_names):
    """Compute and print the four classification metrics."""
    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)

    print("\n=== Test Set Metrics ===")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")

    # Quick explanation of what each number means for our task
    print("\n  Interpreting for 'hot_dog' as the positive class:")
    print(f"  - Of every 'Hotdog!' prediction, {prec*100:.1f}% were actually hot dogs.")
    print(f"  - Of every real hot dog image, we detected {rec*100:.1f}% of them.")
    print(f"  - F1 of {f1:.3f} balances both (1.0 = perfect, 0.5 = coin flip).")

    return acc, prec, rec, f1


def plot_confusion_matrix(labels, preds, class_names):
    """
    Saves a confusion matrix plot.

    The matrix looks like:
                  Predicted hot_dog   Predicted not_hot_dog
    Actual hot_dog       TP                  FN
    Actual not_hot_dog   FP                  TN

    TP = True Positive, FN = False Negative, etc.
    We want TP and TN to be high, FP and FN low.
    """
    cm      = confusion_matrix(labels, preds)
    display = ConfusionMatrixDisplay(confusion_matrix=cm,
                                     display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix -- Test Set")
    plt.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(out)
    plt.close()
    print(f"\nConfusion matrix saved to: {out}")


def save_misclassified(model, loader, class_names, dataset):
    """
    Finds the first N_MISCLF misclassified images and saves them as a
    grid so we can visually inspect why the model failed.

    WHY look at misclassifications?
      Error analysis is how you decide what to fix next. Patterns in
      wrong predictions reveal model weaknesses (e.g., fails on dark images,
      confused by corndogs, etc.)
    """
    os.makedirs(MISCLF_DIR, exist_ok=True)
    _, eval_tf = get_transforms()

    saved     = 0
    idx       = 0   # index into the flat dataset

    with torch.no_grad():
        for images, labels in loader:
            images_gpu = images.to(DEVICE)
            logits     = model(images_gpu)
            preds      = (logits.sigmoid() > 0.5).long().squeeze(1).cpu()

            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    # Undo normalization to get a displayable image
                    img_tensor = images[i]           # [3, 224, 224]
                    img_np     = _denormalize(img_tensor)

                    true_name  = class_names[labels[i].item()]
                    pred_name  = class_names[preds[i].item()]

                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.imshow(img_np)
                    ax.set_title(
                        f"True: {true_name}\nPred: {pred_name}",
                        fontsize=9
                    )
                    ax.axis("off")
                    fname = os.path.join(MISCLF_DIR, f"misclf_{saved:03d}.png")
                    plt.savefig(fname, bbox_inches="tight")
                    plt.close()

                    saved += 1
                    if saved >= N_MISCLF:
                        print(f"\nSaved {saved} misclassified images to: {MISCLF_DIR}")
                        return

            idx += len(labels)

    print(f"\nSaved {saved} misclassified images to: {MISCLF_DIR}")


def _denormalize(tensor):
    """
    Reverses the ImageNet normalization so the image looks natural again.
    Normalization formula was:  x = (pixel - mean) / std
    Reverse is:                 pixel = x * std + mean
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img  = tensor * std + mean          # reverse the normalization
    img  = img.permute(1, 2, 0)        # [C, H, W] -> [H, W, C]  for matplotlib
    img  = img.clamp(0, 1).numpy()     # keep values in [0, 1]
    return img


def evaluate():
    print(f"Device : {DEVICE}")
    print(f"Weights: {WEIGHTS_PATH}")

    # Load data (we only use the test_loader here)
    (_, _, test_loader), class_names = get_dataloaders(PROJECT_ROOT)

    # Load the trained model
    model = load_model()

    # Collect all predictions
    labels, preds = get_all_predictions(model, test_loader)

    # Print metrics
    print_metrics(labels, preds, class_names)

    # Confusion matrix
    plot_confusion_matrix(labels, preds, class_names)

    # Save misclassified images
    save_misclassified(model, test_loader, class_names, None)


if __name__ == "__main__":
    evaluate()
