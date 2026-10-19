"""
src/train.py  --  Step 4: Training Loop
=========================================
Two-phase training strategy:

  Phase 1 (head-only):
    - Backbone is frozen.  Only the new Linear(1280->1) layer trains.
    - Use a normal learning rate (1e-3).
    - WHY: The backbone already has great features.  We just need to wire
      up our new head to those features correctly before touching the backbone.
      Training the whole network from the start risks destroying pre-trained
      weights before the head has learned anything useful.

  Phase 2 (fine-tuning):
    - Unfreeze the top 3 backbone blocks.  Now ~1.2 M params train.
    - Use a MUCH smaller learning rate (1e-4).
    - WHY small LR: The backbone weights are already good -- we only want
      small nudges, not big updates that would destroy what was learned.
      A large LR here would cause "catastrophic forgetting".

Loss: BCEWithLogitsLoss
  WHY: Our model outputs 1 raw logit.  BCE (Binary Cross-Entropy) is the
  standard loss for binary classifiers.  "WithLogits" means the loss
  applies sigmoid internally -- numerically more stable than doing it
  yourself.

Optimizer: Adam
  WHY: Adam adapts the learning rate per parameter using momentum.
  It converges faster than plain SGD and requires less tuning.
  For fine-tuning pre-trained models it is the standard choice.
"""

import os
import pathlib
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")   # Use non-interactive backend -- no display needed
import matplotlib.pyplot as plt

import sys

# Make the project root (seefood/) visible to Python's import system.
# WHY: When you run this file directly (`python src/train.py`), Python only
# adds the src/ folder to sys.path, so `from src.data import ...` fails.
# Inserting the project root (one level up from this file) fixes that.
# This line is harmless when running with `python -m src.train` too.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.data  import get_dataloaders
from src.model import build_model, unfreeze_top_layers

# ── Hyperparameters ────────────────────────────────────────────────────────────
PHASE1_EPOCHS = 5      # Epochs to train the head only
PHASE2_EPOCHS = 5      # Epochs to fine-tune the top backbone layers
PHASE1_LR     = 1e-3   # Learning rate for Phase 1
PHASE2_LR     = 1e-4   # Learning rate for Phase 2 (10x smaller)
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helper: train for one epoch ────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    """
    Runs one full pass over the training DataLoader.
    Returns (avg_loss, accuracy) for the epoch.
    """
    model.train()   # puts model in training mode (enables Dropout, BatchNorm updates)
    running_loss = 0.0
    correct      = 0
    total        = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE).float().unsqueeze(1)  # shape [B] -> [B, 1]
        #                                     ^float() needed: BCE expects float, not long
        #                                              ^unsqueeze: match model output shape [B, 1]

        optimizer.zero_grad()          # clear gradients from previous batch
        logits = model(images)         # forward pass  -> shape [B, 1]
        loss   = criterion(logits, labels)
        loss.backward()                # backpropagate: compute gradients
        optimizer.step()               # update weights using those gradients

        running_loss += loss.item() * images.size(0)   # accumulate total loss

        # Convert logit -> prediction: sigmoid > 0.5  <==>  logit > 0
        preds    = (logits.sigmoid() > 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total   += images.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ── Helper: evaluate on val or test set ───────────────────────────────────────
def evaluate(model, loader, criterion):
    """
    Runs one full pass over a DataLoader WITHOUT updating weights.
    Returns (avg_loss, accuracy).
    """
    model.eval()   # disables Dropout, freezes BatchNorm stats
    running_loss = 0.0
    correct      = 0
    total        = 0

    with torch.no_grad():   # don't build the computation graph -- saves memory
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).float().unsqueeze(1)

            logits = model(images)
            loss   = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            preds    = (logits.sigmoid() > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total   += images.size(0)

    return running_loss / total, correct / total


# ── Helper: run N epochs, logging and checkpointing ───────────────────────────
def run_phase(phase_name, model, train_loader, val_loader,
              criterion, optimizer, num_epochs,
              best_val_loss, weights_path, history):
    """
    Trains for `num_epochs` epochs.
    Saves weights whenever val_loss improves (early-stop style).
    Appends per-epoch metrics to `history` dict.
    """
    print(f"\n{'='*50}")
    print(f"  {phase_name}")
    print(f"{'='*50}")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Save the model if this is the best val loss seen so far
        # WHY val_loss, not val_acc?
        #   Loss is a smoother signal than accuracy (which can plateau).
        #   Saving on best val_loss helps prevent overfitting.
        checkpoint_marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), weights_path)
            checkpoint_marker = "  <-- saved best"

        print(
            f"  Epoch {epoch:02d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.3f}"
            f"{checkpoint_marker}"
        )

    return best_val_loss


# ── Helper: plot and save training curves ─────────────────────────────────────
def plot_curves(history, save_dir):
    """
    Plots train/val loss and accuracy curves and saves them as a PNG.
    WHY: Curves let you visually detect overfitting (val loss rising while
    train loss keeps dropping) and whether you trained long enough.
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss")
    ax1.axvline(PHASE1_EPOCHS + 0.5, color="gray",
                linestyle="--", label="Phase 1 -> Phase 2")
    ax1.set_title("Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.legend()

    # Accuracy plot
    ax2.plot(epochs, history["train_acc"], label="Train Acc")
    ax2.plot(epochs, history["val_acc"],   label="Val Acc")
    ax2.axvline(PHASE1_EPOCHS + 0.5, color="gray",
                linestyle="--", label="Phase 1 -> Phase 2")
    ax2.set_title("Accuracy per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    out_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(out_path)
    plt.close()
    print(f"\nCurves saved to: {out_path}")


# ── Main training entry point ──────────────────────────────────────────────────
def train():
    project_root = str(pathlib.Path(__file__).parent.parent)
    weights_dir  = os.path.join(project_root, "weights")
    plots_dir    = os.path.join(project_root, "plots")
    os.makedirs(weights_dir, exist_ok=True)

    weights_path = os.path.join(weights_dir, "best_model.pt")

    print(f"Device: {DEVICE}")

    # Load data
    (train_loader, val_loader, _), _ = get_dataloaders(project_root)

    # Build model and move to device (CPU or GPU)
    model = build_model(freeze_backbone=True).to(DEVICE)

    # Loss function
    # pos_weight balances the dataset if classes are unequal.
    # Our dataset is balanced (249 vs 249), so weight = 1.0.
    criterion = nn.BCEWithLogitsLoss()

    # History dict to store metrics for plotting
    history = {"train_loss": [], "val_loss": [],
               "train_acc":  [], "val_acc":  []}
    best_val_loss = float("inf")   # start very high; any real loss beats this

    # ── Phase 1: train head only ───────────────────────────────────────────────
    # Only pass parameters where requires_grad=True to the optimizer.
    # WHY: If we pass frozen params, Adam would waste memory tracking
    #      momentum for weights that never change.
    optimizer1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PHASE1_LR
    )

    best_val_loss = run_phase(
        "Phase 1: Head-only training",
        model, train_loader, val_loader,
        criterion, optimizer1, PHASE1_EPOCHS,
        best_val_loss, weights_path, history
    )

    # ── Phase 2: unfreeze top layers and fine-tune ─────────────────────────────
    unfreeze_top_layers(model, num_layers=3)

    # Create a NEW optimizer now that more params are trainable.
    # WHY new optimizer? Adam stores momentum state per parameter.
    # The newly unfrozen params have no momentum history yet --
    # a new optimizer starts them fresh, which is the right behaviour.
    optimizer2 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PHASE2_LR   # 10x smaller than Phase 1
    )

    best_val_loss = run_phase(
        "Phase 2: Fine-tuning top backbone layers",
        model, train_loader, val_loader,
        criterion, optimizer2, PHASE2_EPOCHS,
        best_val_loss, weights_path, history
    )

    print(f"\nBest model saved to: {weights_path}")

    # Plot curves
    plot_curves(history, plots_dir)


if __name__ == "__main__":
    train()
