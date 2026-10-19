"""
src/model.py  —  Step 3: Model Definition
==========================================
We use Transfer Learning:
  1. Load MobileNetV2 pre-trained on ImageNet (1.2 million images, 1000 classes).
  2. FREEZE the backbone — its weights stay fixed during Phase 1 training.
  3. REPLACE the final classifier layer with our own binary output layer.

WHY transfer learning instead of training from scratch?
  We only have ~400 training images. A CNN trained from scratch on that few
  images would overfit badly (memorise the training set, fail on new images).
  MobileNetV2 already knows how to detect edges, textures, shapes, and object
  parts. We just need to teach it: "does this image contain a hot dog?"
"""

import torch
import torch.nn as nn
from torchvision import models


def build_model(freeze_backbone: bool = True) -> nn.Module:
    """
    Returns a MobileNetV2 with a custom binary classification head.

    Args:
        freeze_backbone: If True, all backbone layers are frozen (Phase 1).
                         Set to False to fine-tune the whole network (Phase 2).

    Returns:
        model (nn.Module) ready to be moved to CPU or GPU.
    """

    # ── 1. Load pre-trained MobileNetV2 ──────────────────────────────────────
    # weights=DEFAULT fetches the best available ImageNet checkpoint.
    # This downloads ~14 MB the first time, then caches it locally.
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # ── 2. Freeze the backbone ────────────────────────────────────────────────
    # "Freezing" means: during backpropagation, don't update these weights.
    # We do this by setting requires_grad=False on every backbone parameter.
    #
    # WHY freeze?
    #   The backbone's weights already encode powerful visual features
    #   (edges → textures → shapes → parts) learned from 1.2M images.
    #   If we let them update with only 400 images, we'd destroy that
    #   knowledge — a problem called "catastrophic forgetting".
    #   Freezing forces the model to only learn the new head in Phase 1.
    if freeze_backbone:
        for param in model.features.parameters():  # .features = the backbone
            param.requires_grad = False            # ← this is what "freezing" means

    # ── 3. Replace the classifier head ───────────────────────────────────────
    # MobileNetV2's original head outputs 1000 logits (one per ImageNet class).
    # We only need 1 logit (hot dog vs not hot dog).
    #
    # WHY 1 output instead of 2?
    #   For binary classification, 1 output + BCEWithLogitsLoss is equivalent
    #   to 2 outputs + CrossEntropyLoss, but simpler and slightly more stable.
    #   One number above 0 → hot dog; below 0 → not hot dog.
    #   (We apply sigmoid to convert this to a 0–1 probability at inference.)
    #
    # MobileNetV2's classifier structure: [Dropout, Linear(1280 → 1000)]
    # model.classifier[1] is the Linear layer — we replace just that.
    in_features = model.classifier[1].in_features  # 1280
    model.classifier[1] = nn.Linear(in_features, 1)
    #                                              ^── 1 output neuron

    return model


def unfreeze_top_layers(model: nn.Module, num_layers: int = 3) -> None:
    """
    Unfreezes the last `num_layers` blocks of the backbone for Phase 2
    fine-tuning. Called AFTER Phase 1 training is complete.

    WHY unfreeze only the TOP (last) layers, not all of them?
      Early layers detect simple patterns (edges, colours) that are universal.
      Later layers detect complex, task-specific patterns (shapes, textures).
      The later layers are the ones most worth fine-tuning for our specific task.
      Unfreezing everything risks destroying the universal early features.

    Args:
        model:      the MobileNetV2 returned by build_model()
        num_layers: how many of the last backbone blocks to unfreeze (default 3)
    """
    # model.features is a list of 19 blocks (indices 0–18).
    # We unfreeze the last `num_layers` blocks + the classifier head.
    backbone_blocks = list(model.features.children())
    for block in backbone_blocks[-num_layers:]:
        for param in block.parameters():
            param.requires_grad = True   # ← re-enable gradient updates

    # The classifier head is always trainable (it was never frozen).
    # Nothing extra needed for it.


def count_trainable_params(model: nn.Module) -> int:
    """Returns how many parameters will actually be updated during training."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model: nn.Module) -> int:
    """Returns total parameter count (frozen + trainable)."""
    return sum(p.numel() for p in model.parameters())


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Phase 1: backbone frozen ===")
    model = build_model(freeze_backbone=True)
    total     = count_total_params(model)
    trainable = count_trainable_params(model)
    print(f"  Total params     : {total:,}")
    print(f"  Trainable params : {trainable:,}  <-- only the new head")
    print(f"  Frozen params    : {total - trainable:,}")

    print("\n=== Phase 2: top 3 backbone blocks unfrozen ===")
    unfreeze_top_layers(model, num_layers=3)
    trainable2 = count_trainable_params(model)
    print(f"  Trainable params : {trainable2:,}  <-- head + top 3 blocks")

    print("\n=== Forward pass test ===")
    dummy_input = torch.randn(4, 3, 224, 224)  # batch of 4 fake images
    output = model(dummy_input)
    print(f"  Input shape  : {dummy_input.shape}")
    print(f"  Output shape : {output.shape}")   # expect [4, 1]
    print(f"  Raw logits   : {output.detach().squeeze().tolist()}")
    print("  (positive logit = hot dog, negative logit = not hot dog)")
