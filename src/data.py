"""
src/data.py  —  Step 2: Data Pipeline
======================================
Responsibilities:
  1. Point PyTorch at our train/ and test/ folders.
  2. Apply augmentation ONLY to training images.
  3. Split the training folder into train + validation subsets.
  4. Return ready-to-use DataLoaders for all three splits.
"""

import os
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ── Constants ─────────────────────────────────────────────────────────────────
# ImageNet mean and std — the values MobileNetV2 was pre-trained with.
# WHY: If you normalize your inputs the same way the model was trained,
#      the pre-trained weights "make sense" for your images too.
#      Different normalization = the model sees a completely foreign distribution.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMAGE_SIZE   = 224   # MobileNetV2 expects 224×224 inputs
BATCH_SIZE   = 32    # How many images the GPU processes at once
VAL_FRACTION = 0.2   # 20% of train/ becomes validation
RANDOM_SEED  = 42    # Fixing the seed makes the split reproducible


def get_transforms():
    """
    Returns two transform pipelines:
      - train_tf : augmented  (used only for training)
      - eval_tf  : plain      (used for validation AND test)

    WHY must validation be un-augmented?
      Validation exists to measure how well the model generalises to
      *unseen, real-world* data. If we flip/jitter validation images,
      we're measuring performance on artificial data, not real data —
      which gives us a misleading signal about whether to stop training.
    """

    # ── Training transforms: augmentation first, then resize + normalise ──────
    train_tf = transforms.Compose([
        # Random horizontal flip: a hot dog on the left looks the same flipped.
        # WHY: Doubles the effective dataset size cheaply; teaches the model
        #      that orientation doesn't change the object's identity.
        transforms.RandomHorizontalFlip(),

        # Random rotation up to ±15°.
        # WHY: Real photos aren't always perfectly upright. Small rotations
        #      make the model more robust to tilted shots.
        transforms.RandomRotation(degrees=15),

        # Randomly adjust brightness, contrast, saturation, hue.
        # WHY: Lighting changes a lot between photos. This stops the model
        #      from memorising "hot dogs are always this exact shade of brown".
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.05),

        # Resize to the size the backbone expects.
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        # Convert PIL image (0–255 uint8) → float tensor (0.0–1.0).
        transforms.ToTensor(),

        # Shift pixel values so the model's pre-trained weights apply correctly.
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # ── Eval transforms: NO augmentation — just resize + normalise ────────────
    eval_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_tf, eval_tf


def get_dataloaders(data_root: str):
    """
    Builds and returns three DataLoaders: train, val, test.

    Args:
        data_root: path to the seefood/ project root
                   (the folder that contains train/ and test/)

    Returns:
        (train_loader, val_loader, test_loader), class_names
        class_names is a list like ['hot_dog', 'not_hot_dog']
    """
    train_tf, eval_tf = get_transforms()

    train_dir = os.path.join(data_root, "train")
    test_dir  = os.path.join(data_root, "test")

    # ── Load from disk ─────────────────────────────────────────────────────────
    # ImageFolder expects the folder layout:
    #   train/
    #     hot_dog/      ← class 0
    #     not_hot_dog/  ← class 1
    # It automatically assigns integer labels based on alphabetical order.
    full_train_dataset = datasets.ImageFolder(root=train_dir, transform=train_tf)
    test_dataset       = datasets.ImageFolder(root=test_dir,  transform=eval_tf)

    class_names = full_train_dataset.classes  # ['hot_dog', 'not_hot_dog']

    # ── Train / Validation split ───────────────────────────────────────────────
    # WHY split from train/ only, not from test/?
    #   Test data must NEVER influence training decisions (including early
    #   stopping). If we validated on test data, we'd be "peeking" and the
    #   final test accuracy would be overly optimistic (data leakage).
    total      = len(full_train_dataset)             # e.g. 498
    val_size   = int(total * VAL_FRACTION)           # e.g. ~99
    train_size = total - val_size                    # e.g. ~399

    # random_split with a generator = reproducible split every run.
    import torch
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_subset, val_subset = random_split(
        full_train_dataset, [train_size, val_size], generator=generator
    )

    # ── The validation subset uses eval transforms, not augmented ones ─────────
    # random_split keeps the *same* dataset object (with train_tf).
    # We override the transform on the validation slice here.
    # WHY: See the docstring for get_transforms().
    val_subset.dataset = _copy_dataset_with_transform(full_train_dataset, eval_tf)

    # ── Wrap in DataLoaders ───────────────────────────────────────────────────
    # DataLoader handles batching, shuffling, and parallel loading.
    # shuffle=True for train: prevents the model from memorising image order.
    # num_workers=2: load data on 2 background CPU threads while the GPU trains.
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    print(f"  Train : {train_size} images")
    print(f"  Val   : {val_size} images")
    print(f"  Test  : {len(test_dataset)} images")
    print(f"  Classes: {class_names}")

    return (train_loader, val_loader, test_loader), class_names


# ── Helper ────────────────────────────────────────────────────────────────────
def _copy_dataset_with_transform(original_dataset, new_transform):
    """
    Creates a shallow copy of an ImageFolder dataset with a different transform.
    We need this because random_split shares the dataset object between
    train and val subsets — we can't just set .transform on one without
    affecting the other.
    """
    import copy
    new_ds = copy.copy(original_dataset)   # shallow copy — same file list
    new_ds.transform = new_transform       # but a different transform pipeline
    return new_ds


# ── Quick sanity check (run this file directly to test) ───────────────────────
if __name__ == "__main__":
    import pathlib
    # Go up one level from src/ to reach the project root
    project_root = str(pathlib.Path(__file__).parent.parent)
    (train_loader, val_loader, test_loader), classes = get_dataloaders(project_root)

    # Grab one batch and print its shape to confirm everything works
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape : {images.shape}")   # expect [32, 3, 224, 224]
    print(f"Labels shape: {labels.shape}")     # expect [32]
    print(f"Label values: {labels[:8]}")       # 0 = hot_dog, 1 = not_hot_dog
