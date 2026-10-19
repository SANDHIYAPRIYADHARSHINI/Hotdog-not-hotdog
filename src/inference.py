"""
src/inference.py  --  Step 6: Single-Image Prediction
=======================================================
This module provides ONE public function: predict(image_bytes)

It takes raw bytes (exactly what a web server receives when someone
uploads a photo) and returns:
    {"label": "Hotdog!" or "Not Hotdog!", "confidence": 0.0-1.0}

Journey of an image through this file:
    bytes  ->  PIL Image  ->  Tensor  ->  Model  ->  Logit  ->  Probability  ->  Label
"""

import io
import sys
import pathlib

import torch
from PIL import Image
from torchvision import transforms

# Make project root importable when run as a script
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.model import build_model

# ── Constants (must match data.py exactly) ────────────────────────────────────
# WHY must these match training?
#   The model learned to interpret pixel values that were normalized with
#   these specific numbers. Using different values at inference time is like
#   speaking a different dialect — the model will misread the image.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224

WEIGHTS_PATH  = str(pathlib.Path(__file__).resolve().parent.parent
                    / "weights" / "best_model.pt")
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Preprocessing pipeline (identical to eval_tf in data.py) ─────────────────
# We define it once here so every call to predict() reuses the same object.
_preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ── Model (loaded once when this module is first imported) ────────────────────
# WHY load at module level, not inside predict()?
#   Loading weights from disk is slow (~100ms). If we loaded inside predict(),
#   every API call would pay that cost. Loading once at startup means every
#   subsequent prediction is fast.
_model = None

def _get_model():
    """Lazy-loads the model on first call, then reuses it."""
    global _model
    if _model is None:
        _model = build_model(freeze_backbone=False)
        _model.load_state_dict(
            torch.load(WEIGHTS_PATH, map_location=DEVICE)
        )
        _model.to(DEVICE)
        _model.eval()   # inference mode: no dropout, no gradient tracking
    return _model


# ── Main public function ───────────────────────────────────────────────────────
def predict(image_bytes: bytes) -> dict:
    """
    Classifies a single image from raw bytes.

    Step-by-step journey from bytes to prediction:

    Step A: bytes -> PIL Image
      image_bytes is a raw binary string (e.g., the contents of a .jpg file).
      io.BytesIO wraps it in a file-like object so PIL can read it.
      PIL.Image.open() decodes the JPEG/PNG compression and gives us an
      array of pixels. .convert("RGB") ensures 3 channels — some PNGs have
      4 channels (RGBA, with transparency) which would crash our model.

    Step B: PIL Image -> Tensor
      transforms.Resize()     : scale to 224x224 pixels
      transforms.ToTensor()   : PIL Image (H, W, 3) uint8 [0..255]
                                -> Tensor (3, H, W) float32 [0.0..1.0]
                                   (channels move to first dimension)
      transforms.Normalize()  : x = (x - mean) / std  per channel

    Step C: Tensor -> Batched Tensor
      Our model expects shape [batch_size, 3, 224, 224].
      A single image is shape [3, 224, 224].
      .unsqueeze(0) adds a batch dimension: [1, 3, 224, 224].
      WHY: The model was always trained on batches, so its internal
           math assumes a batch dimension exists.

    Step D: Batched Tensor -> Logit
      model(tensor) runs the forward pass.
      Output shape: [1, 1] — one logit for one image.

    Step E: Logit -> Probability
      sigmoid(logit) squashes the raw score into [0, 1].
      > 0.5 means the model leans toward "hot dog".
      .item() converts a 1-element tensor to a plain Python float.

    Step F: Probability -> Label
      Threshold at 0.5: above = Hotdog!, below = Not Hotdog!
      Confidence is reported as distance from 0.5, mapped to [0, 1]:
        if prob=0.9 -> confident hot dog, confidence=0.9
        if prob=0.1 -> confident not hot dog, confidence=1-0.1=0.9

    Args:
        image_bytes: raw bytes of any image file (JPEG, PNG, WEBP, ...)

    Returns:
        {"label": str, "confidence": float}

    Raises:
        ValueError: if the bytes cannot be decoded as an image
    """

    # ── Step A: bytes -> PIL Image ────────────────────────────────────────────
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # io.BytesIO(image_bytes) : wraps bytes as a file-like object in RAM
        # Image.open(...)         : decodes JPEG/PNG compression
        # .convert("RGB")         : ensure exactly 3 channels (no alpha)
    except Exception as e:
        raise ValueError(f"Could not decode image: {e}")

    # ── Step B: PIL Image -> normalised Tensor ────────────────────────────────
    tensor = _preprocess(pil_image)
    # tensor.shape is now: [3, 224, 224]  (C, H, W)
    # dtype: float32, values roughly in range [-2.1, 2.6] after normalization

    # ── Step C: add batch dimension ───────────────────────────────────────────
    tensor = tensor.unsqueeze(0).to(DEVICE)
    # tensor.shape is now: [1, 3, 224, 224]  (B, C, H, W)

    # ── Step D: forward pass ──────────────────────────────────────────────────
    model = _get_model()
    with torch.no_grad():   # don't build gradient graph — saves memory + speed
        logit = model(tensor)       # shape: [1, 1]

    # ── Step E: logit -> probability ──────────────────────────────────────────
    prob = logit.sigmoid().item()   # float in [0.0, 1.0]
    # prob > 0.5 means the model leans "not_hot_dog" (class 1, alphabetically second)

    # ── Step F: threshold -> label + confidence ───────────────────────────────
    # WHY this direction?
    # ImageFolder assigns labels alphabetically:
    #   hot_dog     -> class 0
    #   not_hot_dog -> class 1
    # BCE trains the model to output HIGH logit for class 1 (not_hot_dog)
    # and LOW logit for class 0 (hot_dog).
    # Therefore: sigmoid > 0.5 means NOT a hot dog.
    if prob > 0.5:
        label      = "Not Hotdog!"
        confidence = prob            # e.g. 0.92 -> 92% sure it's NOT a hot dog
    else:
        label      = "Hotdog!"
        confidence = 1.0 - prob     # e.g. prob=0.1 -> 90% sure it IS a hot dog

    return {
        "label":      label,
        "confidence": round(confidence, 4)
    }


# ── Quick test: run on one real image from the test set ───────────────────────
if __name__ == "__main__":
    import os

    project_root = str(pathlib.Path(__file__).resolve().parent.parent)

    # Grab the first image from test/hot_dog/
    test_hd_dir = os.path.join(project_root, "test", "hot_dog")
    sample_file = os.path.join(test_hd_dir, os.listdir(test_hd_dir)[0])

    print(f"Testing on: {sample_file}")
    with open(sample_file, "rb") as f:
        raw_bytes = f.read()

    result = predict(raw_bytes)
    print(f"Result: {result}")
    # Expected: label=Hotdog! (hopefully), confidence close to 1.0
