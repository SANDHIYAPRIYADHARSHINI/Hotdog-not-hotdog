"""
api/main.py  --  Step 7: FastAPI REST API
==========================================
Two endpoints:

  GET  /health   -- quick check that the server is alive
  POST /predict  -- accepts an image file, returns label + confidence

Request/Response cycle for POST /predict:
  1. Browser sends a multipart/form-data POST request with the image file.
  2. FastAPI receives it and puts the file in an UploadFile object.
  3. We read the raw bytes from the upload.
  4. We validate the file type (reject non-images with HTTP 415).
  5. We call predict(bytes) from inference.py.
  6. We return a JSON response: {"label": "...", "confidence": 0.xx}
  7. If anything goes wrong, we return a meaningful error (HTTP 400/500).

WHY FastAPI and not Flask?
  FastAPI is modern, fast, and automatically generates API docs at /docs.
  It uses Python type hints to validate inputs -- less boilerplate than Flask.
  It is also async-ready, meaning it can handle many requests concurrently.
"""

import sys
import pathlib

# Make project root importable so we can do "from src.inference import predict"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.inference import predict

# ── Create the FastAPI app ────────────────────────────────────────────────────
app = FastAPI(
    title="SeeFood API",
    description="Binary image classifier: Hotdog vs Not Hotdog",
    version="1.0.0",
)

# ── CORS (Cross-Origin Resource Sharing) ─────────────────────────────────────
# WHY do we need CORS?
#   Browsers block JavaScript from making requests to a DIFFERENT origin
#   (domain + port) than the page itself. This is a security feature called
#   the Same-Origin Policy.
#   Our frontend (served at e.g. file:// or localhost:5500) talks to our API
#   at localhost:8000 -- that's a different origin, so CORS blocks it by default.
#   Adding this middleware tells the browser: "yes, I allow requests from
#   these origins -- it's intentional."
#
# allow_origins=["*"] is fine for a dev/demo project.
# In production you would restrict to your actual domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # allow all origins (dev only)
    allow_methods=["*"],       # allow GET, POST, OPTIONS, etc.
    allow_headers=["*"],       # allow any headers
)

# ── Allowed MIME types ────────────────────────────────────────────────────────
# WHY check content_type?
#   Without this check, a user could upload a .txt or .exe and our code
#   would try to decode it as an image, crashing with a confusing error.
#   We return HTTP 415 (Unsupported Media Type) -- the correct status code.
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


# ── Endpoint 1: Health check ──────────────────────────────────────────────────
@app.get("/health")
def health():
    """
    Returns 200 OK if the server is running.
    Used by monitoring systems and by the frontend to check API availability.
    """
    return {"status": "ok", "model": "MobileNetV2", "task": "hotdog-classifier"}


# ── Endpoint 2: Prediction ────────────────────────────────────────────────────
@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Accepts an image upload and returns a classification result.

    - file: the uploaded image (multipart/form-data)

    Returns:
        200 OK:  {"label": str, "confidence": float}
        400 Bad Request:  if the image is corrupt or unreadable
        415 Unsupported Media Type: if the file is not an image
    """

    # ── Validate file type ────────────────────────────────────────────────────
    # file.content_type is the MIME type the browser sends (e.g. "image/jpeg").
    # Note: this can be spoofed -- a more robust check would inspect the
    # file's magic bytes. For this project, MIME check is sufficient.
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,    # 415 = Unsupported Media Type
            detail=(
                f"File type '{file.content_type}' is not supported. "
                f"Please upload a JPEG, PNG, or WEBP image."
            ),
        )

    # ── Read the raw bytes ────────────────────────────────────────────────────
    # await is used because reading from an upload stream is I/O-bound.
    # FastAPI is async, so 'await' lets other requests run while we wait for
    # the file to fully arrive -- important for handling many users at once.
    image_bytes = await file.read()

    # ── Run the model ─────────────────────────────────────────────────────────
    # predict() can raise ValueError if the bytes are not a valid image
    # (e.g. a renamed .txt file that passed the MIME check).
    # We catch it and return HTTP 400 (Bad Request) with a clear message.
    try:
        result = predict(image_bytes)
    except ValueError as e:
        raise HTTPException(
            status_code=400,    # 400 = Bad Request (client sent bad data)
            detail=f"Could not process image: {e}"
        )

    # ── Return JSON response ──────────────────────────────────────────────────
    # FastAPI automatically serializes Python dicts to JSON.
    # The response will look like: {"label": "Hotdog!", "confidence": 0.934}
    return JSONResponse(content=result)


# ── Run directly with: python api/main.py ─────────────────────────────────────
# In normal use, start the server with:
#   uvicorn api.main:app --reload --port 8000
# --reload: automatically restarts the server when you edit code (dev only)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
