"""Quick API test script — run from seefood/ folder."""
import urllib.request, json, os

# ── Test GET /health ──────────────────────────────────────────────────────────
res = urllib.request.urlopen("http://localhost:8000/health")
print("GET /health:", json.loads(res.read()))

# ── Test POST /predict with a real hot dog image ──────────────────────────────
img_dir  = "test/hot_dog"
img_path = os.path.join(img_dir, os.listdir(img_dir)[0])

with open(img_path, "rb") as f:
    img_bytes = f.read()

boundary = b"----seefoodboundary"
CRLF     = b"\r\n"
body = (
    b"--" + boundary + CRLF +
    b'Content-Disposition: form-data; name="file"; filename="test.jpg"' + CRLF +
    b"Content-Type: image/jpeg" + CRLF + CRLF +
    img_bytes + CRLF +
    b"--" + boundary + b"--" + CRLF
)

req = urllib.request.Request(
    "http://localhost:8000/predict",
    data=body,
    headers={"Content-Type": "multipart/form-data; boundary=----seefoodboundary"},
    method="POST",
)
res = urllib.request.urlopen(req)
print("POST /predict:", json.loads(res.read()))
