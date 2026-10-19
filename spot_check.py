"""Quick spot-check: test 10 hot dog + 10 not_hot_dog images."""
import sys, os
sys.path.insert(0, ".")
from src.inference import predict

correct = 0
total   = 0

for fname in os.listdir("test/hot_dog")[:10]:
    with open(os.path.join("test/hot_dog", fname), "rb") as f:
        r = predict(f.read())
    ok = "Hotdog" in r["label"] and "Not" not in r["label"]
    correct += ok
    total   += 1
    status   = "OK" if ok else "WRONG"
    print(f"  hot_dog     | {r['label']:<14s} conf={r['confidence']:.2f}  {status}")

for fname in os.listdir("test/not_hot_dog")[:10]:
    with open(os.path.join("test/not_hot_dog", fname), "rb") as f:
        r = predict(f.read())
    ok = "Not" in r["label"]
    correct += ok
    total   += 1
    status   = "OK" if ok else "WRONG"
    print(f"  not_hot_dog | {r['label']:<14s} conf={r['confidence']:.2f}  {status}")

print(f"\nMini accuracy: {correct}/{total} = {correct/total*100:.0f}%")
