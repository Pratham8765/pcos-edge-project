"""
benchmark.py
Evaluates ResNet18 (.pth), ONNX, and quantized ONNX models on the fixed test set
defined in data/test_files.txt. Reports accuracy, file size (MB), and mean CPU
latency (ms) measured over 50 single-sample runs per model.

Usage (from project root):
    python src/benchmark.py

Docker (via run_sandbox.ps1):
    Mounts ONNX files and cleaned_pcos dataset; runs automatically.
"""

import os
import sys
import time
import json

import numpy as np
from PIL import Image

# ── path resolution ──────────────────────────────────────────────────────────
# Works both locally (project root) and inside the Docker container (/).
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

def _find(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

ONNX_PATH       = _find([os.path.join(PROJECT_ROOT, "pcos_resnet18.onnx"),       "/pcos_resnet18.onnx"])
QUANT_ONNX_PATH = _find([os.path.join(PROJECT_ROOT, "pcos_resnet18_quant.onnx"), "/pcos_resnet18_quant.onnx"])
PTH_PATH        = _find([os.path.join(PROJECT_ROOT, "pcos_resnet18_production.pth"),
                          os.path.join(PROJECT_ROOT, "pcos_resnet18.pth")])
DATA_ROOT       = _find([os.path.join(PROJECT_ROOT, "data", "cleaned_pcos"),     "/data/cleaned_pcos"])
TEST_LIST       = _find([os.path.join(PROJECT_ROOT, "data", "test_files.txt"),   "/data/test_files.txt"])

IMG_SIZE = 256
LATENCY_RUNS = 50   # single-sample warm-start runs for latency measurement

# ── helpers ───────────────────────────────────────────────────────────────────
def preprocess(path: str) -> np.ndarray:
    """Load image, convert to grayscale, resize, normalise -> (1,1,256,256) float32."""
    img = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5                    # same as Normalize(mean=0.5, std=0.5)
    return arr[np.newaxis, np.newaxis, :, :]   # NCHW

def load_test_list():
    with open(TEST_LIST) as f:
        lines = [l.strip() for l in f if l.strip()]
    items = []
    for line in lines:
        rel_path, label = line.rsplit(",", 1)
        full_path = os.path.join(DATA_ROOT, rel_path.replace("\\", os.sep))
        items.append((full_path, int(label)))
    return items

def mb(path: str) -> float:
    return os.path.getsize(path) / (1024 ** 2)

# ── ONNX evaluation ──────────────────────────────────────────────────────────
def eval_onnx(model_path: str, test_items: list, label: str):
    import onnxruntime as ort

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    correct = 0
    for path, true_label in test_items:
        inp = preprocess(path)
        logits = sess.run(None, {inp_name: inp})[0]
        pred = int(np.argmax(logits, axis=1)[0])
        if pred == true_label:
            correct += 1

    accuracy = correct / len(test_items) * 100

    # latency: 50 single-sample runs on first sample
    first_inp = preprocess(test_items[0][0])
    # warm-up
    for _ in range(5):
        sess.run(None, {inp_name: first_inp})
    times = []
    for _ in range(LATENCY_RUNS):
        t0 = time.perf_counter()
        sess.run(None, {inp_name: first_inp})
        times.append((time.perf_counter() - t0) * 1000)

    size = mb(model_path)
    latency = float(np.mean(times))
    print(f"  {label}: accuracy={accuracy:.2f}%  size={size:.2f} MB  latency={latency:.2f} ms")
    return {"model": label, "accuracy": accuracy, "size_mb": size, "latency_ms": latency}

# ── PyTorch .pth evaluation ──────────────────────────────────────────────────
def eval_pth(pth_path: str, test_items: list, label: str):
    try:
        import torch
        import torch.nn as nn
        import torchvision.models as models
    except ImportError:
        print(f"  {label}: PyTorch not available, skipping.")
        return None

    # Rebuild architecture (must match training)
    class PCOSResNet18(nn.Module):
        def __init__(self):
            super().__init__()
            self.resnet = models.resnet18(weights=None)
            orig = self.resnet.conv1
            self.resnet.conv1 = nn.Conv2d(1, orig.out_channels, orig.kernel_size,
                                           orig.stride, orig.padding, bias=orig.bias)
            num_feat = self.resnet.fc.in_features
            self.resnet.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_feat, 2))

        def forward(self, x):
            return self.resnet(x)

    model = PCOSResNet18()
    model.load_state_dict(torch.load(pth_path, map_location="cpu"))
    model.eval()

    correct = 0
    with torch.no_grad():
        for path, true_label in test_items:
            inp = torch.tensor(preprocess(path))
            logits = model(inp)
            pred = int(torch.argmax(logits, dim=1).item())
            if pred == true_label:
                correct += 1

    accuracy = correct / len(test_items) * 100

    # latency
    first_inp = torch.tensor(preprocess(test_items[0][0]))
    for _ in range(5):
        with torch.no_grad():
            model(first_inp)
    times = []
    with torch.no_grad():
        for _ in range(LATENCY_RUNS):
            t0 = time.perf_counter()
            model(first_inp)
            times.append((time.perf_counter() - t0) * 1000)

    size = mb(pth_path)
    latency = float(np.mean(times))
    print(f"  {label}: accuracy={accuracy:.2f}%  size={size:.2f} MB  latency={latency:.2f} ms")
    return {"model": label, "accuracy": accuracy, "size_mb": size, "latency_ms": latency}

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("=== PCOS Edge Benchmark ===")

    if not TEST_LIST:
        sys.exit("ERROR: test_files.txt not found.")
    if not DATA_ROOT:
        sys.exit("ERROR: cleaned_pcos dataset directory not found.")

    test_items = load_test_list()
    print(f"Test samples: {len(test_items)}")

    results = []

    if PTH_PATH:
        print(f"\n[1/3] ResNet18 PyTorch (.pth)  ->  {PTH_PATH}")
        r = eval_pth(PTH_PATH, test_items, "ResNet18 (PyTorch .pth)")
        if r:
            results.append(r)
    else:
        print("\n[1/3] .pth not found, skipping.")

    if ONNX_PATH:
        print(f"\n[2/3] ResNet18 ONNX  ->  {ONNX_PATH}")
        results.append(eval_onnx(ONNX_PATH, test_items, "ResNet18 ONNX"))
    else:
        print("\n[2/3] ONNX model not found, skipping.")

    if QUANT_ONNX_PATH:
        print(f"\n[3/3] ResNet18 ONNX quantized  ->  {QUANT_ONNX_PATH}")
        results.append(eval_onnx(QUANT_ONNX_PATH, test_items, "ResNet18 ONNX quantized"))
    else:
        print("\n[3/3] Quantized ONNX model not found, skipping.")

    print("\n=== Summary ===")
    for r in results:
        print(f"  {r['model']:<35} acc={r['accuracy']:.2f}%  size={r['size_mb']:.2f} MB  latency={r['latency_ms']:.2f} ms")

    out_path = os.path.join(PROJECT_ROOT, "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

if __name__ == "__main__":
    main()
