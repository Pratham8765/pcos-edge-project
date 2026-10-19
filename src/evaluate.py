"""
evaluate.py
-----------
Full per-class evaluation for ResNet18 .pth, ONNX, and quantized ONNX models
on the fixed 401-image test set defined in data/test_files.txt.

Outputs (written to results/):
  metrics.json          -- accuracy, precision, recall, F1, ROC-AUC per model
  confusion_quant.png   -- confusion matrix for the quantized ONNX model

Also re-runs single-sample CPU latency (50 runs, 5 warm-up) for all models
and records the CPU name.

Usage:
    python src/evaluate.py
"""

import os, sys, json, time, platform
import numpy as np
from PIL import Image

# ── path resolution ──────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def _find(*candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None

PTH_PATH        = _find(os.path.join(PROJECT_ROOT, "pcos_resnet18_production.pth"),
                         os.path.join(PROJECT_ROOT, "pcos_resnet18.pth"))
ONNX_PATH       = _find(os.path.join(PROJECT_ROOT, "pcos_resnet18.onnx"))
QUANT_ONNX_PATH = _find(os.path.join(PROJECT_ROOT, "pcos_resnet18_quant.onnx"))
DATA_ROOT       = _find(os.path.join(PROJECT_ROOT, "data", "cleaned_pcos"))
TEST_LIST       = _find(os.path.join(PROJECT_ROOT, "data", "test_files.txt"))

IMG_SIZE      = 256
LATENCY_RUNS  = 50
WARMUP_RUNS   = 5
CLASS_NAMES   = ["infected", "noninfected"]   # label 0 = infected, label 1 = noninfected

# ── helpers ───────────────────────────────────────────────────────────────────
def preprocess(path: str) -> np.ndarray:
    """Load -> grayscale -> resize -> normalise. Returns (1,1,H,W) float32."""
    img = Image.open(path).convert("L").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return arr[np.newaxis, np.newaxis, :, :]

def load_test_list():
    with open(TEST_LIST) as f:
        lines = [l.strip() for l in f if l.strip()]
    items = []
    for line in lines:
        rel_path, label = line.rsplit(",", 1)
        full_path = os.path.join(DATA_ROOT, rel_path.replace("\\", os.sep))
        items.append((full_path, int(label)))
    return items

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def cpu_name():
    try:
        import subprocess
        out = subprocess.check_output(
            ["wmic", "cpu", "get", "Name"], text=True, stderr=subprocess.DEVNULL
        )
        lines = [l.strip() for l in out.strip().splitlines() if l.strip() and l.strip() != "Name"]
        return lines[0] if lines else platform.processor()
    except Exception:
        return platform.processor()

def measure_latency(run_fn, first_input):
    for _ in range(WARMUP_RUNS):
        run_fn(first_input)
    times = []
    for _ in range(LATENCY_RUNS):
        t0 = time.perf_counter()
        run_fn(first_input)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))

def compute_metrics(y_true, y_pred, y_prob_class1):
    from sklearn.metrics import (confusion_matrix, precision_score,
                                  recall_score, f1_score, roc_auc_score)
    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    cm  = confusion_matrix(y_true, y_pred).tolist()
    prec  = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
    rec   = recall_score   (y_true, y_pred, average=None, zero_division=0).tolist()
    f1    = f1_score       (y_true, y_pred, average=None, zero_division=0).tolist()
    try:
        auc = float(roc_auc_score(y_true, y_prob_class1))
    except Exception:
        auc = None
    return {
        "accuracy":         round(acc * 100, 4),
        "confusion_matrix": cm,
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": round(prec[i], 4),
                "recall":    round(rec[i],  4),
                "f1":        round(f1[i],   4),
            } for i in range(len(CLASS_NAMES))
        },
        "roc_auc": round(auc, 4) if auc is not None else None,
    }

def plot_confusion_matrix(cm, model_label, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           xlabel="Predicted label", ylabel="True label",
           title=f"Confusion Matrix\n{model_label}")
    thresh = cm_arr.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm_arr[i, j]),
                    ha="center", va="center",
                    color="white" if cm_arr[i, j] > thresh else "black",
                    fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved confusion matrix -> {out_path}")

# ── ONNX evaluation ───────────────────────────────────────────────────────────
def eval_onnx(model_path, test_items, label):
    import onnxruntime as ort
    sess     = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    y_true, y_pred, y_prob1 = [], [], []
    for path, true_label in test_items:
        inp    = preprocess(path)
        logits = sess.run(None, {inp_name: inp})[0][0]
        probs  = softmax(logits)
        y_true.append(true_label)
        y_pred.append(int(np.argmax(probs)))
        y_prob1.append(float(probs[1]))   # prob of class 1 = noninfected

    metrics = compute_metrics(y_true, y_pred, y_prob1)

    first_inp = preprocess(test_items[0][0])
    lat_mean, lat_std = measure_latency(
        lambda x: sess.run(None, {inp_name: x}), first_inp
    )
    metrics["latency_ms_mean"] = round(lat_mean, 2)
    metrics["latency_ms_std"]  = round(lat_std,  2)
    metrics["size_mb"]         = round(os.path.getsize(model_path) / 1024**2, 2)
    metrics["model"]           = label
    _print_summary(metrics)
    return metrics, (y_true, y_pred, y_prob1)

# ── PyTorch evaluation ─────────────────────────────────────────────────────────
def eval_pth(pth_path, test_items, label):
    import torch, torch.nn as nn
    import torchvision.models as models

    class PCOSResNet18(nn.Module):
        def __init__(self):
            super().__init__()
            self.resnet = models.resnet18(weights=None)
            orig = self.resnet.conv1
            self.resnet.conv1 = nn.Conv2d(1, orig.out_channels, orig.kernel_size,
                                           orig.stride, orig.padding, bias=orig.bias)
            nf = self.resnet.fc.in_features
            self.resnet.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(nf, 2))
        def forward(self, x):
            return self.resnet(x)

    model = PCOSResNet18()
    model.load_state_dict(torch.load(pth_path, map_location="cpu"))
    model.eval()

    y_true, y_pred, y_prob1 = [], [], []
    with torch.no_grad():
        for path, true_label in test_items:
            inp    = torch.tensor(preprocess(path))
            logits = model(inp)[0].numpy()
            probs  = softmax(logits)
            y_true.append(true_label)
            y_pred.append(int(np.argmax(probs)))
            y_prob1.append(float(probs[1]))

    metrics = compute_metrics(y_true, y_pred, y_prob1)

    first_inp = torch.tensor(preprocess(test_items[0][0]))
    with torch.no_grad():
        lat_mean, lat_std = measure_latency(
            lambda x: model(x), first_inp
        )
    metrics["latency_ms_mean"] = round(lat_mean, 2)
    metrics["latency_ms_std"]  = round(lat_std,  2)
    metrics["size_mb"]         = round(os.path.getsize(pth_path) / 1024**2, 2)
    metrics["model"]           = label
    _print_summary(metrics)
    return metrics, (y_true, y_pred, y_prob1)

def _print_summary(m):
    print(f"    accuracy : {m['accuracy']:.2f}%")
    for cls, vals in m["per_class"].items():
        print(f"    {cls:<14} P={vals['precision']:.3f}  R={vals['recall']:.3f}  F1={vals['f1']:.3f}")
    print(f"    ROC-AUC  : {m['roc_auc']}")
    print(f"    latency  : {m['latency_ms_mean']:.2f} +/- {m['latency_ms_std']:.2f} ms")
    print(f"    size     : {m['size_mb']} MB")

# ── majority baseline ─────────────────────────────────────────────────────────
def majority_baseline(test_items):
    labels  = [l for _, l in test_items]
    counts  = {c: labels.count(c) for c in set(labels)}
    maj     = max(counts, key=counts.get)
    acc     = counts[maj] / len(labels) * 100
    print(f"  Majority class: label={maj} ({CLASS_NAMES[maj]}), count={counts[maj]}/{len(labels)}, acc={acc:.2f}%")
    return {"majority_class": CLASS_NAMES[maj], "accuracy": round(acc, 4),
            "class_counts": {CLASS_NAMES[k]: v for k, v in counts.items()}}

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("=== PCOS Per-Class Evaluation ===")
    if not TEST_LIST: sys.exit("ERROR: data/test_files.txt not found.")
    if not DATA_ROOT: sys.exit("ERROR: data/cleaned_pcos not found.")

    test_items = load_test_list()
    print(f"Test samples: {len(test_items)}")
    cpu = cpu_name()
    print(f"CPU: {cpu}")

    print("\n[Majority baseline]")
    majority = majority_baseline(test_items)

    all_metrics = {"cpu": cpu, "test_samples": len(test_items),
                   "majority_baseline": majority, "models": []}

    # 1. PyTorch .pth
    if PTH_PATH:
        print(f"\n[1/3] ResNet18 PyTorch (.pth) -> {PTH_PATH}")
        m, _ = eval_pth(PTH_PATH, test_items, "ResNet18 (PyTorch .pth)")
        all_metrics["models"].append(m)
    else:
        print("\n[1/3] .pth not found, skipping.")

    # 2. ONNX
    if ONNX_PATH:
        print(f"\n[2/3] ResNet18 ONNX -> {ONNX_PATH}")
        m, _ = eval_onnx(ONNX_PATH, test_items, "ResNet18 ONNX")
        all_metrics["models"].append(m)
    else:
        print("\n[2/3] ONNX not found, skipping.")

    # 3. Quantized ONNX + confusion matrix
    if QUANT_ONNX_PATH:
        print(f"\n[3/3] ResNet18 ONNX quantized -> {QUANT_ONNX_PATH}")
        m, (y_true, y_pred, _) = eval_onnx(QUANT_ONNX_PATH, test_items, "ResNet18 ONNX quantized")
        all_metrics["models"].append(m)
        cm_path = os.path.join(RESULTS_DIR, "confusion_quant.png")
        plot_confusion_matrix(m["confusion_matrix"], "ResNet18 ONNX quantized (INT8)", cm_path)
    else:
        print("\n[3/3] Quantized ONNX not found, skipping.")

    # Save JSON
    out_json = os.path.join(RESULTS_DIR, "metrics.json")
    with open(out_json, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nAll metrics saved to: {out_json}")

if __name__ == "__main__":
    main()
