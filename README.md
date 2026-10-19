# PCOS Image Classification and Edge Deployment

CNN-based classifier for PCOS ultrasound images, trained in PyTorch (ResNet18) with a TensorFlow baseline, then exported to ONNX and INT8-quantized for lightweight CPU inference.

> **Disclaimer:** Educational project on a public dataset. Not a medical device and not clinically validated.

## Problem

PCOS is typically identified from ovarian ultrasound images, but expert review is slow and access is uneven. This project explores whether a compact, quantized model can deliver reasonably accurate classification fast enough for edge or resource-constrained deployments, where a large TensorFlow model would be impractical.

## Dataset

- **Source:** [PCOS-XAI Ultrasound Dataset](https://www.kaggle.com/datasets/ibadeus/pcos-xai-ultrasound-dataset) (Kaggle), CC0 Public Domain
- **Raw images:** 11,784 (duplicates, mixed resolutions, inconsistent naming)
- **After MD5 deduplication:** 3,996 images. PCOS-positive (`infected` folder): 3,184. PCOS-negative (`noninfected` folder): 812
- **Split:** 80/10/10 → Train 3,196 / Val 399 / Test 401
- **Split method:** stratified-random by image (`seed=42`). The dataset has no patient identifiers, so a patient-level split was not possible.
- **Preprocessing:** resize to 256×256, grayscale. Training-only augmentation (flips, ±15° rotation, brightness/contrast jitter), applied after the split.

## Approach

1. **TensorFlow baseline:** custom CNN, evaluated on a separate 382-sample hold-out
2. **ResNet18 (PyTorch):** ImageNet-pretrained, adapted to grayscale. Backbone frozen; `conv1` and the head fine-tuned. 15 epochs, Adam (lr=1e-4, weight_decay=1e-4)
3. **ONNX export, then dynamic INT8 quantization** with ONNX Runtime
4. **Docker:** inference-only container limited to 0.5 CPU and 256 MB RAM

## Results (test set, 401 samples)

Majority-class baseline (always predict PCOS-positive): **78.30%**

| Model | Accuracy | PCOS-positive P / R / F1 | PCOS-negative P / R / F1 | ROC-AUC | CPU latency (ms) | Size (MB) |
|---|---|---|---|---|---|---|
| ResNet18 (PyTorch .pth) | 91.52% | 0.917 / 0.981 / 0.948 | 0.908 / 0.678 / 0.776 | 0.9759 | 29.78 ± 0.63 | 42.69 |
| ResNet18 ONNX | 91.52% | 0.917 / 0.981 / 0.948 | 0.908 / 0.678 / 0.776 | 0.9759 | 12.03 ± 0.65 | 42.61 |
| ResNet18 ONNX INT8 | 92.27% | 0.922 / 0.984 / 0.952 | 0.924 / 0.701 / 0.797 | 0.9796 | 3.84 ± 0.25 | 10.69 |
| TensorFlow baseline | not comparable (different split) | n/a | n/a | n/a | 121.57 | 169.86 |

**Reading the results:** the model catches about 98% of PCOS-positive scans but misclassifies about a third of healthy scans (PCOS-negative recall 0.678). The data is roughly 80% PCOS-positive and no re-balancing was applied, so the default 0.5 threshold favors the positive class. Differences under 1 point between the float and quantized models are within noise on 401 images.

Latency: Intel Core i5-13420H, 50 single-sample runs after 5 warm-up runs, measured on the host (Docker not installed at time of measurement; container-constrained latency not measured). All rows come from the same machine and session.

## Limitations

- **Small test set (401):** one error shifts accuracy by about 0.25%.
- **No patient-level split:** images from the same source study may appear in both train and test, which could inflate results.
- **Near-duplicates:** MD5 removes only byte-identical files; resized or re-encoded copies may remain across splits.
- **Class imbalance:** about 80/20, with no re-balancing during training. Weak PCOS-negative recall reflects this.
- **TensorFlow baseline:** its 99.74% was measured on a different 382-sample hold-out and is kept for size and latency reference only, not as an accuracy comparison.
- Public dataset, not clinically validated. Not for medical decisions.

## Future work

- Class weighting or focal loss to improve PCOS-negative recall
- Tune the decision threshold on the validation set only, then report test results once
- Unfreeze more ResNet18 layers
- Patient-level validation if metadata becomes available

## How to run

```bash
git clone https://github.com/Pratham8765/pcos-edge-project.git
cd pcos-edge-project
pip install -r requirements.txt
python src/train.py        # downloads and cleans the dataset via KaggleHub
python src/benchmark.py    # evaluates all models on the fixed test set
```

The first run of `train.py` needs Kaggle credentials (`KAGGLE_USERNAME` and `KAGGLE_KEY`, or `~/.kaggle/kaggle.json`).

`train.py` saves weights as `pcos_resnet18.pth`. The committed file `pcos_resnet18_production.pth` is that output manually renamed; `benchmark.py` looks for `pcos_resnet18_production.pth` first, then falls back to `pcos_resnet18.pth`.

**Docker edge sandbox (Windows PowerShell)** — *commands not tested; Docker was unavailable at time of writing*
```powershell
docker build -t pcos-edge .
docker run --rm --cpus="0.5" --memory="256m" `
  -v "${PWD}/pcos_resnet18.onnx:/pcos_resnet18.onnx" `
  -v "${PWD}/pcos_resnet18_quant.onnx:/pcos_resnet18_quant.onnx" `
  -v "${PWD}/data/cleaned_pcos:/data/cleaned_pcos" `
  -v "${PWD}/data/test_files.txt:/data/test_files.txt" `
  pcos-edge
```

**Docker edge sandbox (Linux / macOS)** — *commands not tested; Docker was unavailable at time of writing*
```bash
docker build -t pcos-edge .
docker run --rm --cpus="0.5" --memory="256m" \
  -v "$PWD/pcos_resnet18.onnx:/pcos_resnet18.onnx" \
  -v "$PWD/pcos_resnet18_quant.onnx:/pcos_resnet18_quant.onnx" \
  -v "$PWD/data/cleaned_pcos:/data/cleaned_pcos" \
  -v "$PWD/data/test_files.txt:/data/test_files.txt" \
  pcos-edge
```

## Repository structure

- `src/`: `clean_data.py` (download + dedupe), `dataset.py` (splits and transforms), `model.py` (grayscale ResNet18), `train.py` (saves `pcos_resnet18.pth`), `benchmark.py`, `evaluate.py` (per-class metrics, confusion matrix, ROC-AUC)
- `data/test_files.txt`: fixed list of the 401 test filenames and labels
- `results/metrics.json`: per-class evaluation output (accuracy, P/R/F1, ROC-AUC, latency)
- `results/confusion_quant.png`: confusion matrix for the INT8 model
- `pcos_resnet18_production.pth`, `pcos_resnet18.onnx`, `pcos_resnet18_quant.onnx`: model files
- `baseline.md`: TensorFlow baseline metrics (382-sample hold-out, different split)
- `Dockerfile`, `run_sandbox.ps1`: edge sandbox

## Tech stack

Python 3.10 · PyTorch · TensorFlow · torchvision · ONNX Runtime · Docker · KaggleHub
