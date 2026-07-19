import os
import torch
import numpy as np
from PIL import Image
from model import PCOSResNet18
import onnxruntime
from onnxruntime.quantization import quantize_static, QuantFormat, QuantType, CalibrationDataReader

def load_and_preprocess_image(img_path, img_size=256):
    """
    Loads an image, converts it to grayscale, resizes it,
    scales to [0,1], and normalizes with mean=0.5, std=0.5.
    Returns a numpy array of shape (1, 1, img_size, img_size).
    """
    img = Image.open(img_path)
    img = img.resize((img_size, img_size))
    img = img.convert('L')  # Grayscale
    
    # Scale to [0, 1]
    x = np.array(img, dtype=np.float32) / 255.0
    
    # Normalize with mean=[0.5], std=[0.5]
    x = (x - 0.5) / 0.5
    
    # Add batch and channel dimensions: shape (1, 1, H, W)
    x = np.expand_dims(x, axis=(0, 1))
    return x

class PCOSCalibrationDataReader(CalibrationDataReader):
    def __init__(self, cleaned_dir, samples, input_name, batch_size=16):
        self.samples = samples
        self.cleaned_dir = cleaned_dir
        self.input_name = input_name
        self.batch_size = batch_size
        self.index = 0
        
    def get_next(self):
        if self.index >= len(self.samples):
            return None
            
        batch = []
        for _ in range(self.batch_size):
            if self.index >= len(self.samples):
                break
            rel_path, _ = self.samples[self.index]
            img_path = os.path.join(self.cleaned_dir, rel_path)
            
            try:
                x = load_and_preprocess_image(img_path)
                batch.append(x)
            except Exception as e:
                print(f"Error loading calibration image {img_path}: {e}")
            self.index += 1
            
        if not batch:
            return None
            
        batch_array = np.vstack(batch)
        return {self.input_name: batch_array}

def export_and_quantize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading PyTorch model and weights (on {device})...")
    
    # Initialize the model and load the trained weights
    model = PCOSResNet18()
    weight_path = "pcos_resnet18.pth"
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Trained weight file '{weight_path}' not found! Please run train.py first.")
        
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Define paths
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    onnx_path = os.path.join(project_root, "pcos_resnet18.onnx")
    quant_onnx_path = os.path.join(project_root, "pcos_resnet18_quant.onnx")
    cleaned_dir = os.path.join(project_root, "data", "cleaned_pcos")
    
    # 1. Export PyTorch Model to ONNX format
    print(f"Exporting model to ONNX format at: {onnx_path}")
    dummy_input = torch.randn(1, 1, 256, 256, device=device)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Baseline ONNX model exported successfully.")
    
    # 2. Export test split paths to data/test_files.txt for PyTorch-free benchmarking
    from torchvision import datasets
    full_dataset = datasets.ImageFolder(root=cleaned_dir, transform=None)
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    generator = torch.Generator().manual_seed(42)
    _, _, test_subset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    
    test_files_path = os.path.join(project_root, "data", "test_files.txt")
    test_files = []
    with open(test_files_path, "w") as f:
        for idx in test_subset.indices:
            path, label = full_dataset.samples[idx]
            rel_path = os.path.relpath(path, cleaned_dir)
            f.write(f"{rel_path},{label}\n")
            test_files.append((rel_path, label))
    print(f"Exported {len(test_files)} test file paths to {test_files_path}")
    
    # 3. Static Quantization (generates CPU-supported QLinearConv instead of unsupported ConvInteger)
    print(f"Starting static INT8 quantization at: {quant_onnx_path}")
    
    # We use a subset of test/val images for calibration (e.g. 100 images)
    calibration_samples = test_files[:100]
    
    # Create the calibration data reader
    # We run an inference session on the baseline model to get the input name dynamically
    session = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    data_reader = PCOSCalibrationDataReader(cleaned_dir, calibration_samples, input_name)
    
    # Execute static quantization
    quantize_static(
        model_input=onnx_path,
        model_output=quant_onnx_path,
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QOperator, # Generates QLinearConv, highly optimized for CPU
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8
    )
    print("Static quantized ONNX model generated successfully.")
    
    # 4. Print Footprint Compression Metrics
    baseline_size = os.path.getsize(onnx_path) / (1024 * 1024)
    quant_size = os.path.getsize(quant_onnx_path) / (1024 * 1024)
    reduction = ((baseline_size - quant_size) / baseline_size) * 100
    
    print("\nFootprint Compression Metrics:")
    print(f"  - Baseline ONNX Size:  {baseline_size:.2f} MB")
    print(f"  - Quantized ONNX Size: {quant_size:.2f} MB")
    print(f"  - Space Savings:       {reduction:.2f}%")

if __name__ == "__main__":
    export_and_quantize()
