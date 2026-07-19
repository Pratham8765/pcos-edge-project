import os
import time
import numpy as np
import onnxruntime
from PIL import Image

def load_and_preprocess_image(img_path, img_size=256):
    """
    Loads an image, converts it to grayscale, resizes it,
    converts it to a float32 numpy array, scales to [0,1],
    and normalizes with mean=0.5, std=0.5.
    
    Returns a numpy array of shape (1, 1, img_size, img_size).
    """
    img = Image.open(img_path)
    img = img.resize((img_size, img_size))
    img = img.convert('L')  # Grayscale
    
    # Scale to [0, 1] (equivalent to torchvision.transforms.ToTensor())
    x = np.array(img, dtype=np.float32) / 255.0
    
    # Normalize with mean=[0.5], std=[0.5] (equivalent to torchvision.transforms.Normalize)
    x = (x - 0.5) / 0.5
    
    # Add batch and channel dimensions: shape (1, 1, H, W)
    x = np.expand_dims(x, axis=(0, 1))
    return x

def evaluate_onnx_model(model_path, test_files, cleaned_dir):
    """
    Evaluates an ONNX model on the test files list and computes:
    - Average latency per frame (ms)
    - Accuracy on the test dataset
    - Memory usage
    """
    # 1. Force single-threaded execution to simulate edge microcontroller limitations
    opts = onnxruntime.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    
    # Load ONNX session (always run on CPU for edge gateway simulation)
    session = onnxruntime.InferenceSession(model_path, opts, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    correct = 0
    total = len(test_files)
    latencies = []
    
    # Import psutil for memory tracking (with standard fallback)
    try:
        import psutil
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 * 1024)  # MB
    except ImportError:
        psutil = None
        start_mem = 0.0
        
    print(f"\nBenchmarking model: {os.path.basename(model_path)}")
    print("-------------------------------------------------")
    
    # Warm-up run
    dummy_input = np.random.randn(1, 1, 256, 256).astype(np.float32)
    session.run(None, {input_name: dummy_input})
    
    # Run evaluation
    for i, (rel_path, label) in enumerate(test_files):
        img_path = os.path.join(cleaned_dir, rel_path)
        x = load_and_preprocess_image(img_path)
        
        # Inference time measurement
        t0 = time.perf_counter()
        outputs = session.run(None, {input_name: x})
        t1 = time.perf_counter()
        
        latencies.append((t1 - t0) * 1000)  # ms
        
        # Get prediction index
        predicted = np.argmax(outputs[0], axis=1)[0]
        if predicted == label:
            correct += 1
            
        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"Processed {i + 1}/{total} samples...")
            
    avg_latency = np.mean(latencies)
    accuracy = (correct / total) * 100
    
    # Fetch peak memory usage
    if psutil:
        peak_mem = process.memory_info().rss / (1024 * 1024)  # MB
        mem_used = peak_mem - start_mem
    else:
        mem_used = 0.0
        peak_mem = 0.0
        
    return {
        "accuracy": accuracy,
        "avg_latency_ms": avg_latency,
        "file_size_mb": os.path.getsize(model_path) / (1024 * 1024),
        "mem_used_mb": mem_used,
        "peak_mem_mb": peak_mem
    }

def main():
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    cleaned_dir = os.path.join(project_root, "data", "cleaned_pcos")
    test_files_path = os.path.join(project_root, "data", "test_files.txt")
    
    # Fallback to local files if path configurations differ
    if not os.path.exists(cleaned_dir):
        cleaned_dir = os.path.join("/data", "cleaned_pcos")
        test_files_path = os.path.join("/data", "test_files.txt")
        
    # Read test files list
    if not os.path.exists(test_files_path):
        raise FileNotFoundError(f"Test files record '{test_files_path}' not found! Please run export_onnx.py first.")
        
    test_files = []
    with open(test_files_path, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split(',')
                # path, label
                test_files.append((parts[0], int(parts[1])))
                
    baseline_model_path = os.path.join(project_root, "pcos_resnet18.onnx")
    quant_model_path = os.path.join(project_root, "pcos_resnet18_quant.onnx")
    
    if not os.path.exists(baseline_model_path):
        baseline_model_path = "pcos_resnet18.onnx"
        quant_model_path = "pcos_resnet18_quant.onnx"

    # Evaluate Baseline Model
    baseline_results = evaluate_onnx_model(baseline_model_path, test_files, cleaned_dir)
    
    # Evaluate Quantized Model
    quant_results = evaluate_onnx_model(quant_model_path, test_files, cleaned_dir)
    
    # Output Comparison Table
    print("\n================== STRATEGIC PERFORMANCE MATRIX ==================")
    print(f"{'Metric':<30} | {'Baseline (Float32)':<20} | {'Optimized (Int8)':<18}")
    print("-" * 76)
    print(f"{'Model File Size (Footprint)':<30} | {baseline_results['file_size_mb']:>16.2f} MB | {quant_results['file_size_mb']:>14.2f} MB")
    
    # Compression Ratio calculation
    compression_ratio = ((baseline_results['file_size_mb'] - quant_results['file_size_mb']) / baseline_results['file_size_mb']) * 100
    print(f"{'Footprint Compression Ratio':<30} | {'-':^20} | {compression_ratio:>13.2f}% saved")
    
    print(f"{'Avg Inference Latency (CPU)':<30} | {baseline_results['avg_latency_ms']:>15.2f} ms | {quant_results['avg_latency_ms']:>13.2f} ms")
    
    # Speedup calculation
    speedup = baseline_results['avg_latency_ms'] / quant_results['avg_latency_ms']
    print(f"{'Inference Speedup':<30} | {'-':^20} | {speedup:>13.2f}x faster")
    
    print(f"{'Validation Accuracy':<30} | {baseline_results['accuracy']:>19.2f}% | {quant_results['accuracy']:>17.2f}%")
    
    accuracy_delta = baseline_results['accuracy'] - quant_results['accuracy']
    print(f"{'Accuracy Sacrifice':<30} | {'-':^20} | {accuracy_delta:>13.2f}% delta")
    
    print(f"{'Peak Process Memory (RSS)':<30} | {baseline_results['peak_mem_mb']:>16.2f} MB | {quant_results['peak_mem_mb']:>14.2f} MB")
    print("==================================================================")

if __name__ == "__main__":
    main()
