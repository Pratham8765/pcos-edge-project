import os
import time
import cv2
import numpy as np
import tensorflow as tf

def evaluate_model():
    print(f"TensorFlow version used: {tf.__version__}")
    
    # Define paths
    model_path = "cyst_detection_final_savedmodel.h5"
    test_image_dir = r"C:\Users\Pratham\python\dataset\dataset\dataset\OTU_2D\test\test_image"
    test_label_dir = r"C:\Users\Pratham\python\dataset\dataset\dataset\OTU_2D\test\label\black_white"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found! Make sure it is copied to the workspace root.")
        
    # 1. Model File Size in MB
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Model file size: {model_size_mb:.2f} MB")
    
    # 2. Load Model
    print("Loading model...")
    t_start = time.time()
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded in {time.time() - t_start:.2f} seconds.")
    
    # 3. Total Parameter Count
    total_params = model.count_params()
    print(f"Total parameter count: {total_params:,}")
    
    # 4. Load and evaluate on the test set
    print("Loading and preprocessing test dataset...")
    images = []
    labels = []
    
    missing_masks = 0
    img_files = [f for f in os.listdir(test_image_dir) if f.lower().endswith(".jpg")]
    
    for filename in img_files:
        img_path = os.path.join(test_image_dir, filename)
        mask_filename = filename.replace(".JPG", ".png").replace(".jpg", ".png")
        mask_path = os.path.join(test_label_dir, mask_filename)
        
        if not os.path.exists(mask_path):
            missing_masks += 1
            continue
            
        # Load and preprocess image (BGR scale to [0,1], resize to (256, 256))
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256)) / 255.0
        
        # Load and preprocess mask to create binary label
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (256, 256)) / 255.0
        label = 1 if np.max(mask) > 0.0 else 0
        
        images.append(img)
        labels.append(label)
        
    X_test = np.array(images)
    y_test = np.array(labels)
    
    test_size = len(X_test)
    print(f"Loaded {test_size} test samples (skipped {missing_masks} missing masks).")
    
    # Run evaluation
    print("Evaluating model on test dataset...")
    loss, accuracy = model.evaluate(X_test, y_test, batch_size=32, verbose=1)
    test_accuracy_pct = accuracy * 100
    print(f"Test Accuracy: {test_accuracy_pct:.2f}%")
    
    # 5. Average Inference Latency on a single sample over 50 runs
    print("Measuring inference latency over 50 runs...")
    single_sample = np.expand_dims(X_test[0], axis=0) # Add batch dimension -> (1, 256, 256, 3)
    
    # Warm-up run
    model.predict(single_sample, verbose=0)
    
    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        model.predict(single_sample, verbose=0)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000) # milliseconds
        
    avg_latency_ms = np.mean(latencies)
    print(f"Average Inference Latency: {avg_latency_ms:.2f} ms")
    
    # 6. Generate baseline.md file
    import datetime
    today = datetime.date.today().strftime("%B %d, %Y")
    
    baseline_content = f"""# Model Performance Baseline

**Date:** {today}
**TensorFlow Version:** {tf.__version__}
**Test Set Size:** {test_size} samples

## Performance Metrics

- **Test Accuracy:** {test_accuracy_pct:.2f}%
- **Total Parameter Count:** {total_params:,}
- **Model File Size:** {model_size_mb:.2f} MB
- **Average Inference Latency (CPU):** {avg_latency_ms:.2f} ms (measured over 50 runs on a single sample)
"""
    
    baseline_path = "baseline.md"
    with open(baseline_path, "w") as f:
        f.write(baseline_content)
    print(f"Created baseline.md successfully at: {os.path.abspath(baseline_path)}")

if __name__ == "__main__":
    evaluate_model()
