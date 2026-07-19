import os
import torch
import torch.nn as nn
from dataset import load_pcos_data
from model import PCOSResNet18
from clean_data import clean_and_get_dataset_path

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")

    # 1. Get cleaned dataset path
    cleaned_path = clean_and_get_dataset_path()
    
    # 2. Load the data splits (we only need the test_loader)
    _, _, test_loader, classes = load_pcos_data(data_dir=cleaned_path, batch_size=32)
    print(f"Dataset classes: {classes}")
    print(f"Number of test samples: {len(test_loader.dataset)}")

    # 3. Initialize model and load trained weights
    model = PCOSResNet18().to(device)
    model_weight_path = "pcos_resnet18.pth"
    
    if not os.path.exists(model_weight_path):
        raise FileNotFoundError(f"Trained model weights '{model_weight_path}' not found! Please run train.py first.")
        
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    total = 0

    # For confusion matrix metrics (binary classification: classes are ['infected', 'noninfected'])
    # infected = positive class (index 0), noninfected = negative class (index 1)
    tp, fp, tn, fn = 0, 0, 0, 0

    print("Evaluating model on test dataset...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Compute confusion matrix values
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                t, p = t.item(), p.item()
                if t == 0:  # infected (positive)
                    if p == 0:
                        tp += 1
                    else:
                        fn += 1
                else:       # noninfected (negative)
                    if p == 1:
                        tn += 1
                    else:
                        fp += 1

    avg_loss = test_loss / total
    accuracy = (correct / total) * 100
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n================ Evaluation Summary ================")
    print(f"Test Loss:        {avg_loss:.4f}")
    print(f"Test Accuracy:    {accuracy:.2f}% ({correct}/{total} correct)")
    print(f"Confusion Matrix:")
    print(f"                 Predicted Infected   Predicted Non-Infected")
    print(f"Actual Infected       TP: {tp:<18} FN: {fn}")
    print(f"Actual Non-Infected   FP: {fp:<18} TN: {tn}")
    print(f"----------------------------------------------------")
    print(f"Precision:        {precision:.4f}")
    print(f"Recall (Sens):    {recall:.4f}")
    print(f"F1-score:         {f1:.4f}")
    print("====================================================")

if __name__ == "__main__":
    test_model()
