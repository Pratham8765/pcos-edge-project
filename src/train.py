import torch
import torch.nn as nn
import torch.optim as optim
from clean_data import clean_and_get_dataset_path
from dataset import load_pcos_data
from model import PCOSResNet18

def run_stabilized_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing ResNet18 training on: {device}")

    # 1. Download and clean dataset
    cleaned_path = clean_and_get_dataset_path()
    print("Cleaned dataset path:", cleaned_path)

    # 2. Feed the cleaned path directly into our processing pipeline
    train_loader, val_loader, _, classes = load_pcos_data(data_dir=cleaned_path, batch_size=32)
    print(f"Dataset Target Labels: {classes}")

    model = PCOSResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

    epochs = 15
    print("Beginning Training Runs...")
    
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = (correct / total) * 100
        
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = (val_correct / val_total) * 100
        print(f"Epoch [{epoch+1}/{epochs}] | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    torch.save(model.state_dict(), "pcos_resnet18.pth")
    print("ResNet18 model successfully saved as 'pcos_resnet18.pth'!")

if __name__ == "__main__":
    run_stabilized_training()