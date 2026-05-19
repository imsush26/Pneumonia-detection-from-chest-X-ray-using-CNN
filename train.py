"""
Pneumonia Detection - Training Script
Uses ResNet50 pretrained on ImageNet, fine-tuned on chest X-rays
Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
"""

import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR   = r"C:\Users\SUSHANT\OneDrive\Desktop\Project\pneumonia-detection\chest_xray\chest_xray"   # update this to your dataset path
MODEL_SAVE = "models/best_model.pth"
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR         = 0.001
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────
# DATA TRANSFORMS & LOADERS
# ─────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def get_dataloaders(data_dir):
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), train_transforms)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"),   val_test_transforms)
    test_ds  = datasets.ImageFolder(os.path.join(data_dir, "test"),  val_test_transforms)

    # Class weights to handle imbalance (more PNEUMONIA than NORMAL)
    class_counts = np.bincount(train_ds.targets)
    weights = 1.0 / class_counts
    sample_weights = weights[train_ds.targets]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    print(f"Classes: {train_ds.classes}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader, train_ds.classes


# ─────────────────────────────────────────────
# MODEL — Fine-tuned ResNet50
# ─────────────────────────────────────────────
def build_model(num_classes=2):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer4 + fc for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model.to(DEVICE)


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
def train_model(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = StepLR(optimizer, step_size=4, gamma=0.5)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc   = 0.0

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 40)

        for phase in ["train", "val"]:
            loader = train_loader if phase == "train" else val_loader
            model.train() if phase == "train" else model.eval()

            running_loss, running_correct = 0.0, 0

            for inputs, labels in loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss    = criterion(outputs, labels)
                    preds   = outputs.argmax(dim=1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss    += loss.item() * inputs.size(0)
                running_correct += (preds == labels).sum().item()

            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc  = running_correct / len(loader.dataset)

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            print(f"  {phase.upper():5s} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_val_acc:
                best_val_acc   = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, MODEL_SAVE)
                print(f"  ✅ Best model saved (val_acc={best_val_acc:.4f})")

        scheduler.step()

    print(f"\nTraining complete. Best Val Acc: {best_val_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model, history


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
def evaluate(model, test_loader, class_names):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            preds   = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png")
    plt.show()
    print("Confusion matrix saved to models/confusion_matrix.png")


# ─────────────────────────────────────────────
# PLOT TRAINING HISTORY
# ─────────────────────────────────────────────
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"],   label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["val_acc"],   label="Val Acc")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("models/training_history.png")
    plt.show()
    print("Training history saved to models/training_history.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    train_loader, val_loader, test_loader, class_names = get_dataloaders(DATA_DIR)
    model = build_model(num_classes=len(class_names))

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable")

    model, history = train_model(model, train_loader, val_loader)
    evaluate(model, test_loader, class_names)
    plot_history(history)
