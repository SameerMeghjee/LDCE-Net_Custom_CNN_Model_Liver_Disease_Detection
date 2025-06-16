import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
from ldce_model import LDCE_Net

def main():
    # CONFIG
    NUM_CLASSES = 3
    BATCH_SIZE = 16
    EPOCHS = 35
    LEARNING_RATE = 1e-3
    PATIENCE = 6
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "ldce_model.pt"
    DATA_DIR = "C:\\Users\\FALCON JNB\\Downloads\\ldce\\Liver Ultrasounds"

    # TRANSFORMS
    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((128, 128)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # LOAD DATASET
    base = datasets.ImageFolder(DATA_DIR, transform=val_transform)
    y = [s[1] for s in base.samples]
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(split.split(np.zeros(len(y)), y))

    train_ds = torch.utils.data.Subset(datasets.ImageFolder(DATA_DIR, transform=transform), train_idx)
    val_ds = torch.utils.data.Subset(datasets.ImageFolder(DATA_DIR, transform=val_transform), val_idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)  # safe for Windows
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # PARSE RESUME FLAG
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    args = parser.parse_args()

    # INIT MODEL
    model = LDCE_Net(num_classes=NUM_CLASSES).to(DEVICE)
    if args.resume and os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✅ Resumed from saved checkpoint.")

    criterion = nn.NLLLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    # TRACKERS
    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []
    best_val_loss = float('inf')
    early_stop_counter = 0

    # TRAINING LOOP
    for epoch in range(EPOCHS):
        model.train()
        correct, total_loss = 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()

        train_loss_list.append(total_loss / len(train_loader))
        train_acc_list.append(100 * correct / len(train_loader.dataset))

        # Validation
        model.eval()
        val_loss, correct, y_true, y_pred = 0, 0, [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                val_loss += criterion(out, y).item()
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                y_true.extend(y.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = 100 * correct / len(val_loader.dataset)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss={train_loss_list[-1]:.4f}, Val Loss={val_loss:.4f}, Train Acc={train_acc_list[-1]:.2f}%, Val Acc={val_acc:.2f}%")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print("✔️ Model saved.")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= PATIENCE:
                print("⛔ Early stopping.")
                break

    # CONFUSION MATRIX & PLOTS
    conf_mat = confusion_matrix(y_true, y_pred)
    os.makedirs("plots", exist_ok=True)

    plt.figure()
    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(val_loss_list, label='Val Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.savefig("plots/loss_curve.png")

    plt.figure()
    plt.plot(train_acc_list, label='Train Accuracy')
    plt.plot(val_acc_list, label='Val Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.savefig("plots/accuracy_curve.png")

    plt.figure(figsize=(6,5))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("plots/confusion_matrix.png")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
