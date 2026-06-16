import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Per-spectrum normalization
# -----------------------------
def normalize_each_spectrum(X):
    row_max = np.max(np.abs(X), axis=1, keepdims=True)
    row_max[row_max == 0] = 1.0
    return X / row_max


# -----------------------------
# Data loading
# -----------------------------
def load_data(file_path, label_col="label"):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path, engine="openpyxl")
    else:
        raise ValueError("Unsupported file format. Use .csv or .xlsx")

    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found. Available columns include: {df.columns[:10].tolist()} ...")

    X = df.drop(columns=[label_col]).copy()
    y = df[label_col].copy()

    X = X.apply(pd.to_numeric, errors="coerce")

    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.mean())

    return X.values.astype(np.float32), y.values, list(X.columns)


# -----------------------------
# Dataset
# -----------------------------
class RamanDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# -----------------------------
# Stronger 1D CNN model
# -----------------------------
class Raman1DCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -----------------------------
# Training
# -----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    progress_bar = tqdm(
        loader,
        desc=f"Epoch {epoch}/{total_epochs}",
        ncols=110,
        leave=True
    )

    for X_batch, y_batch in progress_bar:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(y_batch.detach().cpu().numpy())

        progress_bar.set_postfix(batch_loss=f"{loss.item():.4f}")

    loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return loss, acc, macro_f1


# -----------------------------
# Evaluation
# -----------------------------
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        total_loss += loss.item() * X_batch.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(y_batch.detach().cpu().numpy())

    loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return loss, acc, macro_f1


# -----------------------------
# Early stopping
# -----------------------------
class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.best_loss = None
        self.counter = 0
        self.stop = False

    def step(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


# -----------------------------
# Plot training history
# -----------------------------
def plot_training_history(history_df, output_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_macro_f1"], label="Train Macro F1")
    plt.plot(history_df["epoch"], history_df["val_macro_f1"], label="Val Macro F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title("Training vs Validation Macro F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "macro_f1_curve.png"), dpi=200)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    X, y_raw, feature_names = load_data(args.data_path, label_col=args.label_col)

    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    num_classes = len(label_encoder.classes_)

    print(f"Samples: {len(X)}")
    print(f"Features per spectrum: {X.shape[1]}")
    print(f"Classes: {num_classes}")

    indices = np.arange(len(X))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=args.seed,
        stratify=y
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=args.seed,
        stratify=y[temp_idx]
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # preprocessing
    X_train = normalize_each_spectrum(X_train)
    X_val = normalize_each_spectrum(X_val)
    X_test = normalize_each_spectrum(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # save artifacts
    joblib.dump(label_encoder, os.path.join(args.output_dir, "label_encoder.joblib"))
    joblib.dump(scaler, os.path.join(args.output_dir, "scaler.joblib"))

    np.savez(
        os.path.join(args.output_dir, "splits.npz"),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx
    )

    with open(os.path.join(args.output_dir, "feature_names.json"), "w") as f:
        json.dump([str(c) for c in feature_names], f)

    # loaders
    train_ds = RamanDataset(X_train, y_train)
    val_ds = RamanDataset(X_val, y_val)
    test_ds = RamanDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # model
    model = Raman1DCNN(num_classes=num_classes).to(device)

    # class-weighted loss
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    early_stopper = EarlyStopping(patience=args.patience)

    best_model_path = os.path.join(args.output_dir, "best_model.pt")
    history = []
    best_val_loss = float("inf")

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )

        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, criterion, device
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1
        })

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": num_classes
                },
                best_model_path
            )

        if early_stopper.step(val_loss):
            print(f"Early stopping at epoch {epoch}")
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(args.output_dir, "training_history.csv"), index=False)
    plot_training_history(history_df, args.output_dir)

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)

    summary = {
        "val_loss": float(val_loss),
        "val_accuracy": float(val_acc),
        "val_macro_f1": float(val_f1),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_f1)
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print("\nTraining finished.")
    print("Validation Accuracy:", round(val_acc, 4))
    print("Validation Macro F1:", round(val_f1, 4))
    print("Test Accuracy:", round(test_acc, 4))
    print("Test Macro F1:", round(test_f1, 4))
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    class Args:
        data_path = "/home/ubuntu/Final-Project-SadikSabbir/raman_spectra_api_compounds.xlsx"
        label_col = "label"
        output_dir = "/home/ubuntu/Final-Project-SadikSabbir/raman_output"

        epochs = 50
        batch_size = 32
        lr = 1e-3
        weight_decay = 1e-4
        patience = 10
        seed = 42

    args = Args()
    main(args)