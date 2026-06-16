import os
import json
import joblib
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


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
        raise ValueError(f"Column '{label_col}' not found.")

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
# Model
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
# Evaluation
# -----------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y_batch.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)

    return acc, macro_f1, cm, all_targets, all_preds


# -----------------------------
# Helper: x-axis values
# -----------------------------
def get_x_axis(feature_names):
    try:
        x_values = np.array([float(v) for v in feature_names], dtype=float)
        x_label = "Raman Shift"
    except Exception:
        x_values = np.arange(len(feature_names))
        x_label = "Feature Index"
    return x_values, x_label


# -----------------------------
# Plot confusion matrix
# -----------------------------
def plot_confusion_matrix(cm, class_names, output_path):
    plt.figure(figsize=(16, 14))
    plt.imshow(cm, interpolation="nearest", aspect="auto", cmap="Blues")
    plt.colorbar()
    plt.title("Test Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=8)
    plt.yticks(tick_marks, class_names, fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


# -----------------------------
# Plot misclassified samples
# -----------------------------
def plot_misclassified_samples(
    X_raw_test,
    y_true,
    y_pred,
    class_names,
    feature_names,
    output_dir,
    max_plots=24
):
    mis_mask = y_true != y_pred
    mis_idx = np.where(mis_mask)[0]

    if len(mis_idx) == 0:
        print("No misclassified samples found.")
        return

    x_values, x_label = get_x_axis(feature_names)

    # save one combined grid
    n_to_plot = min(len(mis_idx), max_plots)
    ncols = 3
    nrows = math.ceil(n_to_plot / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax_idx, sample_idx in enumerate(mis_idx[:n_to_plot]):
        ax = axes[ax_idx]
        ax.plot(x_values, X_raw_test[sample_idx])
        ax.set_title(
            f"Idx {sample_idx}\nTrue: {class_names[y_true[sample_idx]]}\nPred: {class_names[y_pred[sample_idx]]}",
            fontsize=9
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel("Intensity")

    for ax_idx in range(n_to_plot, len(axes)):
        axes[ax_idx].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "misclassified_samples_grid.png"), dpi=220)
    plt.close()

    # save each misclassified spectrum separately too
    indiv_dir = os.path.join(output_dir, "misclassified_plots")
    os.makedirs(indiv_dir, exist_ok=True)

    for sample_idx in mis_idx:
        plt.figure(figsize=(8, 4))
        plt.plot(x_values, X_raw_test[sample_idx])
        plt.title(
            f"Sample {sample_idx} | True: {class_names[y_true[sample_idx]]} | Pred: {class_names[y_pred[sample_idx]]}"
        )
        plt.xlabel(x_label)
        plt.ylabel("Intensity")
        plt.tight_layout()
        safe_true = class_names[y_true[sample_idx]].replace("/", "_").replace(" ", "_")
        safe_pred = class_names[y_pred[sample_idx]].replace("/", "_").replace(" ", "_")
        fname = f"idx_{sample_idx}_true_{safe_true}_pred_{safe_pred}.png"
        plt.savefig(os.path.join(indiv_dir, fname), dpi=220)
        plt.close()


# -----------------------------
# Main
# -----------------------------
def main(args):
    # load full raw data
    X_raw, y_raw, feature_names = load_data(args.data_path, label_col=args.label_col)

    # load train-time artifacts
    label_encoder = joblib.load(os.path.join(args.output_dir, "label_encoder.joblib"))
    scaler = joblib.load(os.path.join(args.output_dir, "scaler.joblib"))
    splits = np.load(os.path.join(args.output_dir, "splits.npz"))

    y = label_encoder.transform(y_raw)
    test_idx = splits["test_idx"]

    # raw test set for plotting
    X_raw_test = X_raw[test_idx].copy()
    y_test = y[test_idx]

    # model input preprocessing
    X_test = normalize_each_spectrum(X_raw_test.copy())
    X_test = scaler.transform(X_test)

    test_ds = RamanDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Test samples:", len(test_idx))

    checkpoint = torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=device)
    model = Raman1DCNN(num_classes=checkpoint["num_classes"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    acc, macro_f1, cm, y_true, y_pred = evaluate(model, test_loader, device)
    class_names = list(label_encoder.classes_)

    print("Test Accuracy:", round(acc, 4))
    print("Test Macro F1:", round(macro_f1, 4))

    print("\nClassification Report:\n")
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    print(report)

    # save confusion matrix csv
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(args.output_dir, "test_confusion_matrix.csv"))

    # save confusion matrix image
    plot_confusion_matrix(
        cm,
        class_names,
        os.path.join(args.output_dir, "test_confusion_matrix.png")
    )

    # save per-sample predictions
    pred_df = pd.DataFrame({
        "sample_index_within_test_set": np.arange(len(y_true)),
        "true_label": label_encoder.inverse_transform(y_true),
        "predicted_label": label_encoder.inverse_transform(y_pred),
        "is_correct": y_true == y_pred
    })
    pred_df.to_csv(os.path.join(args.output_dir, "test_predictions.csv"), index=False)

    # save misclassified rows only
    mis_df = pred_df.loc[~pred_df["is_correct"]].copy()
    mis_df.to_csv(os.path.join(args.output_dir, "misclassified_samples.csv"), index=False)

    # plot misclassified spectra
    plot_misclassified_samples(
        X_raw_test=X_raw_test,
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        feature_names=feature_names,
        output_dir=args.output_dir,
        max_plots=args.max_misclassified_grid_plots
    )

    with open(os.path.join(args.output_dir, "test_summary.json"), "w") as f:
        json.dump({
            "test_accuracy": float(acc),
            "test_macro_f1": float(macro_f1),
            "num_test_samples": int(len(y_true)),
            "num_misclassified": int(np.sum(y_true != y_pred))
        }, f, indent=4)

    with open(os.path.join(args.output_dir, "test_classification_report.txt"), "w") as f:
        f.write(report)

    print(f"\nSaved test results to: {args.output_dir}")
    print("Saved files include:")
    print("- test_confusion_matrix.csv")
    print("- test_confusion_matrix.png")
    print("- test_predictions.csv")
    print("- misclassified_samples.csv")
    print("- misclassified_samples_grid.png")
    print("- misclassified_plots/ (individual spectra)")


if __name__ == "__main__":
    class Args:
        data_path = "/home/ubuntu/Final-Project-SadikSabbir/raman_spectra_api_compounds.xlsx"
        label_col = "label"
        output_dir = "/home/ubuntu/Final-Project-SadikSabbir/raman_output"
        batch_size = 32
        max_misclassified_grid_plots = 24

    args = Args()
    main(args)