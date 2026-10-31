"""Evaluate a saved CNN or GFNet checkpoint on ADNI test slices."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from dataset import (
    MRIDataset,
    build_transforms,
    load_split_csvs,
    resolve_image_paths,
)
from modules import CNNBaseline, GFNet


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--model", choices=["cnn", "gfnet"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )

    return parser.parse_args()


def evaluate(model, loader, device):
    """Collect labels, predictions, probabilities, and average loss."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_confidences = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            probabilities = torch.softmax(outputs, dim=1)
            confidence, predictions = probabilities.max(dim=1)

            total_loss += loss.item() * labels.size(0)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_probabilities.extend(
                probabilities[:, 1].cpu().tolist()
            )
            all_confidences.extend(confidence.cpu().tolist())

    if not all_labels:
        raise ValueError("The test dataset is empty.")

    return {
        "loss": total_loss / len(all_labels),
        "labels": np.asarray(all_labels, dtype=int),
        "predictions": np.asarray(all_predictions, dtype=int),
        "ad_probabilities": np.asarray(all_probabilities, dtype=float),
        "confidence": np.asarray(all_confidences, dtype=float),
    }


def save_plots(labels, predictions, ad_probabilities, model_name, output_dir):
    """Save a confusion matrix and ROC curve."""
    cm = confusion_matrix(labels, predictions, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["NC", "AD"],
    ).plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )
    ax.set_title(f"{model_name} Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(
        output_dir / "confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    fpr, tpr, _ = roc_curve(labels, ad_probabilities)
    auroc = roc_auc_score(labels, ad_probabilities)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"{model_name} (AUROC = {auroc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} Test ROC Curve")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(
        output_dir / "roc_curve.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    args = parse_args()

    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Invalid batch size or worker count.")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"{output_dir} is not empty. Choose a new output directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate all split CSVs, but load images only for the test set.
    _, _, test_df = load_split_csvs(args.split_dir)

    test_df = resolve_image_paths(
        test_df,
        source_root=args.source_root,
        image_root=args.image_root,
    )

    _, evaluation_transform = build_transforms(args.model)

    test_dataset = MRIDataset(
        test_df,
        transform=evaluation_transform,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    if args.model == "cnn":
        model = CNNBaseline()
        model_name = "CNN"
    else:
        model = GFNet(
            image_size=256,
            patch_size=16,
            dim=128,
            depth=8,
            dropout=0.1,
        )
        model_name = "GFNet"

    # Load the plain state_dict saved by the notebooks or train.py.
    state_dict = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)

    results = evaluate(model, test_loader, device)

    labels = results["labels"]
    predictions = results["predictions"]
    ad_probabilities = results["ad_probabilities"]

    expected_labels = (
        test_df["label"].map({"NC": 0, "AD": 1}).to_numpy()
    )
    if not np.array_equal(labels, expected_labels):
        raise RuntimeError("Predictions do not match test CSV order.")

    if not np.isfinite(ad_probabilities).all():
        raise RuntimeError("Non-finite prediction probabilities.")

    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Both NC and AD are required to calculate AUROC.")

    report = classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=["NC", "AD"],
        output_dict=True,
        zero_division=0,
    )

    summary = {
        "model": model_name,
        "checkpoint": str(checkpoint.resolve()),
        "test_slices": len(labels),
        "test_patients": int(test_df["patient_id"].nunique()),
        "test_loss": float(results["loss"]),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "auroc": float(roc_auc_score(labels, ad_probabilities)),
    }

    with (output_dir / "metrics.json").open("w") as file:
        json.dump(summary, file, indent=2)

    pd.DataFrame(report).T.to_csv(
        output_dir / "classification_report.csv"
    )

    # Preserve image identifiers and original CSV order for error analysis.
    prediction_table = test_df[
        ["patient_id", "image_id", "path", "label"]
    ].copy()

    prediction_table["true_label"] = labels
    prediction_table["predicted_label"] = predictions
    prediction_table["predicted_class"] = [
        "NC" if value == 0 else "AD" for value in predictions
    ]
    prediction_table["ad_probability"] = ad_probabilities
    prediction_table["confidence"] = results["confidence"]
    prediction_table["correct"] = predictions == labels

    prediction_table.to_csv(
        output_dir / "test_predictions.csv",
        index=False,
    )

    save_plots(
        labels,
        predictions,
        ad_probabilities,
        model_name,
        output_dir,
    )

    print("Loaded checkpoint:", checkpoint)
    print("Model:", model_name)
    print("Device:", device)
    print("Test slices:", summary["test_slices"])
    print(f"Test loss: {summary['test_loss']:.4f}")
    print(f"Test accuracy: {summary['accuracy']:.2%}")
    print(f"Balanced accuracy: {summary['balanced_accuracy']:.2%}")
    print(f"Macro F1: {summary['macro_f1']:.4f}")
    print(f"AUROC: {summary['auroc']:.4f}")

    print("\nClassification report:")
    print(classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=["NC", "AD"],
        digits=4,
        zero_division=0,
    ))

    print("Evaluation files saved to:", output_dir)


if __name__ == "__main__":
    main()