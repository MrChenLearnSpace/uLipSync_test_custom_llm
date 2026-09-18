"""Evaluate a checkpoint on a separate recorder session or held-out dataset."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import LipSyncDataset
from model import LipSyncTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", help="Held-out recorder session directory.")
    parser.add_argument("checkpoint")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", default="artifacts/evaluation.json")
    args = parser.parse_args()

    dataset = LipSyncDataset(args.dataset_dir)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if checkpoint["output_labels"] != dataset.labels:
        raise ValueError("Checkpoint labels and metadata.json outputLabels must match in order.")

    model = LipSyncTransformer(
        len(dataset.labels),
        hidden_size=checkpoint["hidden_size"],
        layers=checkpoint["layers"],
        heads=checkpoint["heads"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    loader = DataLoader(dataset, batch_size=args.batch_size)
    criterion = nn.KLDivLoss(reduction="sum")
    confusion = torch.zeros(len(dataset.labels), len(dataset.labels), dtype=torch.int64)
    total_kl = 0.0
    total_l1 = 0.0
    total_count = 0
    with torch.no_grad():
        for inputs, targets in loader:
            logits = model(inputs)
            probabilities = torch.softmax(logits, dim=1)
            total_kl += criterion(torch.log_softmax(logits, dim=1), targets).item()
            total_l1 += torch.abs(probabilities - targets).sum().item()
            predictions = probabilities.argmax(dim=1)
            expected = targets.argmax(dim=1)
            for target, prediction in zip(expected, predictions):
                confusion[target, prediction] += 1
            total_count += len(inputs)

    metrics = {
        "samples": total_count,
        "kl_divergence": total_kl / max(total_count, 1),
        "mean_l1_per_viseme": total_l1 / max(total_count * len(dataset.labels), 1),
        "top1_accuracy": confusion.diag().sum().item() / max(total_count, 1),
        "labels": dataset.labels,
        "confusion_matrix": confusion.tolist(),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
