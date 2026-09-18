"""Train a Transformer on a recorder session and save a portable PyTorch checkpoint."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from dataset import LipSyncDataset
from model import LipSyncTransformer


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.KLDivLoss(reduction="batchmean")
    loss_sum = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss_sum += criterion(torch.log_softmax(logits, dim=1), targets).item() * len(inputs)
            correct += (logits.argmax(1) == targets.argmax(1)).sum().item()
            count += len(inputs)
    return loss_sum / max(count, 1), correct / max(count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir")
    parser.add_argument("--output", default="artifacts/lipsync_transformer.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = LipSyncDataset(args.dataset_dir)
    validation_size = max(1, int(len(dataset) * args.validation_ratio))
    training_size = len(dataset) - validation_size
    if training_size < 1:
        raise ValueError("Dataset needs at least two non-silent rows.")
    training_set, validation_set = random_split(
        dataset, [training_size, validation_size], generator=torch.Generator().manual_seed(args.seed)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LipSyncTransformer(dataset.output_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.KLDivLoss(reduction="batchmean")
    training_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size)
    best_loss = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for inputs, targets in training_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            loss = criterion(torch.log_softmax(model(inputs), dim=1), targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        validation_loss, validation_accuracy = evaluate(model, validation_loader, device)
        print(f"epoch={epoch:03d} val_kl={validation_loss:.5f} val_top1={validation_accuracy:.3%}")
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_size": dataset.input_size,
                    "output_labels": dataset.labels,
                    "hidden_size": 128,
                    "layers": 3,
                    "heads": 4,
                },
                output_path,
            )

    with output_path.with_suffix(".json").open("w", encoding="utf-8") as output_file:
        json.dump({"best_validation_kl": best_loss, "labels": dataset.labels}, output_file, indent=2)


if __name__ == "__main__":
    main()
