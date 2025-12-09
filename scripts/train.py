from __future__ import annotations

import argparse
import os
from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.optim as optim

from models.cifar_cnn import (
    CIFARCNN,
    CIFARTrainConfig,
    evaluate as evaluate_cifar,
    get_dataloaders as get_cifar_loaders,
    set_seed as set_cifar_seed,
    train_one_epoch as train_cifar_epoch,
)
from models.mnist_cnn import (
    MNISTTrainConfig,
    SimpleCNN,
    evaluate as evaluate_mnist,
    get_dataloaders as get_mnist_loaders,
    set_seed as set_mnist_seed,
    train_one_epoch as train_mnist_epoch,
)


MODEL_REGISTRY: Dict[str, Dict[str, Callable]] = {
    "mnist": {
        "model_class": SimpleCNN,
        "config_class": MNISTTrainConfig,
        "set_seed": set_mnist_seed,
        "loader_fn": get_mnist_loaders,
        "train_fn": train_mnist_epoch,
        "evaluate_fn": evaluate_mnist,
    },
    "cifar10": {
        "model_class": CIFARCNN,
        "config_class": CIFARTrainConfig,
        "set_seed": set_cifar_seed,
        "loader_fn": get_cifar_loaders,
        "train_fn": train_cifar_epoch,
        "evaluate_fn": evaluate_cifar,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CNN 학습 스크립트")
    parser.add_argument("--model", type=str, choices=list(MODEL_REGISTRY.keys()), default="mnist")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="./checkpoints/model.pt")
    return parser.parse_args()


def build_data_config(args: argparse.Namespace, config_class: Callable) -> object:
    return config_class(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        save_path=None,
    )


def main() -> None:
    args = parse_args()
    spec = MODEL_REGISTRY[args.model]

    device = torch.device(args.device)
    spec["set_seed"](args.seed)

    data_cfg = build_data_config(args, spec["config_class"])
    train_loader, test_loader = spec["loader_fn"](data_cfg)

    model = spec["model_class"]().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = spec["train_fn"](model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = spec["evaluate_fn"](model, test_loader, criterion, device)

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()

    if best_state is None:
        raise RuntimeError("Best state was not captured during training.")

    if args.save_path:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save({"model_state": best_state, "val_acc": best_acc}, args.save_path)
        print(f"Best model saved to {args.save_path} (Val Acc: {best_acc * 100:.2f}%)")


if __name__ == "__main__":
    main()

