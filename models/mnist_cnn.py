from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass
class MNISTTrainConfig:
    data_dir: str = "./data"
    batch_size: int = 128
    test_batch_size: int = 256
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 42
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str | None = None


class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 체크포인트 키 구조에 맞춰 모델 구조 수정
        # features.0: Conv2d(1, 32)
        # features.2: Conv2d(32, 64)
        # classifier.1: Linear(in_features, 128) - size mismatch 오류를 통해 in_features=12544임을 확인 (64 * 14 * 14)
        # classifier.4: Linear(128, 10)
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # 0
            nn.ReLU(),                                  # 1
            nn.Conv2d(32, 64, kernel_size=3, padding=1),# 2
            nn.ReLU(),                                  # 3
            nn.MaxPool2d(2)                             # 4: 28x28 -> 14x14
        )
        # features 출력 크기: 64채널 * 14 * 14 = 12544
        
        self.classifier = nn.Sequential(
            nn.Flatten(),                               # 0
            nn.Linear(64 * 14 * 14, 128),               # 1
            nn.ReLU(),                                  # 2
            nn.Dropout(0.25),                           # 3
            nn.Linear(128, 10),                         # 4
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dataloaders(config: MNISTTrainConfig) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root=config.data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=config.data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total

