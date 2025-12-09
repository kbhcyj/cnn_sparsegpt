from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn

from data.calibration import collect_calibration_inputs
from models.cifar_cnn import (
    CIFARCNN,
    CIFARTrainConfig,
    evaluate as evaluate_cifar,
    get_dataloaders as get_cifar_dataloaders,
)
from models.mnist_cnn import (
    MNISTTrainConfig,
    SimpleCNN,
    evaluate as evaluate_mnist,
    get_dataloaders as get_mnist_dataloaders,
)
from models.standard_cifar import ResNet18_CIFAR, VGG16_CIFAR
from .sparsegpt import prune_layer_magnitude, prune_layer_obs


@dataclass
class PruningConfig:
    """
    프루닝 실행을 위한 설정값들을 정의합니다.
    """
    weights: str                # 학습된 모델의 체크포인트 경로
    data_dir: str               # 데이터셋 경로
    batch_size: int = 128
    test_batch_size: int = 256
    num_workers: int = 4
    calib_batches: int = 8      # Hessian 계산을 위해 사용할 캘리브레이션 배치 수
    calib_samples: int = 2048   # (옵션) 최대 샘플 수 제한
    n: int = 2                  # N:M Sparse에서 N
    m: int = 4                  # N:M Sparse에서 M
    lambd: float = 1e-4         # Ridge Regularization 상수 (Dampening)
    device: str = "cpu"
    output: str | None = None   # 프루닝된 모델 저장 경로
    mode: str = "sparsegpt"     # 'sparsegpt' 또는 'magnitude' (단순 절댓값)
    model: str = "mnist"        # 모델 아키텍처 이름
    seed: int = 42
    enforce_nm: bool = True     # True면 N:M 구조적 프루닝, False면 Unstructured


# 모델별 팩토리 및 유틸리티 매핑
MODEL_REGISTRY = {
    "mnist": {
        "model_class": SimpleCNN,
        "config_class": MNISTTrainConfig,
        "loader_fn": get_mnist_dataloaders,
        "evaluate_fn": evaluate_mnist,
    },
    "cifar10": {
        "model_class": CIFARCNN,
        "config_class": CIFARTrainConfig,
        "loader_fn": get_cifar_dataloaders,
        "evaluate_fn": evaluate_cifar,
    },
    "resnet18_cifar": {
        "model_class": ResNet18_CIFAR,
        "config_class": CIFARTrainConfig,
        "loader_fn": get_cifar_dataloaders,
        "evaluate_fn": evaluate_cifar,
    },
    "vgg16_cifar": {
        "model_class": VGG16_CIFAR,
        "config_class": CIFARTrainConfig,
        "loader_fn": get_cifar_dataloaders,
        "evaluate_fn": evaluate_cifar,
    },
}


def load_state_dict(weights_path: str) -> Dict[str, torch.Tensor]:
    """체크포인트 파일에서 state_dict만 안전하게 추출합니다."""
    checkpoint = torch.load(weights_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint:
            return checkpoint["model_state"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        return checkpoint
    raise ValueError(f"지원하지 않는 체크포인트 형식: {type(checkpoint)}")


def get_prunable_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """
    모델에서 프루닝 가능한 레이어(Conv2d, Linear)들을 추출합니다.
    SparseGPT는 레이어별로 순차적으로 적용되므로 순서가 중요합니다.
    """
    prunable: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prunable.append((name, module))
    return prunable


def tensor_sparsity(tensor: torch.Tensor) -> float:
    """텐서 내 0의 비율(%)을 계산합니다."""
    total = tensor.numel()
    zeros = (tensor == 0).sum().item()
    return (zeros / total) * 100.0


def run_pruning(config: PruningConfig) -> Dict[str, float]:
    """
    전체 프루닝 파이프라인을 실행합니다.
    
    SparseGPT (One-Shot Pruning) 흐름:
    1. Pre-trained 모델 로드 및 Baseline 성능 측정
    2. 레이어별 순차 처리 (Sequential Layer-wise Processing):
       - 각 레이어의 입력(Input Activations)을 캘리브레이션 데이터로 수집
       - 수집된 입력으로 Hessian 계산 (H = X X^T)
       - OBS(Optimal Brain Surgeon) 알고리즘으로 가중치 프루닝 및 에러 보정
    3. 최종 성능 측정
    
    Note: 이 과정은 재학습(Fine-tuning) 없이 수행됩니다.
    """
    if config.model not in MODEL_REGISTRY:
        raise ValueError(f"지원하지 않는 모델: {config.model}")

    device = torch.device(config.device)
    spec = MODEL_REGISTRY[config.model]
    model_class = spec["model_class"]
    config_class = spec["config_class"]
    loader_fn = spec["loader_fn"]
    evaluate_fn = spec["evaluate_fn"]

    # 모델 초기화 및 가중치 로드
    state_dict = load_state_dict(config.weights)
    model = model_class()
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Warning: State dict mismatch. Attempting strict=False or ignoring if intended. Error: {e}")
        # 아키텍처 변경 등으로 인한 로드 실패 시 예외 처리
        # (현재 컨텍스트에서는 사용자 요청에 따라 실패하도록 둠)
        raise e
        
    model.to(device)
    model.eval()

    # 데이터 로더 설정 (캘리브레이션 및 테스트용)
    base_cfg = config_class()
    data_cfg = config_class(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        test_batch_size=config.test_batch_size,
        epochs=1,
        lr=base_cfg.lr,
        weight_decay=getattr(base_cfg, "weight_decay", 0.0),
        seed=config.seed,
        num_workers=config.num_workers,
        device=config.device,
        save_path=None,
    )
    train_loader, test_loader = loader_fn(data_cfg)
    criterion = nn.CrossEntropyLoss()

    # 1. Baseline 성능 측정
    _, baseline_acc = evaluate_fn(model, test_loader, criterion, device)
    print(f"Baseline accuracy before pruning: {baseline_acc * 100:.2f}%")

    # 2. 레이어별 프루닝 수행
    for name, layer in get_prunable_layers(model):
        if not hasattr(layer, "weight"):
            continue
        before = tensor_sparsity(layer.weight.data)
        print(f"[{name}] sparsity before pruning: {before:.2f}%")

        if config.mode == "sparsegpt":
            # SparseGPT (OBS based) 모드
            # Hessian 계산을 위해서는 실제 데이터가 통과할 때의 입력값(Activation)이 필요합니다.
            # 이를 위해 소량의 캘리브레이션 데이터를 사용하여 해당 레이어 앞단까지 Forward pass를 수행합니다.
            activations = collect_calibration_inputs(
                model,
                train_loader,
                layer,
                device,
                max_batches=config.calib_batches,
                max_samples=config.calib_samples,
            )
            
            # 수집된 입력(activations)을 바탕으로 Hessian을 계산하고 프루닝 수행
            prune_layer_obs(
                layer,
                activations,
                n=config.n,
                m=config.m,
                lambd=config.lambd,
                enforce_nm=config.enforce_nm,
            )
        else:
            # Magnitude Pruning 모드 (비교군)
            # 데이터 없이 가중치 크기만 보고 자릅니다.
            prune_layer_magnitude(
                layer,
                n=config.n,
                m=config.m,
                enforce_nm=config.enforce_nm,
            )

        after = tensor_sparsity(layer.weight.data)
        print(f"[{name}] sparsity after pruning: {after:.2f}%")

    # 3. 최종 성능 측정 (No Fine-tuning)
    # SparseGPT는 프루닝 과정에서 가중치 업데이트(보정)가 일어나므로, 
    # 별도의 재학습 없이도 성능이 어느 정도 복구되어야 합니다.
    _, pruned_acc = evaluate_fn(model, test_loader, criterion, device)
    print(f"Accuracy after pruning: {pruned_acc * 100:.2f}%")

    if config.output:
        torch.save({"model_state": model.state_dict()}, config.output)
        print(f"Pruned model saved to: {config.output}")

    return {
        "baseline_acc": baseline_acc,
        "pruned_acc": pruned_acc,
    }
