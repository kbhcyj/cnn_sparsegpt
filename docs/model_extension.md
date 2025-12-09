# 모델 확장 및 CIFAR-10 지원 문서

## 개요
이 문서는 기본 Custom CNN 모델 외에 표준 아키텍처인 **ResNet**과 **VGG**를 CIFAR-10 데이터셋 환경에서 SparseGPT 알고리즘으로 검증하기 위해 수행된 확장 작업을 기술합니다.

## 추가된 모델 (`models/standard_cifar.py`)

`torchvision`의 ImageNet용 모델을 CIFAR-10 이미지 크기(32x32)에 맞게 수정하여 구현하였습니다.

### 1. ResNet18_CIFAR
- **Base**: `torchvision.models.resnet18`
- **수정 사항**:
  - **First Conv Layer**: 7x7 kernel, stride 2, padding 3 (ImageNet용) → **3x3 kernel, stride 1, padding 1** (CIFAR-10용 정보 손실 방지)
  - **MaxPooling**: 제거 (저해상도 이미지에서 공간 정보 보존)
  - **Classifier**: CIFAR-10 클래스 개수(10)에 맞춰 `fc` 레이어 재정의

### 2. VGG16_CIFAR
- **Base**: `torchvision.models.vgg16_bn`
- **수정 사항**:
  - **Classifier Input**: ImageNet 입력 시 7x7 feature map이 나오지만, CIFAR-10(32x32) 입력 시 1x1 feature map이 생성됨. 이에 따라 `classifier`의 첫 Linear 레이어 입력 차원을 수정 (512 * 7 * 7 → **512**)

## 파이프라인 통합 (`pruning/pipeline.py`)

`MODEL_REGISTRY`에 새로운 모델 키를 등록하여 설정 파일에서 손쉽게 모델을 교체할 수 있도록 하였습니다.

```python
MODEL_REGISTRY = {
    # ... 기존 모델 ...
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
```

## 사용 방법

### 1. Baseline 학습
프루닝을 수행하기 전, 먼저 해당 구조로 학습된 체크포인트가 필요합니다.
(새로 추가된 `scripts/train_baseline.py` 사용)

```bash
python scripts/train_baseline.py --model resnet18_cifar --epochs 20 --output checkpoints/resnet18_cifar.pt
```

### 2. 프루닝 실행
Config 파일(`configs/exp_cifar_resnet18.yaml`)의 `model` 필드를 지정하여 실행합니다.

```yaml
model: "resnet18_cifar"
weights: "checkpoints/resnet18_cifar.pt"
# ...
```

