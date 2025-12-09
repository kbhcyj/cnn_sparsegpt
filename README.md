# CNN SparseGPT Project

이 프로젝트는 대규모 언어 모델(LLM)에 적용되던 **SparseGPT** 프루닝 알고리즘을 **CNN(Convolutional Neural Networks)**, 특히 **ResNet-18**과 **VGG-16** (CIFAR-10/MNIST) 모델에 적용하고 검증하는 구현체입니다.

Optimal Brain Surgeon (OBS) 프레임워크를 기반으로 가중치(Weight)의 중요도를 계산하고, 한 번의 패스(One-shot)로 모델을 경량화합니다.

---

## 📋 주요 기능

- **OBS 기반 프루닝**: 2차 미분 정보(Hessian)를 활용한 정교한 가지치기
- **구조적/비구조적 프루닝 지원**:
    - **N:M Structured Pruning** (예: 2:4) - 하드웨어 가속에 유리
    - **Unstructured Pruning** (Magnitude/SparseGPT) - 높은 압축률 가능
- **다양한 모델 지원**: ResNet-18, VGG-16, Simple CNN
- **자동화된 벤치마크**: 프루닝 전후의 정확도(Accuracy) 및 희소성(Sparsity) 비교

---

## 🧠 기술적 상세: SparseGPT for CNN

이 프로젝트는 원본 **SparseGPT (Frantar & Alistarh, 2023)**의 핵심 아이디어를 **CNN 모델**에 맞게 재해석하여 적용했습니다.

### 1. Hessian 역행렬 기반 가중치 업데이트 (OBS)
SparseGPT는 각 레이어의 손실 함수에 대한 2차 미분값(Hessian)을 사용하여, 제거되는 가중치가 손실에 미치는 영향을 최소화하도록 남은 가중치를 업데이트합니다.
- **원본 (LLM)**: Transformer의 Linear Layer($W \cdot X$)에서 Row-wise Hessian을 계산.
- **본 구현 (CNN)**: Convolution Layer를 **`im2col`** 형태 또는 채널 단위 행렬 곱으로 해석하여 Hessian을 축적하고, OBS 수식을 적용하여 필터(Kernel) 가중치를 최적화합니다.

### 2. 레이어 단위 로컬 최적화 (Layer-wise Reconstruction)
전체 모델을 한 번에 재학습(Fine-tuning)하지 않고, 각 레이어의 출력을 보존하는 **Local Regression 문제**로 치환하여 풉니다.
$$ \text{argmin}_W || W X - \hat{W} X ||^2_2 $$
이를 통해 Fine-tuning 없이도(Zero-shot/One-shot) 높은 정확도를 유지합니다.

### 3. 효율적인 역행렬 계산 (Adaptive Mask Selection)
Hessian의 역행렬($H^{-1}$)을 효율적으로 업데이트하기 위해 Cholesky 분해와 유사한 방식(Row-by-Row 업데이트)을 사용합니다. 이를 통해 수십억 파라미터 모델이 아닌 CNN에서도 매우 빠른 속도로 최적의 마스크(Mask)를 찾아냅니다.

---

## 🛠️ 설치 및 환경 설정

이 프로젝트는 Python 3.8+ 및 PyTorch 환경에서 실행됩니다.

### 1. 저장소 클론
```bash
git clone https://github.com/kbhcyj/cnn_sparsegpt.git
cd cnn_sparsegpt
```

### 2. Conda 가상환경 생성 및 패키지 설치
```bash
# 가상환경 생성 (이미 pytorch 환경이 있다면 생략 가능)
conda create -n pytorch python=3.10
conda activate pytorch

# 필수 패키지 설치
pip install -r requirements.txt
```

---

## 🚀 사용 가이드

### 1단계: 베이스라인 모델 준비
프루닝을 수행하려면 먼저 학습된 모델(체크포인트)이 필요합니다. 제공된 스크립트로 직접 학습시킬 수 있습니다.

```bash
# CIFAR-10 ResNet-18 학습 (약 93~94% 정확도 목표)
python scripts/train_baseline.py \
    --model resnet18_cifar \
    --epochs 100 \
    --save-path checkpoints/resnet18_cifar.pt
```
> **참고**: 학습된 모델은 `checkpoints/` 폴더에 저장됩니다.

### 2단계: 프루닝 실행 (Pruning)
설정 파일(`configs/*.yaml`)을 사용하여 다양한 프루닝 실험을 진행할 수 있습니다.

**예시 1: ResNet-18에 2:4 SparseGPT 적용**
```bash
python scripts/prune.py --config configs/exp_cifar_resnet18.yaml
```

**예시 2: 명령줄 인수로 직접 실행**
```bash
python scripts/prune.py \
    --model resnet18_cifar \
    --weights checkpoints/resnet18_cifar.pt \
    --mode sparsegpt \
    --sparsity 0.5 \
    --n 2 --m 4
```

### 3단계: 결과 확인 및 시각화
실험 결과는 `experiments/results`에 CSV 형태로 저장되며, 이를 시각화할 수 있습니다.

```bash
# 벤치마크 실행 및 결과 플로팅
python scripts/run_benchmark.py
python scripts/plot_benchmark.py
```
생성된 그래프는 `experiments/plots/` 디렉토리에서 확인할 수 있습니다.

---

## 📂 디렉토리 구조

```text
cnn_sparsegpt/
├── checkpoints/    # 학습된 모델 가중치 저장소 (.gitignore)
├── configs/        # 실험 설정 파일 (YAML)
├── data/           # 데이터셋 및 캘리브레이션 로더
├── docs/           # 프로젝트 문서 및 분석 보고서
├── experiments/    # 실험 결과(logs, csv) 및 그래프
├── models/         # CNN 모델 정의 (ResNet, VGG, SimpleCNN)
├── pruning/        # 핵심 알고리즘 (SparseGPT, OBS, Masking)
└── scripts/        # 실행 스크립트 (train, prune, benchmark)
```

## 📝 라이선스
이 프로젝트는 MIT License를 따릅니다.
