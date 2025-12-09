# CNN SparseGPT 벤치마크 결과 분석 (종합)

본 문서는 CNN 모델(Custom, ResNet, VGG)에 대해 SparseGPT 알고리즘을 적용한 성능 벤치마크 결과를 기술합니다. 특히 **모델의 학습 완성도(Fully Trained 여부)**가 프루닝 성능에 미치는 영향을 중점적으로 분석하였습니다.

## 1. 실험 개요
- **데이터셋**: CIFAR-10
- **조건**: 2:4 Structured Sparsity (50% 희소성), **No Fine-tuning** (재학습 배제)
- **비교 대상**:
  - **Baseline**: 프루닝 전 원본 정확도
  - **Magnitude Pruning**: 가중치 크기 하위 50% 제거 (전통적 방식)
  - **SparseGPT (OBS)**: Hessian 역행렬 기반 Adaptive Mask & Weight Update

## 2. 실험 결과 비교: 학습 완성도에 따른 차이

두 가지 Baseline 조건에서 실험을 수행하여 SparseGPT의 거동 변화를 관찰했습니다.

### Case A: Under-trained (초기 실험)
모델이 충분히 수렴하지 않은 상태 (Epoch 5~7)

| Model | Baseline | Magnitude Acc (Drop) | SparseGPT Acc (Drop) |
| :--- | :---: | :---: | :---: |
| **Custom CNN** | 63.18% | 19.00% (-44.18) | **48.31%** (-14.87) |
| **VGG-16** | 80.40% | 22.52% (-57.88) | **38.20%** (-42.20) |
| **ResNet-18** | 79.45% | 20.80% (-58.65) | **34.27%** (-45.18) |

### Case B: Fully-trained (재학습 실험)
모델을 Local Minima까지 충분히 학습시킨 상태 (Epoch 15~30)

| Model | Baseline | Magnitude Acc (Drop) | SparseGPT Acc (Drop) |
| :--- | :---: | :---: | :---: |
| **Custom CNN** | **78.15%** | 45.48% (-32.67) | **69.25%** (**-8.90**) |
| **VGG-16** | **85.74%** | 31.15% (-54.59) | **68.31%** (**-17.43**) |
| **ResNet-18** | **85.22%** | 20.75% (-64.47) | **42.58%** (-42.64) |

## 3. 심층 분석

### A. "잘 학습된 모델일수록 SparseGPT 효과가 강력하다"
Case A와 Case B를 비교했을 때, Baseline 성능 향상폭보다 **SparseGPT 성능 향상폭이 훨씬 큽니다.**
- **VGG-16**: Baseline은 5%p 올랐지만, SparseGPT 결과는 **30%p** (38% → 68%) 폭등했습니다.
- **Custom CNN**: Baseline 15%p 상승 시, SparseGPT 결과는 **21%p** (48% → 69%) 상승했습니다.

**[원인 분석]**
SparseGPT의 핵심인 OBS(Optimal Brain Surgeon) 알고리즘은 손실 함수(Loss Landscape)를 2차 함수로 근사(Taylor Expansion)하여 최적해를 찾습니다.
- **Under-trained**: 학습 도중이라 Loss Surface가 불안정하고 곡률(Hessian)이 최적점 근처의 특성을 반영하지 못해 근사 오차가 큽니다.
- **Fully-trained**: 모델이 Local Minima에 안착하여 Loss Surface가 안정적입니다. 이때의 Hessian 정보는 매우 정확하며, 이를 바탕으로 한 가중치 보정(Update)이 실제로 성능을 크게 보존합니다.

### B. 모델 구조별 특성 (VGG vs ResNet)
- **VGG-16 (직렬 구조)**: Fully-trained 상태에서 SparseGPT 적용 시 **68.31%**까지 성능을 방어하며 재학습 없는 50% 프루닝의 가능성을 보여주었습니다. (Magnitude 대비 +37%p 우위)
- **ResNet-18 (잔차 구조)**: 학습 상태가 좋아져도 성능 복구율(42%)이 낮았습니다. Skip Connection으로 인해 레이어 단위의 독립적인 보정(Layer-wise Reconstruction)만으로는 전체 네트워크의 정합성을 맞추기 어려운 한계가 확인되었습니다.

## 4. 결론
1. **학습 상태의 중요성**: SparseGPT 알고리즘을 적용할 때는 **Baseline 모델을 최대한 수렴시키는 것**이 프루닝 후 성능 보존에 결정적인 역할을 합니다.
2. **알고리즘 우위**: 모든 케이스에서 SparseGPT는 Magnitude Pruning보다 월등한 성능(최대 **+37%p**)을 보였으며, 특히 잘 학습된 직렬 구조 모델(VGG, Custom CNN)에서 그 효과가 극대화되었습니다.
