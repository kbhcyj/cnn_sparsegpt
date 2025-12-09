# CNN SparseGPT 벤치마크 실험 결과 분석

## 1. 실험 개요
본 실험은 세 가지 CNN 모델에 대해 전통적인 프루닝 방식(Magnitude Pruning)과 제안된 **SparseGPT (OBS 기반 Adaptive Mask Selection)** 방식의 성능을 비교 분석하였습니다.

- **데이터셋**: CIFAR-10
- **조건**: 2:4 Structured Sparsity (50% 희소성), **No Fine-tuning** (재학습 배제)
- **비교 대상**:
  1. **Baseline**: 프루닝 전 원본 모델
  2. **Magnitude**: 가중치 절댓값 하위 50% 제거 (보정 없음)
  3. **SparseGPT**: OBS 알고리즘으로 중요도 산정 및 가중치 보정 (One-Shot)

## 2. 실험 결과 (Accuracy)

| Model | Baseline | Magnitude (Drop) | SparseGPT (Drop) | Gap (SparseGPT vs Mag) |
| :--- | :---: | :---: | :---: | :---: |
| **Custom CNN** | 63.18% | 19.00% (-44.18) | **48.31%** (-14.87) | **+29.31%p** |
| **ResNet-18** | 79.45% | 20.80% (-58.65) | **34.27%** (-45.18) | **+13.47%p** |
| **VGG-16** | 80.40% | 22.52% (-57.88) | **38.20%** (-42.20) | **+15.68%p** |

> **Gap**: SparseGPT가 Magnitude 방식 대비 얼마나 정확도를 더 보존했는지를 나타냄.

## 3. 상세 분석

### A. SparseGPT의 유효성 검증
모든 모델에서 **SparseGPT가 Magnitude Pruning을 압도**하는 성능을 보여주었습니다.
- 단순 크기 기반으로 자르는 것(Magnitude)은 모델을 사실상 붕괴(20% 내외 정확도)시켰습니다.
- 반면 SparseGPT는 Hessian 역행렬 정보를 이용해 **제거된 가중치의 영향을 남은 가중치로 업데이트(Compensation)**함으로써, 재학습 없이도 유의미한 수준의 정확도를 회복했습니다.
- 특히 **Custom CNN**에서는 약 **30%p**의 성능 향상을 가져왔습니다.

### B. 모델 아키텍처별 특성
- **Custom CNN**이 ResNet이나 VGG보다 프루닝에 더 강건한 모습(Drop이 가장 작음)을 보였습니다. 이는 모델 구조가 단순하거나, 파라미터 대비 과적합(Overfitting) 경향이 적어 프루닝의 타격을 덜 받았을 수 있습니다.
- **ResNet-18 / VGG-16**은 Baseline 성능은 높지만, 50%를 날리는 순간 성능이 급격히 하락했습니다. 이는 CNN 모델들이 LLM에 비해 **파라미터 효율성(Compactness)**이 높아, 잉여 파라미터가 적기 때문으로 해석됩니다.

### C. 한계점 및 시사점
- **CNN에서의 One-Shot Pruning 한계**: LLM(GPT)과 달리, 소형 CNN 모델에서 50% 구조적 희소성을 **재학습(Fine-tuning) 없이** 달성하는 것은 매우 어렵다는 것이 확인되었습니다. (Drop -40% 이상)
- 그러나 **재학습이 불가능한 상황(One-Shot)**을 가정한다면, SparseGPT는 Magnitude 방식 대비 필수적인 선택지임이 증명되었습니다.

## 4. 결론
SparseGPT 알고리즘은 **Hessian 기반의 Adaptive Mask Selection과 OBS Update**를 통해, 전통적 방식보다 월등히 뛰어난 정보 보존 능력을 입증하였습니다. 특히 재학습 과정을 완전히 배제한 상황에서도 Magnitude Pruning 대비 **13~30%p** 더 높은 정확도를 달성했습니다.

