# CNN SparseGPT 프로젝트 종합 분석 보고서

**작성일**: 2025-12-17  
**프로젝트**: CNN SparseGPT (cnn_sparsegpt)

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [아키텍처 분석](#2-아키텍처-분석)
3. [핵심 알고리즘 상세](#3-핵심-알고리즘-상세)
4. [N:M Sparsity CNN 적용](#4-nm-sparsity-cnn-적용)
5. [실험 결과](#5-실험-결과)
6. [기술 Q&A](#6-기술-qa)
7. [결론 및 인사이트](#7-결론-및-인사이트)

---

## 1. 프로젝트 개요

### 1.1 목적
대규모 언어 모델(LLM)용 **SparseGPT** 프루닝 알고리즘을 **CNN**에 적용하고 검증하는 구현체.

### 1.2 핵심 특징

| 특징 | 설명 |
|------|------|
| **OBS 기반 프루닝** | Hessian 2차 미분 정보 활용 |
| **One-shot Pruning** | Fine-tuning 없이 한 번에 경량화 |
| **N:M Structured** | 2:4 등 NVIDIA 하드웨어 가속 지원 구조 |
| **Adaptive Mask** | 업데이트된 가중치 기준 동적 마스크 |

### 1.3 지원 모델

| 모델 | 데이터셋 | 파라미터 수 |
|------|---------|-----------|
| SimpleCNN | MNIST | ~6.5M |
| CIFARCNN | CIFAR-10 | ~9.5M |
| ResNet-18 CIFAR | CIFAR-10 | ~11.2M |
| VGG-16 CIFAR | CIFAR-10 | ~134M |

---

## 2. 아키텍처 분석

### 2.1 프로젝트 구조

```
cnn_sparsegpt/
├── pruning/                    ← 핵심 알고리즘
│   ├── sparsegpt.py           # OBS 프루닝, Hessian 계산
│   ├── pipeline.py            # 실행 파이프라인
│   └── mask.py                # 마스크 생성
├── models/                     ← CNN 모델 정의
│   ├── standard_cifar.py      # ResNet-18, VGG-16
│   ├── cifar_cnn.py           # CIFARCNN
│   └── mnist_cnn.py           # SimpleCNN
├── scripts/                    ← 실행 스크립트
│   ├── run_sweep.py           # 전체 모델 스윕 실험
│   ├── prune.py               # 개별 프루닝
│   └── train.py               # 모델 학습
├── configs/                    ← YAML 설정 파일
└── experiments/results/        ← 실험 결과 CSV
```

### 2.2 핵심 모듈 의존성

```
scripts/prune.py
    └── pruning/pipeline.py (run_pruning)
            ├── pruning/sparsegpt.py (prune_layer_obs)
            │       └── pruning/mask.py (flatten_weight, assign_weight)
            └── models/* (모델 클래스)
```

---

## 3. 핵심 알고리즘 상세

### 3.1 Hessian 계산

**파일**: `pruning/sparsegpt.py:12-39`

**수식**:
$$H = \frac{XX^\top}{N} + (\text{damp} + \lambda) I$$

**구현**:
```python
features = inputs.t().contiguous().numpy()  # [Channel, Samples]
hessian = features @ features.T / features.shape[1]

# Adaptive Dampening
mean_diag = np.mean(np.diag(hessian))
damp = 0.01 * mean_diag
hessian += (damp + lambd) * np.eye(features.shape[0])
```

**Dampening 역할**:
- 수치 안정성 확보
- Cholesky 분해 실패 방지
- Ridge 정규화 효과

### 3.2 Hessian 역행렬 (Cholesky 분해)

**수식**:
$$H = LL^\top \implies H^{-1} = (L^{-1})^\top L^{-1}$$

**Cholesky 사용 이유**:

| 항목 | `inv(H)` | Cholesky |
|------|----------|----------|
| 속도 | $O(n^3)$ | $O(2n^3/3)$ ✅ |
| 안정성 | 보통 | ✅ 더 안정적 |
| SPD 검증 | ❌ 불가 | ✅ 가능 |

### 3.3 OBS 프루닝 알고리즘

**에러 점수**:
$$\varepsilon_j = \frac{w_j^2}{[H^{-1}]_{jj}}$$

**보정 업데이트**:
$$W_{:,i+1:} -= \frac{\text{error}}{[H^{-1}]_{ii}} \cdot H^{-1}_{i,i+1:}$$

**구현**:
```python
for i in range(c_start, c_end):
    w = weight_matrix[:, i]
    d = hessian_inv[i, i]
    
    error = w * (1.0 - mask)
    weight_matrix[:, i] = w * mask
    
    if i + 1 < cols:
        correction = error / d
        h_inv_row = hessian_inv[i, i+1:]
        weight_matrix[:, i+1:] -= np.outer(correction, h_inv_row)
```

### 3.4 Adaptive Mask Selection

**핵심**: 블록 처리 중 **업데이트된 가중치**를 기준으로 마스크 재계산

```
블록 0 처리 후 → w4, w5, ... 업데이트됨
블록 1 처리 시 → 업데이트된 w4', w5'로 마스크 계산 ← Adaptive!
```

---

## 4. N:M Sparsity CNN 적용

### 4.1 Conv2d 가중치 변환

**문제**: Conv2d는 4D 텐서, SparseGPT는 2D 행렬 연산 전제

**해결**: 평탄화 (Flatten)

```python
# Conv2d(64, 128, 3, 3) → (64, 1152)
oc, ic, kh, kw = weight.shape  # 64, 128, 3, 3
weight_2d = weight.reshape(oc, ic * kh * kw)  # (64, 1152)
```

**의미**:
- 행 = 출력 채널 (필터 개수)
- 열 = 입력채널 × 커널크기 (필터의 모든 가중치)

### 4.2 변환 이유

| 이유 | 설명 |
|------|------|
| Hessian 계산 | $H = XX^\top$는 2D 행렬 연산 |
| OBS 수식 적용 | 2D 가중치 형태 가정 |
| N:M 블록 처리 | 열 방향으로 M개씩 그룹화 |

### 4.3 N:M 마스크 생성

**2:4 Sparsity 예시**:
```
가중치: [0.8, 0.1, 0.6, 0.2]  ← 4개 블록
중요도: [0.8, 0.1, 0.6, 0.2]
마스크: [1,   0,   1,   0  ]  ← 상위 2개만 보존
결과:   [0.8, 0,   0.6, 0  ]
```

### 4.4 NVIDIA 하드웨어 가속

**지원 조건**:
- GPU: Ampere, Ada Lovelace, Hopper 이상
- 패턴: 2:4 Structured Sparsity
- 연산: FP16, BF16, INT8, TF32

**현재 구현 상태**:
- ✅ 2:4 마스크 생성 지원
- ⚠️ NVIDIA 가속을 위한 변환 파이프라인 필요 (cuSPARSELt, TensorRT)

---

## 5. 실험 결과

### 5.1 전체 모델 스윕 실험 (2:4 Sparsity, 50%)

| 모델 | Method | Baseline | Pruned | **정확도 감소** |
|------|--------|----------|--------|----------------|
| **MNIST** | Magnitude | 98.38% | 97.33% | 1.05% |
| | **SparseGPT** | 98.38% | **98.20%** | **0.18%** ✅ |
| **CIFAR-10 CNN** | Magnitude | 78.15% | 45.48% | 32.67% |
| | **SparseGPT** | 78.15% | **64.64%** | **13.51%** ✅ |
| **ResNet-18** | Magnitude | 85.22% | 20.75% | 64.47% |
| | **SparseGPT** | 85.22% | **46.17%** | **39.05%** ✅ |
| **VGG-16** | Magnitude | 85.74% | 31.16% | 54.58% |
| | **SparseGPT** | 85.74% | **67.91%** | **17.83%** ✅ |

### 5.2 핵심 발견

1. **SparseGPT가 모든 모델에서 Magnitude 대비 우수**
2. **VGG-16**: SparseGPT가 36.75%p 더 높은 정확도
3. **MNIST**: 거의 무손실 프루닝 (0.18% 감소)
4. 복잡한 모델일수록 SparseGPT의 이점 증가

### 5.3 저장된 결과

- CSV: `experiments/results/sweep_results_20251213_013013.csv`
- 스크립트: `scripts/run_sweep.py`

---

## 6. 기술 Q&A

### Q1: Conv2d (64, 128, 3, 3) → (64, 1152) 변환 이유?

**답변**: 
1. SparseGPT는 $Y = WX$ 형태 전제
2. Conv2d도 Im2Col 관점에서 행렬 곱으로 해석 가능
3. 각 필터를 1개 행으로, 필터 내 모든 가중치를 열로 펼침

### Q2: Cholesky 분해로 역행렬을 계산하는 이유?

**답변**:
1. **수치 안정성**: 삼각 행렬 역행렬이 더 안정적
2. **계산 효율**: 약 1.5배 빠름 ($O(2n^3/3)$ vs $O(n^3)$)
3. **SPD 검증**: Cholesky 실패 시 양정치 아님을 즉시 감지

### Q3: NVIDIA N:M 가속을 사용할 수 있는가?

**답변**:
- ✅ 2:4 마스크 생성 가능
- ⚠️ 실제 가속을 위해서는 cuSPARSELt/TensorRT 변환 필요

---

## 7. 결론 및 인사이트

### 7.1 주요 성과

1. **SparseGPT for CNN 성공적 구현**: LLM용 알고리즘을 CNN에 적응
2. **모든 모델에서 우수한 성능**: Magnitude 대비 일관된 우위
3. **체계적 문서화**: 수식-코드 대응 상세 분석

### 7.2 기술적 핵심 포인트

| 항목 | 핵심 |
|------|------|
| **Hessian** | $H = XX^\top/N$ + Adaptive Dampening |
| **역행렬** | Cholesky 분해로 안정성/효율성 확보 |
| **N:M 적용** | 4D→2D 변환 후 열 방향 블록 처리 |
| **Adaptive** | 업데이트된 가중치로 마스크 재계산 |

### 7.3 향후 개선 방향

1. **NVIDIA 가속 통합**: cuSPARSELt/TensorRT 파이프라인
2. **Fine-tuning 추가**: 프루닝 후 정확도 회복
3. **BN 파라미터 고려**: 더 정교한 Hessian 계산
4. **다양한 N:M 비율**: 1:4, 4:8 등 실험

---

## 참고 자료

### 프로젝트 문서
- [algorithm_detailed_analysis.md](file:///home/hoseo/on_the_air/main_cnn_sparsegpt/docs/algorithm_detailed_analysis.md)
- [implementation_details.md](file:///home/hoseo/on_the_air/main_cnn_sparsegpt/docs/implementation_details.md)
- [algorithm_analysis.md](file:///home/hoseo/on_the_air/main_cnn_sparsegpt/docs/algorithm_analysis.md)

### 논문
1. Frantar, E., & Alistarh, D. (2023). *SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot*
2. Hassibi, B., & Stork, D. G. (1992). *Second order derivatives for network pruning: Optimal Brain Surgeon*

---

*본 문서는 CNN SparseGPT 프로젝트 분석 대화의 종합 정리입니다.*
