# CNN SparseGPT 프로젝트 알고리즘 분석 문서

이 문서는 `cnn_sparsegpt` 프로젝트의 핵심 알고리즘 코드를 상세히 분석하고 설명합니다.

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [알고리즘 배경 이론](#2-알고리즘-배경-이론)
3. [핵심 모듈 분석](#3-핵심-모듈-분석)
   - 3.1 [mask.py - 마스크 생성](#31-maskpy---마스크-생성)
   - 3.2 [sparsegpt.py - 핵심 알고리즘](#32-sparsegptpy---핵심-알고리즘)
   - 3.3 [pipeline.py - 실행 파이프라인](#33-pipelinepy---실행-파이프라인)
4. [수식-코드 대응표](#4-수식-코드-대응표)
5. [알고리즘 흐름 다이어그램](#5-알고리즘-흐름-다이어그램)

---

## 1. 프로젝트 개요

### 1.1 목적

이 프로젝트는 대규모 언어 모델(LLM)에 적용되던 **SparseGPT** 프루닝 알고리즘을 **CNN(Convolutional Neural Networks)**에 적용하고 검증하는 구현체입니다.

### 1.2 핵심 특징

- **OBS(Optimal Brain Surgeon) 기반 프루닝**: 2차 미분 정보(Hessian)를 활용한 정교한 가지치기
- **One-shot Pruning**: Fine-tuning 없이 한 번의 패스로 모델 경량화
- **N:M Structured Pruning 지원**: 하드웨어 가속에 유리한 구조적 희소성 (예: 2:4)
- **Adaptive Mask Selection**: 업데이트된 가중치를 기준으로 마스크를 동적으로 재계산

---

## 2. 알고리즘 배경 이론

### 2.1 문제 정의

레이어 단위로 다음과 같은 **재구성(Reconstruction) 문제**를 푸는 것이 목표입니다:

$$
\min_{\tilde{W}} \| WX - \tilde{W}X \|_2^2 \quad \text{s.t. } \tilde{W} \text{ is sparse}
$$

여기서:
- $W$: 원본 가중치 행렬
- $\tilde{W}$: 프루닝 후 가중치 행렬
- $X$: 레이어 입력 (캘리브레이션 데이터로부터 수집)

### 2.2 OBS (Optimal Brain Surgeon) 프레임워크

손실 함수의 2차 Taylor 전개를 사용합니다:

$$
L(\tilde{W}) \approx L(W) + \frac{1}{2} (\tilde{W} - W)^\top H (\tilde{W} - W)
$$

여기서 **Hessian 행렬 H**는 입력 공분산 행렬로 근사됩니다:

$$
H \approx XX^\top
$$

### 2.3 단일 가중치 제거 시 에러

가중치 $w_q$를 0으로 만들 때 발생하는 손실 증가량:

$$
\varepsilon_q = \frac{w_q^2}{[H^{-1}]_{qq}}
$$

- $\varepsilon_q$가 **작을수록** 해당 가중치를 제거해도 손실 증가가 적음
- 이 점수를 기준으로 어떤 가중치를 제거할지 결정

### 2.4 OBS 보정 업데이트

가중치 $w_p$를 제거할 때, 나머지 가중치들을 다음과 같이 보정합니다:

$$
\delta W = -\frac{w_p}{[H^{-1}]_{pp}} \cdot H^{-1}_{:,p}
$$

이 보정을 통해 프루닝으로 인한 출력 오차를 최소화합니다.

---

## 3. 핵심 모듈 분석

### 3.1 mask.py - 마스크 생성

이 모듈은 프루닝 마스크를 생성하는 유틸리티 함수들을 제공합니다.

#### 3.1.1 N:M Structured Pruning 마스크 (`nm_mask_blockwise`)

```python
def nm_mask_blockwise(
    weight: np.ndarray, n: int = 2, m: int = 4, scores: np.ndarray | None = None
) -> np.ndarray:
```

**기능**: M개 연속 컬럼 블록 내에서 중요도가 높은 N개만 보존

**알고리즘 흐름**:
```
1. scores가 주어지지 않으면 |weight|를 중요도로 사용
2. 각 행(row)에 대해:
   3. M개 컬럼씩 블록으로 나눔
   4. 각 블록 내에서 중요도 상위 N개 선택
   5. 선택된 위치의 마스크를 1로 설정
```

**예시** (2:4 Sparsity):
```
입력 블록:  [0.1, 0.8, 0.3, 0.6]
마스크:     [0,   1,   0,   1  ]  ← 상위 2개(0.8, 0.6) 보존
```

#### 3.1.2 Unstructured Pruning 마스크 (`elementwise_topk_mask`)

```python
def elementwise_topk_mask(
    weight: np.ndarray, keep_ratio: float, scores: np.ndarray | None = None
) -> np.ndarray:
```

**기능**: 각 행에서 상위 K%의 가중치만 보존 (구조적 제약 없음)

**알고리즘 흐름**:
```
1. keep_cols = cols × keep_ratio (보존할 개수)
2. 각 행에서 중요도 상위 keep_cols개 선택
3. 선택된 위치의 마스크를 1로 설정
```

#### 3.1.3 가중치 평탄화 및 복원

```python
def flatten_weight(layer: nn.Module) -> tuple[np.ndarray, tuple[int, ...]]:
def assign_weight(layer: nn.Module, weight_matrix: np.ndarray, original_shape: tuple) -> None:
```

| 레이어 타입 | 원본 shape | 평탄화 후 shape |
|------------|-----------|----------------|
| `nn.Linear` | `(out, in)` | `(out, in)` (변화 없음) |
| `nn.Conv2d` | `(OC, IC, KH, KW)` | `(OC, IC×KH×KW)` |

**Conv2d 평탄화 의미**: 각 출력 채널(필터)을 하나의 행으로, 입력 채널×커널 크기를 열로 펼침

---

### 3.2 sparsegpt.py - 핵심 알고리즘

#### 3.2.1 Hessian 계산 (`compute_hessian`)

```python
def compute_hessian(inputs: torch.Tensor, lambd: float = 1e-4) -> np.ndarray:
```

**수식 구현**:

$$
H = \frac{XX^\top}{N} + (\text{damp} + \lambda) \cdot I
$$

**코드 분석**:

```python
# 1. 입력 전치 및 NumPy 변환
features = inputs.t().contiguous().numpy()  # [Channel, Samples]

# 2. 정규화된 공분산 행렬 계산
hessian = features @ features.T / features.shape[1]  # H = X X^T / N

# 3. Adaptive Dampening
mean_diag = np.mean(np.diag(hessian))
damp = 0.01 * mean_diag

# 4. 수치 안정성을 위한 정규화
hessian += (damp + lambd) * np.eye(features.shape[0], dtype=np.float32)
```

**Dampening의 역할**:
- Hessian이 특이(Singular)하거나 조건 수(Condition Number)가 클 때 역행렬 계산 안정화
- Ridge Regression의 정규화 항과 유사한 효과

#### 3.2.2 Hessian 역행렬 계산 (`invert_hessian`)

```python
def invert_hessian(hessian: np.ndarray) -> np.ndarray:
```

**알고리즘**:
1. **Cholesky 분해** 시도: $H = LL^\top$
2. **역행렬 계산**: $H^{-1} = (L^{-1})^\top L^{-1}$
3. 실패 시 추가 Dampening 후 직접 역행렬 계산

**Cholesky 분해를 사용하는 이유**:
- SPD(Symmetric Positive Definite) 행렬에 대해 가장 빠르고 안정적
- 수치 오류 최소화

#### 3.2.3 OBS 기반 프루닝 (`prune_layer_obs`)

```python
def prune_layer_obs(
    layer: nn.Module,
    activations: torch.Tensor,
    n: int = 2,
    m: int = 4,
    lambd: float = 1e-4,
    enforce_nm: bool = True,
) -> Tuple[torch.Tensor, np.ndarray]:
```

**SparseGPT Algorithm 1의 구현**

**Phase 1: 초기화**
```python
# 가중치 평탄화
weight_matrix, original_shape = flatten_weight(layer)
rows, cols = weight_matrix.shape

# Hessian 및 역행렬 계산
hessian = compute_hessian(activations, lambd=lambd)
hessian_inv = invert_hessian(hessian)

# 마스크 저장 공간
final_mask = np.zeros_like(weight_matrix, dtype=np.float32)
block_size = m if enforce_nm else 1
```

**Phase 2: Column-wise 순회 (Fast Reconstruction)**
```python
for c_start in range(0, cols, block_size):
    c_end = min(c_start + block_size, cols)
    
    # 현재 블록 가중치 (이미 이전 루프에서 업데이트됨)
    w_block = weight_matrix[:, c_start:c_end].copy()
```

**Phase 3: Adaptive Mask Selection**
```python
if enforce_nm:
    # 블록에 해당하는 H^-1 대각 성분
    h_inv_block = hessian_inv[c_start:c_end, c_start:c_end]
    diag = np.diagonal(h_inv_block)
    
    # OBS 에러 점수: ε_j = w_j² / [H^-1]_jj
    scores = (w_block ** 2) / diag.reshape(1, -1)
    
    # 각 행에서 점수가 높은 N개 선택
    for r in range(rows):
        idx = np.argsort(scores[r])[-n:]
        mask_block[r, idx] = 1.0
```

**핵심 개념 - Adaptive Mask Selection**:
- 논문 Section 2.3에서 언급된 핵심 개선사항
- 마스크 M을 **업데이트된 가중치**를 기준으로 재계산
- 이전 블록에서의 가중치 보정이 반영된 상태에서 마스크 결정

**Phase 4: OBS 업데이트 (Error Compensation)**
```python
for i in range(c_start, c_end):
    w = weight_matrix[:, i]        # 현재 가중치 컬럼
    d = hessian_inv[i, i]          # [H^-1]_ii
    
    # 마스크 적용 및 에러 계산
    m_val = mask_block[:, i - c_start]
    error = w * (1.0 - m_val)      # 제거되는 가중치 값
    
    # 현재 가중치 프루닝
    weight_matrix[:, i] = w * m_val
    
    # 미래 컬럼들에 대한 보정 (핵심!)
    if i + 1 < cols:
        correction = error / d                      # δw = error / [H^-1]_ii
        h_inv_row = hessian_inv[i, i+1:]           # H^-1의 i번째 행
        weight_matrix[:, i+1:] -= np.outer(correction, h_inv_row)
```

**수식과 코드 대응**:

| 수식 | 코드 |
|------|------|
| $\text{error} = w_i \cdot (1 - m_i)$ | `error = w * (1.0 - m_val)` |
| $\delta w = \frac{\text{error}}{[H^{-1}]_{ii}}$ | `correction = error / d` |
| $W_{:,i+1:} \leftarrow W_{:,i+1:} - \delta w \cdot H^{-1}_{i,i+1:}$ | `weight_matrix[:, i+1:] -= np.outer(correction, h_inv_row)` |

#### 3.2.4 Magnitude Pruning (Baseline)

```python
def prune_layer_magnitude(
    layer: nn.Module, n: int = 2, m: int = 4, enforce_nm: bool = True
) -> Tuple[torch.Tensor, np.ndarray]:
```

**기능**: Hessian 없이 단순히 가중치 절댓값으로 마스크 생성 (비교 실험용)

---

### 3.3 pipeline.py - 실행 파이프라인

#### 3.3.1 설정 구조 (`PruningConfig`)

```python
@dataclass
class PruningConfig:
    weights: str           # 체크포인트 경로
    data_dir: str          # 데이터셋 경로
    batch_size: int = 128
    calib_batches: int = 8 # Hessian 계산용 배치 수
    n: int = 2             # N:M에서 N
    m: int = 4             # N:M에서 M
    lambd: float = 1e-4    # Dampening 상수
    mode: str = "sparsegpt"  # 'sparsegpt' 또는 'magnitude'
    enforce_nm: bool = True  # True: 구조적, False: 비구조적
```

#### 3.3.2 프루닝 가능 레이어 추출

```python
def get_prunable_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    prunable: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prunable.append((name, module))
    return prunable
```

**순서 중요성**: SparseGPT는 레이어별로 순차 처리하므로 순서가 보장되어야 함

#### 3.3.3 전체 프루닝 흐름 (`run_pruning`)

```
[1] 모델 로드 및 Baseline 성능 측정
                ↓
[2] 레이어별 순차 처리 (Sequential Layer-wise)
    ┌─────────────────────────────────────────┐
    │ for each layer in get_prunable_layers(): │
    │   [2.1] 캘리브레이션 데이터로 입력 수집   │
    │   [2.2] Hessian 계산 (H = X X^T)         │
    │   [2.3] OBS 알고리즘으로 프루닝 + 보정   │
    └─────────────────────────────────────────┘
                ↓
[3] 최종 성능 측정 (Fine-tuning 없이)
                ↓
[4] 프루닝된 모델 저장
```

**핵심 코드**:
```python
for name, layer in get_prunable_layers(model):
    # 캘리브레이션 입력 수집
    activations = collect_calibration_inputs(
        model, train_loader, layer, device,
        max_batches=config.calib_batches,
        max_samples=config.calib_samples,
    )
    
    # SparseGPT 프루닝 수행
    prune_layer_obs(
        layer, activations,
        n=config.n, m=config.m,
        lambd=config.lambd,
        enforce_nm=config.enforce_nm,
    )
```

---

## 4. 수식-코드 대응표

### 4.1 Hessian 계산

| 단계 | 수식 | 코드 위치 |
|------|------|----------|
| 입력 변환 | $X \in \mathbb{R}^{d \times n}$ | `features = inputs.t().contiguous().numpy()` |
| 공분산 | $H = \frac{XX^\top}{N}$ | `hessian = features @ features.T / features.shape[1]` |
| Dampening | $H \leftarrow H + (\text{damp} + \lambda)I$ | `hessian += (damp + lambd) * np.eye(...)` |

### 4.2 OBS 에러 메트릭

| 수식 | 의미 | 코드 |
|------|------|------|
| $\varepsilon_j = \frac{w_j^2}{[H^{-1}]_{jj}}$ | 가중치 j 제거 시 에러 | `scores = (w_block ** 2) / diag.reshape(1, -1)` |

### 4.3 OBS 보정 업데이트

| 수식 | 의미 | 코드 |
|------|------|------|
| $\delta w = -\frac{w_p}{[H^{-1}]_{pp}}$ | 보정 계수 | `correction = error / d` |
| $W_{:,j+1:} \leftarrow W_{:,j+1:} - \delta w \cdot H^{-1}_{p,j+1:}$ | 미래 컬럼 보정 | `weight_matrix[:, i+1:] -= np.outer(correction, h_inv_row)` |

---

## 5. 알고리즘 흐름 다이어그램

### 5.1 레이어 단위 프루닝 상세 흐름

```
┌──────────────────────────────────────────────────────────────────┐
│                    레이어 L에 대한 SparseGPT                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [입력] 캘리브레이션 데이터 X, 가중치 W, 희소 비율 (N:M)            │
│                              │                                   │
│                              ▼                                   │
│         ┌─────────────────────────────────────┐                  │
│         │ 1. Hessian 계산: H = X X^T / N      │                  │
│         │    + Adaptive Dampening             │                  │
│         └────────────────┬────────────────────┘                  │
│                          ▼                                       │
│         ┌─────────────────────────────────────┐                  │
│         │ 2. Hessian 역행렬: H^-1             │                  │
│         │    (Cholesky 분해)                  │                  │
│         └────────────────┬────────────────────┘                  │
│                          ▼                                       │
│   ┌──────────────────────────────────────────────────────┐       │
│   │ 3. 블록별 반복 (block_size = M)                       │       │
│   │                                                      │       │
│   │  ┌────────────────────────────────────────────┐      │       │
│   │  │ 3.1 Adaptive Mask Selection                │      │       │
│   │  │     • 현재 블록 가중치로 에러 점수 계산      │      │       │
│   │  │     • ε_j = w_j² / [H^-1]_jj               │      │       │
│   │  │     • 상위 N개 선택 → 마스크                │      │       │
│   │  └────────────────────┬───────────────────────┘      │       │
│   │                       ▼                              │       │
│   │  ┌────────────────────────────────────────────┐      │       │
│   │  │ 3.2 컬럼별 OBS 업데이트                     │      │       │
│   │  │     • 마스크 적용 (0으로 설정)              │      │       │
│   │  │     • 미래 컬럼 보정                        │      │       │
│   │  │       W[:, i+1:] -= outer(δw, H^-1[i, i+1:])│      │       │
│   │  └────────────────────────────────────────────┘      │       │
│   └──────────────────────────────────────────────────────┘       │
│                              │                                   │
│                              ▼                                   │
│  [출력] 프루닝된 가중치 W', 마스크 M                              │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 Adaptive Mask Selection의 핵심

```
블록 0 처리:
  W = [w0, w1, w2, w3, w4, w5, ...]
       └─블록 0─┘

  1. 원본 가중치 w0~w3로 마스크 계산
  2. 마스크 적용 및 OBS 보정
  3. w4, w5, ... 가 업데이트됨 ← 중요!

블록 1 처리:
  W' = [w0', w1', w2', w3', w4', w5', ...]  ← 이미 보정된 상태
                           └─블록 1─┘

  1. **업데이트된** w4'~w7'로 마스크 계산  ← Adaptive!
  2. 마스크 적용 및 OBS 보정
  ...
```

**Adaptive의 의미**: 이전 블록에서 수행한 보정이 반영된 가중치를 기준으로 마스크를 동적으로 결정

---

## 참고 문헌

1. Frantar, E., & Alistarh, D. (2023). SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot.
2. Hassibi, B., & Stork, D. G. (1992). Second order derivatives for network pruning: Optimal Brain Surgeon.
3. NVIDIA. (2021). Achieving FP32 Accuracy for INT8 Inference Using Quantization Aware Training with TensorRT.

---

*이 문서는 cnn_sparsegpt 프로젝트의 알고리즘 이해를 위해 작성되었습니다.*

