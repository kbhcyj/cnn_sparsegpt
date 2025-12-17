# CNN SparseGPT 알고리즘 상세 분석 문서

본 문서는 CNN SparseGPT 프로젝트의 핵심 알고리즘을 **수식-코드 레벨**에서 상세히 분석합니다.

---

## 목차

1. [알고리즘 개요](#1-알고리즘-개요)
2. [Hessian 계산 상세 분석](#2-hessian-계산-상세-분석)
3. [Hessian 역행렬 계산](#3-hessian-역행렬-계산)
4. [OBS 프루닝 알고리즘](#4-obs-프루닝-알고리즘)
5. [Adaptive Mask Selection](#5-adaptive-mask-selection)
6. [마스크 생성 알고리즘](#6-마스크-생성-알고리즘)
7. [가중치 변환 및 할당](#7-가중치-변환-및-할당)
8. [전체 파이프라인](#8-전체-파이프라인)
9. [수식-코드 대응표](#9-수식-코드-대응표)

---

## 1. 알고리즘 개요

### 1.1 SparseGPT의 핵심 아이디어

SparseGPT는 **OBS(Optimal Brain Surgeon)** 프레임워크를 기반으로 한 One-shot 프루닝 알고리즘입니다.

**최적화 목표**:
$$
\min_{\tilde{W}} \| WX - \tilde{W}X \|_2^2 \quad \text{s.t. } \tilde{W} \text{ is sparse}
$$

**핵심 특징**:
- 레이어별 로컬 최적화 (Layer-wise Reconstruction)
- Fine-tuning 없이 가중치 보정
- Hessian 기반 중요도 메트릭

### 1.2 알고리즘 흐름도

```
┌─────────────────────────────────────────────────────────────┐
│                    SparseGPT 프루닝 흐름                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [1] 가중치 평탄화: Conv2d (OC,IC,K,K) → (OC, IC×K×K)         │
│                              │                              │
│                              ▼                              │
│  [2] 캘리브레이션 입력 수집: X ∈ ℝ^{d×n}                      │
│                              │                              │
│                              ▼                              │
│  [3] Hessian 계산: H = XX^T/N + (damp+λ)I                   │
│                              │                              │
│                              ▼                              │
│  [4] 역행렬 계산: H^{-1} (Cholesky 분해)                      │
│                              │                              │
│                              ▼                              │
│  [5] 블록별 순회 (block_size = M)                            │
│      ┌─────────────────────────────────────┐                │
│      │ [5.1] Adaptive Mask Selection       │                │
│      │       scores = w²/[H^{-1}]_{ii}     │                │
│      │       상위 N개 선택 → mask          │                │
│      ├─────────────────────────────────────┤                │
│      │ [5.2] OBS 보정 업데이트             │                │
│      │       error = w × (1-mask)          │                │
│      │       W[:,i+1:] -= outer(err/d, H^{-1}_{i,i+1:})    │
│      └─────────────────────────────────────┘                │
│                              │                              │
│                              ▼                              │
│  [6] 가중치 할당: (OC, IC×K×K) → Conv2d                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Hessian 계산 상세 분석

### 2.1 이론적 배경

손실 함수의 2차 Taylor 전개:
$$
L(\tilde{W}) \approx L(W) + \frac{1}{2} (\tilde{W} - W)^\top H (\tilde{W} - W)
$$

**Hessian 근사**:
$$
H \approx XX^\top
$$

여기서 $X \in \mathbb{R}^{d \times n}$은 레이어 입력, $d$는 입력 차원, $n$은 샘플 수입니다.

### 2.2 구현 분석 (`compute_hessian`)

**파일**: `pruning/sparsegpt.py:12-39`

```python
def compute_hessian(inputs: torch.Tensor, lambd: float = 1e-4) -> np.ndarray:
```

**Step 1: 입력 변환**
```python
features = inputs.t().contiguous().numpy()  # [Samples, Channel] → [Channel, Samples]
```
- 입력 텐서를 전치하여 $X \in \mathbb{R}^{d \times n}$ 형태로 변환
- $d$: 채널 수 (입력 차원)
- $n$: 샘플 수

**Step 2: 공분산 행렬 계산**
```python
hessian = features @ features.T / features.shape[1]
```

**수식**:
$$
H = \frac{XX^\top}{N}
$$

- 샘플 수 $N$으로 나누어 정규화
- 결과: $H \in \mathbb{R}^{d \times d}$ (대칭 행렬)

**Step 3: Adaptive Dampening**
```python
mean_diag = np.mean(np.diag(hessian))
damp = 0.01 * mean_diag
hessian += (damp + lambd) * np.eye(features.shape[0], dtype=np.float32)
```

**수식**:
$$
H_{new} = H + (\text{damp} + \lambda) \cdot I
$$

여기서:
- $\text{damp} = 0.01 \times \text{mean}(\text{diag}(H))$ — Adaptive 상수
- $\lambda = 10^{-4}$ — Ridge 정규화 상수

**Dampening 역할**:
1. 수치 안정성: Hessian이 특이(singular)하거나 조건 수(condition number)가 클 때 역행렬 계산 안정화
2. Ridge 정규화: 과적합 방지 효과
3. Cholesky 분해 실패 방지: 양정치 정칙(SPD) 조건 만족 보장

### 2.3 시간/공간 복잡도

| 연산 | 시간 복잡도 | 공간 복잡도 |
|------|------------|------------|
| `features @ features.T` | $O(d^2 n)$ | $O(d^2)$ |
| Dampening 추가 | $O(d)$ | $O(d^2)$ |
| **총합** | $O(d^2 n)$ | $O(d^2)$ |

---

## 3. Hessian 역행렬 계산

### 3.1 이론적 배경

OBS 수식에서 $H^{-1}$이 필요:
$$
\varepsilon_q = \frac{w_q^2}{[H^{-1}]_{qq}}, \quad \delta W = -\frac{w_p}{[H^{-1}]_{pp}} \cdot H^{-1}_{:,p}
$$

### 3.2 구현 분석 (`invert_hessian`)

**파일**: `pruning/sparsegpt.py:42-60`

```python
def invert_hessian(hessian: np.ndarray) -> np.ndarray:
```

**Primary Path: Cholesky 분해**
```python
try:
    L = np.linalg.cholesky(hessian)  # H = L L^T
    H_inv = np.linalg.inv(L.T) @ np.linalg.inv(L)
```

**수식**:
$$
H = LL^\top \implies H^{-1} = (L^{-1})^\top L^{-1}
$$

**Cholesky 분해 장점**:
1. SPD 행렬에 대해 가장 빠르고 안정적
2. 수치 오류 최소화
3. 시간 복잡도: $O(d^3/3)$

**Fallback Path: 직접 역행렬**
```python
except np.linalg.LinAlgError:
    hessian += 1e-2 * np.mean(np.diag(hessian)) * np.eye(hessian.shape[0])
    H_inv = np.linalg.inv(hessian)
```

- Cholesky 실패 시 추가 Dampening (1% → 더 강한 정규화)
- `np.linalg.inv` 사용 (LU 분해 기반)

### 3.3 복잡도 분석

| 연산 | 시간 복잡도 |
|------|------------|
| Cholesky 분해 | $O(d^3/3)$ |
| 삼각 행렬 역행렬 (×2) | $O(d^3/3) \times 2$ |
| 행렬 곱셈 | $O(d^3)$ |
| **총합** | $O(d^3)$ |

---

## 4. OBS 프루닝 알고리즘

### 4.1 알고리즘 개요

**파일**: `pruning/sparsegpt.py:63-166`

SparseGPT의 핵심인 **Fast Approximate Reconstruction** 구현.

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

### 4.2 Phase 1: 초기화

```python
weight_matrix, original_shape = flatten_weight(layer)
rows, cols = weight_matrix.shape

hessian = compute_hessian(activations, lambd=lambd)
hessian_inv = invert_hessian(hessian)

final_mask = np.zeros_like(weight_matrix, dtype=np.float32)
block_size = m if enforce_nm else 1
```

**설명**:
1. 가중치를 2D 행렬로 평탄화
2. Hessian 및 역행렬 계산
3. N:M 구조적 프루닝 시 M 단위 블록 처리

### 4.3 Phase 2: Unstructured 프루닝 (선택적)

```python
if not enforce_nm:
    diag = np.diag(hessian_inv)
    scores = (weight_matrix ** 2) / diag.reshape(1, -1)
    final_mask = build_pruning_mask(weight_matrix, n=n, m=m, enforce_nm=False, scores=scores)
```

**OBS 에러 메트릭**:
$$
\varepsilon_j = \frac{w_j^2}{[H^{-1}]_{jj}}
$$

- 점수가 **낮을수록** 제거해도 손실 증가가 적음
- 비구조적 프루닝 시 전역 마스크 사전 계산

### 4.4 Phase 3: Column-wise Loop (핵심)

```python
for c_start in range(0, cols, block_size):
    c_end = min(c_start + block_size, cols)
    w_block = weight_matrix[:, c_start:c_end].copy()
```

**블록 단위 처리**:
- N:M 구조적 프루닝: `block_size = M`
- 비구조적 프루닝: `block_size = 1`

### 4.5 Phase 4: Fast Reconstruction Loop

```python
for i in range(c_start, c_end):
    w = weight_matrix[:, i]       # 현재 가중치 컬럼
    d = hessian_inv[i, i]         # [H^-1]_ii
    
    m_val = mask_block[:, i - c_start]
    error = w * (1.0 - m_val)     # 제거되는 가중치 값
    
    weight_matrix[:, i] = w * m_val  # 프루닝 적용
    
    if i + 1 < cols:
        correction = error / d
        h_inv_row = hessian_inv[i, i+1:]
        weight_matrix[:, i+1:] -= np.outer(correction, h_inv_row)
```

**수식 대응**:

| 단계 | 수식 | 코드 |
|------|------|------|
| 에러 계산 | $\text{error} = w_i \cdot (1-m_i)$ | `error = w * (1.0 - m_val)` |
| 보정 계수 | $\delta = \frac{\text{error}}{[H^{-1}]_{ii}}$ | `correction = error / d` |
| 미래 컬럼 업데이트 | $W_{:,j+1:} -= \delta \cdot H^{-1}_{i,j+1:}$ | `weight_matrix[:, i+1:] -= np.outer(...)` |

**`np.outer` 사용 이유**:
- `correction`: [rows] 벡터
- `h_inv_row`: [future_cols] 벡터
- `np.outer`: [rows × future_cols] 행렬 생성
- 한 번의 벡터화된 연산으로 모든 행의 미래 컬럼 업데이트

### 4.6 시간 복잡도

| 단계 | 복잡도 |
|------|--------|
| Column-wise loop | $O(\text{cols})$ |
| Outer product per column | $O(\text{rows} \times \text{remaining\_cols})$ |
| **총합** | $O(\text{rows} \times \text{cols}^2)$ |

---

## 5. Adaptive Mask Selection

### 5.1 핵심 개념

**기존 방식 (Static)**:
- 프루닝 전 원본 가중치로 한 번에 마스크 결정

**Adaptive 방식**:
- 블록 처리 중 **업데이트된 가중치**를 기준으로 마스크 재계산
- 이전 블록의 OBS 보정이 반영된 상태에서 결정

### 5.2 구현 분석

```python
if enforce_nm:
    h_inv_block = hessian_inv[c_start:c_end, c_start:c_end]
    diag = np.diagonal(h_inv_block)
    
    # OBS 에러 점수: ε_j = w_j² / [H^-1]_jj
    scores = (w_block ** 2) / diag.reshape(1, -1)
    
    mask_block = np.zeros_like(w_block)
    for r in range(rows):
        if c_end - c_start < m:
            mask_block[r, :] = 1.0  # 불완전 블록은 모두 보존
        else:
            idx = np.argsort(scores[r])[-n:]  # 상위 N개 선택
            mask_block[r, idx] = 1.0
```

**핵심 포인트**:
1. `w_block`은 이전 루프에서 이미 업데이트된 상태
2. 블록별 Hessian 역행렬 대각 성분 추출
3. OBS 점수 계산 후 행별 Top-N 선택

### 5.3 Adaptive Selection 시각화

```
블록 0 처리:
  W = [w0, w1, w2, w3 | w4, w5, w6, w7 | ...]
       └── 블록 0 ──┘
  
  1. 원본 w0~w3로 마스크 계산
  2. 마스크 적용 및 OBS 보정
  3. w4, w5, ... 업데이트됨 ← 중요!

블록 1 처리:
  W' = [w0', w1', w2', w3' | w4', w5', w6', w7' | ...]
                            └── 보정된 블록 1 ──┘
  
  1. **업데이트된** w4'~w7'로 마스크 계산 ← Adaptive!
  2. 마스크 적용 및 OBS 보정
  ...
```

---

## 6. 마스크 생성 알고리즘

### 6.1 N:M Structured Pruning (`nm_mask_blockwise`)

**파일**: `pruning/mask.py:8-33`

```python
def nm_mask_blockwise(
    weight: np.ndarray, n: int = 2, m: int = 4, scores: np.ndarray | None = None
) -> np.ndarray:
```

**알고리즘**:
```python
for r in range(rows):
    for c in range(0, cols, m):
        block_imp = importance[r, c : c + m]
        if block_imp.size < m:
            mask[r, c : c + m] = 1.0  # 불완전 블록 보존
            continue
        keep_idx = np.argpartition(block_imp, -n)[-n:]  # 상위 N개
        mask[r, c : c + m][keep_idx] = 1.0
```

**`np.argpartition` 사용 이유**:
- 전체 정렬($O(m \log m)$) 대신 부분 분할($O(m)$)
- Top-N 인덱스만 필요하므로 효율적

**예시 (2:4 Sparsity)**:
```
입력:  [0.1, 0.8, 0.3, 0.6]
중요도: [0.1, 0.8, 0.3, 0.6]
마스크: [0,   1,   0,   1  ] ← 0.8, 0.6 보존
```

### 6.2 Unstructured Pruning (`elementwise_topk_mask`)

**파일**: `pruning/mask.py:36-57`

```python
def elementwise_topk_mask(
    weight: np.ndarray, keep_ratio: float, scores: np.ndarray | None = None
) -> np.ndarray:
```

**알고리즘**:
```python
keep_cols = max(1, int(round(cols * keep_ratio)))  # 보존할 개수

for r in range(rows):
    idx = np.argpartition(importance[r], -keep_cols)[-keep_cols:]
    mask[r, idx] = 1.0
```

**특징**:
- 행별(Row-wise) 독립적으로 Top-K% 선택
- N:M 구조적 제약 없음
- 더 높은 유연성, 더 높은 압축률 가능

### 6.3 마스크 생성 헬퍼 (`build_pruning_mask`)

**파일**: `pruning/mask.py:60-74`

```python
def build_pruning_mask(
    weight: np.ndarray, n: int, m: int, enforce_nm: bool, scores: np.ndarray | None = None
) -> np.ndarray:
    if enforce_nm:
        return nm_mask_blockwise(weight, n=n, m=m, scores=scores)
    
    keep_ratio = n / m if m > 0 else 0.0
    return elementwise_topk_mask(weight, keep_ratio=keep_ratio, scores=scores)
```

**조건 분기**:
- `enforce_nm=True`: N:M 구조적 마스크
- `enforce_nm=False`: 비구조적 Top-K 마스크 (비율 = n/m)

---

## 7. 가중치 변환 및 할당

### 7.1 가중치 평탄화 (`flatten_weight`)

**파일**: `pruning/mask.py:77-84`

```python
def flatten_weight(layer: nn.Module) -> tuple[np.ndarray, tuple[int, ...]]:
    weight = layer.weight.detach().cpu().numpy()
    if isinstance(layer, nn.Linear):
        return weight.copy(), weight.shape  # (out, in) 유지
    if isinstance(layer, nn.Conv2d):
        oc, ic, kh, kw = weight.shape
        return weight.reshape(oc, ic * kh * kw).copy(), (oc, ic, kh, kw)
```

**변환 규칙**:

| 레이어 타입 | 원본 Shape | 평탄화 Shape |
|------------|-----------|-------------|
| `nn.Linear` | `(out, in)` | `(out, in)` |
| `nn.Conv2d` | `(OC, IC, KH, KW)` | `(OC, IC×KH×KW)` |

**Conv2d 평탄화 의미**:
- 각 출력 채널(필터) = 하나의 행
- 입력 채널×커널 크기 = 열
- $WX$ 연산으로 해석 가능

### 7.2 가중치 할당 (`assign_weight`)

**파일**: `pruning/mask.py:87-95`

```python
def assign_weight(
    layer: nn.Module, weight_matrix: np.ndarray, original_shape: tuple[int, ...]
) -> None:
    tensor = torch.from_numpy(weight_matrix).to(layer.weight.device, dtype=layer.weight.dtype)
    if isinstance(layer, nn.Conv2d):
        tensor = tensor.view(*original_shape)  # 원래 4D로 복원
    else:
        tensor = tensor.view_as(layer.weight)
    layer.weight.data.copy_(tensor)
```

**과정**:
1. NumPy → PyTorch 텐서 변환
2. 디바이스/dtype 일치
3. Conv2d의 경우 원래 4D shape 복원
4. `copy_`로 가중치 업데이트

---

## 8. 전체 파이프라인

### 8.1 설정 구조 (`PruningConfig`)

**파일**: `pruning/pipeline.py:26-46`

```python
@dataclass
class PruningConfig:
    weights: str           # 체크포인트 경로
    data_dir: str          # 데이터셋 경로
    batch_size: int = 128
    calib_batches: int = 8 # Hessian용 배치 수
    calib_samples: int = 2048
    n: int = 2             # N:M의 N
    m: int = 4             # N:M의 M
    lambd: float = 1e-4    # Dampening
    mode: str = "sparsegpt"  # sparsegpt | magnitude
    enforce_nm: bool = True  # 구조적 vs 비구조적
```

### 8.2 모델 레지스트리 (`MODEL_REGISTRY`)

```python
MODEL_REGISTRY = {
    "mnist": {"model_class": SimpleCNN, ...},
    "cifar10": {"model_class": CIFARCNN, ...},
    "resnet18_cifar": {"model_class": ResNet18_CIFAR, ...},
    "vgg16_cifar": {"model_class": VGG16_CIFAR, ...},
}
```

### 8.3 프루닝 실행 흐름 (`run_pruning`)

**파일**: `pruning/pipeline.py:109-223`

```
1. 모델 로드 및 Baseline 성능 측정
           ↓
2. 레이어별 순차 처리
   ┌─────────────────────────────────────────┐
   │ for layer in get_prunable_layers():     │
   │   1. collect_calibration_inputs()       │
   │   2. prune_layer_obs() 또는             │
   │      prune_layer_magnitude()            │
   └─────────────────────────────────────────┘
           ↓
3. 최종 성능 측정 (Fine-tuning 없음)
           ↓
4. 프루닝된 모델 저장 (선택)
```

---

## 9. 수식-코드 대응표

### 9.1 Hessian 계산

| 수식 | 의미 | 코드 위치 |
|------|------|----------|
| $X \in \mathbb{R}^{d \times n}$ | 입력 행렬 | `inputs.t().contiguous().numpy()` |
| $H = \frac{XX^\top}{N}$ | 공분산 | `features @ features.T / features.shape[1]` |
| $H_{new} = H + (\text{damp}+\lambda)I$ | Dampened Hessian | `hessian += (damp + lambd) * np.eye(...)` |

### 9.2 OBS 에러 메트릭

| 수식 | 의미 | 코드 |
|------|------|------|
| $\varepsilon_j = \frac{w_j^2}{[H^{-1}]_{jj}}$ | 프루닝 에러 점수 | `scores = (w_block ** 2) / diag` |

### 9.3 OBS 보정 업데이트

| 수식 | 의미 | 코드 |
|------|------|------|
| $\text{error} = w_i \cdot (1-m_i)$ | 제거되는 값 | `error = w * (1.0 - m_val)` |
| $\delta = \frac{\text{error}}{[H^{-1}]_{ii}}$ | 보정 계수 | `correction = error / d` |
| $W_{:,j+1:} -= \delta \cdot H^{-1}_{i,j+1:}$ | 미래 컬럼 보정 | `weight_matrix[:, i+1:] -= np.outer(correction, h_inv_row)` |

### 9.4 N:M 마스크 생성

| 연산 | 의미 | 코드 |
|------|------|------|
| M 블록 추출 | 연속 M개 가중치 | `importance[r, c:c+m]` |
| Top-N 선택 | 중요도 상위 N개 | `np.argpartition(block_imp, -n)[-n:]` |

---

## 10. 성능 최적화 포인트

### 10.1 벡터화된 연산

```python
# 비효율적 (루프)
for r in range(rows):
    for c in range(remaining_cols):
        weight_matrix[r, i+1+c] -= correction[r] * h_inv_row[c]

# 효율적 (벡터화)
weight_matrix[:, i+1:] -= np.outer(correction, h_inv_row)
```

### 10.2 Cholesky 분해 활용

- LU 분해 대신 Cholesky 사용
- SPD 행렬에서 2배 빠름
- 수치 안정성 향상

### 10.3 부분 정렬 (`argpartition`)

- Full sort: $O(n \log n)$
- Partial partition: $O(n)$
- Top-K만 필요할 때 효율적

---

## 참고 문헌

1. Frantar, E., & Alistarh, D. (2023). *SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot.*
2. Hassibi, B., & Stork, D. G. (1992). *Second order derivatives for network pruning: Optimal Brain Surgeon.*
3. NVIDIA. (2021). *Achieving FP32 Accuracy for INT8 Inference Using Quantization Aware Training with TensorRT.*

---

*본 문서는 CNN SparseGPT 프로젝트의 알고리즘을 수식-코드 레벨에서 상세히 분석한 기술 문서입니다.*
