# SparseGPT 알고리즘 구현 분석 및 CNN_SparseGPT 적용 보고서

본 문서는 SparseGPT 논문(*SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot*)에서 제안된 핵심 알고리즘들을 분석하고, 이를 공식 구현체와 본 프로젝트(`CNN_SparseGPT`)에서 어떻게 구현했는지 비교 설명합니다. 또한, CNN 모델(ResNet, VGG 등)에 적용하기 위해 변경된 사항과 구현의 특징을 심층적으로 다룹니다.

---

## 1. SparseGPT 핵심 알고리즘 개요

SparseGPT는 대규모 언어 모델(LLM)을 재학습(Fine-tuning) 없이 높은 희소성(Sparsity)으로 압축하기 위해 제안된 기법입니다. 핵심은 레이어 단위의 역전파 없는(layer-wise) 최적화 문제로 정의하고, 이를 **Closed-form**으로 빠르게 푸는 것입니다.

### 최적화 목표
각 레이어의 가중치 $W$에 대해, 입력 $X$가 주어졌을 때 프루닝된 가중치 $\hat{W}$와의 출력 오차를 최소화합니다.
$$ \min_{\hat{W}} || WX - \hat{W}X ||_2^2 \quad \text{s.t.} \quad \hat{W} \text{ is sparse} $$

### 핵심 알고리즘 3요소
1.  **OBS (Optimal Brain Surgeon) Closed-form Solution**:
    Hessian Inverse $H^{-1} = (XX^T)^{-1}$를 이용하여, 가중치 하나를 제거할 때 발생하는 에러를 최소화하는 최적의 업데이트 식을 유도합니다.
    $$ \delta W = - \frac{w_p}{[H^{-1}]_{pp}} H^{-1}_{:, p} $$
2.  **Adaptive Mask Selection**:
    가중치를 제거할 때 정적인(Static) 값이 아닌, 이전 가중치 제거로 인해 **업데이트된 값**을 기준으로 마스크를 동적으로 결정합니다.
3.  **Fast Approximate Reconstruction (Row-Hessian update)**:
    가중치 업데이트를 효율적으로 수행하기 위해, Cholesky 분해와 벡터화된 연산을 사용하여 $O(d^3)$ 복잡도를 $O(d^3)$ (하지만 매우 작은 상수) 또는 블록 단위 처리로 가속화합니다.

---

## 2. 알고리즘 구현 비교: 공식 구현체 vs CNN_SparseGPT

### 2.1 Hessian 계산 및 Adaptive Dampening

**공식 구현체 (LLM)**:
-   `H = 2 * X @ X.T / N` 형태로 계산 (비율 상수는 구현마다 다를 수 있음)
-   **Adaptive Dampening**: 수치적 안정성을 위해 대각 성분에 `damp = 0.01 * mean(diag(H))`를 더합니다.

**CNN_SparseGPT**:
-   `obs.py/compute_hessian` 함수에서 동일하게 구현되었습니다.
-   CNN의 경우 입력이 `(N, C, H, W)` 4차원 텐서이므로, 이를 `(N * H * W, C)` 형태의 2D 행렬로 변환(Unfolding)하여 $X$를 구성합니다. 이는 1x1 Conv를 수행하는 것과 수학적으로 동일한 형태를 만들어 Hessian을 계산하게 해줍니다.

### 2.2 Fast Approximate Reconstruction (핵심 알고리즘)

논문의 Algorithm 1에 해당하는 부분으로, 가장 중요한 구현 포인트입니다.

**공식 구현체**:
```python
# Pseudo-code logic
InverseH = CholeskyInverse(H)
For col i in range(d):
    w = W[:, i] 
    d = InverseH[i, i]
    # ... Mask Selection ...
    Error = (w - masked_w) 
    # Update all future weights in one go using outer product
    W[:, i+1:] -= (Error / d) * InverseH[i, i+1:] 
```

**CNN_SparseGPT (`pruning/obs.py`)**:
-   초기에는 이중 루프를 사용하는 정석적인 OBS 방식을 사용했으나, **최적화된 벡터 연산** 방식으로 수정되었습니다.
-   `prune_layer_obs` 함수 내부의 `Fast Reconstruction Loop`:
    ```python
    # Future columns update logic
    if i + 1 < cols:
        correction = error / d  # [Rows]
        h_inv_row = hessian_inv[i, i+1:] # [FutureCols]
        weight_matrix[:, i+1:] -= np.outer(correction, h_inv_row)
    ```
-   `np.outer`를 사용하여 현재 컬럼의 에러를 남은 모든 컬럼에 한 번에 전파합니다. 이는 Python 루프 오버헤드를 줄이고 Numpy의 BLAS 최적화를 활용합니다.

### 2.3 Adaptive Mask Selection

**공식 구현체**:
-   루프 안에서 현재 시점의 가중치 `W` (이미 이전 단계의 보정이 반영된 상태)를 기준으로 중요도 점수 $\frac{w^2}{[H^{-1}]_{ii}}$ 를 계산합니다.
-   이 점수를 기반으로 마스크를 생성하므로, 마스크가 프루닝 순서에 따라 달라질 수 있는 **동적(Adaptive)** 특성을 가집니다.

**CNN_SparseGPT**:
-   **N:M Structured Pruning**: `obs.py` 내에서 `enforce_nm=True`일 때, 블록 단위로 루프를 돌며 현재 블록의 **업데이트된 가중치**를 기준으로 마스크를 계산합니다.
    ```python
    scores = (w_block ** 2) / diag.reshape(1, -1)
    # ... Select Top-N indices per row ...
    ```
-   **Unstructured Pruning**: 공식 구현체와 유사하게, 초기 가중치 기반으로 Global Mask를 생성하거나(속도 우선), 루프 내에서 처리하는 Hybrid 방식을 지원하도록 설계되었습니다. 현재 구현은 N:M에 최적화되어 있습니다.

---

## 3. CNN 적용을 위한 변경 사항 및 심층 설명

LLM(Linear Layer 위주)을 위한 SparseGPT를 CNN(Conv2d Layer 위주)에 적용하기 위해 `CNN_SparseGPT`에서 수행한 주요 변경 사항입니다.

### 3.1 4D Tensor to 2D Matrix (Im2Col 관점)

SparseGPT는 기본적으로 행렬 연산($W X$)을 가정합니다. CNN의 합성곱 연산은 행렬 곱으로 표현될 수 있습니다.

-   **가중치 펼치기 (`mask.py/flatten_weight`)**:
    -   `Conv2d(Out, In, K, K)` 가중치를 `(Out, In * K * K)` 형태의 2D 행렬로 변환합니다.
    -   여기서 각 행(Row)은 하나의 필터(Output Channel)에 해당하며, 각 열(Col)은 입력 채널과 공간적 위치(Spatial Kernel)에 해당합니다.
    -   이렇게 하면 Conv2d 연산이 `MatMul` 형태가 되어 SparseGPT 알고리즘을 그대로 적용할 수 있습니다.

-   **입력 활성화 처리 (`compute_hessian`)**:
    -   입력 텐서 `(Batch, In, H, W)`를 `(Batch * H * W, In)` 형태로 Reshape(또는 `im2col`)하여 Hessian $X^T X$를 계산합니다.
    -   이는 합성곱의 Sliding Window 특성을 통계적으로 반영하는 효과를 줍니다.

### 3.2 Channel-wise vs Kernel-wise Sparsity

-   LLM에서는 Linear Layer의 Input Dimension(Hidden Size) 방향으로 프루닝이 일어납니다.
-   CNN에서는 `In * K * K` 차원 방향으로 프루닝이 일어납니다.
-   **N:M 적용 시 의미**:
    -   `CNN_SparseGPT`에서는 2:4 Sparsity를 적용할 때, 4개의 연속된 가중치(커널의 공간적 픽셀 또는 입력 채널) 중 2개를 살립니다.
    -   이는 커널 내부의 미세한 패턴(Micro-structure)을 희소하게 만드는 것으로, 실제 하드웨어 가속(Tensor Core)을 받기 적합한 구조입니다.

### 3.3 Batch Normalization 처리

-   CNN은 LLM과 달리 Conv 레이어 직후에 Batch Normalization(BN)이 오는 경우가 많습니다.
-   SparseGPT는 $W X$의 에러를 최소화하지만, 실제로는 $BN(W X)$가 다음 레이어로 전달됩니다.
-   **현재 구현**: BN을 명시적으로 고려하지 않고 Conv 가중치 자체의 Reconstruction Error를 최소화합니다. 실험 결과(CIFAR/MNIST)에서 보듯이 이 방식만으로도 충분히 우수한 성능을 보이지만, 더 정교한 튜닝을 위해서는 BN의 Scale 파라미터를 Hessian 계산에 반영하거나(Folded BN), 프루닝 후 BN 통계량을 재조정(Re-calibration)하는 과정이 도움이 될 수 있습니다. (현재 구현은 Re-calibration 없이도 높은 성능을 달성함)

---

## 4. 결론

`CNN_SparseGPT` 프로젝트는 SparseGPT의 공식 구현체(LLM용)가 가진 수학적 원리와 핵심 알고리즘(Cholesky Update, Adaptive Masking)을 **CNN 아키텍처의 특성(4D Tensor 처리, Sliding Window)**에 맞게 정확하게 이식하고 구현했습니다.

특히, 단순 반복문이 아닌 **Vectorized Fast Reconstruction**을 구현함으로써 Python 환경에서도 효율적인 연산이 가능하도록 했으며, CIFAR-10 및 MNIST 실험을 통해 기존 Magnitude Pruning 대비 압도적인 성능 우위를 입증하였습니다.

