# CNN SparseGPT 구현 분석 보고서

본 문서는 `cnn_sparsegpt` 프로젝트의 소스 코드를 분석하여, SparseGPT 알고리즘이 실제 코드로 어떻게 구현되었는지 상세히 기술합니다.

## 1. 프로젝트 개요

SparseGPT는 대규모 언어 모델(LLM) 및 CNN 모델을 위한 **One-Shot Pruning** 기법입니다. 재학습(Fine-tuning) 없이도 모델의 정확도를 최대한 보존하면서 가중치를 희소화(Sparsity)하는 것이 목표입니다.

- **핵심 철학**: Pruning을 최적화 문제로 정의하고, 가중치를 제거할 때 발생하는 손실(Reconstruction Error)을 남은 가중치를 업데이트하여 보정합니다.
- **수학적 배경**: Optimal Brain Surgeon (OBS) 프레임워크를 대규모 모델에 적용 가능하도록 효율화했습니다.

## 2. 전체 파이프라인 흐름 (`pruning/pipeline.py`)

이 모듈은 전체 프루닝 프로세스를 관장하는 지휘자 역할을 합니다.

### 2.1. 주요 프로세스
1.  **설정 및 로딩 (`run_pruning`)**:
    - Pre-trained 모델 가중치와 설정을 로드합니다.
    - 데이터 로더를 초기화합니다.
2.  **데이터 준비 (Not Data-Free)**:
    - SparseGPT는 완전한 Data-Free가 아닙니다. 가중치 업데이트 방향을 결정하는 Hessian 행렬을 계산하기 위해 **소량의 실제 데이터(Calibration Data)**가 반드시 필요합니다.
    - `collect_calibration_inputs` 함수를 통해 약 128~1024개 정도의 샘플을 모델에 통과시켜 각 레이어의 **입력값(Activation)**을 수집합니다.
3.  **순차적 레이어 처리**:
    - 모델의 레이어를 앞쪽부터 하나씩 순회하며 프루닝을 수행합니다.
    - 앞 레이어의 프루닝 결과(오차)가 뒤 레이어의 입력에 영향을 줄 수 있으므로 순차적 처리가 중요합니다.

## 3. 핵심 알고리즘 상세 (`pruning/sparsegpt.py`)

논문의 **Algorithm 1**이 구현된 핵심 모듈입니다.

### 3.1. Hessian 계산 및 안정화 (`compute_hessian`)
가중치의 중요도를 판단하고 업데이트하기 위해 2차 미분 정보인 Hessian($H$)을 근사합니다.

- **수식**: $H = X X^\top$ (입력 공분산 행렬)
- **구현 디테일**:
  ```python
  hessian = features @ features.T / N
  hessian += (damp + lambd) * np.eye(features.shape[0])
  ```
  - **Dampening**: 역행렬 계산 시 수치적 불안정성(Singularity)을 방지하기 위해 대각 성분에 작은 값($\lambda$)을 더해줍니다. (Ridge Regularization 기법)

### 3.2. 가중치 업데이트 로직 (`prune_layer_obs`) - **Core Logic**
가중치를 0으로 만들 때 발생하는 손실을 최소화하기 위해 **남은 가중치를 수정**합니다.

1.  **Block-wise Processing**:
    - 효율적인 처리를 위해 가중치 행렬을 블록(Column 그룹) 단위로 나누어 처리합니다.
2.  **Adaptive Mask Selection**:
    - 단순히 가중치 크기($|w|$)만 보는 것이 아니라, Hessian 정보를 포함한 중요도 Metric을 사용합니다.
    - **Metric**: $\frac{w^2}{[H^{-1}]_{ii}}$
    - $[H^{-1}]_{ii}$가 작을수록(Hessian 곡률이 클수록) 해당 가중치는 에러에 민감하므로 중요하게 취급됩니다.
3.  **Optimal Update (보정)**:
    - 가장 중요한 단계입니다. 가중치 하나를 자르면, 그 오차를 다른 가중치들이 나눠서 부담하도록 값을 변경합니다.
    - **코드 매핑**:
      ```python
      # Error: 잘려나가는 가중치 값
      correction = error / diag_element
      
      # 남은 가중치들을 한 번에 업데이트 (Vectorized)
      weight_matrix[:, i+1:] -= np.outer(correction, h_inv_row)
      ```
    - 즉, "내가 사라지지만, 내 역할은 너희들이 조금씩 나눠서 대신 해줘"라는 논리로 동작합니다.

## 4. 마스크 생성 및 형상 관리 (`pruning/mask.py`)

### 4.1. CNN 필터의 2D 변환 (`flatten_weight`)
SparseGPT 알고리즘은 2차원 행렬 연산을 기반으로 합니다. 따라서 4차원 CNN 커널을 2차원으로 변환해야 합니다.

- **변환 방식**: `(Out, In, K, K)` $\rightarrow$ `(Out, In * K * K)`
    - **Row (행)**: 출력 채널 (Filter 개수)
    - **Col (열)**: 입력 채널 $\times$ 커널 높이 $\times$ 커널 너비 (Flattened Kernel)
- 이 형태에서 각 행은 하나의 독립적인 뉴런(필터) 역할을 하며, SparseGPT는 이 행렬 내에서 연산을 수행합니다.

### 4.2. 마스크 생성 전략
변환된 2차원 행렬에 대해 두 가지 방식의 마스크를 지원합니다.

1.  **N:M Structured Pruning (`nm_mask_blockwise`)**:
    - 연속된 $M$개의 가중치 그룹마다 정확히 $N$개만 남깁니다. (예: 2:4 Sparsity)
    - NVIDIA Ampere 아키텍처 등 최신 하드웨어 가속기를 활용하기 위한 정형화된 패턴입니다.
2.  **Row-wise Top-K (Unstructured) (`elementwise_topk_mask`)**:
    - **행 단위 적용**: 전체 행렬을 통틀어 하위 X%를 자르는 것이 아니라, **각 행(출력 필터)마다** 상위 K개를 살립니다.
    - **목적**:
        - **Load Balancing**: 모든 필터가 동일한 연산량을 가지게 하여 병렬 처리 효율을 높입니다.
        - **Layer Collapse 방지**: 특정 필터가 완전히 0이 되어 정보가 소실되는 것을 막습니다.

### 4.3. 복구 (`assign_weight`)
- 모든 연산이 끝난 2차원 행렬을 다시 원래의 4차원 텐서 `(Out, In, K, K)`로 변환하여 PyTorch 모델에 주입합니다.

## 5. 데이터 처리 흐름 요약

| 단계 | 모듈 | 작업 내용 | 데이터 형태 (예: Conv2d) |
| :--- | :--- | :--- | :--- |
| **1. 준비** | `pipeline` | 모델 로드 및 캘리브레이션 데이터(Activation) 수집 | Tensor `(Batch, In, H, W)` |
| **2. 변환** | `mask` | CNN 가중치를 2D 행렬로 펼침 (Flatten) | Matrix `(Out, In*K*K)` |
| **3. 계산** | `sparsegpt` | $H = X X^\top$ 계산 및 역행렬 $H^{-1}$ 준비 | Matrix `(Dim, Dim)` |
| **4. 실행** | `sparsegpt` | 마스크 결정 $\rightarrow$ 가중치 Pruning $\rightarrow$ **남은 가중치 Update** | Matrix `(Out, In*K*K)` |
| **5. 복구** | `mask` | 처리된 행렬을 다시 4D 텐서로 변환하여 모델에 적용 | Tensor `(Out, In, K, K)` |

---
*Last Updated: 2025-12-10*

