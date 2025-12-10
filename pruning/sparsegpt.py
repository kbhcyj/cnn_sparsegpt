from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from .mask import assign_weight, flatten_weight, build_pruning_mask


def compute_hessian(inputs: torch.Tensor, lambd: float = 1e-4) -> np.ndarray:
    """
    Hessian Matrix (H)를 계산합니다.
    
    논문 배경 (Section 2.1):
    SparseGPT는 Layer-wise reconstruction error를 최소화하는 것을 목표로 합니다.
    E = || W X - \hat{W} X ||^2_2
    
    이 식의 Taylor expansion에서 2차 항(Hessian)은 입력 공분산 행렬과 비례합니다.
    H = 2 * X X^T (상수 2는 최적화 해에 영향을 주지 않으므로 생략 가능)
    """
    # inputs: [Batch, Channel, ...] -> features: [Channel, Samples]
    features = inputs.t().contiguous().numpy()
    
    # H = X @ X.T / N
    # 샘플 수 N으로 나누어 정규화 (논문 구현체 방식)
    hessian = features @ features.T / features.shape[1]
    
    # Adaptive Dampening (SparseGPT Implementation Detail)
    # 논문에서는 수치적 안정성을 위해 Hessian 대각 성분에 작은 값을 더하는 Dampening을 사용합니다.
    # damp = \lambda * mean(diag(H))
    mean_diag = np.mean(np.diag(hessian))
    damp = 0.01 * mean_diag
    
    # H_new = H + (damp + \lambda) * I
    # Ridge regularization과 유사한 역할을 하며, 역행렬 계산 시 Singularity를 방지합니다.
    hessian += (damp + lambd) * np.eye(features.shape[0], dtype=np.float32)
    return hessian


def invert_hessian(hessian: np.ndarray) -> np.ndarray:
    """
    Hessian의 역행렬(H^-1)을 계산합니다.
    
    Fast Approximate Reconstruction을 위해서는 H^-1가 필요합니다.
    OBS(Optimal Brain Surgeon) 수식에서 분모에 [H^-1]_{qq} 항이 등장하기 때문입니다.
    """
    try:
        # Cholesky Decomposition: H = L L^T
        # H^-1 = (L^-1)^T L^-1
        # Cholesky는 SPD(Symmetric Positive Definite) 행렬에 대해 가장 빠르고 안정적입니다.
        L = np.linalg.cholesky(hessian)
        H_inv = np.linalg.inv(L.T) @ np.linalg.inv(L)
    except np.linalg.LinAlgError:
        print("Warning: Cholesky failed, using numpy.linalg.inv with extra dampening")
        # 실패 시 더 강한 Dampening을 적용하여 역행렬 계산
        hessian += 1e-2 * np.mean(np.diag(hessian)) * np.eye(hessian.shape[0])
        H_inv = np.linalg.inv(hessian)
    return H_inv


def prune_layer_obs(
    layer: nn.Module,
    activations: torch.Tensor,
    n: int = 2,
    m: int = 4,
    lambd: float = 1e-4,
    enforce_nm: bool = True,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    SparseGPT 공식 구현을 따르는 Fast & Adaptive Pruning 알고리즘.
    
    논문 [Algorithm 1: Fast Approximate Reconstruction] 구현
    """
    weight_matrix, original_shape = flatten_weight(layer)
    rows, cols = weight_matrix.shape
    
    # 1. Hessian & Inverse Hessian 계산
    # Step 1 in Algorithm 1: Compute H^-1
    hessian = compute_hessian(activations, lambd=lambd)
    hessian_inv = invert_hessian(hessian)
    
    # 최종 마스크 저장용
    final_mask = np.zeros_like(weight_matrix, dtype=np.float32)
    
    # N:M Sparsity의 경우 M개 컬럼 단위로 블록 처리가 필요함
    block_size = m if enforce_nm else 1
    
    # Unstructured(enforce_nm=False)일 때의 초기 마스크 계산
    if not enforce_nm:
        diag = np.diag(hessian_inv)
        # 논문 Eq (3): Selection Metric
        # \sigma_q = w_q^2 / [H^{-1}]_{qq}
        scores = (weight_matrix ** 2) / diag.reshape(1, -1)
        final_mask = build_pruning_mask(weight_matrix, n=n, m=m, enforce_nm=False, scores=scores)
    
    # 2. Column-wise Loop (Fast Reconstruction)
    # Step 2 in Algorithm 1: Iterate over columns j = 1 ... d_col
    for c_start in range(0, cols, block_size):
        c_end = min(c_start + block_size, cols)
        
        # 현재 블록의 가중치 복사 (이미 이전 루프의 업데이트가 반영된 상태)
        w_block = weight_matrix[:, c_start:c_end].copy()
        
        # --- [Adaptive Mask Selection] ---
        # 논문 Section 2.3: Adaptive Mask Selection
        # "The mask M is chosen based on the current weights W"
        # 즉, 업데이트된 가중치(w_block)를 기준으로 마스크를 다시 계산합니다.
        
        if enforce_nm:
            # 해당 블록에 대응하는 H^-1의 대각 성분 추출
            h_inv_block = hessian_inv[c_start:c_end, c_start:c_end]
            diag = np.diagonal(h_inv_block)
            
            # Metric: w^2 / [H^-1]_ii
            scores = (w_block ** 2) / diag.reshape(1, -1)
            
            mask_block = np.zeros_like(w_block)
            for r in range(rows):
                if c_end - c_start < m:
                    mask_block[r, :] = 1.0
                else:
                    # 각 행별로 점수가 가장 높은 N개 선택
                    idx = np.argsort(scores[r])[-n:]
                    mask_block[r, idx] = 1.0
        else:
            # Unstructured는 위에서 미리 계산한 마스크 사용 (여기서 Adaptive하게 할 수도 있음)
            mask_block = final_mask[:, c_start:c_end]

        # 마스크 저장
        if enforce_nm:
            final_mask[:, c_start:c_end] = mask_block
        
        # --- [Fast Reconstruction Loop] ---
        # 블록 내부의 각 컬럼(또는 단일 컬럼)에 대해 업데이트 수행
        for i in range(c_start, c_end):
            w = weight_matrix[:, i]       # 현재 가중치 컬럼 W_i
            d = hessian_inv[i, i]         # [H^-1]_ii
            
            # 마스크 적용 및 에러 계산
            # Error = w - mask(w)  (제거되는 값)
            m_val = mask_block[:, i - c_start]  # 현재 처리 중인 컬럼의 마스크 (0 또는 1)
            error = w * (1.0 - m_val)           # 잘려나가는 값 (Mask가 0인 곳의 가중치 값)
            
            # 현재 가중치 프루닝 (0으로 만듦)
            weight_matrix[:, i] = w * m_val
            
            # [Core of SparseGPT]
            # 논문 Eq (2): Optimal Update \delta w = - (w_p / [H^{-1}]_{pp}) * H^{-1}_{:, p}
            # Algorithm 1 Line 9-10: Update future columns
            # W_{:, j+1:} -= (Error / [H^{-1}]_{jj}) * H^{-1}_{j, j+1:}
            
            if i + 1 < cols:
                # correction term: Error / [H^{-1}]_{ii}
                correction = error / d      # 보정값 크기 계산 (d는 Hessian 역행렬 대각성분)
                
                # H^{-1} Row vector for future columns: H^{-1}_{i, i+1:}
                h_inv_row = hessian_inv[i, i+1:] 
                
                # Outer product로 한 번에 업데이트 (Vectorized Operation)
                # w_next = w_curr - correction * h_inv_row
                weight_matrix[:, i+1:] -= np.outer(correction, h_inv_row)

    assign_weight(layer, weight_matrix, original_shape)
    return layer.weight.data, final_mask


def prune_layer_magnitude(
    layer: nn.Module, n: int = 2, m: int = 4, enforce_nm: bool = True
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    단순 Magnitude Pruning (데이터/Hessian 사용 안함).
    비교 실험을 위한 Baseline 알고리즘입니다.
    """
    weight_matrix, original_shape = flatten_weight(layer)
    
    # scores=None이면 내부에서 abs(weight)를 중요도로 사용함
    mask = build_pruning_mask(
        weight_matrix, n=n, m=m, enforce_nm=enforce_nm, scores=None
    )
    
    # Pruning 적용
    weight_matrix *= mask
    
    assign_weight(layer, weight_matrix, original_shape)
    return layer.weight.data, mask
