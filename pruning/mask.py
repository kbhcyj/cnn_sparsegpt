from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def nm_mask_blockwise(
    weight: np.ndarray, n: int = 2, m: int = 4, scores: np.ndarray | None = None
) -> np.ndarray:
    """
    N:M Structured Pruning을 위한 블록 단위 마스크를 생성합니다.
    """
    if m <= 0 or n <= 0 or n > m:
        raise ValueError(f"잘못된 n:m 설정입니다. (n={n}, m={m})")

    if scores is None:
        importance = np.abs(weight)
    else:
        importance = scores

    mask = np.zeros_like(weight, dtype=np.float32)
    rows, cols = weight.shape
    
    for r in range(rows):
        for c in range(0, cols, m):
            block_imp = importance[r, c : c + m]
            if block_imp.size < m:
                mask[r, c : c + m] = 1.0
                continue
            keep_idx = np.argpartition(block_imp, -n)[-n:]
            mask[r, c : c + m][keep_idx] = 1.0
    return mask


def elementwise_topk_mask(
    weight: np.ndarray, keep_ratio: float, scores: np.ndarray | None = None
) -> np.ndarray:
    """
    Unstructured Pruning을 위한 행(Row) 단위 Top-K 마스크를 생성합니다.
    """
    keep_ratio = float(np.clip(keep_ratio, 0.0, 1.0))
    
    if scores is None:
        importance = np.abs(weight)
    else:
        importance = scores

    mask = np.zeros_like(weight, dtype=np.float32)
    rows, cols = weight.shape
    keep_cols = max(1, int(round(cols * keep_ratio)))

    for r in range(rows):
        idx = np.argpartition(importance[r], -keep_cols)[-keep_cols:]
        mask[r, idx] = 1.0
    return mask


def build_pruning_mask(
    weight: np.ndarray, 
    n: int, 
    m: int, 
    enforce_nm: bool, 
    scores: np.ndarray | None = None
) -> np.ndarray:
    """
    조건에 따른 프루닝 마스크 생성 헬퍼 함수
    """
    if enforce_nm:
        return nm_mask_blockwise(weight, n=n, m=m, scores=scores)
    
    keep_ratio = n / m if m > 0 else 0.0
    return elementwise_topk_mask(weight, keep_ratio=keep_ratio, scores=scores)


def flatten_weight(layer: nn.Module) -> tuple[np.ndarray, tuple[int, ...]]:
    weight = layer.weight.detach().cpu().numpy()
    if isinstance(layer, nn.Linear):
        return weight.copy(), weight.shape
    if isinstance(layer, nn.Conv2d):
        oc, ic, kh, kw = weight.shape
        return weight.reshape(oc, ic * kh * kw).copy(), (oc, ic, kh, kw)
    raise TypeError(f"지원되지 않는 레이어: {type(layer)}")


def assign_weight(
    layer: nn.Module, weight_matrix: np.ndarray, original_shape: tuple[int, ...]
) -> None:
    tensor = torch.from_numpy(weight_matrix).to(layer.weight.device, dtype=layer.weight.dtype)
    if isinstance(layer, nn.Conv2d):
        tensor = tensor.view(*original_shape)
    else:
        tensor = tensor.view_as(layer.weight)
    layer.weight.data.copy_(tensor)
