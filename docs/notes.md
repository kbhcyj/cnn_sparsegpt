## SparseGPT & OBS 주요 개념 정리

- **SparseGPT (Frantar & Alistarh, 2023)**  
  - GPT-175B/BLOOM-176B까지 단일 패스로 50~60% 희소화를 달성.  
  - OBS(Optimal Brain Surgeon) 기반 컬럼 단위 그리디 접근으로 행마다 다른 마스크를 허용하면서도 공유 Hessian 역행렬 시퀀스를 활용해 계산량을 `O(d_hidden^3)`로 완화.  
  - n:m(2:4, 4:8) 패턴 및 4-bit 양자화를 동시에 수행 가능.

- **OBS Layer Reconstruction**  
  - 고정된 마스크 `M`에서 가중치 최소제곱 해: `w_M = (X_M X_M^T)^{-1} X_M (w X_M)^T`.  
  - OBS 업데이트: 제거 대상 가중치 `w_m`에 대해 `δ_m = - w_m / [H^{-1}]_{mm} * H^{-1}_{:,m}` 로 나머지 가중치를 보상.  
  - Schur 보수 기반으로 열 순서별 부분 Hessian 역행렬 `(H_{U_j})^{-1}`를 재귀 계산하여 행마다 상이한 마스크 문제 해결.

- **Adaptive Mask Selection**  
  - OBS 오차 `ε_m = w_m^2 / [H^{-1}]_{mm}` 를 기준으로 열 블록 단위(`B_s`) 마스크를 선택해 비균일 sparsity를 허용.  
  - n:m 패턴에서는 블록 크기를 `m`으로 맞추고 각 블록에서 `n`개 비제로를 유지.

- **프로젝트 적용 포인트**  
  - `pruning/obs.py`는 OBS 보상, `pruning/mask.py`는 n:m 마스크 생성을 담당.  
  - `pruning/pipeline.py`는 모델/데이터 로더/캘리브레이션/프루닝/저장을 한 번에 실행.  
  - 추후 개선: 열 블록 기반 Hessian 재사용, lazy update, 양자화 결합 등 추가 최적화 계획.

