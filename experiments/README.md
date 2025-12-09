## Experiments

이 디렉터리는 SparseGPT CNN 프루닝 실험 로그와 결과를 기록하기 위한 공간입니다.

### 예시 워크플로우
1. `configs/`에서 실험 YAML을 선택/복사하여 수정
2. `python scripts/prune.py --config configs/mnist_obs.yaml`
3. 출력 로그와 정확도, 프루닝 후 체크포인트를 `experiments/` 하위 폴더에 정리

### 권장 기록 항목
- 사용한 데이터셋/가중치 경로
- n:m 설정, λ, 캘리브레이션 샘플 수
- 프루닝 전/후 정확도 및 각 레이어 희소도
- 추가 메모(추가 최적화, 문제점 등)

