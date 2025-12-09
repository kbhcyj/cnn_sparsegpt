# CNN SparseGPT Project

이 프로젝트는 SparseGPT 논문(Frantar & Alistarh, 2023)과 내부 정리 자료를 토대로 OBS(Optimal Brain Surgeon) 기반 n:m 프루닝을 MNIST/CIFAR CNN 모델에 적용하는 예제입니다.

## 디렉터리 구조
- `models/`: MNIST/CIFAR CNN 및 학습 설정
- `data/`: 캘리브레이션/데이터 유틸리티
- `pruning/`: OBS, 마스크, 파이프라인 로직
- `configs/`: 예제 구성 파일(YAML)
- `scripts/`: 실행 엔트리포인트 (예: `prune.py`)
- `experiments/`: 실험 기록 및 리포트
- `docs/`: SparseGPT/OBS 이론 정리

## 빠른 시작
```bash
conda activate pytorch

# 1. 베이스라인 모델 학습 (체크포인트 생성)
python scripts/train_baseline.py --model resnet18_cifar --epochs 100 --save-path checkpoints/resnet18_cifar.pt

# 2. 프루닝 실행
python scripts/prune.py --config configs/exp_cifar_resnet18.yaml
```

## 참고 문헌
- SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot (arXiv:2301.00774)
- SparseGPT Review Notes (docs/notes.md)
