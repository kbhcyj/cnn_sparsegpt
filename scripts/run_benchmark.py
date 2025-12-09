import argparse
import os
import sys
import csv
import torch
from typing import List, Dict

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pruning.pipeline import PruningConfig, run_pruning

def run_benchmark():
    # 실험 대상 모델 및 체크포인트 (Fully Trained)
    models = {
        "cifar10": "cnn_sparsegpt/checkpoints/cifar10_cnn_full.pt",
        "resnet18_cifar": "cnn_sparsegpt/checkpoints/resnet18_cifar_full.pt",
        "vgg16_cifar": "cnn_sparsegpt/checkpoints/vgg16_cifar_full.pt"
    }
    
    modes = ["magnitude", "sparsegpt"]
    results = []
    
    print(f"{'Model':<20} | {'Method':<10} | {'Base Acc':<10} | {'Pruned Acc':<10} | {'Drop':<10}")
    print("-" * 75)
    
    for model_name, checkpoint_path in models.items():
        if not os.path.exists(checkpoint_path):
            print(f"Skipping {model_name}: checkpoint not found at {checkpoint_path}")
            continue
            
        for mode in modes:
            # 설정 생성
            config = PruningConfig(
                weights=checkpoint_path,
                data_dir="data",
                model=model_name,
                mode=mode,
                device="cuda" if torch.cuda.is_available() else "cpu",
                batch_size=128,
                test_batch_size=256,
                calib_batches=8,   # SparseGPT용
                calib_samples=1024,
                n=2, m=4,          # 2:4 Sparsity
                enforce_nm=True,
                lambd=0.01,        # Dampening
                output=None        # 벤치마킹이므로 저장은 생략 (원하면 경로 지정 가능)
            )
            
            try:
                # 프루닝 실행
                # run_pruning returns: {"baseline_acc": float, "pruned_acc": float}
                metrics = run_pruning(config)
                
                base_acc = metrics["baseline_acc"] * 100
                pruned_acc = metrics["pruned_acc"] * 100
                drop = base_acc - pruned_acc
                
                print(f"{model_name:<20} | {mode:<10} | {base_acc:>9.2f}% | {pruned_acc:>9.2f}% | {drop:>9.2f}%")
                
                results.append({
                    "Model": model_name,
                    "Method": mode,
                    "Baseline Acc": f"{base_acc:.2f}",
                    "Pruned Acc": f"{pruned_acc:.2f}",
                    "Accuracy Drop": f"{drop:.2f}"
                })
                
            except Exception as e:
                print(f"Error running {model_name} with {mode}: {e}")

    # 결과 CSV 저장
    csv_path = "cnn_sparsegpt/experiments/results/benchmark_results_full.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Model", "Method", "Baseline Acc", "Pruned Acc", "Accuracy Drop"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nBenchmark results saved to {csv_path}")

if __name__ == "__main__":
    run_benchmark()

