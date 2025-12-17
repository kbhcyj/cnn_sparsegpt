#!/usr/bin/env python3
"""
전체 모델 스윕 실험 스크립트
- 모든 모델 (MNIST, CIFAR-10, ResNet-18, VGG-16)
- 두 가지 방법 (Magnitude, SparseGPT) 비교
- 결과를 CSV로 저장
"""

import os
import sys
import csv
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pruning.pipeline import PruningConfig, run_pruning

# 실험 설정
EXPERIMENTS = [
    # MNIST
    {
        "name": "MNIST",
        "model": "mnist",
        "weights": "checkpoints/mnist_cnn.pt",
        "batch_size": 128,
    },
    # CIFAR-10 CNN
    {
        "name": "CIFAR-10 CNN",
        "model": "cifar10",
        "weights": "checkpoints/cifar10_cnn_full.pt",
        "batch_size": 128,
    },
    # ResNet-18 CIFAR
    {
        "name": "ResNet-18 CIFAR",
        "model": "resnet18_cifar",
        "weights": "checkpoints/resnet18_cifar_full.pt",
        "batch_size": 64,
    },
    # VGG-16 CIFAR
    {
        "name": "VGG-16 CIFAR",
        "model": "vgg16_cifar",
        "weights": "checkpoints/vgg16_cifar_full.pt",
        "batch_size": 32,
    },
]

METHODS = ["magnitude", "sparsegpt"]


def run_sweep():
    results = []
    
    print("=" * 80)
    print("전체 모델 스윕 실험 시작")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    for exp in EXPERIMENTS:
        if not os.path.exists(exp["weights"]):
            print(f"\n⚠️ 스킵: {exp['name']} - 체크포인트 없음 ({exp['weights']})")
            continue
            
        for method in METHODS:
            print(f"\n{'='*60}")
            print(f"▶ {exp['name']} - {method.upper()}")
            print("=" * 60)
            
            config = PruningConfig(
                weights=exp["weights"],
                data_dir="data",
                batch_size=exp["batch_size"],
                test_batch_size=256,
                num_workers=4,
                calib_batches=8,
                calib_samples=1024,
                n=2,
                m=4,
                lambd=1e-4,
                device="cuda",
                output=f"checkpoints/{exp['model']}_pruned_{method}.pt",
                mode=method,
                model=exp["model"],
                seed=42,
                enforce_nm=True,
            )
            
            try:
                metrics = run_pruning(config)
                
                result = {
                    "Model": exp["name"],
                    "Method": method,
                    "Baseline Acc (%)": f"{metrics['baseline_acc'] * 100:.2f}",
                    "Pruned Acc (%)": f"{metrics['pruned_acc'] * 100:.2f}",
                    "Acc Drop (%)": f"{(metrics['baseline_acc'] - metrics['pruned_acc']) * 100:.2f}",
                }
                results.append(result)
                
                print(f"✅ 완료: Baseline={result['Baseline Acc (%)']}%, Pruned={result['Pruned Acc (%)']}%")
                
            except Exception as e:
                print(f"❌ 오류: {e}")
                results.append({
                    "Model": exp["name"],
                    "Method": method,
                    "Baseline Acc (%)": "ERROR",
                    "Pruned Acc (%)": "ERROR",
                    "Acc Drop (%)": "ERROR",
                })
    
    # 결과 저장
    os.makedirs("experiments/results", exist_ok=True)
    csv_path = f"experiments/results/sweep_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Model", "Method", "Baseline Acc (%)", "Pruned Acc (%)", "Acc Drop (%)"])
        writer.writeheader()
        writer.writerows(results)
    
    print("\n" + "=" * 80)
    print("실험 결과 요약")
    print("=" * 80)
    print(f"{'Model':<20} | {'Method':<10} | {'Baseline':>10} | {'Pruned':>10} | {'Drop':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['Model']:<20} | {r['Method']:<10} | {r['Baseline Acc (%)']:>10} | {r['Pruned Acc (%)']:>10} | {r['Acc Drop (%)']:>10}")
    
    print(f"\n결과 저장: {csv_path}")
    print(f"완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results


if __name__ == "__main__":
    run_sweep()
