import argparse
import os
import sys

# 프로젝트 루트 경로를 sys.path에 추가하여 모듈 import 가능하게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from pruning.pipeline import MODEL_REGISTRY
from models.cifar_cnn import train_one_epoch

def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline models for SparseGPT experiments")
    parser.add_argument("--model", type=str, required=True, choices=MODEL_REGISTRY.keys(),
                        help="Model architecture to train")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to dataset")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--output", type=str, required=True, help="Path to save the trained checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 디렉토리 생성
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # 시드 설정
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    print(f"Training {args.model} on {args.device}...")
    
    # 모델 스펙 가져오기
    if args.model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {args.model}")
        
    spec = MODEL_REGISTRY[args.model]
    model_class = spec["model_class"]
    config_class = spec["config_class"]
    loader_fn = spec["loader_fn"]
    evaluate_fn = spec["evaluate_fn"]
    
    # Config 및 데이터 로더 초기화
    # 학습 설정은 인자값으로 덮어씌움
    train_config = config_class(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        seed=args.seed
    )
    
    train_loader, test_loader = loader_fn(train_config)
    
    # 모델 초기화
    model = model_class()
    model.to(args.device)
    
    # Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training Loop
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, torch.device(args.device))
        _, test_acc = evaluate_fn(model, test_loader, criterion, torch.device(args.device))
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")
        
        # Best model 저장
        if test_acc > best_acc:
            best_acc = test_acc
            save_dict = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "accuracy": best_acc,
            }
            torch.save(save_dict, args.output)
            print(f"  -> Model saved to {args.output} (Acc: {best_acc*100:.2f}%)")
            
    print(f"\nTraining finished. Best Accuracy: {best_acc*100:.2f}%")

if __name__ == "__main__":
    main()

