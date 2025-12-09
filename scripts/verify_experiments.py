import argparse
import copy
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from data.calibration import collect_calibration_inputs
from models.mnist_cnn import SimpleCNN, MNISTTrainConfig, evaluate, get_dataloaders
from pruning.mask import flatten_weight
from pruning.obs import prune_layer_magnitude, prune_layer_obs
from pruning.pipeline import tensor_sparsity, load_state_dict, get_prunable_layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify CNN SparseGPT Experiments")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--model", type=str, default="mnist")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--calib-batches", type=int, default=8)
    parser.add_argument("--calib-samples", type=int, default=2048)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--lambda", dest="lambd", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def get_layer_shape_info(layer: nn.Module) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    weight = layer.weight
    if isinstance(layer, nn.Linear):
        return tuple(weight.shape), tuple(weight.shape)
    if isinstance(layer, nn.Conv2d):
        oc, ic, kh, kw = weight.shape
        return (oc, ic, kh, kw), (oc, ic * kh * kw)
    raise TypeError(f"Unsupported layer: {type(layer)}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading weights from: {args.weights}")
    state_dict = load_state_dict(args.weights)
    model = SimpleCNN()
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading state dict: {e}")
        print("Ensure the checkpoint matches the model architecture (SimpleCNN with named layers).")
        sys.exit(1)
        
    model.to(device)
    model.eval()

    # Config for dataloaders
    data_cfg = MNISTTrainConfig(
        data_dir=args.data_dir,
        batch_size=128,
        test_batch_size=256,
        num_workers=2,
        device=args.device
    )
    train_loader, test_loader = get_dataloaders(data_cfg)
    criterion = nn.CrossEntropyLoss()

    print("\n=== Model Architecture ===")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,} | Trainable parameters: {trainable_params:,}")

    _, baseline_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nBaseline accuracy before pruning: {baseline_acc * 100:.2f}%")

    pruning_targets = get_prunable_layers(model)
    
    print("\n=== Prunable Layer Shapes ===")
    for name, layer in pruning_targets:
        if not hasattr(layer, "weight"): continue
        orig, flat = get_layer_shape_info(layer)
        print(f"[{name}] original: {orig} -> flattened: {flat}")

    baseline_state = copy.deepcopy(model.state_dict())

    # 4 Experiments: OBS/Mag x NM/Dense
    experiments = [
        ("obs", True),
        ("obs", False),
        ("magnitude", True),
        ("magnitude", False),
    ]
    total_experiments = len(experiments)
    results = []

    for idx, (mode, enforce_nm) in enumerate(experiments, 1):
        model.load_state_dict(copy.deepcopy(baseline_state))
        model.to(device)
        model.eval()
        
        nm_status = "ON" if enforce_nm else "OFF"
        print(f"\n=== Experiment {idx}/{total_experiments} | mode={mode} | N:M={nm_status} ===")

        for name, layer in pruning_targets:
            if not hasattr(layer, "weight"): continue
            before = tensor_sparsity(layer.weight.data)
            print(f"\n[{name}] sparsity before pruning: {before:.2f}%")

            if mode == "obs":
                activations = collect_calibration_inputs(
                    model, train_loader, layer, device,
                    max_batches=args.calib_batches,
                    max_samples=args.calib_samples
                )
                prune_layer_obs(
                    layer, activations, 
                    n=args.n, m=args.m, lambd=args.lambd, 
                    enforce_nm=enforce_nm
                )
            else:
                prune_layer_magnitude(
                    layer, 
                    n=args.n, m=args.m, 
                    enforce_nm=enforce_nm
                )
            
            after = tensor_sparsity(layer.weight.data)
            delta = after - before
            print(f"[{name}] sparsity after pruning: {after:.2f}% (Δ {delta:+.2f}%)")

        _, pruned_acc = evaluate(model, test_loader, criterion, device)
        acc_drop = (pruned_acc - baseline_acc) * 100
        print(f"\nAccuracy after pruning: {pruned_acc * 100:.2f}% (Δ {acc_drop:+.2f}pp)")

        results.append({
            "mode": mode,
            "enforce_nm": enforce_nm,
            "acc": pruned_acc,
            "delta": acc_drop,
            "tag": f"{mode}_{'nm' if enforce_nm else 'dense'}"
        })

    print("\n=== Experiment Summary ===")
    for res in results:
        nm_status = "ON" if res["enforce_nm"] else "OFF"
        print(f"{res['tag']:<20} | mode={res['mode']:<9} | N:M={nm_status:<3} | acc={res['acc']*100:.2f}% (Δ {res['delta']:+.2f}pp)")

if __name__ == "__main__":
    main()

