import argparse
import copy
import sys
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn

from data.calibration import collect_calibration_inputs
from models.mnist_cnn import SimpleCNN, MNISTTrainConfig, evaluate, get_dataloaders
from pruning.mask import flatten_weight
from pruning.obs import prune_layer_magnitude, prune_layer_obs, compute_hessian, invert_hessian
from pruning.pipeline import tensor_sparsity, load_state_dict, get_prunable_layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug OBS Stats")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--model", type=str, default="mnist")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--calib-batches", type=int, default=8)
    parser.add_argument("--calib-samples", type=int, default=2048)
    parser.add_argument("--lambda", dest="lambd", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading weights from: {args.weights}")
    state_dict = load_state_dict(args.weights)
    model = SimpleCNN()
    try:
        model.load_state_dict(state_dict)
    except Exception:
        pass
    model.to(device)
    model.eval()

    data_cfg = MNISTTrainConfig(
        data_dir=args.data_dir,
        batch_size=128,
        test_batch_size=256,
        num_workers=2,
        device=args.device
    )
    train_loader, _ = get_dataloaders(data_cfg)
    
    pruning_targets = get_prunable_layers(model)
    
    print("\n=== Analyzing OBS Stats (First Layer) ===")
    # Analyze only the first prunable layer to inspect stats
    name, layer = pruning_targets[0] 
    print(f"Layer: {name}")
    
    if hasattr(layer, "weight"):
        activations = collect_calibration_inputs(
            model, train_loader, layer, device,
            max_batches=args.calib_batches,
            max_samples=args.calib_samples
        )
        
        weight_matrix, _ = flatten_weight(layer)
        hessian = compute_hessian(activations, lambd=args.lambd)
        hessian_inv = invert_hessian(hessian)
        
        diag_H = np.diag(hessian)
        diag_Hinv = np.diag(hessian_inv)
        
        obs_scores = (weight_matrix ** 2) / (diag_Hinv.reshape(1, -1) + 1e-10)
        mag_scores = (weight_matrix ** 2)
        
        print(f"Hessian Diag: min={diag_H.min():.2e}, max={diag_H.max():.2e}, mean={diag_H.mean():.2e}")
        print(f"H_inv Diag:   min={diag_Hinv.min():.2e}, max={diag_Hinv.max():.2e}, mean={diag_Hinv.mean():.2e}")
        print(f"OBS Scores:   min={obs_scores.min():.2e}, max={obs_scores.max():.2e}, mean={obs_scores.mean():.2e}")
        print(f"Mag Scores:   min={mag_scores.min():.2e}, max={mag_scores.max():.2e}, mean={mag_scores.mean():.2e}")
        
        # Correlation between OBS and Magnitude scores
        # Flatten to 1D
        obs_flat = obs_scores.flatten()
        mag_flat = mag_scores.flatten()
        correlation = np.corrcoef(obs_flat, mag_flat)[0, 1]
        print(f"\nCorrelation between OBS Scores and Magnitude^2: {correlation:.4f}")
        
        if correlation > 0.9:
            print("-> OBS and Magnitude are highly correlated. Adaptive mask might behave similarly to Magnitude.")
        elif correlation < 0.5:
            print("-> Low correlation. OBS is finding different importance structure.")
        else:
            print("-> Moderate correlation.")

if __name__ == "__main__":
    main()

