"""
Main entry point for GNN graph classification experiments.

Usage:
    python main.py --dataset PROTEINS --model gcn --pool mean --hidden 64 --layers 4 --seeds 0 1 2

This script:
1. Loads the specified dataset
2. Creates the specified model
3. Trains with early stopping across multiple seeds
4. Reports mean ± std results
"""

import argparse
import time
import json
import os
import torch
import numpy as np

from data import get_dataset, get_data_loaders, print_dataset_info
from models import get_model, MODEL_REGISTRY, count_parameters
from train import train_and_evaluate
from utils.utilities import set_seed, get_device, print_config, format_results, count_parameters


def parse_args():
    parser = argparse.ArgumentParser(description='GNN Graph Classification Experiments')
    
    # Dataset
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['ogbg-molhiv', 'PROTEINS', 'IMDB-MULTI', 'REDDIT-BINARY'],
                        help='Dataset name')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for datasets')
    
    # Model
    parser.add_argument('--model', type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Model architecture')
    parser.add_argument('--pool', type=str, default='mean',
                        choices=['mean', 'sum', 'max'],
                        help='Global pooling method')
    
    # Architecture
    parser.add_argument('--hidden', type=int, default=64,
                        help='Hidden dimension')
    parser.add_argument('--layers', type=int, default=4,
                        help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    
    # Training
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2],
                        help='Random seeds to run')
    
    # Output
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--save_plots', action='store_true',
                        help='Save training curve plots')
    
    return parser.parse_args()


def run_experiment(args):
    """Run the full experiment for all seeds."""
    device = get_device()
    
    # Print configuration
    print_config(
        dataset_name=args.dataset,
        model_name=args.model,
        pool=args.pool,
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        seeds=args.seeds
    )
    
    # Load dataset
    print(f"\nDownloading/loading {args.dataset}...")
    dataset, num_features, num_classes, task_type, metric = get_dataset(
        args.dataset, root=args.data_root
    )
    
    # Results storage
    val_metrics = []
    test_metrics = []
    
    for seed in args.seeds:
        print(f"\n{'─'*50}")
        print(f"Model: {args.model.upper()} | Pool: {args.pool} | "
              f"Params: {{params}} | Seed: {seed} | Device: {device}")
        print(f"{'─'*50}")
        
        # Set seed
        set_seed(seed)
        
        # Get data loaders
        train_loader, val_loader, test_loader = get_data_loaders(
            dataset, args.dataset, batch_size=args.batch_size, seed=seed
        )
        
        # Print dataset info (only first seed)
        if seed == args.seeds[0]:
            print_dataset_info(args.dataset, dataset, train_loader, val_loader, test_loader)
        
        # Create model
        model = get_model(
            args.model,
            num_features=num_features,
            num_classes=num_classes,
            hidden_dim=args.hidden,
            num_layers=args.layers,
            dropout=args.dropout,
            pool=args.pool
        ).to(device)
        
        params = count_parameters(model)
        print(f"Model: {args.model.upper()} | Pool: {args.pool} | "
              f"Params: {params:,} | Seed: {seed} | Device: {device}")
        
        # Train and evaluate
        best_val, test_metric, logger = train_and_evaluate(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            task_type=task_type,
            metric=metric,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            seed=seed,
            verbose=True
        )
        
        val_metrics.append(best_val)
        test_metrics.append(test_metric)
        
        # Save plot
        if args.save_plots:
            plot_path = os.path.join(
                args.save_dir, 'plots',
                f"{args.dataset}_{args.model}_{args.pool}_seed{seed}.png"
            )
            logger.plot_curves(
                save_path=plot_path,
                dataset_name=args.dataset,
                model_name=args.model.upper(),
                seed=seed
            )
    
    # Print summary
    metric_name = 'ROC-AUC' if metric == 'rocauc' else 'Accuracy'
    print(f"\n{'='*60}")
    print(f"RESULTS: {args.dataset} | {args.model.upper()} | Pool: {args.pool}")
    print(f"{'='*60}")
    print(f"  Val {metric_name}:  {np.mean(val_metrics):.4f} ± {np.std(val_metrics):.4f}")
    print(f"  Test {metric_name}: {np.mean(test_metrics):.4f} ± {np.std(test_metrics):.4f}")
    print(f"  Per-seed val:  {[f'{v:.4f}' for v in val_metrics]}")
    print(f"  Per-seed test: {[f'{v:.4f}' for v in test_metrics]}")
    print(f"{'='*60}")
    
    # Save results to JSON
    os.makedirs(args.save_dir, exist_ok=True)
    result = {
        'dataset': args.dataset,
        'model': args.model,
        'pool': args.pool,
        'hidden': args.hidden,
        'layers': args.layers,
        'dropout': args.dropout,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'seeds': args.seeds,
        'params': params,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'val_mean': float(np.mean(val_metrics)),
        'val_std': float(np.std(val_metrics)),
        'test_mean': float(np.mean(test_metrics)),
        'test_std': float(np.std(test_metrics)),
        'metric': metric,
    }
    
    result_file = os.path.join(
        args.save_dir, f"{args.dataset}_{args.model}_{args.pool}.json"
    )
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {result_file}")
    
    return result


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)
