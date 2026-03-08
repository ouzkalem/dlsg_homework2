"""
Utility functions for the GNN graph classification experiments.
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving."""
    
    def __init__(self, patience=20, mode='max', min_delta=0.0):
        """
        Args:
            patience: Number of epochs to wait after last improvement.
            mode: 'min' for loss, 'max' for accuracy/ROC-AUC.
            min_delta: Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False


class ExperimentLogger:
    """Logger for tracking experiment metrics per epoch."""
    
    def __init__(self):
        self.history = defaultdict(list)
    
    def log(self, epoch, **kwargs):
        """Log metrics for an epoch."""
        self.history['epoch'].append(epoch)
        for key, value in kwargs.items():
            self.history[key].append(value)
    
    def get_best(self, metric, mode='max'):
        """Get the best value and epoch for a given metric."""
        values = self.history[metric]
        if mode == 'max':
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        return values[best_idx], self.history['epoch'][best_idx]
    
    def plot_curves(self, save_path=None, dataset_name='', model_name='', seed=0):
        """Plot training and validation curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = self.history['epoch']
        
        # Loss plot
        axes[0].plot(epochs, self.history['train_loss'], label='Train Loss', color='tab:blue')
        axes[0].plot(epochs, self.history['val_loss'], label='Val Loss', color='tab:orange')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{dataset_name} - {model_name} (Seed {seed}) - Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Metric plot
        metric_key = 'val_rocauc' if 'val_rocauc' in self.history else 'val_acc'
        metric_name = 'ROC-AUC' if metric_key == 'val_rocauc' else 'Accuracy'
        axes[1].plot(epochs, self.history[metric_key], label=f'Val {metric_name}', color='tab:green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel(metric_name)
        axes[1].set_title(f'{dataset_name} - {model_name} (Seed {seed}) - {metric_name}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()


def print_config(dataset_name, model_name, pool, hidden, layers, dropout, lr, batch_size, seeds):
    """Print experiment configuration in a formatted way."""
    print("=" * 60)
    print(f"  Dataset : {dataset_name}")
    print(f"  Model   : {model_name.upper()} | Pool: {pool}")
    print(f"  Hidden  : {hidden}  | Layers: {layers}  | Dropout: {dropout}")
    print(f"  LR      : {lr}   | BS: {batch_size}")
    print(f"  Seeds   : {seeds}")
    print("=" * 60)


def format_results(results_dict):
    """Format results dictionary as mean ± std string."""
    formatted = {}
    for key, values in results_dict.items():
        mean = np.mean(values)
        std = np.std(values)
        formatted[key] = f"{mean:.4f} ± {std:.4f}"
    return formatted
