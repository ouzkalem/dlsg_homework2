"""
Data loading and preprocessing for graph classification datasets.

Handles:
- ogbg-molhiv (OGB molecular dataset)
- PROTEINS, IMDB-MULTI, REDDIT-BINARY (TUDatasets)
"""

import torch
import numpy as np
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform
from ogb.graphproppred import PygGraphPropPredDataset


class AddDegreeFeature(BaseTransform):
    """Add node degree as feature for datasets without node features."""
    
    def __call__(self, data):
        if data.x is None:
            # Use degree as node feature
            deg = torch.zeros(data.num_nodes, dtype=torch.long)
            row, col = data.edge_index
            for node in range(data.num_nodes):
                deg[node] = (row == node).sum()
            # One-hot encode degree (cap at max_degree)
            max_degree = 100
            deg = deg.clamp(max=max_degree)
            data.x = torch.zeros(data.num_nodes, max_degree + 1)
            data.x.scatter_(1, deg.unsqueeze(1), 1.0)
        elif data.x.dtype == torch.long:
            # If features are categorical integers, convert to float
            data.x = data.x.float()
        return data


class AddConstantFeature(BaseTransform):
    """Add constant feature (all ones) for datasets without node features."""
    
    def __call__(self, data):
        if data.x is None:
            data.x = torch.ones(data.num_nodes, 1)
        elif data.x.dtype == torch.long:
            data.x = data.x.float()
        return data


def get_dataset(name, root='./data'):
    """
    Load a graph classification dataset.
    
    Args:
        name: Dataset name ('ogbg-molhiv', 'PROTEINS', 'IMDB-MULTI', 'REDDIT-BINARY')
        root: Root directory for data storage
        
    Returns:
        dataset: PyG dataset object
        num_features: Number of node features
        num_classes: Number of output classes
        task_type: 'binary' for ogbg-molhiv, 'multiclass' for others
        metric: 'rocauc' for ogbg-molhiv, 'accuracy' for others
    """
    if name == 'ogbg-molhiv':
        dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root=root)
        num_features = dataset[0].x.shape[1]
        num_classes = 2  # Binary classification
        task_type = 'binary'
        metric = 'rocauc'
    elif name in ['PROTEINS', 'IMDB-MULTI', 'REDDIT-BINARY']:
        # Use degree features for datasets without node features
        transform = AddDegreeFeature()
        dataset = TUDataset(root=root, name=name, transform=transform)
        
        # Get feature dim from first sample
        sample = dataset[0]
        num_features = sample.x.shape[1]
        num_classes = dataset.num_classes
        task_type = 'multiclass'
        metric = 'accuracy'
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    return dataset, num_features, num_classes, task_type, metric


def get_data_loaders(dataset, name, batch_size=64, seed=0):
    """
    Create train/val/test data loaders.
    
    For ogbg-molhiv: uses official scaffold split.
    For TUDatasets: uses random 80/10/10 split.
    
    Args:
        dataset: PyG dataset
        name: Dataset name
        batch_size: Batch size for DataLoader
        seed: Random seed for splitting
        
    Returns:
        train_loader, val_loader, test_loader
    """
    if name == 'ogbg-molhiv':
        # Use official OGB split
        split_idx = dataset.get_idx_split()
        train_loader = DataLoader(dataset[split_idx['train']], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset[split_idx['valid']], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset[split_idx['test']], batch_size=batch_size, shuffle=False)
    else:
        # Random split for TU datasets
        torch.manual_seed(seed)
        num_graphs = len(dataset)
        indices = torch.randperm(num_graphs)
        
        train_size = int(0.8 * num_graphs)
        val_size = int(0.1 * num_graphs)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        train_loader = DataLoader(dataset[train_idx], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset[val_idx], batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset[test_idx], batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def print_dataset_info(name, dataset, train_loader, val_loader, test_loader):
    """Print dataset statistics."""
    sample = dataset[0]
    print(f"\n{'='*50}")
    print(f"Dataset: {name}")
    print(f"{'='*50}")
    print(f"  Number of graphs    : {len(dataset)}")
    print(f"  Number of features  : {sample.x.shape[1]}")
    print(f"  Number of classes   : {dataset.num_classes if hasattr(dataset, 'num_classes') else 2}")
    print(f"  Train / Val / Test  : {len(train_loader.dataset)} / {len(val_loader.dataset)} / {len(test_loader.dataset)}")
    
    # Graph statistics
    num_nodes = [d.num_nodes for d in dataset]
    num_edges = [d.num_edges for d in dataset]
    print(f"  Avg. nodes per graph: {np.mean(num_nodes):.1f} ± {np.std(num_nodes):.1f}")
    print(f"  Avg. edges per graph: {np.mean(num_edges):.1f} ± {np.std(num_edges):.1f}")
    print(f"{'='*50}\n")
