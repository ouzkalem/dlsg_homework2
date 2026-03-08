"""
Model definitions for graph classification.

Feature-Only Models (no message passing):
  - LinearBaseline: Global pool -> Linear
  - MLPBaseline: Global pool -> MLP
  - DeepSets: Per-node MLP -> Global pool -> MLP

Message Passing Models:
  - GCN: Graph Convolutional Network
  - GraphSAGE: GraphSAGE
  - GAT: Graph Attention Network
  - GIN: Graph Isomorphism Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, SAGEConv, GATConv, GINConv,
    global_mean_pool, global_add_pool, global_max_pool
)


def get_pooling(pool_type):
    """Get the pooling function by name."""
    if pool_type == 'mean':
        return global_mean_pool
    elif pool_type == 'sum':
        return global_add_pool
    elif pool_type == 'max':
        return global_max_pool
    else:
        raise ValueError(f"Unknown pooling: {pool_type}")


# ============================================================================
#  Feature-Only Models
# ============================================================================

class LinearBaseline(nn.Module):
    """
    Linear Baseline: Global pool node features -> Linear classifier.
    No hidden layers, no message passing.
    """
    
    def __init__(self, num_features, num_classes, hidden_dim=64, num_layers=4,
                 dropout=0.5, pool='mean', **kwargs):
        super().__init__()
        self.pool = get_pooling(pool)
        self.classifier = nn.Linear(num_features, num_classes)
    
    def forward(self, x, edge_index, batch):
        # Global pooling directly on input features
        x = self.pool(x, batch)
        x = self.classifier(x)
        return x


class MLPBaseline(nn.Module):
    """
    MLP Baseline: Global pool node features -> MLP classifier.
    Uses hidden layers but no message passing.
    """
    
    def __init__(self, num_features, num_classes, hidden_dim=64, num_layers=4,
                 dropout=0.5, pool='mean', **kwargs):
        super().__init__()
        self.pool = get_pooling(pool)
        self.dropout = dropout
        
        layers = []
        in_dim = num_features
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, edge_index, batch):
        # Global pooling first (ignoring structure)
        x = self.pool(x, batch)
        x = self.mlp(x)
        x = self.classifier(x)
        return x


class DeepSets(nn.Module):
    """
    DeepSets: Per-node MLP -> Global pool -> Classifier MLP.
    Processes each node independently, then aggregates.
    Permutation invariant but ignores graph structure.
    """
    
    def __init__(self, num_features, num_classes, hidden_dim=64, num_layers=4,
                 dropout=0.5, pool='mean', **kwargs):
        super().__init__()
        self.pool = get_pooling(pool)
        self.dropout = dropout
        
        # Per-node encoder (phi)
        enc_layers = []
        in_dim = num_features
        for i in range(num_layers // 2):
            enc_layers.append(nn.Linear(in_dim, hidden_dim))
            enc_layers.append(nn.BatchNorm1d(hidden_dim))
            enc_layers.append(nn.ReLU())
            enc_layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*enc_layers)
        
        # Post-aggregation network (rho)
        dec_layers = []
        for i in range(num_layers - num_layers // 2 - 1):
            dec_layers.append(nn.Linear(hidden_dim, hidden_dim))
            dec_layers.append(nn.BatchNorm1d(hidden_dim))
            dec_layers.append(nn.ReLU())
            dec_layers.append(nn.Dropout(dropout))
        self.decoder = nn.Sequential(*dec_layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, edge_index, batch):
        # Per-node transformation (ignoring edges)
        x = self.encoder(x)
        # Aggregate
        x = self.pool(x, batch)
        # Post-aggregation
        x = self.decoder(x)
        x = self.classifier(x)
        return x


# ============================================================================
#  Message Passing Models
# ============================================================================

class GCN(nn.Module):
    """Graph Convolutional Network for graph classification."""
    
    def __init__(self, num_features, num_classes, hidden_dim=64, num_layers=4,
                 dropout=0.5, pool='mean', **kwargs):
        super().__init__()
        self.pool = get_pooling(pool)
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.convs.append(GCNConv(num_features, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.pool(x, batch)
        x = self.classifier(x)
        return x


class GraphSAGE(nn.Module):
    """GraphSAGE for graph classification."""
    
    def __init__(self, num_features, num_classes, hidden_dim=64, num_layers=4,
                 dropout=0.5, pool='mean', **kwargs):
        super().__init__()
        self.pool = get_pooling(pool)
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.convs.append(SAGEConv(num_features, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.pool(x, batch)
        x = self.classifier(x)
        return x


class GAT(nn.Module):
    """Graph Attention Network for graph classification."""
    
    def __init__(self, num_features, num_classes, hidden_dim=64, num_layers=4,
                 dropout=0.5, pool='mean', heads=4, **kwargs):
        super().__init__()
        self.pool = get_pooling(pool)
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(num_features, hidden_dim // heads, heads=heads, dropout=dropout))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Last conv layer (single head)
        self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.pool(x, batch)
        x = self.classifier(x)
        return x


class GIN(nn.Module):
    """Graph Isomorphism Network for graph classification."""
    
    def __init__(self, num_features, num_classes, hidden_dim=64, num_layers=4,
                 dropout=0.5, pool='mean', **kwargs):
        super().__init__()
        self.pool = get_pooling(pool)
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First GIN layer
        mlp = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(mlp))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Remaining GIN layers
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.pool(x, batch)
        x = self.classifier(x)
        return x


# ============================================================================
#  Model Registry
# ============================================================================

MODEL_REGISTRY = {
    # Feature-Only Models
    'linear': LinearBaseline,
    'mlp': MLPBaseline,
    'deepsets': DeepSets,
    # Message Passing Models
    'gcn': GCN,
    'graphsage': GraphSAGE,
    'gat': GAT,
    'gin': GIN,
}


def get_model(name, **kwargs):
    """Create a model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)
