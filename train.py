"""
Training and evaluation functions for graph classification.
"""

import time
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from utils.utilities import EarlyStopping, ExperimentLogger


def train_epoch(model, loader, optimizer, device, task_type='multiclass'):
    """
    Train for one epoch.
    
    Returns:
        avg_loss: Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index, data.batch)
        
        if task_type == 'binary':
            # Binary classification (ogbg-molhiv)
            target = data.y.float().view(-1)
            # Handle potential NaN labels
            is_valid = ~torch.isnan(target)
            if is_valid.sum() == 0:
                continue
            loss = F.binary_cross_entropy_with_logits(
                out[is_valid, 1] if out.shape[1] > 1 else out[is_valid].view(-1),
                target[is_valid]
            )
        else:
            # Multi-class classification
            target = data.y.long().view(-1)
            loss = F.cross_entropy(out, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, task_type='multiclass', metric='accuracy'):
    """
    Evaluate model on a data loader.
    
    Returns:
        avg_loss: Average loss
        metric_value: ROC-AUC or Accuracy depending on task
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        
        if task_type == 'binary':
            target = data.y.float().view(-1)
            is_valid = ~torch.isnan(target)
            if is_valid.sum() == 0:
                continue
            pred_score = out[is_valid, 1] if out.shape[1] > 1 else out[is_valid].view(-1)
            loss = F.binary_cross_entropy_with_logits(pred_score, target[is_valid])
            
            all_probs.append(torch.sigmoid(pred_score).cpu().numpy())
            all_labels.append(target[is_valid].cpu().numpy())
        else:
            target = data.y.long().view(-1)
            loss = F.cross_entropy(out, target)
            
            pred = out.argmax(dim=1)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(target.cpu().numpy())
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    
    if metric == 'rocauc':
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)
        try:
            metric_value = roc_auc_score(all_labels, all_probs)
        except ValueError:
            metric_value = 0.0
    else:
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        metric_value = accuracy_score(all_labels, all_preds)
    
    return avg_loss, metric_value


def train_and_evaluate(model, train_loader, val_loader, test_loader,
                       device, task_type='multiclass', metric='accuracy',
                       epochs=100, lr=0.001, patience=20, seed=0,
                       verbose=True):
    """
    Full training loop with early stopping.
    
    Args:
        model: PyTorch model
        train_loader, val_loader, test_loader: DataLoaders
        device: Device to use
        task_type: 'binary' or 'multiclass'
        metric: 'rocauc' or 'accuracy'
        epochs: Maximum number of epochs
        lr: Learning rate
        patience: Early stopping patience
        seed: Random seed (for display)
        verbose: Whether to print per-epoch logs
        
    Returns:
        best_val_metric: Best validation metric value
        test_metric: Test metric at best validation epoch
        logger: ExperimentLogger with full training history
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=patience, mode='max')
    logger = ExperimentLogger()
    
    best_val_metric = 0.0
    best_model_state = None
    start_time = time.time()
    
    metric_name = 'ROC-AUC' if metric == 'rocauc' else 'Accuracy'
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, task_type)
        
        # Evaluate
        val_loss, val_metric = evaluate(model, val_loader, device, task_type, metric)
        
        elapsed = time.time() - start_time
        
        # Log
        log_kwargs = {
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        if metric == 'rocauc':
            log_kwargs['val_rocauc'] = val_metric
        else:
            log_kwargs['val_acc'] = val_metric
        logger.log(epoch, **log_kwargs)
        
        # Track best
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d} | tr_loss {train_loss:.4f} | "
                  f"val_loss {val_loss:.4f} | val_{metric} {val_metric:.4f} | "
                  f"{elapsed:.1f}s")
        
        # Early stopping
        if early_stopping(val_metric, epoch):
            if verbose:
                print(f"  Early stop at epoch {epoch}.")
            break
    
    # Load best model and evaluate on test set
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    test_loss, test_metric = evaluate(model, test_loader, device, task_type, metric)
    
    if verbose:
        print(f"  Best val {metric_name}: {best_val_metric:.4f} "
              f"(epoch {early_stopping.best_epoch})")
        print(f"  Test {metric_name}: {test_metric:.4f}")
    
    return best_val_metric, test_metric, logger
