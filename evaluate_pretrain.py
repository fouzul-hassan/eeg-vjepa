"""
Enhanced Pretraining Evaluation Script for EEG-VJEPA

This script provides comprehensive evaluation of pretrained models:
1. Loss curves (train/val)
2. Embedding quality metrics (variance, effective rank, collapse detection)
3. **NEW: k-NN classification evaluation**
4. **NEW: Linear probe evaluation**
5. **NEW: Uniformity and Alignment metrics**
6. Visualizations (t-SNE, PCA)

Usage:
    python evaluate_pretrain.py --config configs/pretrain/zuco_pretrain.yaml \
                                --checkpoint output/zuco_pretrain/zuco-jepa-best.pth.tar

    # Or just analyze training logs:
    python evaluate_pretrain.py --log_only --log_file output/zuco_pretrain/zuco-jepa_train.csv
"""

import os
import sys
import copy
import argparse
from pathlib import Path
from collections import defaultdict

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.datasets.data_manager import init_data
from src.masks.multiblock3d import MaskCollator as MB3DMaskCollator
from src.masks.utils import apply_masks
from app.vjepa.utils import init_video_model


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_checkpoint(checkpoint_path, encoder, predictor, target_encoder, device):
    """Load model weights from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    encoder.load_state_dict(checkpoint['encoder'])
    predictor.load_state_dict(checkpoint['predictor'])
    target_encoder.load_state_dict(checkpoint['target_encoder'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', 0)
    print(f"Loaded checkpoint from epoch {epoch}, loss={loss:.4f}")
    
    return epoch, loss


def compute_jepa_loss(clips, encoder, predictor, target_encoder, masks_enc, masks_pred, device, loss_exp=1.0):
    """Compute JEPA loss for a batch."""
    _masks_enc = [m.to(device, non_blocking=True) for m in masks_enc]
    _masks_pred = [m.to(device, non_blocking=True) for m in masks_pred]
    
    with torch.no_grad():
        # Target encoder forward
        h = target_encoder(clips)
        h = F.layer_norm(h, (h.size(-1),))
        h = apply_masks(h, _masks_pred, concat=False)
        
        # Context encoder forward
        z = encoder(clips, _masks_enc)
        z = predictor(z, h, _masks_enc, _masks_pred)
        
        # Compute loss
        loss = 0.
        for zi, hi in zip(z, h):
            loss += torch.mean(torch.abs(zi - hi)**loss_exp) / loss_exp
        loss /= len(_masks_pred)
        
    return float(loss)


def compute_embeddings_with_labels(data_loader, encoder, device, max_batches=100):
    """
    Extract embeddings and labels from the encoder for evaluation.
    
    Returns:
        embeddings: [N, D] tensor
        labels: dict with 'subject' and 'sentence' labels if available
    """
    encoder.eval()
    all_embeddings = []
    all_subjects = []
    all_sent_idx = []
    
    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(data_loader, desc="Extracting embeddings", 
                                            total=min(max_batches, len(data_loader)))):
            if i >= max_batches:
                break
            
            # Handle different batch formats with metadata extraction
            if isinstance(batch_data, tuple):
                batch = batch_data[0]
                # Check if there's metadata in the batch
                if len(batch_data) > 1 and isinstance(batch_data[1], dict):
                    metadata = batch_data[1]
                    if 'subject' in metadata:
                        all_subjects.extend(metadata['subject'])
                    if 'sentence_idx' in metadata:
                        all_sent_idx.extend(metadata['sentence_idx'])
                
                if isinstance(batch, (list, tuple)):
                    if isinstance(batch[0], torch.Tensor):
                        clips = batch[0]
                    else:
                        clips = batch[0][0] if isinstance(batch[0], (list, tuple)) else batch[0]
                else:
                    clips = batch
            else:
                clips = batch_data
            
            if isinstance(clips, list):
                clips = clips[0]
            
            clips = clips.to(device)
            
            # Get embeddings (no masking)
            embeddings = encoder(clips)  # [B, N, D]
            
            # Global average pooling
            embeddings = embeddings.mean(dim=1)  # [B, D]
            all_embeddings.append(embeddings.cpu())
    
    embeddings = torch.cat(all_embeddings, dim=0)
    
    # Create labels dict
    labels = {}
    if all_subjects:
        # Convert subject names to numeric labels
        unique_subjects = list(set(all_subjects))
        subject_to_idx = {s: i for i, s in enumerate(unique_subjects)}
        labels['subject'] = torch.tensor([subject_to_idx[s] for s in all_subjects])
        labels['subject_names'] = unique_subjects
    
    return embeddings, labels


# =============================================================================
# K-NN EVALUATION
# =============================================================================

def knn_evaluate(embeddings, labels, k=5):
    """
    Evaluate embeddings using k-Nearest Neighbors classification.
    
    Args:
        embeddings: [N, D] tensor of embeddings
        labels: [N] tensor of class labels
        k: Number of neighbors
        
    Returns:
        accuracy: k-NN accuracy
    """
    if len(labels) == 0:
        return 0.0
    
    embeddings = F.normalize(embeddings, dim=1)  # L2 normalize
    
    # Compute pairwise similarities
    similarities = torch.mm(embeddings, embeddings.t())  # [N, N]
    
    # For each sample, find k nearest neighbors (excluding itself)
    n_samples = embeddings.shape[0]
    correct = 0
    
    for i in range(n_samples):
        # Get similarities to other samples
        sims = similarities[i].clone()
        sims[i] = -float('inf')  # Exclude self
        
        # Get top-k neighbors
        _, topk_indices = torch.topk(sims, k)
        
        # Majority vote
        neighbor_labels = labels[topk_indices]
        predicted_label = torch.mode(neighbor_labels).values.item()
        
        if predicted_label == labels[i].item():
            correct += 1
    
    accuracy = 100.0 * correct / n_samples
    return accuracy


def knn_evaluation_report(embeddings, labels_dict, k_values=[1, 5, 10, 20]):
    """Run k-NN evaluation for multiple k values."""
    print("\n" + "="*60)
    print("K-NN EVALUATION")
    print("="*60)
    
    results = {}
    
    for label_name, labels in labels_dict.items():
        if label_name == 'subject_names':
            continue
            
        print(f"\n[{label_name.upper()}] ({len(set(labels.numpy()))} classes)")
        
        results[label_name] = {}
        for k in k_values:
            if k > len(labels) - 1:
                continue
            acc = knn_evaluate(embeddings, labels, k=k)
            results[label_name][f'k={k}'] = acc
            print(f"  k={k:2d}: {acc:.2f}%")
    
    return results


# =============================================================================
# LINEAR PROBE EVALUATION
# =============================================================================

class LinearProbe(nn.Module):
    """Simple linear classifier for probe evaluation."""
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)


def train_linear_probe(embeddings, labels, num_classes, epochs=100, lr=0.01, val_split=0.2):
    """
    Train a linear classifier on frozen embeddings.
    
    Args:
        embeddings: [N, D] tensor
        labels: [N] tensor
        num_classes: Number of classes
        epochs: Training epochs
        lr: Learning rate
        val_split: Validation split ratio
        
    Returns:
        train_acc, val_acc, probe
    """
    # Split data
    n_samples = len(labels)
    indices = torch.randperm(n_samples)
    n_val = int(n_samples * val_split)
    
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    train_emb = embeddings[train_indices]
    train_labels = labels[train_indices]
    val_emb = embeddings[val_indices]
    val_labels = labels[val_indices]
    
    # Initialize probe
    embed_dim = embeddings.shape[1]
    probe = LinearProbe(embed_dim, num_classes)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0
    for epoch in range(epochs):
        probe.train()
        optimizer.zero_grad()
        logits = probe(train_emb)
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()
        
        # Validation
        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            probe.eval()
            with torch.no_grad():
                train_pred = probe(train_emb).argmax(dim=1)
                train_acc = 100.0 * (train_pred == train_labels).float().mean().item()
                
                val_pred = probe(val_emb).argmax(dim=1)
                val_acc = 100.0 * (val_pred == val_labels).float().mean().item()
                
                best_val_acc = max(best_val_acc, val_acc)
    
    return train_acc, best_val_acc, probe


def linear_probe_evaluation(embeddings, labels_dict):
    """Run linear probe evaluation for all available labels."""
    print("\n" + "="*60)
    print("LINEAR PROBE EVALUATION")
    print("="*60)
    
    results = {}
    
    for label_name, labels in labels_dict.items():
        if label_name == 'subject_names':
            continue
        
        num_classes = len(set(labels.numpy()))
        print(f"\n[{label_name.upper()}] ({num_classes} classes)")
        
        train_acc, val_acc, _ = train_linear_probe(
            embeddings, labels, num_classes, epochs=100, lr=0.01
        )
        
        results[label_name] = {'train_acc': train_acc, 'val_acc': val_acc}
        print(f"  Train: {train_acc:.2f}%, Val: {val_acc:.2f}%")
    
    return results


# =============================================================================
# UNIFORMITY & ALIGNMENT METRICS
# =============================================================================

def compute_uniformity(embeddings, t=2):
    """
    Compute uniformity loss (Wang & Isola, 2020).
    Lower is better - measures how uniformly distributed embeddings are on the unit sphere.
    
    uniformity = log(E[exp(-t * ||z_i - z_j||^2)])
    """
    embeddings = F.normalize(embeddings, dim=1)
    sq_pdist = torch.pdist(embeddings, p=2).pow(2)
    uniformity = torch.log(torch.exp(-t * sq_pdist).mean())
    return uniformity.item()


def compute_alignment(embeddings, labels, alpha=2):
    """
    Compute alignment loss (Wang & Isola, 2020).
    Lower is better - measures how close embeddings of same class are.
    
    alignment = E[||z_i - z_j||^alpha] for positive pairs (same class)
    """
    embeddings = F.normalize(embeddings, dim=1)
    
    # Find positive pairs (same class)
    alignment_sum = 0
    n_pairs = 0
    
    for label in labels.unique():
        mask = labels == label
        class_embeddings = embeddings[mask]
        if len(class_embeddings) > 1:
            # Compute pairwise distances within class
            dists = torch.pdist(class_embeddings, p=2).pow(alpha)
            alignment_sum += dists.sum().item()
            n_pairs += len(dists)
    
    if n_pairs > 0:
        alignment = alignment_sum / n_pairs
    else:
        alignment = 0
    
    return alignment


def uniformity_alignment_report(embeddings, labels_dict):
    """Compute uniformity and alignment metrics."""
    print("\n" + "="*60)
    print("UNIFORMITY & ALIGNMENT METRICS")
    print("="*60)
    
    uniformity = compute_uniformity(embeddings)
    print(f"\n[Uniformity] (lower is better)")
    print(f"  Score: {uniformity:.4f}")
    
    if uniformity > -1:
        print(f"  ⚠️  WARNING: Low uniformity - embeddings may be clustered")
    else:
        print(f"  ✓ Good uniformity - embeddings are well distributed")
    
    results = {'uniformity': uniformity}
    
    for label_name, labels in labels_dict.items():
        if label_name == 'subject_names':
            continue
        
        alignment = compute_alignment(embeddings, labels)
        results[f'alignment_{label_name}'] = alignment
        
        print(f"\n[Alignment - {label_name}] (lower is better)")
        print(f"  Score: {alignment:.4f}")
    
    return results


# =============================================================================
# EMBEDDING QUALITY ANALYSIS (from original)
# =============================================================================

def analyze_embeddings(embeddings):
    """Analyze embedding quality for collapse detection."""
    print("\n" + "="*60)
    print("EMBEDDING QUALITY ANALYSIS")
    print("="*60)
    
    embeddings_np = embeddings.numpy()
    n_samples, embed_dim = embeddings_np.shape
    
    # 1. Variance per dimension
    var_per_dim = np.var(embeddings_np, axis=0)
    mean_variance = np.mean(var_per_dim)
    std_variance = np.std(var_per_dim)
    
    print(f"\n[Variance Analysis]")
    print(f"  Embedding dimension: {embed_dim}")
    print(f"  Number of samples: {n_samples}")
    print(f"  Mean variance per dim: {mean_variance:.6f}")
    print(f"  Std of variance: {std_variance:.6f}")
    
    # 2. Dead dimensions
    dead_threshold = 1e-6
    dead_dims = np.sum(var_per_dim < dead_threshold)
    print(f"  Dead dimensions: {dead_dims}/{embed_dim} ({100*dead_dims/embed_dim:.1f}%)")
    
    # 3. Effective rank
    try:
        embeddings_centered = embeddings_np - np.mean(embeddings_np, axis=0)
        _, singular_values, _ = np.linalg.svd(embeddings_centered, full_matrices=False)
        singular_values = singular_values / np.sum(singular_values)
        singular_values = singular_values[singular_values > 1e-10]
        effective_rank = np.exp(-np.sum(singular_values * np.log(singular_values + 1e-10)))
        
        print(f"\n[Effective Rank]")
        print(f"  Effective rank: {effective_rank:.1f}/{embed_dim} ({100*effective_rank/embed_dim:.1f}%)")
    except Exception as e:
        print(f"  SVD failed: {e}")
        effective_rank = 0
        singular_values = None
    
    # 4. Cosine similarity
    print(f"\n[Collapse Detection]")
    embeddings_norm = embeddings_np / (np.linalg.norm(embeddings_np, axis=1, keepdims=True) + 1e-8)
    n_pairs = min(1000, n_samples * (n_samples - 1) // 2)
    indices = np.random.choice(n_samples, size=(n_pairs, 2), replace=True)
    cos_sims = np.sum(embeddings_norm[indices[:, 0]] * embeddings_norm[indices[:, 1]], axis=1)
    mean_cos_sim = np.mean(cos_sims)
    
    print(f"  Mean cosine similarity: {mean_cos_sim:.4f}")
    
    if mean_cos_sim > 0.9:
        print(f"  ⚠️  WARNING: REPRESENTATION COLLAPSE detected!")
    elif mean_cos_sim > 0.7:
        print(f"  ⚡ CAUTION: Moderately high similarity")
    else:
        print(f"  ✓ Good: Embeddings show healthy diversity")
    
    return {
        'mean_variance': mean_variance,
        'dead_dims': dead_dims,
        'effective_rank': effective_rank,
        'mean_cos_sim': mean_cos_sim,
        'singular_values': singular_values,
        'var_per_dim': var_per_dim,
        'cos_sims': cos_sims,
    }


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_training_curves(log_file, output_dir):
    """Plot training curves from CSV log."""
    print(f"\nPlotting training curves from {log_file}")
    
    df = pd.read_csv(log_file)
    
    epoch_stats = df.groupby('epoch').agg({
        'loss': ['mean', 'std'],
        'loss-jepa': ['mean', 'std'],
        'reg-loss': ['mean', 'std'],
        'lr': 'last'
    }).reset_index()
    
    epoch_stats.columns = ['epoch', 'loss_mean', 'loss_std', 'jepa_mean', 'jepa_std', 
                           'reg_mean', 'reg_std', 'lr']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Pretraining Curves', fontsize=14, fontweight='bold')
    
    # Total Loss
    ax = axes[0, 0]
    ax.plot(epoch_stats['epoch'], epoch_stats['loss_mean'], 'b-', linewidth=2)
    ax.fill_between(epoch_stats['epoch'], 
                    epoch_stats['loss_mean'] - epoch_stats['loss_std'],
                    epoch_stats['loss_mean'] + epoch_stats['loss_std'], alpha=0.3)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('Total Loss')
    ax.grid(True, alpha=0.3)
    
    # JEPA Loss
    ax = axes[0, 1]
    ax.plot(epoch_stats['epoch'], epoch_stats['jepa_mean'], 'g-', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('JEPA Loss'); ax.set_title('JEPA Prediction Loss')
    ax.grid(True, alpha=0.3)
    
    # Reg Loss
    ax = axes[1, 0]
    ax.plot(epoch_stats['epoch'], epoch_stats['reg_mean'], 'r-', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Reg Loss'); ax.set_title('Regularization Loss')
    ax.grid(True, alpha=0.3)
    
    # LR
    ax = axes[1, 1]
    ax.plot(epoch_stats['epoch'], epoch_stats['lr'], 'purple', linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('LR'); ax.set_title('Learning Rate')
    ax.set_yscale('log'); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'pretraining_curves.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    return epoch_stats


def plot_evaluation_results(analysis_results, knn_results, probe_results, embeddings, output_dir):
    """Plot comprehensive evaluation results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Pretraining Evaluation Results', fontsize=14, fontweight='bold')
    
    # 1. Variance histogram
    ax = axes[0, 0]
    ax.hist(analysis_results['var_per_dim'], bins=50, color='steelblue', alpha=0.7)
    ax.axvline(np.mean(analysis_results['var_per_dim']), color='red', linestyle='--')
    ax.set_xlabel('Variance'); ax.set_ylabel('Count'); ax.set_title('Variance per Dimension')
    ax.grid(True, alpha=0.3)
    
    # 2. Cosine similarity histogram
    ax = axes[0, 1]
    ax.hist(analysis_results['cos_sims'], bins=50, color='coral', alpha=0.7)
    ax.axvline(analysis_results['mean_cos_sim'], color='red', linestyle='--')
    ax.set_xlabel('Cosine Similarity'); ax.set_ylabel('Count')
    ax.set_title('Pairwise Cosine Similarity')
    ax.grid(True, alpha=0.3)
    
    # 3. Singular values
    ax = axes[0, 2]
    if analysis_results['singular_values'] is not None:
        sv = analysis_results['singular_values'][:min(50, len(analysis_results['singular_values']))]
        ax.bar(range(len(sv)), sv, color='teal', alpha=0.7)
        ax.set_xlabel('Index'); ax.set_ylabel('Singular Value')
        ax.set_title(f'Top Singular Values (Eff. Rank: {analysis_results["effective_rank"]:.1f})')
    ax.grid(True, alpha=0.3)
    
    # 4. k-NN accuracy
    ax = axes[1, 0]
    if knn_results:
        for label_name, results in knn_results.items():
            k_vals = [int(k.split('=')[1]) for k in results.keys()]
            accs = list(results.values())
            ax.plot(k_vals, accs, marker='o', label=label_name)
        ax.set_xlabel('k'); ax.set_ylabel('Accuracy (%)')
        ax.set_title('k-NN Classification'); ax.legend(); ax.grid(True, alpha=0.3)
    
    # 5. Linear probe accuracy
    ax = axes[1, 1]
    if probe_results:
        labels = list(probe_results.keys())
        train_accs = [r['train_acc'] for r in probe_results.values()]
        val_accs = [r['val_acc'] for r in probe_results.values()]
        x = np.arange(len(labels))
        ax.bar(x - 0.2, train_accs, 0.4, label='Train', color='steelblue')
        ax.bar(x + 0.2, val_accs, 0.4, label='Val', color='coral')
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel('Accuracy (%)'); ax.set_title('Linear Probe'); ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 6. PCA visualization
    ax = axes[1, 2]
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings.numpy())
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5, s=10)
        ax.set_xlabel(f'PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)')
        ax.set_ylabel(f'PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)')
        ax.set_title('PCA of Embeddings')
    except ImportError:
        ax.text(0.5, 0.5, 'sklearn not installed', ha='center', va='center')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'evaluation_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def print_summary(analysis_results, knn_results=None, probe_results=None, ua_results=None):
    """Print comprehensive evaluation summary."""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    print(f"\n[Representation Quality]")
    print(f"  Effective Rank: {analysis_results['effective_rank']:.1f}")
    print(f"  Mean Cosine Sim: {analysis_results['mean_cos_sim']:.4f}")
    print(f"  Dead Dimensions: {analysis_results['dead_dims']}")
    
    if knn_results:
        print(f"\n[k-NN Accuracy]")
        for label, results in knn_results.items():
            best_acc = max(results.values())
            print(f"  {label}: {best_acc:.2f}%")
    
    if probe_results:
        print(f"\n[Linear Probe Accuracy]")
        for label, results in probe_results.items():
            print(f"  {label}: Train={results['train_acc']:.2f}%, Val={results['val_acc']:.2f}%")
    
    if ua_results:
        print(f"\n[Uniformity/Alignment]")
        print(f"  Uniformity: {ua_results['uniformity']:.4f}")
    
    # Quality verdict
    print(f"\n[Overall Assessment]")
    issues = []
    
    if analysis_results['mean_cos_sim'] > 0.9:
        issues.append("COLLAPSE: Very high cosine similarity")
    if analysis_results['effective_rank'] < 20:
        issues.append("LOW RANK: Few dimensions used")
    if analysis_results['dead_dims'] > analysis_results['effective_rank'] * 0.3:
        issues.append("DEAD DIMS: Many unused dimensions")
    
    if knn_results:
        for label, results in knn_results.items():
            if max(results.values()) < 30:
                issues.append(f"LOW k-NN: Poor {label} separation")
    
    if issues:
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print(f"  ✓ Model appears to be learning useful representations!")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate EEG-VJEPA Pretraining')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--log_only', action='store_true', help='Only plot training logs')
    parser.add_argument('--log_file', type=str, help='Path to training log CSV')
    parser.add_argument('--output_dir', type=str, default='./eval_output', help='Output directory')
    parser.add_argument('--max_batches', type=int, default=50, help='Max batches for embedding extraction')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Log-only mode
    if args.log_only and args.log_file:
        plot_training_curves(args.log_file, args.output_dir)
        print("\nDone!")
        return
    
    # Full evaluation
    if not args.config or not args.checkpoint:
        parser.error("--config and --checkpoint are required for full evaluation")
    
    config = load_config(args.config)
    
    # Plot training curves
    log_dir = config.get('logging', {}).get('folder', './output')
    tag = config.get('logging', {}).get('write_tag', 'jepa')
    log_file = os.path.join(log_dir, f'{tag}_train.csv')
    
    if os.path.exists(log_file):
        plot_training_curves(log_file, args.output_dir)
    
    # Load model
    cfgs_data = config.get('data', {})
    cfgs_model = config.get('model', {})
    cfgs_mask = config.get('mask', [])
    cfgs_meta = config.get('meta', {})
    
    num_frames = cfgs_data.get('num_frames', 28)
    tubelet_size = cfgs_data.get('tubelet_size', 4)
    crop_size = tuple(cfgs_data.get('crop_size', [105, 33]))
    patch_size = tuple(cfgs_data.get('patch_size', [7, 8]))
    dataset_type = cfgs_data.get('dataset_type', 'EEGDataset')
    dataset_paths = cfgs_data.get('datasets', [])
    batch_size = cfgs_data.get('batch_size', 16)
    
    model_name = cfgs_model.get('model_name', 'vit_small')
    in_chans = cfgs_model.get('in_chans', 1)
    pred_depth = cfgs_model.get('pred_depth', 6)
    pred_embed_dim = cfgs_model.get('pred_embed_dim', 384)
    
    print("\nInitializing model...")
    encoder, predictor = init_video_model(
        device=device,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_embed_dim=pred_embed_dim,
        uniform_power=cfgs_model.get('uniform_power', True),
        use_mask_tokens=cfgs_model.get('use_mask_tokens', True),
        num_mask_tokens=len(cfgs_mask),
        zero_init_mask_tokens=cfgs_model.get('zero_init_mask_tokens', True),
        use_sdpa=cfgs_meta.get('use_sdpa', False),
        in_chans=in_chans,
    )
    target_encoder = copy.deepcopy(encoder)
    
    load_checkpoint(args.checkpoint, encoder, predictor, target_encoder, device)
    encoder.eval()
    
    # Load data and extract embeddings
    print("\nLoading data...")
    data_loader, _ = init_data(
        data=dataset_type,
        root_path=dataset_paths,
        batch_size=batch_size,
        num_workers=2,
        world_size=1,
        rank=0,
        pin_mem=True,
        collator=None,
        drop_last=False,
    )
    
    embeddings, labels_dict = compute_embeddings_with_labels(
        data_loader, encoder, device, max_batches=args.max_batches
    )
    
    # Run evaluations
    analysis_results = analyze_embeddings(embeddings)
    
    knn_results = {}
    probe_results = {}
    ua_results = {}
    
    if labels_dict:
        knn_results = knn_evaluation_report(embeddings, labels_dict)
        probe_results = linear_probe_evaluation(embeddings, labels_dict)
        ua_results = uniformity_alignment_report(embeddings, labels_dict)
    
    # Plot results
    plot_evaluation_results(analysis_results, knn_results, probe_results, embeddings, args.output_dir)
    
    # Print summary
    print_summary(analysis_results, knn_results, probe_results, ua_results)
    
    print(f"\n✓ Evaluation complete! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
