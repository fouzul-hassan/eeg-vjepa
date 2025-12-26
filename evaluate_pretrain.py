"""
Pretraining Evaluation Script for EEG-VJEPA

This standalone script evaluates the quality of pretrained models by:
1. Computing train/val loss curves (overfitting detection)
2. Analyzing embedding quality (variance, effective rank, collapse detection)
3. Visualizing embeddings (t-SNE/PCA)
4. Plotting training curves from CSV logs

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

import yaml
import numpy as np
import torch
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


def compute_embeddings(data_loader, encoder, device, max_batches=50):
    """Extract embeddings from the encoder for analysis."""
    encoder.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(data_loader, desc="Extracting embeddings", total=min(max_batches, len(data_loader)))):
            if i >= max_batches:
                break
            
            # Handle different batch formats
            if isinstance(batch_data, tuple):
                batch = batch_data[0]
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
    
    return torch.cat(all_embeddings, dim=0)


def analyze_embeddings(embeddings):
    """Analyze embedding quality for collapse detection."""
    print("\n" + "="*60)
    print("EMBEDDING QUALITY ANALYSIS")
    print("="*60)
    
    # Compute statistics
    embeddings = embeddings.numpy()
    n_samples, embed_dim = embeddings.shape
    
    # 1. Variance per dimension
    var_per_dim = np.var(embeddings, axis=0)
    mean_variance = np.mean(var_per_dim)
    std_variance = np.std(var_per_dim)
    
    print(f"\n[Variance Analysis]")
    print(f"  Embedding dimension: {embed_dim}")
    print(f"  Number of samples: {n_samples}")
    print(f"  Mean variance per dim: {mean_variance:.6f}")
    print(f"  Std of variance: {std_variance:.6f}")
    print(f"  Min variance: {np.min(var_per_dim):.6f}")
    print(f"  Max variance: {np.max(var_per_dim):.6f}")
    
    # 2. Dead dimensions (near zero variance)
    dead_threshold = 1e-6
    dead_dims = np.sum(var_per_dim < dead_threshold)
    print(f"  Dead dimensions (<{dead_threshold}): {dead_dims}/{embed_dim} ({100*dead_dims/embed_dim:.1f}%)")
    
    # 3. Effective rank (via SVD)
    try:
        # Center the embeddings
        embeddings_centered = embeddings - np.mean(embeddings, axis=0)
        _, singular_values, _ = np.linalg.svd(embeddings_centered, full_matrices=False)
        
        # Normalize singular values
        singular_values = singular_values / np.sum(singular_values)
        
        # Compute effective rank (entropy-based)
        singular_values = singular_values[singular_values > 1e-10]  # Filter near-zero
        effective_rank = np.exp(-np.sum(singular_values * np.log(singular_values + 1e-10)))
        
        print(f"\n[Effective Rank Analysis]")
        print(f"  Effective rank: {effective_rank:.1f}/{embed_dim} ({100*effective_rank/embed_dim:.1f}%)")
        print(f"  Top 10 singular values: {singular_values[:10]}")
        
        # Cumulative explained variance
        cumsum = np.cumsum(singular_values**2)
        dims_90 = np.searchsorted(cumsum, 0.9) + 1
        dims_99 = np.searchsorted(cumsum, 0.99) + 1
        print(f"  Dims for 90% variance: {dims_90}")
        print(f"  Dims for 99% variance: {dims_99}")
        
    except Exception as e:
        print(f"  SVD failed: {e}")
        effective_rank = 0
        singular_values = None
    
    # 4. Cosine similarity collapse check
    print(f"\n[Collapse Detection]")
    # Sample random pairs
    n_pairs = min(1000, n_samples * (n_samples - 1) // 2)
    indices = np.random.choice(n_samples, size=(n_pairs, 2), replace=True)
    
    # Compute cosine similarities
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    cos_sims = np.sum(embeddings_norm[indices[:, 0]] * embeddings_norm[indices[:, 1]], axis=1)
    
    mean_cos_sim = np.mean(cos_sims)
    std_cos_sim = np.std(cos_sims)
    
    print(f"  Mean cosine similarity: {mean_cos_sim:.4f}")
    print(f"  Std cosine similarity: {std_cos_sim:.4f}")
    
    # Collapse warning
    if mean_cos_sim > 0.9:
        print(f"  ⚠️  WARNING: High cosine similarity suggests REPRESENTATION COLLAPSE!")
    elif mean_cos_sim > 0.7:
        print(f"  ⚡ CAUTION: Moderately high similarity, monitor for collapse.")
    else:
        print(f"  ✓ Good: Embeddings show healthy diversity.")
    
    return {
        'mean_variance': mean_variance,
        'dead_dims': dead_dims,
        'effective_rank': effective_rank,
        'mean_cos_sim': mean_cos_sim,
        'std_cos_sim': std_cos_sim,
        'singular_values': singular_values,
        'var_per_dim': var_per_dim,
        'cos_sims': cos_sims,
    }


def plot_training_curves(log_file, output_dir):
    """Plot training curves from CSV log."""
    print(f"\nPlotting training curves from {log_file}")
    
    df = pd.read_csv(log_file)
    
    # Group by epoch
    epoch_stats = df.groupby('epoch').agg({
        'loss': ['mean', 'std'],
        'loss-jepa': ['mean', 'std'],
        'reg-loss': ['mean', 'std'],
        'lr': 'last'
    }).reset_index()
    
    # Flatten column names
    epoch_stats.columns = ['epoch', 'loss_mean', 'loss_std', 'jepa_mean', 'jepa_std', 
                           'reg_mean', 'reg_std', 'lr']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Pretraining Curves', fontsize=14, fontweight='bold')
    
    # Total Loss
    ax = axes[0, 0]
    ax.plot(epoch_stats['epoch'], epoch_stats['loss_mean'], 'b-', linewidth=2, label='Mean Loss')
    ax.fill_between(epoch_stats['epoch'], 
                    epoch_stats['loss_mean'] - epoch_stats['loss_std'],
                    epoch_stats['loss_mean'] + epoch_stats['loss_std'],
                    alpha=0.3, color='blue')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # JEPA Loss
    ax = axes[0, 1]
    ax.plot(epoch_stats['epoch'], epoch_stats['jepa_mean'], 'g-', linewidth=2, label='JEPA Loss')
    ax.fill_between(epoch_stats['epoch'], 
                    epoch_stats['jepa_mean'] - epoch_stats['jepa_std'],
                    epoch_stats['jepa_mean'] + epoch_stats['jepa_std'],
                    alpha=0.3, color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('JEPA Loss')
    ax.set_title('JEPA Prediction Loss (Lower = Better Prediction)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Regularization Loss
    ax = axes[1, 0]
    ax.plot(epoch_stats['epoch'], epoch_stats['reg_mean'], 'r-', linewidth=2, label='Reg Loss')
    ax.fill_between(epoch_stats['epoch'], 
                    epoch_stats['reg_mean'] - epoch_stats['reg_std'],
                    epoch_stats['reg_mean'] + epoch_stats['reg_std'],
                    alpha=0.3, color='red')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Regularization Loss')
    ax.set_title('Regularization Loss (Should Stay Low)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Learning Rate
    ax = axes[1, 1]
    ax.plot(epoch_stats['epoch'], epoch_stats['lr'], 'purple', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'pretraining_curves.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    return epoch_stats


def plot_embedding_analysis(analysis_results, embeddings, output_dir):
    """Plot embedding quality visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Embedding Quality Analysis', fontsize=14, fontweight='bold')
    
    # 1. Variance histogram
    ax = axes[0, 0]
    ax.hist(analysis_results['var_per_dim'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(analysis_results['var_per_dim']), color='red', linestyle='--', 
               label=f"Mean: {np.mean(analysis_results['var_per_dim']):.4f}")
    ax.set_xlabel('Variance')
    ax.set_ylabel('Count')
    ax.set_title('Variance per Dimension')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Cosine similarity histogram
    ax = axes[0, 1]
    ax.hist(analysis_results['cos_sims'], bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax.axvline(analysis_results['mean_cos_sim'], color='red', linestyle='--',
               label=f"Mean: {analysis_results['mean_cos_sim']:.4f}")
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Count')
    ax.set_title('Pairwise Cosine Similarity Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Singular values
    ax = axes[1, 0]
    if analysis_results['singular_values'] is not None:
        sv = analysis_results['singular_values'][:min(100, len(analysis_results['singular_values']))]
        ax.bar(range(len(sv)), sv, color='teal', alpha=0.7)
        ax.set_xlabel('Singular Value Index')
        ax.set_ylabel('Normalized Singular Value')
        ax.set_title(f'Top Singular Values (Effective Rank: {analysis_results["effective_rank"]:.1f})')
    else:
        ax.text(0.5, 0.5, 'SVD Failed', ha='center', va='center', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # 4. t-SNE / PCA visualization
    ax = axes[1, 1]
    try:
        from sklearn.decomposition import PCA
        
        # Use PCA for speed (t-SNE is slow)
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings.numpy())
        
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5, s=10, c='purple')
        ax.set_xlabel(f'PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)')
        ax.set_ylabel(f'PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)')
        ax.set_title('PCA of Embeddings')
        
    except ImportError:
        ax.text(0.5, 0.5, 'sklearn not installed\nfor PCA visualization', 
                ha='center', va='center', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'embedding_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def compute_train_val_loss(config, checkpoint_path, device):
    """Compute loss on both train and validation splits."""
    # Extract config
    cfgs_data = config.get('data', {})
    cfgs_model = config.get('model', {})
    cfgs_mask = config.get('mask', [])
    cfgs_meta = config.get('meta', {})
    cfgs_loss = config.get('loss', {})
    
    # Data params
    batch_size = cfgs_data.get('batch_size', 16)
    num_frames = cfgs_data.get('num_frames', 28)
    tubelet_size = cfgs_data.get('tubelet_size', 4)
    crop_size = cfgs_data.get('crop_size', [105, 33])
    patch_size = cfgs_data.get('patch_size', [7, 8])
    dataset_type = cfgs_data.get('dataset_type', 'EEGDataset')
    dataset_paths = cfgs_data.get('datasets', [])
    
    if isinstance(crop_size, list):
        crop_size = tuple(crop_size)
    if isinstance(patch_size, list):
        patch_size = tuple(patch_size)
    
    # Model params
    model_name = cfgs_model.get('model_name', 'vit_small')
    in_chans = cfgs_model.get('in_chans', 1)
    pred_depth = cfgs_model.get('pred_depth', 6)
    pred_embed_dim = cfgs_model.get('pred_embed_dim', 384)
    uniform_power = cfgs_model.get('uniform_power', True)
    use_mask_tokens = cfgs_model.get('use_mask_tokens', True)
    zero_init_mask_tokens = cfgs_model.get('zero_init_mask_tokens', True)
    use_sdpa = cfgs_meta.get('use_sdpa', False)
    
    loss_exp = cfgs_loss.get('loss_exp', 1.0)
    
    # Initialize model
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
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=len(cfgs_mask),
        zero_init_mask_tokens=zero_init_mask_tokens,
        use_sdpa=use_sdpa,
        in_chans=in_chans,
    )
    target_encoder = copy.deepcopy(encoder)
    
    # Load checkpoint
    load_checkpoint(checkpoint_path, encoder, predictor, target_encoder, device)
    
    encoder.eval()
    predictor.eval()
    target_encoder.eval()
    
    # Mask collator
    mask_collator = MB3DMaskCollator(
        crop_size=crop_size,
        num_frames=num_frames,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        cfgs_mask=cfgs_mask
    )
    
    results = {}
    
    # Compute loss on each split
    for split_name, split_suffix in [('train', 'train'), ('val', 'val')]:
        # Construct split path
        if len(dataset_paths) > 0:
            base_path = dataset_paths[0]
            if 'train' in base_path:
                split_path = base_path.replace('train', split_suffix)
            else:
                split_path = os.path.join(base_path, split_suffix)
        else:
            continue
            
        if not os.path.exists(split_path):
            print(f"  {split_name}: Path not found: {split_path}")
            continue
        
        print(f"\nComputing {split_name} loss on: {split_path}")
        
        try:
            data_loader, _ = init_data(
                data=dataset_type,
                root_path=[split_path],
                batch_size=batch_size,
                num_workers=2,
                world_size=1,
                rank=0,
                pin_mem=True,
                collator=mask_collator,
                drop_last=False,
            )
            
            losses = []
            for batch, masks_enc, masks_pred in tqdm(data_loader, desc=f"Evaluating {split_name}"):
                if isinstance(batch, (list, tuple)):
                    clips = batch[0]
                    if isinstance(clips, list):
                        clips = torch.cat([c.to(device) for c in clips], dim=0)
                    else:
                        clips = clips.to(device)
                else:
                    clips = batch.to(device)
                
                loss = compute_jepa_loss(clips, encoder, predictor, target_encoder, 
                                         masks_enc, masks_pred, device, loss_exp)
                losses.append(loss)
            
            results[split_name] = {
                'mean_loss': np.mean(losses),
                'std_loss': np.std(losses),
                'n_batches': len(losses)
            }
            print(f"  {split_name} loss: {results[split_name]['mean_loss']:.4f} ± {results[split_name]['std_loss']:.4f}")
            
        except Exception as e:
            print(f"  Error loading {split_name}: {e}")
    
    return results, encoder


def print_summary(analysis_results, train_val_results=None):
    """Print a summary of the evaluation."""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    # Overfitting check
    if train_val_results and 'train' in train_val_results and 'val' in train_val_results:
        train_loss = train_val_results['train']['mean_loss']
        val_loss = train_val_results['val']['mean_loss']
        gap = val_loss - train_loss
        gap_pct = 100 * gap / train_loss if train_loss > 0 else 0
        
        print(f"\n[Overfitting Check]")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Gap:        {gap:.4f} ({gap_pct:.1f}%)")
        
        if gap_pct > 20:
            print(f"  ⚠️  WARNING: Large gap suggests OVERFITTING!")
        elif gap_pct > 10:
            print(f"  ⚡ CAUTION: Moderate gap, consider early stopping.")
        elif gap_pct < -5:
            print(f"  🔍 NOTE: Val loss lower than train - check data split.")
        else:
            print(f"  ✓ Good: Train/Val losses are close.")
    
    # Collapse check
    print(f"\n[Representation Quality]")
    print(f"  Effective Rank: {analysis_results['effective_rank']:.1f}")
    print(f"  Mean Cosine Sim: {analysis_results['mean_cos_sim']:.4f}")
    print(f"  Dead Dimensions: {analysis_results['dead_dims']}")
    
    if analysis_results['mean_cos_sim'] > 0.9:
        print(f"  ⚠️  WARNING: High similarity - REPRESENTATION COLLAPSE detected!")
    elif analysis_results['effective_rank'] < 10:
        print(f"  ⚠️  WARNING: Low effective rank - model using few dimensions!")
    else:
        print(f"  ✓ Good: Representations appear healthy.")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate EEG-VJEPA Pretraining')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--log_only', action='store_true', help='Only plot training logs')
    parser.add_argument('--log_file', type=str, help='Path to training log CSV')
    parser.add_argument('--output_dir', type=str, default='./eval_output', help='Output directory for plots')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Log-only mode
    if args.log_only and args.log_file:
        plot_training_curves(args.log_file, args.output_dir)
        print("\nDone! Check the plots in", args.output_dir)
        return
    
    # Full evaluation
    if not args.config or not args.checkpoint:
        parser.error("--config and --checkpoint are required for full evaluation")
    
    config = load_config(args.config)
    
    # Plot training curves if log file exists
    log_dir = config.get('logging', {}).get('folder', './output')
    tag = config.get('logging', {}).get('write_tag', 'jepa')
    log_file = os.path.join(log_dir, f'{tag}_train.csv')
    
    if os.path.exists(log_file):
        plot_training_curves(log_file, args.output_dir)
    else:
        print(f"Training log not found: {log_file}")
    
    # Compute train/val loss
    train_val_results, encoder = compute_train_val_loss(config, args.checkpoint, device)
    
    # Extract embeddings for analysis
    print("\nExtracting embeddings for quality analysis...")
    cfgs_data = config.get('data', {})
    dataset_paths = cfgs_data.get('datasets', [])
    
    if dataset_paths:
        # Simple data loader without masking for embedding extraction
        data_loader, _ = init_data(
            data=cfgs_data.get('dataset_type', 'EEGDataset'),
            root_path=dataset_paths,
            batch_size=cfgs_data.get('batch_size', 16),
            num_workers=2,
            world_size=1,
            rank=0,
            pin_mem=True,
            collator=None,
            drop_last=False,
        )
        
        embeddings = compute_embeddings(data_loader, encoder, device, max_batches=50)
        analysis_results = analyze_embeddings(embeddings)
        plot_embedding_analysis(analysis_results, embeddings, args.output_dir)
        
        # Print summary
        print_summary(analysis_results, train_val_results)
    
    print(f"\n✓ Evaluation complete! Check outputs in: {args.output_dir}")


if __name__ == '__main__':
    main()
