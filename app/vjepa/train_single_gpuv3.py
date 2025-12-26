"""
Single-GPU Training Script v3 for EEG-VJEPA with Comprehensive Evaluation

This version includes:
- Validation loss computation every N epochs
- **Online k-NN evaluation** (no training required)
- **Online linear probe evaluation**
- **Embedding quality metrics** (variance, effective rank, collapse detection)
- Overfitting detection & Early stopping
- Comprehensive logging and plotting

Usage:
    python app/vjepa/train_single_gpuv3.py --config configs/pretrain/zuco_pretrain.yaml
"""

import os
import sys
import copy
import time
import argparse
from datetime import datetime

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.datasets.data_manager import init_data
from src.masks.multiblock3d import MaskCollator as MB3DMaskCollator
from src.masks.utils import apply_masks
from src.utils.logging import AverageMeter, CSVLogger

from app.vjepa.utils import init_video_model, init_opt


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# =============================================================================
# ONLINE EVALUATION FUNCTIONS
# =============================================================================

def extract_embeddings(data_loader, encoder, device, max_batches=20):
    """Extract embeddings from encoder for evaluation."""
    encoder.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for i, batch_data in enumerate(data_loader):
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
            embeddings = encoder(clips)  # [B, N, D]
            embeddings = embeddings.mean(dim=1)  # [B, D] - global avg pool
            all_embeddings.append(embeddings.cpu())
    
    encoder.train()
    return torch.cat(all_embeddings, dim=0) if all_embeddings else None


def knn_accuracy(embeddings, k=5):
    """
    Self-supervised k-NN evaluation using index as pseudo-label.
    Measures how well nearby samples cluster together.
    Returns consistency score (0-100).
    """
    if embeddings is None or len(embeddings) < k + 1:
        return 0.0
    
    embeddings = F.normalize(embeddings, dim=1)
    similarities = torch.mm(embeddings, embeddings.t())
    
    n_samples = embeddings.shape[0]
    consistency = 0
    
    for i in range(n_samples):
        sims = similarities[i].clone()
        sims[i] = -float('inf')
        _, topk_indices = torch.topk(sims, min(k, n_samples - 1))
        
        # Check if neighbors are close in index (temporal consistency)
        neighbor_distances = torch.abs(topk_indices.float() - i)
        consistency += (neighbor_distances < n_samples * 0.1).float().mean().item()
    
    return 100.0 * consistency / n_samples


class LinearProbe(nn.Module):
    """Simple linear classifier for probe evaluation."""
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)


def quick_linear_probe(embeddings, num_classes=10, epochs=50, lr=0.01):
    """
    Quick linear probe using random labels to test feature separability.
    Creates synthetic classification task to measure feature quality.
    """
    if embeddings is None or len(embeddings) < num_classes * 2:
        return 0.0
    
    # Create pseudo-labels based on embedding clusters
    n_samples = len(embeddings)
    labels = torch.arange(n_samples) % num_classes
    
    # Shuffle
    perm = torch.randperm(n_samples)
    embeddings = embeddings[perm]
    labels = labels[perm]
    
    # Split
    split_idx = int(0.8 * n_samples)
    train_emb, val_emb = embeddings[:split_idx], embeddings[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]
    
    # Train probe
    embed_dim = embeddings.shape[1]
    probe = LinearProbe(embed_dim, num_classes)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(epochs):
        probe.train()
        optimizer.zero_grad()
        logits = probe(train_emb)
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()
    
    # Evaluate
    probe.eval()
    with torch.no_grad():
        val_pred = probe(val_emb).argmax(dim=1)
        accuracy = 100.0 * (val_pred == val_labels).float().mean().item()
    
    return accuracy


def compute_embedding_metrics(embeddings):
    """Compute embedding quality metrics."""
    if embeddings is None or len(embeddings) < 10:
        return {'variance': 0, 'effective_rank': 0, 'mean_cos_sim': 1.0}
    
    embeddings_np = embeddings.numpy()
    n_samples, embed_dim = embeddings_np.shape
    
    # Variance
    var_per_dim = np.var(embeddings_np, axis=0)
    mean_variance = np.mean(var_per_dim)
    
    # Effective rank
    try:
        embeddings_centered = embeddings_np - np.mean(embeddings_np, axis=0)
        _, singular_values, _ = np.linalg.svd(embeddings_centered, full_matrices=False)
        singular_values = singular_values / np.sum(singular_values)
        singular_values = singular_values[singular_values > 1e-10]
        effective_rank = np.exp(-np.sum(singular_values * np.log(singular_values + 1e-10)))
    except:
        effective_rank = 0
    
    # Cosine similarity
    embeddings_norm = embeddings_np / (np.linalg.norm(embeddings_np, axis=1, keepdims=True) + 1e-8)
    n_pairs = min(500, n_samples * (n_samples - 1) // 2)
    indices = np.random.choice(n_samples, size=(n_pairs, 2), replace=True)
    cos_sims = np.sum(embeddings_norm[indices[:, 0]] * embeddings_norm[indices[:, 1]], axis=1)
    mean_cos_sim = np.mean(cos_sims)
    
    return {
        'variance': mean_variance,
        'effective_rank': effective_rank,
        'mean_cos_sim': mean_cos_sim
    }


def compute_val_loss(val_loader, encoder, predictor, target_encoder, device, dtype, mixed_precision, loss_exp=1.0):
    """Compute validation loss."""
    encoder.eval()
    predictor.eval()
    
    val_losses = []
    
    with torch.no_grad():
        for batch, masks_enc, masks_pred in tqdm(val_loader, desc="Validation", leave=False):
            if isinstance(batch, (list, tuple)):
                clips = batch[0]
                if isinstance(clips, list):
                    clips = torch.cat([c.to(device, non_blocking=True) for c in clips], dim=0)
                else:
                    clips = clips.to(device, non_blocking=True)
            else:
                clips = batch.to(device, non_blocking=True)
            
            _masks_enc = [m.to(device, non_blocking=True) for m in masks_enc]
            _masks_pred = [m.to(device, non_blocking=True) for m in masks_pred]
            
            with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                h = target_encoder(clips)
                h = F.layer_norm(h, (h.size(-1),))
                h = apply_masks(h, _masks_pred, concat=False)
                
                z = encoder(clips, _masks_enc)
                z = predictor(z, h, _masks_enc, _masks_pred)
                
                loss = 0.
                for zi, hi in zip(z, h):
                    loss += torch.mean(torch.abs(zi - hi)**loss_exp) / loss_exp
                loss /= len(_masks_pred)
            
            val_losses.append(float(loss))
    
    encoder.train()
    predictor.train()
    
    return np.mean(val_losses), np.std(val_losses)


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='EEG-VJEPA Single GPU Pretraining v3')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    
    # Extract config sections
    cfgs_data = config.get('data', {})
    cfgs_model = config.get('model', {})
    cfgs_opt = config.get('optimization', {})
    cfgs_mask = config.get('mask', [])
    cfgs_loss = config.get('loss', {})
    cfgs_logging = config.get('logging', {})
    cfgs_meta = config.get('meta', {})

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup dtype
    which_dtype = cfgs_meta.get('dtype', 'float32')
    if which_dtype.lower() == 'bfloat16':
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == 'float16':
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False
    print(f"Using dtype: {dtype}, mixed_precision: {mixed_precision}")

    # Seed
    seed = cfgs_meta.get('seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Create output directory
    output_dir = cfgs_logging.get('folder', './output')
    os.makedirs(output_dir, exist_ok=True)
    tag = cfgs_logging.get('write_tag', 'jepa')
    
    # Evaluation frequency
    val_freq = cfgs_meta.get('eval_freq', 5)
    early_stopping_patience = cfgs_meta.get('early_stopping_patience', 20)
    
    # Setup logging - iteration level
    log_file = os.path.join(output_dir, f'{tag}_train.csv')
    csv_logger = CSVLogger(
        log_file,
        ('%d', 'epoch'),
        ('%d', 'itr'),
        ('%.5f', 'loss'),
        ('%.5f', 'loss-jepa'),
        ('%.5f', 'reg-loss'),
        ('%.5f', 'lr'),
    )
    
    # Epoch-level logger with evaluation metrics
    epoch_log_file = os.path.join(output_dir, f'{tag}_epochs.csv')
    epoch_logger = CSVLogger(
        epoch_log_file,
        ('%d', 'epoch'),
        ('%.5f', 'train_loss'),
        ('%.5f', 'val_loss'),
        ('%.2f', 'knn_score'),
        ('%.2f', 'probe_acc'),
        ('%.4f', 'variance'),
        ('%.1f', 'eff_rank'),
        ('%.4f', 'cos_sim'),
        ('%.5f', 'lr'),
        ('%.1f', 'time'),
    )
    
    # Data params
    batch_size = cfgs_data.get('batch_size', 16)
    num_frames = cfgs_data.get('num_frames', 28)
    tubelet_size = cfgs_data.get('tubelet_size', 4)
    crop_size = cfgs_data.get('crop_size', [105, 33])
    patch_size = cfgs_data.get('patch_size', [7, 8])
    num_workers = cfgs_data.get('num_workers', 4)
    pin_mem = cfgs_data.get('pin_mem', True)
    dataset_type = cfgs_data.get('dataset_type', 'EEGDataset')
    dataset_paths = cfgs_data.get('datasets', [])
    
    if isinstance(crop_size, list):
        crop_size = tuple(crop_size)
    if isinstance(patch_size, list):
        patch_size = tuple(patch_size)
    
    print(f"\nData config:")
    print(f"  Dataset type: {dataset_type}")
    print(f"  Batch size: {batch_size}")
    print(f"  Crop size: {crop_size}")
    print(f"  Num frames: {num_frames}")
    
    # Model params
    model_name = cfgs_model.get('model_name', 'vit_small')
    in_chans = cfgs_model.get('in_chans', 1)
    pred_depth = cfgs_model.get('pred_depth', 6)
    pred_embed_dim = cfgs_model.get('pred_embed_dim', 384)
    uniform_power = cfgs_model.get('uniform_power', True)
    use_mask_tokens = cfgs_model.get('use_mask_tokens', True)
    zero_init_mask_tokens = cfgs_model.get('zero_init_mask_tokens', True)
    use_sdpa = cfgs_meta.get('use_sdpa', False)
    
    print(f"\nModel config:")
    print(f"  Model name: {model_name}")
    print(f"  Input channels: {in_chans}")
    
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
    for p in target_encoder.parameters():
        p.requires_grad = False
    
    # Initialize mask collator
    mask_collator = MB3DMaskCollator(
        crop_size=crop_size,
        num_frames=num_frames,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        cfgs_mask=cfgs_mask
    )
    
    # Initialize data loaders
    print("Initializing data loaders...")
    train_loader, _ = init_data(
        data=dataset_type,
        root_path=dataset_paths,
        batch_size=batch_size,
        num_workers=num_workers,
        world_size=1,
        rank=0,
        pin_mem=pin_mem,
        collator=mask_collator,
        drop_last=True,
    )
    print(f"Training loader: {len(train_loader)} batches")
    
    # Eval loader (no masking, for embedding extraction)
    eval_loader, _ = init_data(
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
    print(f"Eval loader: {len(eval_loader)} batches")
    
    # Validation loader
    val_loader = None
    if len(dataset_paths) > 0:
        val_path = dataset_paths[0].replace('train', 'val')
        if os.path.exists(val_path):
            print(f"Validation loader from: {val_path}")
            val_loader, _ = init_data(
                data=dataset_type,
                root_path=[val_path],
                batch_size=batch_size,
                num_workers=num_workers,
                world_size=1,
                rank=0,
                pin_mem=pin_mem,
                collator=mask_collator,
                drop_last=False,
            )
    
    # Optimization params
    num_epochs = cfgs_opt.get('epochs', 100)
    warmup = cfgs_opt.get('warmup', 10)
    start_lr = cfgs_opt.get('start_lr', 1e-5)
    lr = cfgs_opt.get('lr', 1e-4)
    final_lr = cfgs_opt.get('final_lr', 1e-6)
    wd = cfgs_opt.get('weight_decay', 0.04)
    final_wd = cfgs_opt.get('final_weight_decay', 0.4)
    clip_grad = cfgs_opt.get('clip_grad', 10.0)
    ema = cfgs_opt.get('ema', [0.996, 1.0])
    
    # Initialize optimizer
    ipe = len(train_loader)
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        mixed_precision=mixed_precision,
    )
    
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs)
                          for i in range(int(ipe*num_epochs)+1))
    
    # Loss params
    loss_exp = cfgs_loss.get('loss_exp', 1.0)
    reg_coeff = cfgs_loss.get('reg_coeff', 0.0)
    
    # Checkpointing
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    epochs_without_improvement = 0
    start_epoch = 0
    
    # Resume from checkpoint
    latest_path = os.path.join(output_dir, f'{tag}-latest.pth.tar')
    if args.resume and os.path.exists(latest_path):
        print(f"Resuming from {latest_path}")
        checkpoint = torch.load(latest_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        predictor.load_state_dict(checkpoint['predictor'])
        target_encoder.load_state_dict(checkpoint['target_encoder'])
        optimizer.load_state_dict(checkpoint['opt'])
        if scaler is not None and checkpoint.get('scaler'):
            scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        best_train_loss = checkpoint.get('loss', float('inf'))
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
        print(f"Resumed from epoch {start_epoch}")
    
    print(f"\n{'='*60}")
    print(f"Starting training for {num_epochs} epochs")
    print(f"Evaluation every {val_freq} epochs")
    print(f"{'='*60}\n")
    
    # Track history
    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [], 
        'knn_score': [], 'probe_acc': [], 'eff_rank': [], 'cos_sim': []
    }
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        
        loss_meter = AverageMeter()
        jepa_loss_meter = AverageMeter()
        reg_loss_meter = AverageMeter()
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                    desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for itr, (batch, masks_enc, masks_pred) in pbar:
            if isinstance(batch, (list, tuple)):
                clips = batch[0]
                if isinstance(clips, list):
                    clips = torch.cat([c.to(device, non_blocking=True) for c in clips], dim=0)
                else:
                    clips = clips.to(device, non_blocking=True)
            else:
                clips = batch.to(device, non_blocking=True)
            
            _masks_enc = [m.to(device, non_blocking=True) for m in masks_enc]
            _masks_pred = [m.to(device, non_blocking=True) for m in masks_pred]
            
            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                
                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    with torch.no_grad():
                        h = target_encoder(clips)
                        h = F.layer_norm(h, (h.size(-1),))
                        h = apply_masks(h, _masks_pred, concat=False)
                    
                    z = encoder(clips, _masks_enc)
                    z = predictor(z, h, _masks_enc, _masks_pred)
                    
                    loss_jepa = 0.
                    for zi, hi in zip(z, h):
                        loss_jepa += torch.mean(torch.abs(zi - hi)**loss_exp) / loss_exp
                    loss_jepa /= len(_masks_pred)
                    
                    pstd_z = sum([torch.sqrt(zi.var(dim=1) + 0.0001) for zi in z]) / len(z)
                    loss_reg = torch.mean(F.relu(1.-pstd_z))
                
                loss = loss_jepa + reg_coeff * loss_reg
                
                optimizer.zero_grad()
                if mixed_precision and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                
                if epoch > warmup and clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad)
                    torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad)
                
                if mixed_precision and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                m = next(momentum_scheduler)
                with torch.no_grad():
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
                
                return float(loss), float(loss_jepa), float(loss_reg), _new_lr
            
            loss, loss_jepa, loss_reg, current_lr = train_step()
            
            loss_meter.update(loss)
            jepa_loss_meter.update(loss_jepa)
            reg_loss_meter.update(loss_reg)
            
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'lr': f'{current_lr:.2e}',
                'mem': f'{torch.cuda.max_memory_allocated()/1e9:.1f}GB'
            })
            
            csv_logger.log(epoch+1, itr, loss, loss_jepa, loss_reg, current_lr)
            
            if np.isnan(loss):
                print(f"\nNaN loss at epoch {epoch+1}, iter {itr}")
                return
        
        epoch_time = time.time() - epoch_start
        
        # Run evaluation every val_freq epochs
        val_loss = 0
        knn_score = 0
        probe_acc = 0
        emb_metrics = {'variance': 0, 'effective_rank': 0, 'mean_cos_sim': 1.0}
        
        should_eval = (epoch + 1) % val_freq == 0 or epoch == num_epochs - 1
        
        if should_eval:
            print(f"\n--- Epoch {epoch+1} Evaluation ---")
            
            # Validation loss
            if val_loader is not None:
                val_loss, _ = compute_val_loss(
                    val_loader, encoder, predictor, target_encoder,
                    device, dtype, mixed_precision, loss_exp
                )
                gap_pct = 100 * (val_loss - loss_meter.avg) / loss_meter.avg if loss_meter.avg > 0 else 0
                print(f"  Val Loss: {val_loss:.4f} (gap: {gap_pct:.1f}%)")
            
            # Extract embeddings for evaluation
            embeddings = extract_embeddings(eval_loader, encoder, device, max_batches=30)
            
            if embeddings is not None:
                # k-NN evaluation
                knn_score = knn_accuracy(embeddings, k=5)
                print(f"  k-NN Score: {knn_score:.2f}%")
                
                # Linear probe
                probe_acc = quick_linear_probe(embeddings, num_classes=10)
                print(f"  Probe Acc: {probe_acc:.2f}%")
                
                # Embedding metrics
                emb_metrics = compute_embedding_metrics(embeddings)
                print(f"  Eff. Rank: {emb_metrics['effective_rank']:.1f}")
                print(f"  Cos Sim: {emb_metrics['mean_cos_sim']:.4f}")
                
                # Collapse warning
                if emb_metrics['mean_cos_sim'] > 0.9:
                    print(f"  ⚠️  WARNING: High cosine similarity - possible collapse!")
            
            print(f"---")
            
            # Check for improvement
            metric_to_check = val_loss if val_loader else loss_meter.avg
            best_metric = best_val_loss if val_loader else best_train_loss
            
            if metric_to_check < best_metric:
                if val_loader:
                    best_val_loss = val_loss
                else:
                    best_train_loss = metric_to_check
                epochs_without_improvement = 0
                
                # Save best
                best_path = os.path.join(output_dir, f'{tag}-best.pth.tar')
                torch.save({
                    'encoder': encoder.state_dict(),
                    'predictor': predictor.state_dict(),
                    'target_encoder': target_encoder.state_dict(),
                    'opt': optimizer.state_dict(),
                    'scaler': scaler.state_dict() if scaler else None,
                    'epoch': epoch + 1,
                    'loss': loss_meter.avg,
                    'val_loss': val_loss,
                    'knn_score': knn_score,
                    'probe_acc': probe_acc,
                }, best_path)
                print(f"  ✓ New best! Saved to {best_path}")
            else:
                epochs_without_improvement += val_freq
                if epochs_without_improvement >= early_stopping_patience:
                    print(f"\n⚠️  Early stopping at epoch {epoch+1}")
                    break
        else:
            print(f"Epoch {epoch+1}: loss={loss_meter.avg:.4f}, time={epoch_time:.1f}s")
        
        # Log epoch stats
        epoch_logger.log(
            epoch+1, loss_meter.avg, val_loss, knn_score, probe_acc,
            emb_metrics['variance'], emb_metrics['effective_rank'],
            emb_metrics['mean_cos_sim'], current_lr, epoch_time
        )
        
        # Track history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(loss_meter.avg)
        history['val_loss'].append(val_loss if val_loss > 0 else np.nan)
        history['knn_score'].append(knn_score)
        history['probe_acc'].append(probe_acc)
        history['eff_rank'].append(emb_metrics['effective_rank'])
        history['cos_sim'].append(emb_metrics['mean_cos_sim'])
        
        # Save latest
        torch.save({
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': scaler.state_dict() if scaler else None,
            'epoch': epoch + 1,
            'loss': loss_meter.avg,
            'best_val_loss': best_val_loss,
        }, latest_path)
    
    # Plot training curves
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Training Progress', fontsize=14, fontweight='bold')
        
        # Loss
        ax = axes[0, 0]
        ax.plot(history['epoch'], history['train_loss'], 'b-', label='Train', linewidth=2)
        val_losses = [v for v in history['val_loss'] if not np.isnan(v)]
        val_epochs = [e for e, v in zip(history['epoch'], history['val_loss']) if not np.isnan(v)]
        if val_losses:
            ax.plot(val_epochs, val_losses, 'r-', label='Val', linewidth=2)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('Train vs Val Loss')
        ax.legend(); ax.grid(True, alpha=0.3)
        
        # k-NN Score
        ax = axes[0, 1]
        knn_scores = [k for k in history['knn_score'] if k > 0]
        knn_epochs = [e for e, k in zip(history['epoch'], history['knn_score']) if k > 0]
        if knn_scores:
            ax.plot(knn_epochs, knn_scores, 'g-', marker='o', linewidth=2)
        ax.set_xlabel('Epoch'); ax.set_ylabel('k-NN Score (%)'); ax.set_title('k-NN Evaluation')
        ax.grid(True, alpha=0.3)
        
        # Probe accuracy
        ax = axes[0, 2]
        probe_accs = [p for p in history['probe_acc'] if p > 0]
        probe_epochs = [e for e, p in zip(history['epoch'], history['probe_acc']) if p > 0]
        if probe_accs:
            ax.plot(probe_epochs, probe_accs, 'm-', marker='o', linewidth=2)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Probe Acc (%)'); ax.set_title('Linear Probe')
        ax.grid(True, alpha=0.3)
        
        # Effective rank
        ax = axes[1, 0]
        ranks = [r for r in history['eff_rank'] if r > 0]
        rank_epochs = [e for e, r in zip(history['epoch'], history['eff_rank']) if r > 0]
        if ranks:
            ax.plot(rank_epochs, ranks, 'c-', marker='o', linewidth=2)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Effective Rank'); ax.set_title('Embedding Effective Rank')
        ax.grid(True, alpha=0.3)
        
        # Cosine similarity
        ax = axes[1, 1]
        sims = [s for s in history['cos_sim'] if s > 0]
        sim_epochs = [e for e, s in zip(history['epoch'], history['cos_sim']) if s > 0]
        if sims:
            ax.plot(sim_epochs, sims, 'orange', marker='o', linewidth=2)
            ax.axhline(y=0.9, color='red', linestyle='--', label='Collapse threshold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Mean Cos Sim'); ax.set_title('Collapse Detection')
        ax.legend(); ax.grid(True, alpha=0.3)
        
        # Summary text
        ax = axes[1, 2]
        ax.axis('off')
        summary = f"""Training Summary
        
Final Train Loss: {history['train_loss'][-1]:.4f}
Best Val Loss: {best_val_loss:.4f if best_val_loss < float('inf') else 'N/A'}
Final k-NN Score: {knn_scores[-1]:.1f}% if knn_scores else 'N/A'
Final Probe Acc: {probe_accs[-1]:.1f}% if probe_accs else 'N/A'
Final Eff. Rank: {ranks[-1]:.1f if ranks else 'N/A'}
Final Cos Sim: {sims[-1]:.4f if sims else 'N/A'}
"""
        ax.text(0.1, 0.5, summary, fontsize=12, verticalalignment='center', family='monospace')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_progress.png'), dpi=150)
        print(f"\nSaved training progress to {output_dir}/training_progress.png")
        plt.close()
    except Exception as e:
        print(f"Could not plot: {e}")
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
