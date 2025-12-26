"""
Single-GPU Training Script v2 for EEG-VJEPA with Validation Tracking

This version includes:
- Validation loss computation every N epochs
- Overfitting detection
- Enhanced logging with train/val curves
- Early stopping support

Usage:
    python app/vjepa/train_single_gpuv2.py --config configs/pretrain/zuco_pretrain.yaml
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


def compute_val_loss(val_loader, encoder, predictor, target_encoder, device, dtype, mixed_precision, loss_exp=1.0):
    """Compute validation loss."""
    encoder.eval()
    predictor.eval()
    
    val_losses = []
    
    with torch.no_grad():
        for batch, masks_enc, masks_pred in tqdm(val_loader, desc="Validation", leave=False):
            # Get data
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
                # Target forward
                h = target_encoder(clips)
                h = F.layer_norm(h, (h.size(-1),))
                h = apply_masks(h, _masks_pred, concat=False)
                
                # Context forward
                z = encoder(clips, _masks_enc)
                z = predictor(z, h, _masks_enc, _masks_pred)
                
                # Loss
                loss = 0.
                for zi, hi in zip(z, h):
                    loss += torch.mean(torch.abs(zi - hi)**loss_exp) / loss_exp
                loss /= len(_masks_pred)
            
            val_losses.append(float(loss))
    
    encoder.train()
    predictor.train()
    
    return np.mean(val_losses), np.std(val_losses)


def main():
    parser = argparse.ArgumentParser(description='EEG-VJEPA Single GPU Pretraining v2')
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
    
    # Validation frequency
    val_freq = cfgs_meta.get('eval_freq', 5)  # Validate every N epochs
    early_stopping_patience = cfgs_meta.get('early_stopping_patience', 10)
    
    # Setup logging - now with train AND val columns
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
    
    # Separate logger for epoch-level stats with validation
    epoch_log_file = os.path.join(output_dir, f'{tag}_epochs.csv')
    epoch_logger = CSVLogger(
        epoch_log_file,
        ('%d', 'epoch'),
        ('%.5f', 'train_loss'),
        ('%.5f', 'train_jepa'),
        ('%.5f', 'train_reg'),
        ('%.5f', 'val_loss'),
        ('%.5f', 'val_std'),
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
    
    # Convert to tuples
    if isinstance(crop_size, list):
        crop_size = tuple(crop_size)
    if isinstance(patch_size, list):
        patch_size = tuple(patch_size)
    
    print(f"\nData config:")
    print(f"  Dataset type: {dataset_type}")
    print(f"  Dataset paths: {dataset_paths}")
    print(f"  Batch size: {batch_size}")
    print(f"  Crop size (H, W): {crop_size}")
    print(f"  Num frames (T): {num_frames}")
    print(f"  Patch size: {patch_size}")
    print(f"  Tubelet size: {tubelet_size}")
    
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
    print(f"  Predictor depth: {pred_depth}")
    
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
    print("Initializing mask collator...")
    mask_collator = MB3DMaskCollator(
        crop_size=crop_size,
        num_frames=num_frames,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        cfgs_mask=cfgs_mask
    )
    
    # Initialize TRAINING data loader
    print("Initializing training data loader...")
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
    
    # Initialize VALIDATION data loader
    val_loader = None
    if len(dataset_paths) > 0:
        val_path = dataset_paths[0].replace('train', 'val')
        if os.path.exists(val_path):
            print(f"Initializing validation data loader from: {val_path}")
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
            print(f"Validation loader: {len(val_loader)} batches")
        else:
            print(f"Validation path not found: {val_path}")
            print("  Proceeding without validation (can't detect overfitting)")
    
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
    print("\nInitializing optimizer...")
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
    
    # Momentum scheduler for EMA
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
    print(f"Validation every {val_freq} epochs")
    print(f"Early stopping patience: {early_stopping_patience}")
    print(f"{'='*60}\n")
    
    # Track metrics history for plotting
    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [], 'train_jepa': [], 'train_reg': []
    }
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        
        loss_meter = AverageMeter()
        jepa_loss_meter = AverageMeter()
        reg_loss_meter = AverageMeter()
        
        # Progress bar
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), 
                    desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for itr, (batch, masks_enc, masks_pred) in pbar:
            # Get data
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
                
                def forward_target(x):
                    with torch.no_grad():
                        h = target_encoder(x)
                        h = F.layer_norm(h, (h.size(-1),))
                        h = apply_masks(h, _masks_pred, concat=False)
                        return h
                
                def forward_context(x, h):
                    z = encoder(x, _masks_enc)
                    z = predictor(z, h, _masks_enc, _masks_pred)
                    return z
                
                def loss_fn(z, h):
                    loss = 0.
                    for zi, hi in zip(z, h):
                        loss += torch.mean(torch.abs(zi - hi)**loss_exp) / loss_exp
                    loss /= len(_masks_pred)
                    return loss
                
                def reg_fn(z):
                    return sum([torch.sqrt(zi.var(dim=1) + 0.0001) for zi in z]) / len(z)
                
                # Forward pass
                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    h = forward_target(clips)
                    z = forward_context(clips, h)
                    loss_jepa = loss_fn(z, h)
                    pstd_z = reg_fn(z)
                    loss_reg = torch.mean(F.relu(1.-pstd_z))
                
                loss = loss_jepa + reg_coeff * loss_reg
                
                # Backward pass
                optimizer.zero_grad()
                if mixed_precision and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                
                # Gradient clipping
                if epoch > warmup and clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad)
                    torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad)
                
                # Optimizer step
                if mixed_precision and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                # EMA update
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
                print(f"\nNaN loss detected at epoch {epoch+1}, iter {itr}")
                return
        
        epoch_time = time.time() - epoch_start
        
        # Compute validation loss
        val_loss = 0
        val_std = 0
        should_validate = (epoch + 1) % val_freq == 0 or epoch == num_epochs - 1
        
        if val_loader is not None and should_validate:
            val_loss, val_std = compute_val_loss(
                val_loader, encoder, predictor, target_encoder, 
                device, dtype, mixed_precision, loss_exp
            )
            
            # Overfitting check
            train_val_gap = val_loss - loss_meter.avg
            gap_pct = 100 * train_val_gap / loss_meter.avg if loss_meter.avg > 0 else 0
            
            print(f"Epoch {epoch+1}: train={loss_meter.avg:.4f}, val={val_loss:.4f}±{val_std:.4f}, "
                  f"gap={gap_pct:.1f}%, time={epoch_time:.1f}s")
            
            if gap_pct > 20:
                print(f"  ⚠️  WARNING: Large train/val gap suggests overfitting!")
            
            # Track best val loss and early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                
                # Save best model based on validation
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
                    'best_val_loss': best_val_loss,
                }, best_path)
                print(f"  ✓ New best val loss! Saved to {best_path}")
            else:
                epochs_without_improvement += val_freq
                print(f"  No improvement for {epochs_without_improvement} epochs")
                
                if epochs_without_improvement >= early_stopping_patience:
                    print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                    break
        else:
            print(f"Epoch {epoch+1}: loss={loss_meter.avg:.4f}, "
                  f"jepa={jepa_loss_meter.avg:.4f}, reg={reg_loss_meter.avg:.4f}, "
                  f"time={epoch_time:.1f}s")
            
            # Save best based on train loss if no validation
            if loss_meter.avg < best_train_loss:
                best_train_loss = loss_meter.avg
                best_path = os.path.join(output_dir, f'{tag}-best.pth.tar')
                torch.save({
                    'encoder': encoder.state_dict(),
                    'predictor': predictor.state_dict(),
                    'target_encoder': target_encoder.state_dict(),
                    'opt': optimizer.state_dict(),
                    'scaler': scaler.state_dict() if scaler else None,
                    'epoch': epoch + 1,
                    'loss': loss_meter.avg,
                }, best_path)
                print(f"  New best loss saved: {best_train_loss:.4f}")
        
        # Log epoch stats
        epoch_logger.log(epoch+1, loss_meter.avg, jepa_loss_meter.avg, reg_loss_meter.avg,
                         val_loss, val_std, current_lr, epoch_time)
        
        # Track history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(loss_meter.avg)
        history['val_loss'].append(val_loss if val_loss > 0 else np.nan)
        history['train_jepa'].append(jepa_loss_meter.avg)
        history['train_reg'].append(reg_loss_meter.avg)
        
        # Save latest checkpoint
        latest_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': scaler.state_dict() if scaler else None,
            'epoch': epoch + 1,
            'loss': loss_meter.avg,
            'best_val_loss': best_val_loss,
        }
        torch.save(latest_dict, latest_path)
    
    # Plot final curves
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Train vs Val Loss
        ax = axes[0]
        ax.plot(history['epoch'], history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        if val_loader is not None:
            val_losses = [v for v in history['val_loss'] if not np.isnan(v)]
            val_epochs = [e for e, v in zip(history['epoch'], history['val_loss']) if not np.isnan(v)]
            ax.plot(val_epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Train vs Val Loss (Overfitting Detection)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # JEPA Loss
        ax = axes[1]
        ax.plot(history['epoch'], history['train_jepa'], 'g-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('JEPA Loss')
        ax.set_title('JEPA Prediction Loss')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
        print(f"\nSaved training curves to {output_dir}/training_curves.png")
        plt.close()
    except Exception as e:
        print(f"Could not plot curves: {e}")
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best train loss: {best_train_loss:.4f}")
    if val_loader is not None:
        print(f"Best val loss: {best_val_loss:.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
