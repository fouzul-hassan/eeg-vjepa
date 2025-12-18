"""
Single-GPU Training Script for EEG-VJEPA

This script provides a simplified training loop for pretraining on a single GPU,
without the need for distributed training setup.

Usage:
    conda activate zuco-hdf5
    python app/vjepa/train_single_gpu.py --config configs/pretrain/zuco_pretrain.yaml
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


def main():
    parser = argparse.ArgumentParser(description='EEG-VJEPA Single GPU Pretraining')
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
    
    # Setup logging
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
    in_chans = cfgs_model.get('in_chans', 1)  # Default 1 for EEG (not 3 like video)
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
        in_chans=in_chans,  # Pass input channels
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
    
    # Initialize data loader
    print("Initializing data loader...")
    data_loader, sampler = init_data(
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
    print(f"Data loader created: {len(data_loader)} batches")
    
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
    ipe = len(data_loader)
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
    best_loss = float('inf')
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
        best_loss = checkpoint.get('loss', float('inf'))
        # Advance schedulers
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
        print(f"Resumed from epoch {start_epoch}")
    
    print(f"\n{'='*60}")
    print(f"Starting training for {num_epochs} epochs")
    print(f"{'='*60}\n")
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        
        loss_meter = AverageMeter()
        jepa_loss_meter = AverageMeter()
        reg_loss_meter = AverageMeter()
        
        # Progress bar
        pbar = tqdm(enumerate(data_loader), total=len(data_loader), 
                    desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for itr, (batch, masks_enc, masks_pred) in pbar:
            # Get data - handle different batch formats
            if isinstance(batch, (list, tuple)):
                # VideoDataset format: (clips, labels, indices)
                clips = batch[0]
                if isinstance(clips, list):
                    clips = torch.cat([c.to(device, non_blocking=True) for c in clips], dim=0)
                else:
                    clips = clips.to(device, non_blocking=True)
            else:
                # EEGDataset format: just the tensor
                clips = batch.to(device, non_blocking=True)
            
            # Move masks to device
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
                
                # EMA update of target encoder
                m = next(momentum_scheduler)
                with torch.no_grad():
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
                
                return float(loss), float(loss_jepa), float(loss_reg), _new_lr
            
            loss, loss_jepa, loss_reg, current_lr = train_step()
            
            loss_meter.update(loss)
            jepa_loss_meter.update(loss_jepa)
            reg_loss_meter.update(loss_reg)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'lr': f'{current_lr:.2e}',
                'mem': f'{torch.cuda.max_memory_allocated()/1e9:.1f}GB'
            })
            
            # Log to CSV
            csv_logger.log(epoch+1, itr, loss, loss_jepa, loss_reg, current_lr)
            
            # Check for NaN
            if np.isnan(loss):
                print(f"\nNaN loss detected at epoch {epoch+1}, iter {itr}")
                return
        
        # End of epoch
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}: loss={loss_meter.avg:.4f}, "
              f"jepa={jepa_loss_meter.avg:.4f}, reg={reg_loss_meter.avg:.4f}, "
              f"time={epoch_time:.1f}s")
        
        # Save checkpoint
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': scaler.state_dict() if scaler else None,
            'epoch': epoch + 1,
            'loss': loss_meter.avg,
        }
        
        # Save latest
        torch.save(save_dict, latest_path)
        
        # Save best
        if loss_meter.avg < best_loss:
            best_loss = loss_meter.avg
            best_path = os.path.join(output_dir, f'{tag}-best.pth.tar')
            torch.save(save_dict, best_path)
            print(f"  New best model saved: loss={best_loss:.4f}")
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
