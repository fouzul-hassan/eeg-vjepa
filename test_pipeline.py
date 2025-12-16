"""Test script for EEG-VJEPA pipeline verification"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

print("=" * 60)
print("EEG-VJEPA Pipeline Verification")
print("=" * 60)

# Test 1: Spectrogram computation
print("\n[1] Testing spectrogram computation...")
from src.datasets.eeg_dataset import compute_spectrogram

test_eeg = np.random.randn(105, 512).astype('float32')
spectrogram = compute_spectrogram(test_eeg, n_fft=64, hop_length=16)
print(f"    Input shape: (105, 512)")
print(f"    Spectrogram shape: {spectrogram.shape}")
print(f"    Expected: (105, 33, 28)")  # (channels, freq_bins, time_bins)

# Test 2: Dataset class import
print("\n[2] Testing dataset class...")
from src.datasets.eeg_dataset import ZuCoSpectrogramDataset
print("    ZuCoSpectrogramDataset imported successfully")

# Test 3: Model initialization
print("\n[3] Testing model initialization...")
from app.vjepa.utils import init_video_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"    Device: {device}")

try:
    encoder, predictor = init_video_model(
        device=device,
        patch_size=(7, 8),
        num_frames=28,
        tubelet_size=4,
        model_name='vit_small',
        crop_size=(105, 33),
        pred_depth=6,
        pred_embed_dim=384,
        uniform_power=True,
        use_mask_tokens=True,
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
        use_sdpa=False,
    )
    print("    Encoder and Predictor initialized successfully")
    
    # Count parameters
    enc_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    pred_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    print(f"    Encoder params: {enc_params:,}")
    print(f"    Predictor params: {pred_params:,}")
    
except Exception as e:
    print(f"    ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Forward pass
print("\n[4] Testing forward pass...")
try:
    # Create dummy input: (batch, in_channels, time_bins, channels, freq_bins)
    # Shape: (B, 1, T, H, W) = (2, 1, 28, 105, 33)
    dummy_input = torch.randn(2, 1, 28, 105, 33).to(device)
    print(f"    Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = encoder(dummy_input)
    print(f"    Encoder output shape: {output.shape}")
    print("    Forward pass successful!")
    
except Exception as e:
    print(f"    ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Verification complete!")
print("=" * 60)
