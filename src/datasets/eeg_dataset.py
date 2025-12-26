"""
ZuCo EEG Spectrogram Dataset for EEG-VJEPA Pretraining

This module provides:
1. Preprocessing script to convert raw EEG pickle to .pt spectrograms
2. Dataset class for efficient loading of preprocessed .pt files
"""

import os
import pickle
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from logging import getLogger

logger = getLogger(__name__)


def compute_spectrogram(eeg_data: np.ndarray, n_fft: int = 64, hop_length: int = 16) -> np.ndarray:
    """
    Compute STFT spectrogram for each EEG channel.
    
    Args:
        eeg_data: Raw EEG data of shape (n_channels, n_samples)
        n_fft: FFT window size
        hop_length: Hop length between windows
        
    Returns:
        Spectrogram of shape (n_channels, n_freq_bins, n_time_bins)
    """
    n_channels, n_samples = eeg_data.shape
    
    # Compute number of output dimensions
    n_freq_bins = n_fft // 2 + 1
    n_time_bins = (n_samples - n_fft) // hop_length + 1
    
    if n_time_bins <= 0:
        # Pad if too short
        pad_len = n_fft + hop_length - n_samples
        eeg_data = np.pad(eeg_data, ((0, 0), (0, pad_len)), mode='constant')
        n_samples = eeg_data.shape[1]
        n_time_bins = (n_samples - n_fft) // hop_length + 1
    
    # Hanning window
    window = np.hanning(n_fft)
    
    spectrogram = np.zeros((n_channels, n_freq_bins, n_time_bins), dtype=np.float32)
    
    for ch in range(n_channels):
        for t in range(n_time_bins):
            start_idx = t * hop_length
            segment = eeg_data[ch, start_idx:start_idx + n_fft] * window
            fft_result = np.fft.rfft(segment)
            spectrogram[ch, :, t] = np.abs(fft_result)
    
    # Log-scale magnitude (add small epsilon for stability)
    spectrogram = np.log1p(spectrogram)
    
    return spectrogram


def preprocess_zuco_to_pt(
    pickle_path: str,
    output_dir: str,
    time_window: int = 512,
    n_fft: int = 64,
    hop_length: int = 16,
    overlap: float = 0.5
):
    """
    Preprocess ZuCo pickle file to individual .pt spectrogram files.
    
    This extracts overlapping time windows from each sentence and computes
    spectrograms, saving each as a separate .pt file for fast loading.
    
    Args:
        pickle_path: Path to ZuCo pickle file
        output_dir: Directory to save .pt files
        time_window: Number of raw EEG samples per window
        n_fft: FFT size for spectrogram
        hop_length: Hop length for spectrogram
        overlap: Overlap ratio between consecutive windows (0.0 to 0.9)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading pickle from: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    step_size = int(time_window * (1 - overlap))
    sample_idx = 0
    metadata = []
    
    print(f"Processing {len(data)} subjects...")
    for subject_name, sentences in tqdm(data.items()):
        for sent_idx, sent in enumerate(sentences):
            if sent is None:
                continue
            
            raw_eeg = sent['sentence_level_EEG']['rawData']  # (channels, time)
            content = sent.get('content', '')
            n_channels, n_samples = raw_eeg.shape
            
            # Skip very short sentences
            if n_samples < time_window // 2:
                continue
            
            # Extract windows
            start = 0
            while start + time_window <= n_samples:
                window_data = raw_eeg[:, start:start + time_window]
                
                # Compute spectrogram
                spectrogram = compute_spectrogram(window_data, n_fft, hop_length)
                
                # Z-score normalize per channel
                mean = spectrogram.mean(axis=(1, 2), keepdims=True)
                std = spectrogram.std(axis=(1, 2), keepdims=True) + 1e-8
                spectrogram = (spectrogram - mean) / std
                
                # Save as .pt
                pt_path = os.path.join(output_dir, f"sample_{sample_idx:06d}.pt")
                torch.save({
                    'spectrogram': torch.from_numpy(spectrogram).float(),
                    'subject': subject_name,
                    'sentence_idx': sent_idx,
                    'window_start': start,
                }, pt_path)
                
                metadata.append({
                    'path': pt_path,
                    'subject': subject_name,
                    'sent_idx': sent_idx,
                    'content': content[:100],
                })
                
                sample_idx += 1
                start += step_size
            
            # Also get one window from the end if we have leftovers
            if n_samples >= time_window and start < n_samples - time_window // 2:
                window_data = raw_eeg[:, -time_window:]
                spectrogram = compute_spectrogram(window_data, n_fft, hop_length)
                mean = spectrogram.mean(axis=(1, 2), keepdims=True)
                std = spectrogram.std(axis=(1, 2), keepdims=True) + 1e-8
                spectrogram = (spectrogram - mean) / std
                
                pt_path = os.path.join(output_dir, f"sample_{sample_idx:06d}.pt")
                torch.save({
                    'spectrogram': torch.from_numpy(spectrogram).float(),
                    'subject': subject_name,
                    'sentence_idx': sent_idx,
                    'window_start': n_samples - time_window,
                }, pt_path)
                
                metadata.append({
                    'path': pt_path,
                    'subject': subject_name,
                    'sent_idx': sent_idx,
                })
                sample_idx += 1
    
    # Save metadata
    metadata_path = os.path.join(output_dir, 'metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    # Get shape info from first sample
    first_sample = torch.load(os.path.join(output_dir, 'sample_000000.pt'))
    shape = first_sample['spectrogram'].shape
    
    print(f"\nPreprocessing complete!")
    print(f"Total samples: {sample_idx}")
    print(f"Spectrogram shape: {shape} (channels, freq_bins, time_bins)")
    print(f"Saved to: {output_dir}")
    
    return sample_idx, shape


class ZuCoSpectrogramDataset(Dataset):
    """
    Dataset for loading preprocessed ZuCo EEG spectrograms from .pt files.
    
    Expected spectrogram shape: (n_channels, n_freq_bins, n_time_bins)
    Output shape for 3D ViT: (1, n_time_bins, n_channels, n_freq_bins)
    
    Supports:
    - Local directory loading
    - HuggingFace download (zip files)
    - Train/val/test split (70/15/15)
    """
    
    def __init__(
        self,
        data_dir: str = None,
        hf_repo: str = None,
        hf_filename: str = None,
        hf_token: str = None,
        split: str = 'train',  # 'train', 'val', 'test', or 'all'
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        transform=None,
        load_to_ram: bool = False
    ):
        """
        Args:
            data_dir: Local directory containing preprocessed .pt files
            hf_repo: HuggingFace repo ID (e.g., 'fouzul-hassan/zuco-preprocessed')
            hf_filename: Filename of zip on HF (e.g., 'task2-NR.zip')
            hf_token: HuggingFace token (optional)
            split: 'train', 'val', 'test', or 'all'
            train_ratio: Ratio for training (default 0.70 = 70%)
            val_ratio: Ratio for validation (default 0.15 = 15%)
            test_ratio: Ratio for testing (default 0.15 = 15%)
            seed: Random seed for reproducible splits
            transform: Optional transform to apply
            load_to_ram: If True, load all data to RAM for faster access
        """
        self.transform = transform
        self.load_to_ram = load_to_ram
        self.split = split
        
        # Determine data directory
        if data_dir is not None:
            self.data_dir = data_dir
        elif hf_repo is not None and hf_filename is not None:
            self.data_dir = self._download_from_hf(hf_repo, hf_filename, hf_token)
        else:
            raise ValueError("Must provide either data_dir or hf_repo+hf_filename")
        
        # Find all .pt files
        all_pt_files = sorted([
            os.path.join(self.data_dir, f)
            for f in os.listdir(self.data_dir)
            if f.endswith('.pt')
        ])
        
        if len(all_pt_files) == 0:
            raise ValueError(f"No .pt files found in {self.data_dir}")
        
        # Apply train/val/test split
        self.pt_files = self._apply_split(all_pt_files, split, train_ratio, val_ratio, test_ratio, seed)
        
        logger.info(f"[{split}] Found {len(self.pt_files)}/{len(all_pt_files)} samples in {self.data_dir}")
        
        # Optionally load all to RAM
        self.data_cache = None
        if load_to_ram:
            logger.info("Loading all data to RAM...")
            self.data_cache = []
            for pt_file in tqdm(self.pt_files, desc=f"Loading {split} data"):
                self.data_cache.append(torch.load(pt_file))
            logger.info("Data loaded to RAM")
    
    def _download_from_hf(self, repo_id: str, filename: str, token: str = None) -> str:
        """Download and extract zip file from HuggingFace."""
        from huggingface_hub import hf_hub_download
        import zipfile
        
        logger.info(f"Downloading {filename} from {repo_id}...")
        
        # Download zip file
        zip_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            token=token
        )
        
        # Extract to a local directory
        extract_dir = os.path.join(
            os.path.dirname(zip_path),
            os.path.splitext(os.path.basename(filename))[0]
        )
        
        if not os.path.exists(extract_dir):
            logger.info(f"Extracting to {extract_dir}...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_dir)
        
        # Find the directory with .pt files
        for root, dirs, files in os.walk(extract_dir):
            if any(f.endswith('.pt') for f in files):
                return root
        
        return extract_dir
    
    def _apply_split(self, all_files: list, split: str, train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> list:
        """Apply train/val/test split to file list (70/15/15)."""
        if split == 'all':
            return all_files
        
        # Reproducible shuffle
        import random
        rng = random.Random(seed)
        indices = list(range(len(all_files)))
        rng.shuffle(indices)
        
        # Calculate split boundaries
        n_total = len(all_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        # n_test = remainder
        
        if split == 'train':
            selected_indices = indices[:n_train]
        elif split == 'val':
            selected_indices = indices[n_train:n_train + n_val]
        elif split == 'test':
            selected_indices = indices[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split: {split}. Use 'train', 'val', 'test', or 'all'")
        
        return [all_files[i] for i in sorted(selected_indices)]
    
    def __len__(self):
        return len(self.pt_files)
    
    def __getitem__(self, idx):
        if self.data_cache is not None:
            data = self.data_cache[idx]
        else:
            data = torch.load(self.pt_files[idx])
        
        spectrogram = data['spectrogram']  # (channels, freq_bins, time_bins)
        
        # Reshape for 3D ViT: (1, time_bins, channels, freq_bins)
        # This treats: in_channels=1, T=time_bins, H=channels, W=freq_bins
        spectrogram = spectrogram.permute(2, 0, 1).unsqueeze(0)  # (1, T, C, F)
        
        if self.transform is not None:
            spectrogram = self.transform(spectrogram)
        
        return spectrogram


def make_eeg_dataset(
    data_dir: str = None,
    hf_repo: str = None,
    hf_filename: str = None,
    hf_token: str = None,
    split: str = 'train',
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    batch_size: int = 16,
    num_workers: int = 4,
    world_size: int = 1,
    rank: int = 0,
    collator=None,
    pin_mem: bool = True,
    drop_last: bool = True,
    load_to_ram: bool = False,
):
    """
    Create EEG spectrogram dataset and dataloader.
    
    Args:
        data_dir: Local directory with .pt files
        hf_repo: HuggingFace repo ID (alternative to data_dir)
        hf_filename: Zip filename on HF
        hf_token: HuggingFace token
        split: 'train', 'val', 'test', or 'all'
        train_ratio: Training set ratio (default 70%)
        val_ratio: Validation set ratio (default 15%)
        test_ratio: Test set ratio (default 15%)
        ...
    
    Returns:
        dataset, dataloader, sampler
    """
    dataset = ZuCoSpectrogramDataset(
        data_dir=data_dir,
        hf_repo=hf_repo,
        hf_filename=hf_filename,
        hf_token=hf_token,
        split=split,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        load_to_ram=load_to_ram
    )
    
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=(split == 'train')
        )
    else:
        if split == 'train':
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=pin_mem,
        drop_last=drop_last if split == 'train' else False,
        persistent_workers=num_workers > 0
    )
    
    logger.info(f"Created EEG dataloader [{split}]: {len(dataset)} samples, batch_size={batch_size}")
    
    return dataset, dataloader, sampler



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess ZuCo data to .pt spectrograms")
    parser.add_argument('--pickle', type=str, required=True, help='Path to ZuCo pickle file')
    parser.add_argument('--output', type=str, required=True, help='Output directory for .pt files')
    parser.add_argument('--time_window', type=int, default=512, help='Time window size in samples')
    parser.add_argument('--n_fft', type=int, default=64, help='FFT size')
    parser.add_argument('--hop_length', type=int, default=16, help='Hop length')
    parser.add_argument('--overlap', type=float, default=0.5, help='Overlap between windows')
    
    args = parser.parse_args()
    
    preprocess_zuco_to_pt(
        pickle_path=args.pickle,
        output_dir=args.output,
        time_window=args.time_window,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        overlap=args.overlap
    )
