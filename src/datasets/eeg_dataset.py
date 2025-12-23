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
    
    Supports TWO formats:
    1. Old format: Individual sample files (sample_000000.pt, sample_000001.pt, ...)
       - Each file contains: {'spectrogram': tensor(C, F, T), ...}
    2. Subject format: Per-subject files (ZAB.pt, ZMG.pt, ...)
       - Each file contains: {'spectrograms': tensor(N, 1, T, C, F), 'metadata': [...]}
       - Uses LAZY LOADING to avoid memory issues with large files
    
    Output shape for 3D ViT: (1, T, C, F)
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
        load_to_ram: bool = False,
        cache_size: int = 2  # Number of subject files to keep in cache
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
            load_to_ram: If True, load all data to RAM (only for sample format)
            cache_size: Number of subject files to cache (for subject format)
        """
        self.transform = transform
        self.load_to_ram = load_to_ram
        self.split = split
        self.format = None  # Will be 'sample' or 'subject'
        self.cache_size = cache_size
        
        # Determine data directory
        if data_dir is not None:
            self.data_dir = data_dir
        elif hf_repo is not None and hf_filename is not None:
            self.data_dir = self._download_from_hf(hf_repo, hf_filename, hf_token)
        else:
            raise ValueError("Must provide either data_dir or hf_repo+hf_filename")
        
        # Detect format and find .pt files
        all_pt_files = sorted([
            os.path.join(self.data_dir, f) 
            for f in os.listdir(self.data_dir) 
            if f.endswith('.pt')
        ])
        
        # Check for sample-based format (sample_*.pt)
        sample_files = [f for f in all_pt_files if os.path.basename(f).startswith('sample_')]
        
        # Check for subject-based format (e.g., ZAB.pt, ZMG.pt)
        subject_files = [f for f in all_pt_files if not os.path.basename(f).startswith('sample_')]
        
        if len(sample_files) > 0:
            # Old sample-based format
            self.format = 'sample'
            self.pt_files = self._apply_split(sample_files, split, train_ratio, val_ratio, test_ratio, seed)
            logger.info(f"[{split}] Found {len(self.pt_files)} sample files in {self.data_dir}")
            
            # Optionally load all to RAM
            self.data_cache = None
            if load_to_ram:
                logger.info("Loading all data to RAM...")
                self.data_cache = []
                for pt_file in tqdm(self.pt_files, desc=f"Loading {split} data"):
                    self.data_cache.append(torch.load(pt_file))
                logger.info("Data loaded to RAM")
            
        elif len(subject_files) > 0:
            # New subject-based format - LAZY LOADING
            self.format = 'subject'
            self.subject_files = subject_files
            
            # Build index: scan files to get sample counts WITHOUT loading full data
            # This creates a mapping: global_idx -> (file_idx, local_idx)
            self.file_info = []  # List of (filepath, num_samples, start_idx)
            self.total_samples = 0
            
            logger.info(f"Scanning {len(subject_files)} subject files...")
            for pt_file in subject_files:
                # Quick scan: load just metadata, not full tensor
                # For very large files, we can use torch.load with map_location
                data = torch.load(pt_file, weights_only=False)
                if 'spectrograms' in data:
                    num_samples = data['spectrograms'].shape[0]
                    self.file_info.append({
                        'path': pt_file,
                        'num_samples': num_samples,
                        'start_idx': self.total_samples,
                        'subject': data.get('subject', os.path.basename(pt_file).replace('.pt', ''))
                    })
                    self.total_samples += num_samples
                del data  # Free memory immediately
            
            logger.info(f"[{split}] Found {self.total_samples} samples across {len(self.file_info)} subject files (LAZY LOADING)")
            
            # LRU cache for loaded files
            self._file_cache = {}
            self._cache_order = []
            
        else:
            raise ValueError(f"No .pt files found in {self.data_dir}")
    
    def _get_file_data(self, file_idx: int):
        """Load file data with LRU caching."""
        if file_idx in self._file_cache:
            # Move to end of cache order (most recently used)
            self._cache_order.remove(file_idx)
            self._cache_order.append(file_idx)
            return self._file_cache[file_idx]
        
        # Load the file
        file_info = self.file_info[file_idx]
        data = torch.load(file_info['path'], weights_only=False)
        
        # Add to cache
        self._file_cache[file_idx] = data
        self._cache_order.append(file_idx)
        
        # Evict oldest if cache is full
        while len(self._cache_order) > self.cache_size:
            oldest = self._cache_order.pop(0)
            del self._file_cache[oldest]
        
        return data
    
    def _global_to_local(self, global_idx: int):
        """Convert global sample index to (file_idx, local_idx)."""
        for file_idx, info in enumerate(self.file_info):
            if global_idx < info['start_idx'] + info['num_samples']:
                local_idx = global_idx - info['start_idx']
                return file_idx, local_idx
        raise IndexError(f"Index {global_idx} out of range")
    
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
        if self.format == 'subject':
            return self.total_samples
        else:
            return len(self.pt_files)
    
    def __getitem__(self, idx):
        if self.format == 'subject':
            # Subject format with lazy loading
            file_idx, local_idx = self._global_to_local(idx)
            data = self._get_file_data(file_idx)
            spectrogram = data['spectrograms'][local_idx]  # Already (1, T, C, F)
        else:
            # Sample format: load from file
            if self.data_cache is not None:
                data = self.data_cache[idx]
            else:
                data = torch.load(self.pt_files[idx])
            
            spectrogram = data['spectrogram']  # (channels, freq_bins, time_bins)
            
            # Reshape for 3D ViT: (1, time_bins, channels, freq_bins)
            spectrogram = spectrogram.permute(2, 0, 1).unsqueeze(0)  # (1, T, C, F)
        
        if self.transform is not None:
            spectrogram = self.transform(spectrogram)
        
        return spectrogram


def make_eeg_dataset(
    data_dir = None,  # Can be str or list of str
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
        data_dir: Local directory (str) or list of directories with .pt files
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
    from torch.utils.data import ConcatDataset
    
    # Handle multiple directories
    if isinstance(data_dir, list):
        datasets = []
        for d in data_dir:
            try:
                ds = ZuCoSpectrogramDataset(
                    data_dir=d,
                    split=split,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                    load_to_ram=load_to_ram
                )
                datasets.append(ds)
                logger.info(f"Loaded {len(ds)} samples from {d}")
            except Exception as e:
                logger.warning(f"Could not load from {d}: {e}")
        
        if len(datasets) == 0:
            raise ValueError(f"No valid datasets found in {data_dir}")
        elif len(datasets) == 1:
            dataset = datasets[0]
        else:
            dataset = ConcatDataset(datasets)
            logger.info(f"Total combined: {len(dataset)} samples from {len(datasets)} directories")
    else:
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
