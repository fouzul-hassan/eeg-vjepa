# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ZuCo Spector Pickle Preprocessing Pipeline
# 
# This notebook preprocesses ZuCo spectro.pickle files for:
# 1. **EEG-VJEPA** pretraining (spectrograms from time windows)
# 2. **EEG-to-Text** decoding (full sentence EEG + text labels)
# 
# **Subject-Based Splits:**
# - Train: ZAB, ZDM, ZGW, ZJM, ZJN, ZJS, ZKB, ZKH, ZKW (9 subjects)
# - Val: ZMG (1 subject)  
# - Test: ZPH (1 subject)

# %% [markdown]
# ## 1. Imports & Configuration

# %%
import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Configuration
TRAIN_SUBJECTS = ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW']
VAL_SUBJECTS = ['ZMG']
TEST_SUBJECTS = ['ZPH']

# Paths - Update these to match your setup
BASE_DIR = Path(r"c:\MSc Files\MSc Project\E2T-w-VJEPA\e2t-cloned-amirhojati\eeg-vjepa")
PICKLE_DIR = BASE_DIR / "src" / "datasets" / "ZuCo"

# Pickle file paths
PICKLE_FILES = {
    'task1-SR': PICKLE_DIR / "task1-SR" / "pickle" / "task1-SR-dataset-spectro.pickle",
    'task2-NR': PICKLE_DIR / "task2-NR" / "pickle" / "task2-NR-dataset-spectro.pickle",
    'task3-TSR': PICKLE_DIR / "task3-TSR" / "pickle" / "task3-TSR-dataset-spectro.pickle",
}

# Output directories
OUTPUT_DIR_VJEPA = BASE_DIR / "src" / "datasets" / "preprocessed_vjepa"
OUTPUT_DIR_E2T = BASE_DIR / "src" / "datasets" / "preprocessed_e2t"

# EEG-VJEPA preprocessing parameters
TIME_WINDOW = 512  # ~1 sec at 500 Hz
N_FFT = 64
HOP_LENGTH = 16
OVERLAP = 0.5  # 50% overlap between windows

# EEG-to-Text preprocessing parameters
MAX_EEG_LENGTH = 8000  # Maximum EEG sequence length (pad/truncate)
TARGET_CHANNELS = 105  # Expected number of EEG channels

print(f"Base Directory: {BASE_DIR}")
print(f"Output VJEPA: {OUTPUT_DIR_VJEPA}")
print(f"Output E2T: {OUTPUT_DIR_E2T}")

# %% [markdown]
# ## 2. Load & Explore Pickle Files

# %%
def load_pickle_data(pickle_path: Path) -> Dict:
    """Load a pickle file and return the data."""
    print(f"Loading: {pickle_path}")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data


def explore_pickle_data(data: Dict, task_name: str):
    """Explore the structure of a pickle file."""
    print(f"\n{'='*60}")
    print(f"Exploring: {task_name}")
    print(f"{'='*60}")
    
    subjects = list(data.keys())
    print(f"Subjects ({len(subjects)}): {subjects}")
    
    total_samples = 0
    sample_shapes = []
    
    for subj in subjects:
        valid_count = 0
        for sent in data[subj]:
            if sent is not None:
                valid_count += 1
                total_samples += 1
                raw = sent['sentence_level_EEG']['rawData']
                sample_shapes.append(raw.shape)
        print(f"  {subj}: {valid_count} valid sentences")
    
    if sample_shapes:
        channels = set(s[0] for s in sample_shapes)
        time_lengths = [s[1] for s in sample_shapes]
        print(f"\nData Statistics:")
        print(f"  Channels: {channels}")
        print(f"  Time range: [{min(time_lengths)}, {max(time_lengths)}]")
        print(f"  Mean length: {np.mean(time_lengths):.1f}")
        print(f"  Total valid samples: {total_samples}")
    
    return total_samples


# Load and explore all pickle files
all_data = {}
for task_name, pickle_path in PICKLE_FILES.items():
    if pickle_path.exists():
        all_data[task_name] = load_pickle_data(pickle_path)
        explore_pickle_data(all_data[task_name], task_name)
    else:
        print(f"⚠ Not found: {pickle_path}")

# %% [markdown]
# ## 3. EEG-VJEPA Preprocessing Functions

# %%
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


def extract_windows_and_spectrograms(
    raw_eeg: np.ndarray,
    time_window: int = 512,
    overlap: float = 0.5,
    n_fft: int = 64,
    hop_length: int = 16
) -> List[np.ndarray]:
    """
    Extract overlapping time windows and compute spectrograms.
    
    Args:
        raw_eeg: Raw EEG data (channels, time)
        time_window: Window size in samples
        overlap: Overlap ratio (0.0 to 0.9)
        n_fft: FFT size
        hop_length: Hop length
        
    Returns:
        List of spectrograms, each of shape (channels, freq_bins, time_bins)
    """
    n_channels, n_samples = raw_eeg.shape
    step_size = int(time_window * (1 - overlap))
    
    spectrograms = []
    
    # Skip if too short
    if n_samples < time_window // 2:
        return spectrograms
    
    # Extract overlapping windows
    start = 0
    while start + time_window <= n_samples:
        window_data = raw_eeg[:, start:start + time_window]
        spectrogram = compute_spectrogram(window_data, n_fft, hop_length)
        
        # Z-score normalize per channel
        mean = spectrogram.mean(axis=(1, 2), keepdims=True)
        std = spectrogram.std(axis=(1, 2), keepdims=True) + 1e-8
        spectrogram = (spectrogram - mean) / std
        
        spectrograms.append(spectrogram)
        start += step_size
    
    # Also get one window from the end if we have leftovers
    if n_samples >= time_window and start < n_samples - time_window // 2:
        window_data = raw_eeg[:, -time_window:]
        spectrogram = compute_spectrogram(window_data, n_fft, hop_length)
        mean = spectrogram.mean(axis=(1, 2), keepdims=True)
        std = spectrogram.std(axis=(1, 2), keepdims=True) + 1e-8
        spectrogram = (spectrogram - mean) / std
        spectrograms.append(spectrogram)
    
    return spectrograms


def get_subject_split(subject_id: str) -> str:
    """Determine which split a subject belongs to."""
    if subject_id in TRAIN_SUBJECTS:
        return 'train'
    elif subject_id in VAL_SUBJECTS:
        return 'val'
    elif subject_id in TEST_SUBJECTS:
        return 'test'
    else:
        print(f"⚠ Unknown subject {subject_id}, defaulting to train")
        return 'train'

# %% [markdown]
# ## 4. Process EEG-VJEPA Data

# %%
def process_vjepa_data(all_data: Dict, output_dir: Path):
    """
    Process all pickle data for EEG-VJEPA pretraining.
    
    Creates folder structure:
    output_dir/
    ├── task1-SR/
    │   ├── train/
    │   │   ├── ZAB_s0_w0.pt
    │   │   └── ...
    │   ├── val/
    │   └── test/
    ├── task2-NR/
    │   └── ...
    └── task3-TSR/
        └── ...
    """
    print("\n" + "="*60)
    print("EEG-VJEPA PREPROCESSING")
    print("="*60)
    
    stats = defaultdict(lambda: defaultdict(int))
    
    for task_name, task_data in all_data.items():
        print(f"\nProcessing {task_name}...")
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            split_dir = output_dir / task_name / split
            split_dir.mkdir(parents=True, exist_ok=True)
        
        sample_counts = {'train': 0, 'val': 0, 'test': 0}
        
        for subject_id, sentences in tqdm(task_data.items(), desc=task_name):
            split = get_subject_split(subject_id)
            split_dir = output_dir / task_name / split
            
            for sent_idx, sent in enumerate(sentences):
                if sent is None:
                    continue
                
                raw_eeg = sent['sentence_level_EEG']['rawData']
                
                # Extract spectrograms from overlapping windows
                spectrograms = extract_windows_and_spectrograms(
                    raw_eeg,
                    time_window=TIME_WINDOW,
                    overlap=OVERLAP,
                    n_fft=N_FFT,
                    hop_length=HOP_LENGTH
                )
                
                # Save each spectrogram as separate .pt file
                for win_idx, spec in enumerate(spectrograms):
                    # Reshape for 3D ViT: (1, time_bins, channels, freq_bins)
                    spec_tensor = torch.from_numpy(spec).float()
                    spec_tensor = spec_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, T, C, F)
                    
                    filename = f"{subject_id}_s{sent_idx}_w{win_idx}.pt"
                    pt_path = split_dir / filename
                    
                    torch.save({
                        'spectrogram': spec_tensor,
                        'subject': subject_id,
                        'sentence_idx': sent_idx,
                        'window_idx': win_idx,
                        'task': task_name
                    }, pt_path)
                    
                    sample_counts[split] += 1
        
        # Print stats for this task
        print(f"  {task_name} complete:")
        for split, count in sample_counts.items():
            print(f"    {split}: {count} samples")
            stats[task_name][split] = count
    
    return stats

# Process EEG-VJEPA data
vjepa_stats = process_vjepa_data(all_data, OUTPUT_DIR_VJEPA)

# %% [markdown]
# ## 5. EEG-to-Text Preprocessing Functions

# %%
def preprocess_eeg_for_e2t(
    raw_eeg: np.ndarray,
    max_length: int = MAX_EEG_LENGTH,
    target_channels: int = TARGET_CHANNELS
) -> torch.Tensor:
    """
    Preprocess raw EEG for EEG-to-Text decoding.
    
    - Pad or truncate to max_length
    - Z-score normalize
    
    Args:
        raw_eeg: Raw EEG data (channels, time)
        max_length: Target time dimension
        target_channels: Expected number of channels
        
    Returns:
        Preprocessed EEG tensor (channels, time)
    """
    n_channels, n_samples = raw_eeg.shape
    
    # Validate channels
    if n_channels != target_channels:
        print(f"⚠ Channel mismatch: expected {target_channels}, got {n_channels}")
    
    # Pad or truncate time dimension
    if n_samples < max_length:
        # Pad with zeros
        padded = np.zeros((n_channels, max_length), dtype=np.float32)
        padded[:, :n_samples] = raw_eeg
        eeg_data = padded
        actual_length = n_samples
    else:
        # Truncate
        eeg_data = raw_eeg[:, :max_length].astype(np.float32)
        actual_length = max_length
    
    # Z-score normalize per channel
    mean = eeg_data.mean(axis=1, keepdims=True)
    std = eeg_data.std(axis=1, keepdims=True) + 1e-8
    eeg_data = (eeg_data - mean) / std
    
    return torch.from_numpy(eeg_data), actual_length

# %% [markdown]
# ## 6. Process EEG-to-Text Data

# %%
def process_e2t_data(all_data: Dict, output_dir: Path):
    """
    Process all pickle data for EEG-to-Text decoding.
    
    Creates folder structure:
    output_dir/
    ├── task1-SR/
    │   ├── train/
    │   │   ├── ZAB_s0.pt
    │   │   └── ...
    │   ├── val/
    │   └── test/
    ├── task2-NR/
    │   └── ...
    └── task3-TSR/
        └── ...
    
    Each .pt file contains:
        - eeg: (channels, time) preprocessed EEG
        - text: sentence content
        - actual_length: original EEG length before padding
        - subject: subject ID
        - task: task name
    """
    print("\n" + "="*60)
    print("EEG-TO-TEXT PREPROCESSING")
    print("="*60)
    
    stats = defaultdict(lambda: defaultdict(int))
    
    for task_name, task_data in all_data.items():
        print(f"\nProcessing {task_name}...")
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            split_dir = output_dir / task_name / split
            split_dir.mkdir(parents=True, exist_ok=True)
        
        sample_counts = {'train': 0, 'val': 0, 'test': 0}
        text_lengths = []
        
        for subject_id, sentences in tqdm(task_data.items(), desc=task_name):
            split = get_subject_split(subject_id)
            split_dir = output_dir / task_name / split
            
            for sent_idx, sent in enumerate(sentences):
                if sent is None:
                    continue
                
                raw_eeg = sent['sentence_level_EEG']['rawData']
                content = sent.get('content', '')
                
                # Skip if no text content
                if not content or len(content.strip()) == 0:
                    continue
                
                # Preprocess EEG
                eeg_tensor, actual_length = preprocess_eeg_for_e2t(raw_eeg)
                
                # Save as .pt file
                filename = f"{subject_id}_s{sent_idx}.pt"
                pt_path = split_dir / filename
                
                torch.save({
                    'eeg': eeg_tensor,
                    'text': content,
                    'actual_length': actual_length,
                    'subject': subject_id,
                    'sentence_idx': sent_idx,
                    'task': task_name
                }, pt_path)
                
                sample_counts[split] += 1
                text_lengths.append(len(content))
        
        # Print stats for this task
        print(f"  {task_name} complete:")
        for split, count in sample_counts.items():
            print(f"    {split}: {count} samples")
            stats[task_name][split] = count
        
        if text_lengths:
            print(f"  Text length: min={min(text_lengths)}, max={max(text_lengths)}, mean={np.mean(text_lengths):.1f}")
    
    return stats

# Process EEG-to-Text data
e2t_stats = process_e2t_data(all_data, OUTPUT_DIR_E2T)

# %% [markdown]
# ## 7. Create PyTorch Dataset Classes

# %%
from torch.utils.data import Dataset, DataLoader

class EEGVJEPADataset(Dataset):
    """
    PyTorch Dataset for EEG-VJEPA pretraining.
    
    Loads preprocessed spectrogram .pt files.
    Output shape: (1, T, C, F) - ready for 3D ViT
    """
    
    def __init__(self, data_dir: str, split: str = 'train', tasks: List[str] = None):
        """
        Args:
            data_dir: Path to preprocessed_vjepa directory
            split: 'train', 'val', or 'test'
            tasks: List of tasks to include, e.g., ['task1-SR', 'task2-NR']
                   If None, includes all available tasks
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.pt_files = []
        
        # Find all .pt files for specified tasks
        if tasks is None:
            tasks = ['task1-SR', 'task2-NR', 'task3-TSR']
        
        for task in tasks:
            task_split_dir = self.data_dir / task / split
            if task_split_dir.exists():
                files = sorted(task_split_dir.glob('*.pt'))
                self.pt_files.extend(files)
        
        print(f"[{split}] Loaded {len(self.pt_files)} samples from {len(tasks)} tasks")
    
    def __len__(self):
        return len(self.pt_files)
    
    def __getitem__(self, idx):
        data = torch.load(self.pt_files[idx])
        return data['spectrogram']  # (1, T, C, F)


class EEGToTextDataset(Dataset):
    """
    PyTorch Dataset for EEG-to-Text decoding.
    
    Returns:
        eeg: (channels, time) normalized EEG
        text: sentence content string
        actual_length: original EEG length before padding
    """
    
    def __init__(self, data_dir: str, split: str = 'train', tasks: List[str] = None):
        """
        Args:
            data_dir: Path to preprocessed_e2t directory
            split: 'train', 'val', or 'test'
            tasks: List of tasks to include
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.pt_files = []
        
        if tasks is None:
            tasks = ['task1-SR', 'task2-NR', 'task3-TSR']
        
        for task in tasks:
            task_split_dir = self.data_dir / task / split
            if task_split_dir.exists():
                files = sorted(task_split_dir.glob('*.pt'))
                self.pt_files.extend(files)
        
        print(f"[{split}] Loaded {len(self.pt_files)} samples from {len(tasks)} tasks")
    
    def __len__(self):
        return len(self.pt_files)
    
    def __getitem__(self, idx):
        data = torch.load(self.pt_files[idx])
        return {
            'eeg': data['eeg'],
            'text': data['text'],
            'actual_length': data['actual_length'],
            'subject': data['subject']
        }

# %% [markdown]
# ## 8. Validation & Statistics

# %%
def validate_preprocessing():
    """Validate the preprocessed data."""
    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)
    
    # Check EEG-VJEPA data
    print("\n📊 EEG-VJEPA Preprocessed Data:")
    for task in ['task1-SR', 'task2-NR', 'task3-TSR']:
        task_dir = OUTPUT_DIR_VJEPA / task
        if task_dir.exists():
            for split in ['train', 'val', 'test']:
                split_dir = task_dir / split
                if split_dir.exists():
                    count = len(list(split_dir.glob('*.pt')))
                    if count > 0:
                        # Load one sample to check shape
                        sample_file = list(split_dir.glob('*.pt'))[0]
                        sample = torch.load(sample_file)
                        shape = sample['spectrogram'].shape
                        print(f"  {task}/{split}: {count} samples, shape={shape}")
    
    # Check EEG-to-Text data
    print("\n📊 EEG-to-Text Preprocessed Data:")
    for task in ['task1-SR', 'task2-NR', 'task3-TSR']:
        task_dir = OUTPUT_DIR_E2T / task
        if task_dir.exists():
            for split in ['train', 'val', 'test']:
                split_dir = task_dir / split
                if split_dir.exists():
                    count = len(list(split_dir.glob('*.pt')))
                    if count > 0:
                        # Load one sample to check
                        sample_file = list(split_dir.glob('*.pt'))[0]
                        sample = torch.load(sample_file)
                        shape = sample['eeg'].shape
                        text_preview = sample['text'][:50] + "..." if len(sample['text']) > 50 else sample['text']
                        print(f"  {task}/{split}: {count} samples, EEG shape={shape}")
                        print(f"    Example text: \"{text_preview}\"")
    
    # Test dataset classes
    print("\n🔧 Testing Dataset Classes:")
    try:
        vjepa_ds = EEGVJEPADataset(OUTPUT_DIR_VJEPA, split='train')
        if len(vjepa_ds) > 0:
            sample = vjepa_ds[0]
            print(f"  EEGVJEPADataset: ✓ ({len(vjepa_ds)} samples, shape={sample.shape})")
    except Exception as e:
        print(f"  EEGVJEPADataset: ✗ ({e})")
    
    try:
        e2t_ds = EEGToTextDataset(OUTPUT_DIR_E2T, split='train')
        if len(e2t_ds) > 0:
            sample = e2t_ds[0]
            print(f"  EEGToTextDataset: ✓ ({len(e2t_ds)} samples)")
            print(f"    EEG shape: {sample['eeg'].shape}")
            print(f"    Text: \"{sample['text'][:50]}...\"")
    except Exception as e:
        print(f"  EEGToTextDataset: ✗ ({e})")

# Run validation
validate_preprocessing()

# %% [markdown]
# ## 9. Summary

# %%
print("\n" + "="*60)
print("PREPROCESSING COMPLETE!")
print("="*60)

print(f"""
📁 Output Directories:
  EEG-VJEPA: {OUTPUT_DIR_VJEPA}
  EEG-to-Text: {OUTPUT_DIR_E2T}

📊 Subject Splits:
  Train ({len(TRAIN_SUBJECTS)} subjects): {TRAIN_SUBJECTS}
  Val ({len(VAL_SUBJECTS)} subjects): {VAL_SUBJECTS}
  Test ({len(TEST_SUBJECTS)} subjects): {TEST_SUBJECTS}

🔧 EEG-VJEPA Parameters:
  Time Window: {TIME_WINDOW} samples
  Overlap: {OVERLAP*100:.0f}%
  FFT Size: {N_FFT}
  Hop Length: {HOP_LENGTH}
  Output Shape: (1, T, C, F) for 3D ViT

🔧 EEG-to-Text Parameters:
  Max EEG Length: {MAX_EEG_LENGTH}
  Channels: {TARGET_CHANNELS}
  Output Shape: (channels, time) + text

📚 Usage:
```python
# Load EEG-VJEPA dataset
from preprocess_spector import EEGVJEPADataset
train_ds = EEGVJEPADataset('{OUTPUT_DIR_VJEPA}', split='train')
val_ds = EEGVJEPADataset('{OUTPUT_DIR_VJEPA}', split='val')

# Load EEG-to-Text dataset
from preprocess_spector import EEGToTextDataset
train_ds = EEGToTextDataset('{OUTPUT_DIR_E2T}', split='train')
```
""")
