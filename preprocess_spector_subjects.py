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
# # ZuCo Spector Pickle Preprocessing Pipeline (Subject-Based)
# 
# This notebook preprocesses ZuCo spectro.pickle files for:
# 1. **EEG-VJEPA** pretraining (spectrograms from time windows)
# 2. **EEG-to-Text** decoding (full sentence EEG + text labels)
# 
# **OPTIMIZED**: Creates ONE .pt file PER SUBJECT per task.
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
ALL_SUBJECTS = TRAIN_SUBJECTS + VAL_SUBJECTS + TEST_SUBJECTS

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
OUTPUT_DIR_VJEPA = BASE_DIR / "src" / "datasets" / "preprocessed_vjepa_subjects"
OUTPUT_DIR_E2T = BASE_DIR / "src" / "datasets" / "preprocessed_e2t_subjects"

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
    """
    n_channels, n_samples = eeg_data.shape
    n_freq_bins = n_fft // 2 + 1
    n_time_bins = (n_samples - n_fft) // hop_length + 1
    
    if n_time_bins <= 0:
        pad_len = n_fft + hop_length - n_samples
        eeg_data = np.pad(eeg_data, ((0, 0), (0, pad_len)), mode='constant')
        n_samples = eeg_data.shape[1]
        n_time_bins = (n_samples - n_fft) // hop_length + 1
    
    window = np.hanning(n_fft)
    spectrogram = np.zeros((n_channels, n_freq_bins, n_time_bins), dtype=np.float32)
    
    for ch in range(n_channels):
        for t in range(n_time_bins):
            start_idx = t * hop_length
            segment = eeg_data[ch, start_idx:start_idx + n_fft] * window
            fft_result = np.fft.rfft(segment)
            spectrogram[ch, :, t] = np.abs(fft_result)
    
    spectrogram = np.log1p(spectrogram)
    return spectrogram


def extract_windows_and_spectrograms(
    raw_eeg: np.ndarray,
    time_window: int = 512,
    overlap: float = 0.5,
    n_fft: int = 64,
    hop_length: int = 16
) -> List[np.ndarray]:
    """Extract overlapping time windows and compute spectrograms."""
    n_channels, n_samples = raw_eeg.shape
    step_size = int(time_window * (1 - overlap))
    spectrograms = []
    
    if n_samples < time_window // 2:
        return spectrograms
    
    start = 0
    while start + time_window <= n_samples:
        window_data = raw_eeg[:, start:start + time_window]
        spectrogram = compute_spectrogram(window_data, n_fft, hop_length)
        mean = spectrogram.mean(axis=(1, 2), keepdims=True)
        std = spectrogram.std(axis=(1, 2), keepdims=True) + 1e-8
        spectrogram = (spectrogram - mean) / std
        spectrograms.append(spectrogram)
        start += step_size
    
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
        return 'train'

# %% [markdown]
# ## 4. Process EEG-VJEPA Data (SUBJECT-BASED)

# %%
def process_vjepa_data_by_subject(all_data: Dict, output_dir: Path):
    """
    Process all pickle data for EEG-VJEPA pretraining.
    
    Creates SUBJECT-BASED folder structure:
    output_dir/
    ├── task1-SR/
    │   ├── train/
    │   │   ├── ZAB.pt      # All spectrograms for subject ZAB
    │   │   ├── ZDM.pt
    │   │   └── ...
    │   ├── val/
    │   │   └── ZMG.pt
    │   └── test/
    │       └── ZPH.pt
    ├── task2-NR/
    │   └── ...
    └── task3-TSR/
        └── ...
    
    Each .pt file contains:
        - spectrograms: tensor of shape (N, 1, T, C, F) for this subject
        - metadata: list of dicts with sentence_idx, window_idx, task
        - subject: subject ID
        - split: train/val/test
    """
    print("\n" + "="*60)
    print("EEG-VJEPA PREPROCESSING (SUBJECT-BASED)")
    print("="*60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    
    for task_name, task_data in all_data.items():
        print(f"\n{'='*40}")
        print(f"Processing {task_name}...")
        print(f"{'='*40}")
        
        task_dir = output_dir / task_name
        
        # Create split directories
        for split in ['train', 'val', 'test']:
            (task_dir / split).mkdir(parents=True, exist_ok=True)
        
        task_stats = {'train': {}, 'val': {}, 'test': {}}
        
        for subject_id, sentences in tqdm(task_data.items(), desc=f"{task_name} subjects"):
            split = get_subject_split(subject_id)
            
            # Collect all spectrograms for this subject
            subject_spectrograms = []
            subject_metadata = []
            
            for sent_idx, sent in enumerate(sentences):
                if sent is None:
                    continue
                
                raw_eeg = sent['sentence_level_EEG']['rawData']
                
                spectrograms = extract_windows_and_spectrograms(
                    raw_eeg,
                    time_window=TIME_WINDOW,
                    overlap=OVERLAP,
                    n_fft=N_FFT,
                    hop_length=HOP_LENGTH
                )
                
                for win_idx, spec in enumerate(spectrograms):
                    spec_tensor = torch.from_numpy(spec).float()
                    spec_tensor = spec_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, T, C, F)
                    
                    subject_spectrograms.append(spec_tensor)
                    subject_metadata.append({
                        'sentence_idx': sent_idx,
                        'window_idx': win_idx,
                        'task': task_name
                    })
            
            # Save subject file
            if len(subject_spectrograms) > 0:
                spectrograms_tensor = torch.stack(subject_spectrograms, dim=0)
                
                pt_path = task_dir / split / f"{subject_id}.pt"
                torch.save({
                    'spectrograms': spectrograms_tensor,
                    'metadata': subject_metadata,
                    'subject': subject_id,
                    'split': split
                }, pt_path)
                
                file_size_mb = pt_path.stat().st_size / (1024 * 1024)
                task_stats[split][subject_id] = len(subject_spectrograms)
                print(f"  {split}/{subject_id}.pt: {len(subject_spectrograms)} samples, {file_size_mb:.1f} MB")
        
        stats[task_name] = task_stats
    
    return stats

# Process EEG-VJEPA data
vjepa_stats = process_vjepa_data_by_subject(all_data, OUTPUT_DIR_VJEPA)

# %% [markdown]
# ## 5. EEG-to-Text Preprocessing Functions

# %%
def preprocess_eeg_for_e2t(
    raw_eeg: np.ndarray,
    max_length: int = MAX_EEG_LENGTH,
    target_channels: int = TARGET_CHANNELS
) -> Tuple[torch.Tensor, int]:
    """Preprocess raw EEG for EEG-to-Text decoding."""
    n_channels, n_samples = raw_eeg.shape
    
    if n_samples < max_length:
        padded = np.zeros((n_channels, max_length), dtype=np.float32)
        padded[:, :n_samples] = raw_eeg
        eeg_data = padded
        actual_length = n_samples
    else:
        eeg_data = raw_eeg[:, :max_length].astype(np.float32)
        actual_length = max_length
    
    mean = eeg_data.mean(axis=1, keepdims=True)
    std = eeg_data.std(axis=1, keepdims=True) + 1e-8
    eeg_data = (eeg_data - mean) / std
    
    return torch.from_numpy(eeg_data), actual_length

# %% [markdown]
# ## 6. Process EEG-to-Text Data (SUBJECT-BASED)

# %%
def process_e2t_data_by_subject(all_data: Dict, output_dir: Path):
    """
    Process all pickle data for EEG-to-Text decoding.
    
    Creates SUBJECT-BASED folder structure:
    output_dir/
    ├── task1-SR/
    │   ├── train/
    │   │   ├── ZAB.pt      # All EEG+text for subject ZAB
    │   │   ├── ZDM.pt
    │   │   └── ...
    │   ├── val/
    │   │   └── ZMG.pt
    │   └── test/
    │       └── ZPH.pt
    └── ...
    
    Each .pt file contains:
        - eeg: tensor of shape (N, channels, time) for this subject
        - texts: list of sentence strings
        - actual_lengths: list of original EEG lengths
        - metadata: list of dicts with sentence_idx, task
        - subject: subject ID
        - split: train/val/test
    """
    print("\n" + "="*60)
    print("EEG-TO-TEXT PREPROCESSING (SUBJECT-BASED)")
    print("="*60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    
    for task_name, task_data in all_data.items():
        print(f"\n{'='*40}")
        print(f"Processing {task_name}...")
        print(f"{'='*40}")
        
        task_dir = output_dir / task_name
        
        for split in ['train', 'val', 'test']:
            (task_dir / split).mkdir(parents=True, exist_ok=True)
        
        task_stats = {'train': {}, 'val': {}, 'test': {}}
        
        for subject_id, sentences in tqdm(task_data.items(), desc=f"{task_name} subjects"):
            split = get_subject_split(subject_id)
            
            subject_eeg = []
            subject_texts = []
            subject_lengths = []
            subject_metadata = []
            
            for sent_idx, sent in enumerate(sentences):
                if sent is None:
                    continue
                
                raw_eeg = sent['sentence_level_EEG']['rawData']
                content = sent.get('content', '')
                
                if not content or len(content.strip()) == 0:
                    continue
                
                eeg_tensor, actual_length = preprocess_eeg_for_e2t(raw_eeg)
                
                subject_eeg.append(eeg_tensor)
                subject_texts.append(content)
                subject_lengths.append(actual_length)
                subject_metadata.append({
                    'sentence_idx': sent_idx,
                    'task': task_name
                })
            
            # Save subject file
            if len(subject_eeg) > 0:
                eeg_tensor = torch.stack(subject_eeg, dim=0)
                
                pt_path = task_dir / split / f"{subject_id}.pt"
                torch.save({
                    'eeg': eeg_tensor,
                    'texts': subject_texts,
                    'actual_lengths': subject_lengths,
                    'metadata': subject_metadata,
                    'subject': subject_id,
                    'split': split
                }, pt_path)
                
                file_size_mb = pt_path.stat().st_size / (1024 * 1024)
                task_stats[split][subject_id] = len(subject_eeg)
                print(f"  {split}/{subject_id}.pt: {len(subject_eeg)} samples, {file_size_mb:.1f} MB")
        
        stats[task_name] = task_stats
    
    return stats

# Process EEG-to-Text data
e2t_stats = process_e2t_data_by_subject(all_data, OUTPUT_DIR_E2T)

# %% [markdown]
# ## 7. Create PyTorch Dataset Classes (Subject-Based)

# %%
from torch.utils.data import Dataset, DataLoader

class EEGVJEPADatasetSubjects(Dataset):
    """
    PyTorch Dataset for EEG-VJEPA pretraining (loads subject-based .pt files).
    """
    
    def __init__(self, data_dir: str, split: str = 'train', tasks: List[str] = None,
                 subjects: List[str] = None):
        """
        Args:
            data_dir: Path to preprocessed_vjepa_subjects directory
            split: 'train', 'val', or 'test'
            tasks: List of tasks to include (default: all)
            subjects: List of specific subjects to include (default: all in split)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        
        if tasks is None:
            tasks = ['task1-SR', 'task2-NR', 'task3-TSR']
        
        # Determine which subjects to load
        if subjects is None:
            if split == 'train':
                subjects = TRAIN_SUBJECTS
            elif split == 'val':
                subjects = VAL_SUBJECTS
            else:
                subjects = TEST_SUBJECTS
        
        # Load all subject files
        all_spectrograms = []
        all_metadata = []
        
        for task in tasks:
            for subject in subjects:
                pt_path = self.data_dir / task / split / f"{subject}.pt"
                if pt_path.exists():
                    data = torch.load(pt_path, weights_only=False)
                    all_spectrograms.append(data['spectrograms'])
                    # Add subject to each metadata entry
                    for m in data['metadata']:
                        m['subject'] = subject
                    all_metadata.extend(data['metadata'])
        
        if len(all_spectrograms) > 0:
            self.spectrograms = torch.cat(all_spectrograms, dim=0)
            self.metadata = all_metadata
        else:
            self.spectrograms = torch.tensor([])
            self.metadata = []
        
        print(f"[{split}] Loaded {len(self)} samples from {len(subjects)} subjects, {len(tasks)} tasks")
    
    def __len__(self):
        return len(self.spectrograms)
    
    def __getitem__(self, idx):
        return self.spectrograms[idx]
    
    def get_metadata(self, idx):
        return self.metadata[idx]


class EEGToTextDatasetSubjects(Dataset):
    """
    PyTorch Dataset for EEG-to-Text decoding (loads subject-based .pt files).
    """
    
    def __init__(self, data_dir: str, split: str = 'train', tasks: List[str] = None,
                 subjects: List[str] = None):
        """
        Args:
            data_dir: Path to preprocessed_e2t_subjects directory
            split: 'train', 'val', or 'test'
            tasks: List of tasks to include (default: all)
            subjects: List of specific subjects to include (default: all in split)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        
        if tasks is None:
            tasks = ['task1-SR', 'task2-NR', 'task3-TSR']
        
        if subjects is None:
            if split == 'train':
                subjects = TRAIN_SUBJECTS
            elif split == 'val':
                subjects = VAL_SUBJECTS
            else:
                subjects = TEST_SUBJECTS
        
        all_eeg = []
        all_texts = []
        all_lengths = []
        all_metadata = []
        
        for task in tasks:
            for subject in subjects:
                pt_path = self.data_dir / task / split / f"{subject}.pt"
                if pt_path.exists():
                    data = torch.load(pt_path, weights_only=False)
                    all_eeg.append(data['eeg'])
                    all_texts.extend(data['texts'])
                    all_lengths.extend(data['actual_lengths'])
                    for m in data['metadata']:
                        m['subject'] = subject
                    all_metadata.extend(data['metadata'])
        
        if len(all_eeg) > 0:
            self.eeg = torch.cat(all_eeg, dim=0)
            self.texts = all_texts
            self.actual_lengths = all_lengths
            self.metadata = all_metadata
        else:
            self.eeg = torch.tensor([])
            self.texts = []
            self.actual_lengths = []
            self.metadata = []
        
        print(f"[{split}] Loaded {len(self)} samples from {len(subjects)} subjects, {len(tasks)} tasks")
    
    def __len__(self):
        return len(self.eeg)
    
    def __getitem__(self, idx):
        return {
            'eeg': self.eeg[idx],
            'text': self.texts[idx],
            'actual_length': self.actual_lengths[idx],
            'subject': self.metadata[idx]['subject']
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
    print("\n📊 EEG-VJEPA Preprocessed Data (Subject-Based):")
    total_vjepa_size = 0
    for task in ['task1-SR', 'task2-NR', 'task3-TSR']:
        print(f"\n  {task}:")
        task_dir = OUTPUT_DIR_VJEPA / task
        if task_dir.exists():
            for split in ['train', 'val', 'test']:
                split_dir = task_dir / split
                if split_dir.exists():
                    files = list(split_dir.glob('*.pt'))
                    split_total = 0
                    for pt_path in sorted(files):
                        data = torch.load(pt_path, weights_only=False)
                        n_samples = data['spectrograms'].shape[0]
                        size_mb = pt_path.stat().st_size / (1024 * 1024)
                        total_vjepa_size += size_mb
                        split_total += n_samples
                        print(f"    {split}/{pt_path.name}: {n_samples} samples, {size_mb:.1f} MB")
                    print(f"    {split} total: {split_total} samples")
    print(f"\n  📦 Total VJEPA size: {total_vjepa_size:.1f} MB ({total_vjepa_size/1024:.2f} GB)")
    
    # Check EEG-to-Text data
    print("\n📊 EEG-to-Text Preprocessed Data (Subject-Based):")
    total_e2t_size = 0
    for task in ['task1-SR', 'task2-NR', 'task3-TSR']:
        print(f"\n  {task}:")
        task_dir = OUTPUT_DIR_E2T / task
        if task_dir.exists():
            for split in ['train', 'val', 'test']:
                split_dir = task_dir / split
                if split_dir.exists():
                    files = list(split_dir.glob('*.pt'))
                    split_total = 0
                    for pt_path in sorted(files):
                        data = torch.load(pt_path, weights_only=False)
                        n_samples = data['eeg'].shape[0]
                        size_mb = pt_path.stat().st_size / (1024 * 1024)
                        total_e2t_size += size_mb
                        split_total += n_samples
                        print(f"    {split}/{pt_path.name}: {n_samples} samples, {size_mb:.1f} MB")
                    print(f"    {split} total: {split_total} samples")
    print(f"\n  📦 Total E2T size: {total_e2t_size:.1f} MB ({total_e2t_size/1024:.2f} GB)")
    
    # Test dataset classes
    print("\n🔧 Testing Dataset Classes:")
    try:
        vjepa_ds = EEGVJEPADatasetSubjects(OUTPUT_DIR_VJEPA, split='train')
        if len(vjepa_ds) > 0:
            sample = vjepa_ds[0]
            print(f"  EEGVJEPADatasetSubjects: ✓ ({len(vjepa_ds)} samples, shape={sample.shape})")
    except Exception as e:
        print(f"  EEGVJEPADatasetSubjects: ✗ ({e})")
    
    try:
        e2t_ds = EEGToTextDatasetSubjects(OUTPUT_DIR_E2T, split='train')
        if len(e2t_ds) > 0:
            sample = e2t_ds[0]
            print(f"  EEGToTextDatasetSubjects: ✓ ({len(e2t_ds)} samples)")
            print(f"    EEG shape: {sample['eeg'].shape}")
            text_preview = sample['text'][:50] + "..." if len(sample['text']) > 50 else sample['text']
            print(f"    Text: \"{text_preview}\"")
    except Exception as e:
        print(f"  EEGToTextDatasetSubjects: ✗ ({e})")

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

📊 File Structure (SUBJECT-BASED):
  preprocessed_vjepa_subjects/
  ├── task1-SR/
  │   ├── train/
  │   │   ├── ZAB.pt    # All spectrograms for ZAB
  │   │   ├── ZDM.pt    # All spectrograms for ZDM
  │   │   ├── ZGW.pt
  │   │   ├── ZJM.pt
  │   │   ├── ZJN.pt
  │   │   ├── ZJS.pt
  │   │   ├── ZKB.pt
  │   │   ├── ZKH.pt
  │   │   └── ZKW.pt    # (9 train subjects)
  │   ├── val/
  │   │   └── ZMG.pt    # (1 val subject)
  │   └── test/
  │       └── ZPH.pt    # (1 test subject)
  ├── task2-NR/
  │   └── ...
  └── task3-TSR/
      └── ...

📊 Subject Splits:
  Train ({len(TRAIN_SUBJECTS)} subjects): {TRAIN_SUBJECTS}
  Val ({len(VAL_SUBJECTS)} subjects): {VAL_SUBJECTS}
  Test ({len(TEST_SUBJECTS)} subjects): {TEST_SUBJECTS}

📦 Total files per format:
  ~{len(ALL_SUBJECTS) * 3} files for VJEPA (11 subjects × 3 tasks)
  ~{len(ALL_SUBJECTS) * 3} files for E2T (11 subjects × 3 tasks)

📚 Usage:
```python
# Load EEG-VJEPA dataset (all train subjects)
from preprocess_spector_subjects import EEGVJEPADatasetSubjects
train_ds = EEGVJEPADatasetSubjects('{OUTPUT_DIR_VJEPA}', split='train')

# Load specific subjects only
custom_ds = EEGVJEPADatasetSubjects(
    '{OUTPUT_DIR_VJEPA}', 
    split='train', 
    subjects=['ZAB', 'ZDM']  # Only load these subjects
)

# Load EEG-to-Text dataset  
from preprocess_spector_subjects import EEGToTextDatasetSubjects
train_ds = EEGToTextDatasetSubjects('{OUTPUT_DIR_E2T}', split='train')
```

✨ Benefits of subject-based format:
  - ~33 files per task instead of thousands
  - Easy to add/remove subjects
  - Good file size (~1-5 GB each)
  - Natural split organization
""")
