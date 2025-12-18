"""
Download preprocessed ZuCo .pt files from HuggingFace, extract and split them.

Downloads from: https://huggingface.co/datasets/fouzulhassan/zuco/tree/main/preprocessed

Usage:
    conda activate zuco-hdf5
    python download_preprocessed_hf.py
    
    # Or with token for private repos:
    python download_preprocessed_hf.py --token YOUR_HF_TOKEN
"""

import os
import argparse
import tarfile
import random
import shutil
from huggingface_hub import hf_hub_download

# Configuration
REPO_ID = "fouzulhassan/zuco"
REPO_TYPE = "dataset"

# Files to download (from preprocessed folder) - .tar format
FILES = [
    "preprocessed/task1-SR.tar",
    "preprocessed/task2-NR.tar",
    "preprocessed/task3-TSR.tar",
]

# Output directory
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "src", "datasets", "preprocessed"
)

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42


def split_data(task_dir: str, task_name: str):
    """Split .pt files into train/val/test folders (70/15/15)."""
    
    # Find all .pt files
    pt_files = sorted([f for f in os.listdir(task_dir) if f.endswith('.pt')])
    
    if len(pt_files) == 0:
        print(f"  No .pt files found in {task_dir}")
        return
    
    # Shuffle with fixed seed
    random.seed(SEED)
    indices = list(range(len(pt_files)))
    random.shuffle(indices)
    
    # Calculate split sizes
    n_total = len(pt_files)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    
    # Split indices
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Create split directories
    splits = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }
    
    print(f"  Splitting {n_total} files: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    
    for split_name, split_indices in splits.items():
        split_dir = os.path.join(task_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        for idx in split_indices:
            src = os.path.join(task_dir, pt_files[idx])
            dst = os.path.join(split_dir, pt_files[idx])
            shutil.move(src, dst)
        
        print(f"    {split_name}: {len(split_indices)} files")
    
    return len(train_indices), len(val_indices), len(test_indices)


def download_and_extract(token: str = None, do_split: bool = True):
    """Download zip files from HuggingFace, extract and optionally split them."""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    for file_path in FILES:
        filename = os.path.basename(file_path)
        task_name = filename.replace('.zip', '')
        extract_dir = os.path.join(OUTPUT_DIR, task_name)
        
        print(f"[{task_name}]")
        
        # Check if already split
        train_dir = os.path.join(extract_dir, 'train')
        if os.path.exists(train_dir) and len(os.listdir(train_dir)) > 0:
            pt_count = len([f for f in os.listdir(train_dir) if f.endswith('.pt')])
            print(f"  Already split (train has {pt_count} files), skipping...")
            continue
        
        # Check if already extracted (but not split)
        needs_download = True
        if os.path.exists(extract_dir):
            pt_files = [f for f in os.listdir(extract_dir) if f.endswith('.pt')]
            if len(pt_files) > 0:
                print(f"  Already extracted ({len(pt_files)} files)")
                needs_download = False
        
        if needs_download:
            # Download
            print(f"  Downloading {filename}...")
            try:
                zip_path = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=file_path,
                    repo_type=REPO_TYPE,
                    token=token,
                )
                print(f"  Downloaded to cache")
                
                # Extract
                print(f"  Extracting...")
                os.makedirs(extract_dir, exist_ok=True)
                
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    file_list = zf.namelist()
                    print(f"  Extracting {len(file_list)} files...")
                    zf.extractall(extract_dir)
                
                # Count .pt files
                pt_files = [f for f in os.listdir(extract_dir) if f.endswith('.pt')]
                print(f"  ✓ Extracted {len(pt_files)} .pt files")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue
        
        # Split into train/val/test
        if do_split:
            print(f"  Splitting (70/15/15)...")
            split_data(extract_dir, task_name)
    
    print(f"\n{'='*60}")
    print("Download and split complete!")
    print(f"{'='*60}")
    
    # Summary
    print("\nSummary:")
    for task in ["task1-SR", "task2-NR", "task3-TSR"]:
        task_dir = os.path.join(OUTPUT_DIR, task)
        if os.path.exists(task_dir):
            for split in ['train', 'val', 'test']:
                split_dir = os.path.join(task_dir, split)
                if os.path.exists(split_dir):
                    pt_count = len([f for f in os.listdir(split_dir) if f.endswith('.pt')])
                    print(f"  {task}/{split}: {pt_count} samples")
    
    print(f"\n{'='*60}")
    print("Ready for training!")
    print("Use these paths in your config:")
    print(f"  Train: {os.path.join(OUTPUT_DIR, 'task2-NR', 'train')}")
    print(f"  Val:   {os.path.join(OUTPUT_DIR, 'task2-NR', 'val')}")
    print(f"  Test:  {os.path.join(OUTPUT_DIR, 'task2-NR', 'test')}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Download and split preprocessed ZuCo data")
    parser.add_argument(
        '--token',
        type=str,
        default=os.environ.get('HF_TOKEN'),
        help='HuggingFace token (or set HF_TOKEN env var)'
    )
    parser.add_argument(
        '--no-split',
        action='store_true',
        help='Skip splitting into train/val/test'
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("Download & Split Preprocessed ZuCo Data from HuggingFace")
    print("=" * 60)
    print(f"Repo: {REPO_ID}")
    print(f"Split: {int(TRAIN_RATIO*100)}/{int(VAL_RATIO*100)}/{int(TEST_RATIO*100)} (train/val/test)")
    print()
    
    download_and_extract(token=args.token, do_split=not args.no_split)


if __name__ == "__main__":
    main()
