"""
Download preprocessed ZuCo .pt files from HuggingFace, extract and split them.

Downloads from: https://huggingface.co/datasets/fouzulhassan/zuco/tree/main/preprocessed

Subject-based splitting:
  - Train: ZAB, ZDM, ZGW, ZJM, ZJN, ZJS, ZKB, ZKH, ZKW (9 subjects)
  - Val:   ZMG (1 subject)
  - Test:  ZPH (1 subject)

Usage:
    conda activate zuco-hdf5
    python download_preprocessed_hf.py
    
    # Or with token for private repos:
    python download_preprocessed_hf.py --token YOUR_HF_TOKEN
"""

import os
import argparse
import tarfile
import shutil
from collections import defaultdict
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
    "src", "datasets", "preprocessedv2"
)

# Subject-based splits
TRAIN_SUBJECTS = ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW']
VAL_SUBJECTS = ['ZMG']
TEST_SUBJECTS = ['ZPH']


def extract_subject_id(filename):
    """Extract subject ID from filename (e.g., 'ZAB_task1_sample0.pt' -> 'ZAB')."""
    basename = os.path.splitext(filename)[0]
    parts = basename.split('_')
    if parts:
        return parts[0]
    return None


def split_data(task_dir: str, task_name: str):
    """Split .pt files into train/val/test folders based on subject ID."""
    
    # Find all .pt files (recursively, in case they're in subdirectories)
    pt_files = []
    pt_dir = task_dir  # Directory where .pt files are located
    
    # Check if files are directly in task_dir or in a subdirectory
    direct_files = [f for f in os.listdir(task_dir) if f.endswith('.pt')]
    if len(direct_files) > 0:
        pt_files = sorted(direct_files)
        pt_dir = task_dir
    else:
        # Look in subdirectories (tar might extract with folder structure)
        for subdir in os.listdir(task_dir):
            subdir_path = os.path.join(task_dir, subdir)
            if os.path.isdir(subdir_path):
                sub_files = [f for f in os.listdir(subdir_path) if f.endswith('.pt')]
                if len(sub_files) > 0:
                    pt_files = sorted(sub_files)
                    pt_dir = subdir_path
                    break
    
    if len(pt_files) == 0:
        print(f"  No .pt files found in {task_dir}")
        return
    
    print(f"  Found .pt files in: {pt_dir}")
    
    # Group files by subject
    subject_files = defaultdict(list)
    for f in pt_files:
        subject_id = extract_subject_id(f)
        if subject_id:
            subject_files[subject_id].append(f)
    
    print(f"  Found {len(subject_files)} unique subjects")
    
    # Split by subject
    splits = {
        'train': [],
        'val': [],
        'test': []
    }
    
    for subject_id, files in subject_files.items():
        if subject_id in TRAIN_SUBJECTS:
            splits['train'].extend(files)
        elif subject_id in VAL_SUBJECTS:
            splits['val'].extend(files)
        elif subject_id in TEST_SUBJECTS:
            splits['test'].extend(files)
        else:
            print(f"  WARNING: Subject {subject_id} not in any split, adding to train")
            splits['train'].extend(files)
    
    n_total = len(pt_files)
    print(f"  Splitting {n_total} files by subject: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    
    # Create split directories and move files
    for split_name, split_files in splits.items():
        split_dir = os.path.join(task_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        for filename in split_files:
            src = os.path.join(pt_dir, filename)
            dst = os.path.join(split_dir, filename)
            shutil.move(src, dst)
        
        print(f"    {split_name}: {len(split_files)} files")
    
    return len(splits['train']), len(splits['val']), len(splits['test'])



def download_and_extract(token: str = None, do_split: bool = True):
    """Download tar files from HuggingFace, extract and optionally split them."""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    for file_path in FILES:
        filename = os.path.basename(file_path)
        task_name = filename.replace('.tar', '')
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
                tar_path = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=file_path,
                    repo_type=REPO_TYPE,
                    token=token,
                )
                print(f"  Downloaded to cache")
                
                # Extract - flatten nested paths, only extract .pt files
                print(f"  Extracting (flattening nested paths)...")
                os.makedirs(extract_dir, exist_ok=True)
                
                with tarfile.open(tar_path, 'r') as tf:
                    members = tf.getmembers()
                    pt_members = [m for m in members if m.name.endswith('.pt')]
                    print(f"  Found {len(pt_members)} .pt files in archive")
                    
                    # Extract .pt files directly to extract_dir (flatten paths)
                    for member in pt_members:
                        # Get just the filename, ignore the nested path
                        member.name = os.path.basename(member.name)
                        tf.extract(member, extract_dir)
                
                # Count extracted .pt files
                pt_files = [f for f in os.listdir(extract_dir) if f.endswith('.pt')]
                print(f"  ✓ Extracted {len(pt_files)} .pt files to {extract_dir}")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue
        
        # Split into train/val/test by subject
        if do_split:
            print(f"  Splitting by subject...")
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
    print(f"Split: Subject-based (Train: {len(TRAIN_SUBJECTS)}, Val: {len(VAL_SUBJECTS)}, Test: {len(TEST_SUBJECTS)} subjects)")
    print()
    
    download_and_extract(token=args.token, do_split=not args.no_split)


if __name__ == "__main__":
    main()
