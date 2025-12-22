"""
Split .pt files into train/val/test folders based on SUBJECT ID.

Subject-based splitting:
  - Train: ZAB, ZDM, ZGW, ZJM, ZJN, ZJS, ZKB, ZKH, ZKW (9 subjects)
  - Val:   ZMG (1 subject)
  - Test:  ZPH (1 subject)

Usage:
    python split_data.py --data_dir /path/to/folder/with/pt/files
"""

import os
import shutil
import argparse
from collections import defaultdict

# Subject-based splits
TRAIN_SUBJECTS = ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW']
VAL_SUBJECTS = ['ZMG']
TEST_SUBJECTS = ['ZPH']


def find_pt_files_dir(base_dir):
    """Recursively find the directory containing .pt files."""
    # Check if .pt files are directly here
    direct_files = [f for f in os.listdir(base_dir) if f.endswith('.pt') and os.path.isfile(os.path.join(base_dir, f))]
    if len(direct_files) > 0:
        return base_dir, direct_files
    
    # Search subdirectories
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item not in ['train', 'val', 'test']:
            result = find_pt_files_dir(item_path)
            if result[0] is not None:
                return result
    
    return None, []


def extract_subject_id(filename):
    """Extract subject ID from filename (e.g., 'ZAB_task1_sample0.pt' -> 'ZAB')."""
    # Subject ID is the first part before underscore
    basename = os.path.splitext(filename)[0]
    parts = basename.split('_')
    if parts:
        return parts[0]
    return None


def split_data(data_dir):
    """Split .pt files into train/val/test folders based on subject ID."""
    
    print(f"Looking for .pt files in: {data_dir}")
    
    # Find the actual directory with .pt files
    pt_dir, pt_files = find_pt_files_dir(data_dir)
    
    if pt_dir is None or len(pt_files) == 0:
        print(f"ERROR: No .pt files found in {data_dir} or its subdirectories")
        return False
    
    pt_files = sorted(pt_files)
    print(f"Found {len(pt_files)} .pt files in: {pt_dir}")
    
    # Check if already split
    train_dir = os.path.join(data_dir, 'train')
    if os.path.exists(train_dir) and len(os.listdir(train_dir)) > 0:
        print(f"Already split! train folder has {len(os.listdir(train_dir))} files")
        return True
    
    # Group files by subject
    subject_files = defaultdict(list)
    unknown_files = []
    
    for f in pt_files:
        subject_id = extract_subject_id(f)
        if subject_id:
            subject_files[subject_id].append(f)
        else:
            unknown_files.append(f)
    
    print(f"\nFound {len(subject_files)} unique subjects: {sorted(subject_files.keys())}")
    for subj in sorted(subject_files.keys()):
        print(f"  {subj}: {len(subject_files[subj])} files")
    
    if unknown_files:
        print(f"\nWARNING: {len(unknown_files)} files with unknown subject ID")
    
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
            print(f"WARNING: Subject {subject_id} not in any split, adding to train")
            splits['train'].extend(files)
    
    n_total = len(pt_files)
    print(f"\nSplitting {n_total} files by subject:")
    print(f"  train: {len(splits['train'])} files ({len(TRAIN_SUBJECTS)} subjects: {TRAIN_SUBJECTS})")
    print(f"  val:   {len(splits['val'])} files ({len(VAL_SUBJECTS)} subjects: {VAL_SUBJECTS})")
    print(f"  test:  {len(splits['test'])} files ({len(TEST_SUBJECTS)} subjects: {TEST_SUBJECTS})")
    
    # Create split directories and move files
    for split_name, split_files in splits.items():
        split_dir = os.path.join(data_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        for filename in split_files:
            src = os.path.join(pt_dir, filename)
            dst = os.path.join(split_dir, filename)
            shutil.move(src, dst)
        
        print(f"  Moved {len(split_files)} files to {split_name}/")
    
    print("\n✓ Split complete!")
    print(f"  Train: {os.path.join(data_dir, 'train')}")
    print(f"  Val:   {os.path.join(data_dir, 'val')}")
    print(f"  Test:  {os.path.join(data_dir, 'test')}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Split .pt files into train/val/test by subject")
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Directory containing .pt files (or parent directory)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Split Data into Train/Val/Test (Subject-Based)")
    print("=" * 60)
    print(f"Train subjects: {TRAIN_SUBJECTS}")
    print(f"Val subjects:   {VAL_SUBJECTS}")
    print(f"Test subjects:  {TEST_SUBJECTS}")
    print("=" * 60)
    
    split_data(args.data_dir)


if __name__ == "__main__":
    main()
