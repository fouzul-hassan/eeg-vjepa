"""
Simple script to split .pt files into train/val/test folders (70/15/15).
Run this in Colab after extracting the tar files.

Usage:
    python split_data.py --data_dir /path/to/folder/with/pt/files
"""

import os
import random
import shutil
import argparse

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42


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


def split_data(data_dir):
    """Split .pt files into train/val/test folders (70/15/15)."""
    
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
    
    splits = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }
    
    print(f"\nSplitting {n_total} files:")
    print(f"  train: {len(train_indices)} ({TRAIN_RATIO*100:.0f}%)")
    print(f"  val:   {len(val_indices)} ({VAL_RATIO*100:.0f}%)")
    print(f"  test:  {len(test_indices)} ({TEST_RATIO*100:.0f}%)")
    
    # Create split directories and move files
    for split_name, split_indices in splits.items():
        split_dir = os.path.join(data_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        for idx in split_indices:
            src = os.path.join(pt_dir, pt_files[idx])
            dst = os.path.join(split_dir, pt_files[idx])
            shutil.move(src, dst)
        
        print(f"  Moved {len(split_indices)} files to {split_name}/")
    
    print("\n✓ Split complete!")
    print(f"  Train: {os.path.join(data_dir, 'train')}")
    print(f"  Val:   {os.path.join(data_dir, 'val')}")
    print(f"  Test:  {os.path.join(data_dir, 'test')}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Split .pt files into train/val/test")
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Directory containing .pt files (or parent directory)')
    args = parser.parse_args()
    
    print("=" * 50)
    print("Split Data into Train/Val/Test (70/15/15)")
    print("=" * 50)
    
    split_data(args.data_dir)


if __name__ == "__main__":
    main()
