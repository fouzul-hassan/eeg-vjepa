"""
Merge all task splits into one combined train/val/test folder.

Customize the TASK_PATHS below with your actual paths.

Usage:
    python merge_splits.py
"""

import os
import shutil

# ============================================================
# CONFIGURE YOUR PATHS HERE
# ============================================================

# Base output directory for combined data
OUTPUT_DIR = "/content/drive/MyDrive/2. MSc Files/0. MSc-Research/EEGtoText-Spectro/eeg-vjepa/src/datasets/preprocessed/combined"

# Paths to each task's split directories
# Format: { 'task_name': '/path/to/task/folder/containing/train/val/test' }
TASK_PATHS = {
    'task1-SR': '/content/drive/MyDrive/2. MSc Files/0. MSc-Research/EEGtoText-Spectro/eeg-vjepa/src/datasets/preprocessed/task1-SR',
    'task2-NR': '/content/drive/MyDrive/2. MSc Files/0. MSc-Research/EEGtoText-Spectro/eeg-vjepa/src/datasets/preprocessed/task2-NR',
    'task3-TSR': '/content/drive/MyDrive/2. MSc Files/0. MSc-Research/EEGtoText-Spectro/eeg-vjepa/src/datasets/preprocessed/task3-TSR',
}

# ============================================================


def merge_splits():
    """Merge train/val/test splits from all tasks into one combined folder."""
    
    splits = ['train', 'val', 'test']
    
    print("=" * 60)
    print("Merging Task Splits into Combined Folder")
    print("=" * 60)
    print(f"Output: {OUTPUT_DIR}")
    
    # Create output split directories
    for split in splits:
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)
    
    # Track counts
    counts = {split: 0 for split in splits}
    
    # Copy files from each task
    for task_name, task_dir in TASK_PATHS.items():
        
        if not os.path.exists(task_dir):
            print(f"\n[{task_name}] NOT FOUND: {task_dir}")
            continue
        
        print(f"\n[{task_name}]")
        print(f"  Path: {task_dir}")
        
        for split in splits:
            split_dir = os.path.join(task_dir, split)
            
            if not os.path.exists(split_dir):
                print(f"  {split}: No split folder found")
                continue
            
            # Get .pt files
            pt_files = [f for f in os.listdir(split_dir) if f.endswith('.pt')]
            
            if len(pt_files) == 0:
                print(f"  {split}: Empty")
                continue
            
            # Copy files with task prefix to avoid name collisions
            dest_dir = os.path.join(OUTPUT_DIR, split)
            for pt_file in pt_files:
                # Add task prefix to filename
                new_name = f"{task_name}_{pt_file}"
                src = os.path.join(split_dir, pt_file)
                dst = os.path.join(dest_dir, new_name)
                shutil.copy2(src, dst)
            
            counts[split] += len(pt_files)
            print(f"  {split}: {len(pt_files)} files copied")
    
    # Summary
    print(f"\n{'='*60}")
    print("✓ Merge complete!")
    print(f"{'='*60}")
    print(f"\nCombined counts:")
    for split in splits:
        split_dir = os.path.join(OUTPUT_DIR, split)
        if os.path.exists(split_dir):
            actual_count = len([f for f in os.listdir(split_dir) if f.endswith('.pt')])
            print(f"  {split}: {actual_count} samples")
    
    print(f"\nPaths for your config:")
    print(f"  Train: {os.path.join(OUTPUT_DIR, 'train')}")
    print(f"  Val:   {os.path.join(OUTPUT_DIR, 'val')}")
    print(f"  Test:  {os.path.join(OUTPUT_DIR, 'test')}")


if __name__ == "__main__":
    merge_splits()
