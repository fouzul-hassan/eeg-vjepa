"""
Preprocess ZuCo data to .pt spectrogram files.

Run this script after downloading data from HuggingFace to convert 
the raw EEG pickle files to individual .pt spectrograms for fast training.

Usage:
    1. First download data: python download_zuco_hf.py --token YOUR_TOKEN
    2. Then preprocess: python preprocess_zuco.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.datasets.eeg_dataset import preprocess_zuco_to_pt

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input paths - check both local and downloaded locations
PICKLE_LOCATIONS = [
    # Downloaded from HuggingFace
    os.path.join(BASE_DIR, "src", "datasets", "ZuCo", "downloads"),
    # Original local location
    os.path.join(BASE_DIR, "src", "datasets", "ZuCo", "task2-NR", "pickle"),
]

# Tasks to process
TASKS = [
    "task1-SR-dataset-spectro.pickle",
    "task2-NR-dataset-spectro.pickle",
    "task3-TSR-dataset-spectro.pickle",
]

# Output base directory
OUTPUT_BASE = os.path.join(BASE_DIR, "src", "datasets", "ZuCo", "preprocessed")


def find_pickle_file(filename):
    """Find pickle file in known locations."""
    for location in PICKLE_LOCATIONS:
        path = os.path.join(location, filename)
        if os.path.exists(path):
            return path
    return None


def main():
    print("=" * 60)
    print("ZuCo EEG Spectrogram Preprocessing")
    print("=" * 60)
    
    # Find and process each task
    processed = 0
    for task_file in TASKS:
        pickle_path = find_pickle_file(task_file)
        
        if pickle_path is None:
            print(f"\n⚠ {task_file} not found, skipping...")
            continue
        
        # Extract task name for output directory
        task_name = task_file.replace("-dataset-spectro.pickle", "")
        output_dir = os.path.join(OUTPUT_BASE, task_name)
        
        print(f"\n[{task_name}]")
        print(f"  Input: {pickle_path}")
        print(f"  Output: {output_dir}")
        
        try:
            num_samples, shape = preprocess_zuco_to_pt(
                pickle_path=pickle_path,
                output_dir=output_dir,
                time_window=512,      # ~1 sec @ 500Hz
                n_fft=64,             # FFT window size
                hop_length=16,        # Hop between windows
                overlap=0.5           # 50% overlap
            )
            print(f"  ✓ Created {num_samples} samples, shape: {shape}")
            processed += 1
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\n{'='*60}")
    print(f"Preprocessing complete! Processed {processed}/{len(TASKS)} tasks")
    print(f"Output directory: {OUTPUT_BASE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
