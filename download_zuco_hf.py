"""
Download ZuCo dataset files from HuggingFace.

This script downloads the preprocessed pickle files from:
https://huggingface.co/datasets/fouzulhassan/zuco/

Usage:
    conda activate zuco-hdf5
    python download_zuco_hf.py --token YOUR_HF_TOKEN

Or set HF_TOKEN environment variable:
    set HF_TOKEN=your_token
    python download_zuco_hf.py
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download, login

# Files to download
FILES = [
    "data/task1-SR-dataset-spectro.pickle",
    "data/task2-NR-dataset-spectro.pickle", 
    "data/task3-TSR-dataset-spectro.pickle",
]

# HuggingFace dataset info
REPO_ID = "fouzulhassan/zuco"
REPO_TYPE = "dataset"

# Default output directory
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "src", "datasets", "ZuCo", "downloads"
)


def download_zuco_files(output_dir: str, token: str = None):
    """
    Download ZuCo pickle files from HuggingFace.
    
    Args:
        output_dir: Directory to save downloaded files
        token: HuggingFace token (optional if already logged in)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Login if token provided
    if token:
        print("Logging in to HuggingFace...")
        login(token=token)
    
    print(f"\nDownloading ZuCo files from: {REPO_ID}")
    print(f"Output directory: {output_dir}\n")
    
    downloaded_files = []
    
    for file_path in FILES:
        filename = os.path.basename(file_path)
        print(f"Downloading {filename}...")
        
        try:
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=file_path,
                repo_type=REPO_TYPE,
                local_dir=output_dir,
                token=token,
            )
            
            # Move file from nested structure to output dir root
            final_path = os.path.join(output_dir, filename)
            if local_path != final_path and os.path.exists(local_path):
                import shutil
                shutil.move(local_path, final_path)
                local_path = final_path
            
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            print(f"  ✓ Downloaded: {filename} ({size_mb:.1f} MB)")
            downloaded_files.append(local_path)
            
        except Exception as e:
            print(f"  ✗ Error downloading {filename}: {e}")
    
    print(f"\n{'='*50}")
    print(f"Download complete! Files saved to: {output_dir}")
    print(f"Downloaded {len(downloaded_files)}/{len(FILES)} files")
    print(f"{'='*50}")
    
    return downloaded_files


def main():
    parser = argparse.ArgumentParser(description="Download ZuCo dataset from HuggingFace")
    parser.add_argument(
        '--token', 
        type=str, 
        default=os.environ.get('HF_TOKEN'),
        help='HuggingFace token (or set HF_TOKEN env var)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default=DEFAULT_OUTPUT_DIR,
        help='Output directory for downloaded files'
    )
    args = parser.parse_args()
    
    if not args.token:
        print("Warning: No HuggingFace token provided.")
        print("If the dataset is private, provide token via --token or HF_TOKEN env var")
        print()
    
    download_zuco_files(output_dir=args.output, token=args.token)


if __name__ == "__main__":
    main()
