#!/usr/bin/env python3
"""
Extract COCO images from zip files for VQA v2 dataset
"""

import os
import zipfile
from pathlib import Path
import argparse

def extract_coco_images(vqa_dir: str):
    """Extract COCO images from zip files in VQA v2 directory"""
    vqa_path = Path(vqa_dir)
    
    # Check for train2014 images
    train_zip = vqa_path / "train2014.zip"
    if train_zip.exists():
        print(f"📦 Found train2014.zip, extracting...")
        with zipfile.ZipFile(train_zip, 'r') as zip_ref:
            zip_ref.extractall(vqa_path)
        print("✅ Extracted train2014 images")
    else:
        print(f"⚠️ train2014.zip not found in {vqa_path}")
    
    # Check for val2014 images  
    val_zip = vqa_path / "val2014.zip"
    if val_zip.exists():
        print(f"📦 Found val2014.zip, extracting...")
        with zipfile.ZipFile(val_zip, 'r') as zip_ref:
            zip_ref.extractall(vqa_path)
        print("✅ Extracted val2014 images")
    else:
        print(f"⚠️ val2014.zip not found in {vqa_path}")
    
    # Check if images were extracted
    train_dir = vqa_path / "train2014"
    val_dir = vqa_path / "val2014"
    
    if train_dir.exists():
        num_train = len(list(train_dir.glob("*.jpg")))
        print(f"📊 Found {num_train} training images")
    
    if val_dir.exists():
        num_val = len(list(val_dir.glob("*.jpg")))
        print(f"📊 Found {num_val} validation images")
    
    print("\n✅ COCO image extraction complete!")
    print(f"   Images should now be available for VQA v2 training")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract COCO images for VQA v2")
    parser.add_argument("--vqa-dir", type=str, default="./data/vqa_datasets/vqa_v2",
                        help="Path to VQA v2 directory containing COCO zip files")
    args = parser.parse_args()
    
    extract_coco_images(args.vqa_dir)