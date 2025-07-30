#!/usr/bin/env python3
"""
Create dataset configuration for real VQA datasets
This script creates a configuration file that maps to actual downloaded datasets

SHIRG-FIX: 2025-07-30 - Configure real VQA datasets for training
ISSUE: Training using synthetic data instead of real datasets
SOLUTION: Create proper configuration for downloaded datasets
LAVIDA IMPACT: None
SHIRG IMPACT: Enables training on real VQA data with millions of samples
"""

import json
import os
from pathlib import Path

def create_dataset_config(data_dir="./data/vqa_datasets"):
    """Create configuration for real VQA datasets"""
    
    data_path = Path(data_dir)
    
    # Check what datasets are available
    available_datasets = {}
    
    # ChartQA
    chartqa_path = data_path / "chartqa" / "ChartQA Dataset"
    if chartqa_path.exists():
        train_json = chartqa_path / "train" / "train_augmented.json"
        if train_json.exists():
            with open(train_json, 'r') as f:
                data = json.load(f)
                available_datasets['chartqa'] = {
                    'train_samples': len(data),
                    'path': str(chartqa_path),
                    'format': 'chartqa'
                }
                print(f"✅ ChartQA found: {len(data)} training samples")
    
    # DocVQA
    docvqa_path = data_path / "docvqa"
    if docvqa_path.exists():
        train_json = docvqa_path / "train" / "train_v1.0.json"
        if train_json.exists():
            with open(train_json, 'r') as f:
                data = json.load(f)
                available_datasets['docvqa'] = {
                    'train_samples': len(data['data']),
                    'path': str(docvqa_path),
                    'format': 'docvqa'
                }
                print(f"✅ DocVQA found: {len(data['data'])} training samples")
    
    # VQA v2
    vqa2_path = data_path / "vqa_v2"
    if vqa2_path.exists():
        train_q = vqa2_path / "v2_mscoco_train2014_questions.json"
        if train_q.exists():
            with open(train_q, 'r') as f:
                data = json.load(f)
                available_datasets['vqa_v2'] = {
                    'train_samples': len(data['questions']),
                    'path': str(vqa2_path),
                    'format': 'vqa_v2'
                }
                print(f"✅ VQA v2 found: {len(data['questions'])} training samples")
    
    # TextVQA
    textvqa_path = data_path / "textvqa"
    if textvqa_path.exists():
        train_json = textvqa_path / "TextVQA_0.5.1_train.json"
        if train_json.exists():
            with open(train_json, 'r') as f:
                data = json.load(f)
                available_datasets['textvqa'] = {
                    'train_samples': len(data['data']),
                    'path': str(textvqa_path),
                    'format': 'textvqa'
                }
                print(f"✅ TextVQA found: {len(data['data'])} training samples")
    
    # OCR-VQA
    ocrvqa_path = data_path / "ocrvqa"
    if ocrvqa_path.exists():
        train_json = ocrvqa_path / "train.json"
        if train_json.exists():
            with open(train_json, 'r') as f:
                data = json.load(f)
                available_datasets['ocrvqa'] = {
                    'train_samples': len(data),
                    'path': str(ocrvqa_path),
                    'format': 'ocrvqa'
                }
                print(f"✅ OCR-VQA found: {len(data)} training samples")
    
    # Calculate total samples
    total_samples = sum(d['train_samples'] for d in available_datasets.values())
    print(f"\n📊 Total training samples available: {total_samples:,}")
    
    # Create training configuration
    if total_samples > 0:
        # For 8 GPU training with ~45k steps per epoch
        # With batch size 64, we need ~2.88M samples per epoch
        # Scale weights to get reasonable epoch size
        
        config = {
            'data_dir': data_dir,
            'available_datasets': available_datasets,
            'total_samples': total_samples,
            'training_config': {
                'batch_size': 64,  # Total across 8 GPUs
                'num_epochs': 3,
                'expected_steps_per_epoch': 45000,
                'dataset_weights': {
                    'chartqa': 0.15,
                    'docvqa': 0.15,
                    'vqa_v2': 0.40,
                    'textvqa': 0.15,
                    'ocrvqa': 0.15
                }
            }
        }
        
        # Save configuration
        config_path = Path(data_dir) / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✅ Configuration saved to: {config_path}")
        print(f"\n📝 To use real datasets, run training with:")
        print(f"   bash shirg/run_8gpu_training.sh")
        print(f"\nThe training script will automatically use datasets from: {data_dir}")
    else:
        print("\n❌ No datasets found! Please run:")
        print(f"   python shirg/download_vqa_datasets.py --data-dir {data_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data/vqa_datasets", 
                       help="Directory containing VQA datasets")
    args = parser.parse_args()
    
    create_dataset_config(args.data_dir)