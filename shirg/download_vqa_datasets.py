#!/usr/bin/env python3
"""
Download real VQA datasets for SHIRG training
Downloads ChartQA, DocVQA, VQA v2, TextVQA, and other datasets

SHIRG-FIX: 2025-07-30 - Download real VQA datasets for proper training
ISSUE: Training running on synthetic data with only 300 steps
SOLUTION: Download and prepare real VQA datasets with millions of samples
LAVIDA IMPACT: Ensures proper training data for LaViDa integration
SHIRG IMPACT: Enables real training with expected 45k steps per epoch
"""

import os
import sys
import json
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import shutil
from typing import Dict, List, Optional
import hashlib

# Add paths
sys.path.append('./')
sys.path.append('./shirg')

class VQADatasetDownloader:
    """Download and prepare VQA datasets for SHIRG training"""
    
    def __init__(self, base_dir: str = "./data/vqa_datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset configurations
        self.datasets = {
            "chartqa": {
                "train_url": "https://github.com/vis-nlp/ChartQA/releases/download/v1.0/ChartQA_Dataset.zip",
                "description": "ChartQA - Chart question answering dataset",
                "expected_samples": {"train": 18317, "val": 1250, "test": 2500},
                "data_format": "chartqa"
            },
            "docvqa": {
                "train_url": "https://datasets.cvc.uab.es/rrc/DocVQA/train.tar.gz",
                "val_url": "https://datasets.cvc.uab.es/rrc/DocVQA/val.tar.gz", 
                "test_url": "https://datasets.cvc.uab.es/rrc/DocVQA/test.tar.gz",
                "description": "DocVQA - Document visual question answering",
                "expected_samples": {"train": 39463, "val": 5349, "test": 5188},
                "data_format": "docvqa"
            },
            "vqa_v2": {
                "train_questions": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
                "val_questions": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
                "train_annotations": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
                "val_annotations": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
                "train_images": "http://images.cocodataset.org/zips/train2014.zip",
                "val_images": "http://images.cocodataset.org/zips/val2014.zip",
                "description": "VQA v2 - Visual Question Answering v2.0",
                "expected_samples": {"train": 443757, "val": 214354},
                "data_format": "vqa_v2"
            },
            "textvqa": {
                "train_url": "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_train.json",
                "val_url": "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json",
                "train_images": "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip",
                "description": "TextVQA - Text-based visual question answering",
                "expected_samples": {"train": 34602, "val": 5000},
                "data_format": "textvqa"
            },
            "ocrvqa": {
                "dataset_url": "https://huggingface.co/datasets/howard-hou/OCR-VQA/resolve/main/ocrvqa.json",
                "description": "OCR-VQA - OCR-specific visual question answering",
                "expected_samples": {"train": 207572, "val": 10000},
                "data_format": "ocrvqa"
            },
            "infovqa": {
                "train_url": "https://github.com/doc-analysis/InfographicsVQA/releases/download/v1.0/infographicsVQA_train_v1.0.json",
                "val_url": "https://github.com/doc-analysis/InfographicsVQA/releases/download/v1.0/infographicsVQA_val_v1.0.json",
                "description": "InfoVQA - Infographics visual question answering",
                "expected_samples": {"train": 23946, "val": 2801},
                "data_format": "infovqa"
            }
        }
    
    def download_file(self, url: str, dest_path: Path, desc: str = None) -> bool:
        """Download file with progress bar"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dest_path, 'wb') as f:
                with tqdm(total=total_size, unit='iB', unit_scale=True, desc=desc) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        size = f.write(chunk)
                        pbar.update(size)
            
            return True
        except Exception as e:
            print(f"❌ Error downloading {url}: {e}")
            return False
    
    def extract_archive(self, archive_path: Path, extract_to: Path) -> bool:
        """Extract zip or tar archive"""
        try:
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_to)
            else:
                print(f"❌ Unknown archive format: {archive_path}")
                return False
            
            return True
        except Exception as e:
            print(f"❌ Error extracting {archive_path}: {e}")
            return False
    
    def download_chartqa(self) -> Dict[str, int]:
        """Download ChartQA dataset"""
        print("\n📊 Downloading ChartQA dataset...")
        dataset_dir = self.base_dir / "chartqa"
        dataset_dir.mkdir(exist_ok=True)
        
        # Download main dataset
        zip_path = dataset_dir / "ChartQA_Dataset.zip"
        if not zip_path.exists():
            if not self.download_file(
                self.datasets["chartqa"]["train_url"],
                zip_path,
                "ChartQA dataset"
            ):
                return {}
        
        # Extract
        if not (dataset_dir / "ChartQA Dataset").exists():
            print("📦 Extracting ChartQA...")
            self.extract_archive(zip_path, dataset_dir)
        
        # Count samples
        counts = {}
        for split in ["train", "val", "test"]:
            json_path = dataset_dir / "ChartQA Dataset" / f"{split}/{split}_augmented.json"
            if json_path.exists():
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    counts[split] = len(data)
                    print(f"✅ ChartQA {split}: {counts[split]} samples")
        
        return counts
    
    def download_docvqa(self) -> Dict[str, int]:
        """Download DocVQA dataset"""
        print("\n📄 Downloading DocVQA dataset...")
        dataset_dir = self.base_dir / "docvqa"
        dataset_dir.mkdir(exist_ok=True)
        
        counts = {}
        
        # Download each split
        for split in ["train", "val", "test"]:
            url_key = f"{split}_url"
            if url_key not in self.datasets["docvqa"]:
                continue
            
            archive_path = dataset_dir / f"{split}.tar.gz"
            if not archive_path.exists():
                if not self.download_file(
                    self.datasets["docvqa"][url_key],
                    archive_path,
                    f"DocVQA {split}"
                ):
                    continue
            
            # Extract
            split_dir = dataset_dir / split
            if not split_dir.exists():
                print(f"📦 Extracting DocVQA {split}...")
                self.extract_archive(archive_path, split_dir)
            
            # Count samples
            json_path = split_dir / f"{split}_v1.0.json"
            if json_path.exists():
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    counts[split] = len(data['data'])
                    print(f"✅ DocVQA {split}: {counts[split]} samples")
        
        return counts
    
    def download_vqa_v2(self) -> Dict[str, int]:
        """Download VQA v2 dataset"""
        print("\n🖼️ Downloading VQA v2 dataset...")
        dataset_dir = self.base_dir / "vqa_v2"
        dataset_dir.mkdir(exist_ok=True)
        
        counts = {}
        
        # Download questions and annotations
        for split in ["train", "val"]:
            # Questions
            questions_zip = dataset_dir / f"v2_Questions_{split.title()}_mscoco.zip"
            if not questions_zip.exists():
                if not self.download_file(
                    self.datasets["vqa_v2"][f"{split}_questions"],
                    questions_zip,
                    f"VQA v2 {split} questions"
                ):
                    continue
            
            # Annotations
            annotations_zip = dataset_dir / f"v2_Annotations_{split.title()}_mscoco.zip"
            if not annotations_zip.exists():
                if not self.download_file(
                    self.datasets["vqa_v2"][f"{split}_annotations"],
                    annotations_zip,
                    f"VQA v2 {split} annotations"
                ):
                    continue
            
            # Extract
            if not (dataset_dir / f"v2_mscoco_{split}2014_questions.json").exists():
                print(f"📦 Extracting VQA v2 {split} data...")
                self.extract_archive(questions_zip, dataset_dir)
                self.extract_archive(annotations_zip, dataset_dir)
            
            # Count samples
            questions_path = dataset_dir / f"v2_mscoco_{split}2014_questions.json"
            if questions_path.exists():
                with open(questions_path, 'r') as f:
                    data = json.load(f)
                    counts[split] = len(data['questions'])
                    print(f"✅ VQA v2 {split}: {counts[split]} samples")
        
        # Note: Images download is optional and large (~13GB each)
        print("ℹ️ VQA v2 images not downloaded (13GB+ each). Use COCO dataset if needed.")
        
        return counts
    
    def download_textvqa(self) -> Dict[str, int]:
        """Download TextVQA dataset"""
        print("\n📝 Downloading TextVQA dataset...")
        dataset_dir = self.base_dir / "textvqa"
        dataset_dir.mkdir(exist_ok=True)
        
        counts = {}
        
        # Download JSON files
        for split in ["train", "val"]:
            json_path = dataset_dir / f"TextVQA_0.5.1_{split}.json"
            if not json_path.exists():
                if not self.download_file(
                    self.datasets["textvqa"][f"{split}_url"],
                    json_path,
                    f"TextVQA {split}"
                ):
                    continue
            
            # Count samples
            if json_path.exists():
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    counts[split] = len(data['data'])
                    print(f"✅ TextVQA {split}: {counts[split]} samples")
        
        # Note: Images are shared with VQA v2
        print("ℹ️ TextVQA uses COCO images. Download separately if needed.")
        
        return counts
    
    def download_ocrvqa(self) -> Dict[str, int]:
        """Download OCR-VQA dataset"""
        print("\n🔤 Downloading OCR-VQA dataset...")
        dataset_dir = self.base_dir / "ocrvqa"
        dataset_dir.mkdir(exist_ok=True)
        
        # Download main JSON
        json_path = dataset_dir / "ocrvqa.json"
        if not json_path.exists():
            if not self.download_file(
                self.datasets["ocrvqa"]["dataset_url"],
                json_path,
                "OCR-VQA dataset"
            ):
                return {}
        
        # Parse and split dataset
        counts = {}
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            # Split into train/val (95/5)
            total_samples = len(data)
            train_size = int(0.95 * total_samples)
            
            train_data = dict(list(data.items())[:train_size])
            val_data = dict(list(data.items())[train_size:])
            
            # Save splits
            with open(dataset_dir / "train.json", 'w') as f:
                json.dump(train_data, f)
            with open(dataset_dir / "val.json", 'w') as f:
                json.dump(val_data, f)
            
            counts["train"] = len(train_data)
            counts["val"] = len(val_data)
            
            print(f"✅ OCR-VQA train: {counts['train']} samples")
            print(f"✅ OCR-VQA val: {counts['val']} samples")
        
        return counts
    
    def create_dataset_config(self, dataset_counts: Dict[str, Dict[str, int]]) -> None:
        """Create configuration file for training"""
        config_path = self.base_dir / "dataset_config.json"
        
        config = {
            "dataset_paths": {
                "chartqa": str(self.base_dir / "chartqa"),
                "docvqa": str(self.base_dir / "docvqa"),
                "vqa_v2": str(self.base_dir / "vqa_v2"),
                "textvqa": str(self.base_dir / "textvqa"),
                "ocrvqa": str(self.base_dir / "ocrvqa"),
                "infovqa": str(self.base_dir / "infovqa")
            },
            "dataset_counts": dataset_counts,
            "total_train_samples": sum(
                counts.get("train", 0) 
                for counts in dataset_counts.values()
            ),
            "total_val_samples": sum(
                counts.get("val", 0)
                for counts in dataset_counts.values()
            ),
            "dataset_weights": {
                "chartqa": 0.15,
                "docvqa": 0.15,
                "vqa_v2": 0.40,
                "textvqa": 0.15,
                "ocrvqa": 0.10,
                "infovqa": 0.05
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✅ Dataset configuration saved to: {config_path}")
        print(f"📊 Total training samples: {config['total_train_samples']:,}")
        print(f"📊 Total validation samples: {config['total_val_samples']:,}")
    
    def download_all(self) -> None:
        """Download all datasets"""
        print("🚀 Starting VQA dataset downloads...")
        print(f"📁 Base directory: {self.base_dir}")
        
        dataset_counts = {}
        
        # Download each dataset
        dataset_counts["chartqa"] = self.download_chartqa()
        dataset_counts["docvqa"] = self.download_docvqa()
        dataset_counts["vqa_v2"] = self.download_vqa_v2()
        dataset_counts["textvqa"] = self.download_textvqa()
        dataset_counts["ocrvqa"] = self.download_ocrvqa()
        
        # Create configuration
        self.create_dataset_config(dataset_counts)
        
        print("\n✅ All datasets downloaded successfully!")
        print("\n📝 Next steps:")
        print("1. Update training script to use real dataset paths")
        print("2. Adjust batch size for larger dataset")
        print("3. Run training with proper dataset configuration")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download VQA datasets for SHIRG training")
    parser.add_argument("--data-dir", type=str, default="./data/vqa_datasets",
                       help="Base directory for datasets")
    parser.add_argument("--datasets", nargs="+", 
                       choices=["chartqa", "docvqa", "vqa_v2", "textvqa", "ocrvqa", "all"],
                       default=["all"],
                       help="Datasets to download")
    
    args = parser.parse_args()
    
    downloader = VQADatasetDownloader(args.data_dir)
    
    if "all" in args.datasets:
        downloader.download_all()
    else:
        dataset_counts = {}
        for dataset in args.datasets:
            if dataset == "chartqa":
                dataset_counts[dataset] = downloader.download_chartqa()
            elif dataset == "docvqa":
                dataset_counts[dataset] = downloader.download_docvqa()
            elif dataset == "vqa_v2":
                dataset_counts[dataset] = downloader.download_vqa_v2()
            elif dataset == "textvqa":
                dataset_counts[dataset] = downloader.download_textvqa()
            elif dataset == "ocrvqa":
                dataset_counts[dataset] = downloader.download_ocrvqa()
        
        downloader.create_dataset_config(dataset_counts)


if __name__ == "__main__":
    main()