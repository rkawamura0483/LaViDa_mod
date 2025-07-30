#!/usr/bin/env python3
"""
SHIRG Dataset Loaders for ChartQA, DocVQA, and VQA-v2
Proper dataset loading implementation for SHIRG LoRA training

Author: Research Implementation
Date: 2025-07-30
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, List, Optional, Any, Tuple
from datasets import load_dataset
import numpy as np


class ChartQADataset(Dataset):
    """ChartQA dataset loader for high-resolution chart understanding"""
    
    def __init__(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        image_size: int = 672,
        cache_dir: str = "./data/chartqa",
    ):
        """
        Initialize ChartQA dataset
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            max_samples: Maximum number of samples to load
            image_size: Target image size for resizing
            cache_dir: Directory to cache dataset
        """
        self.split = split
        self.image_size = image_size
        self.cache_dir = cache_dir
        
        # Load dataset from HuggingFace
        try:
            # ChartQA is available on HuggingFace datasets
            dataset = load_dataset("ahmed-masry/ChartQA", split=split, cache_dir=cache_dir)
            self.data = dataset
            
            # Limit samples if requested
            if max_samples and len(self.data) > max_samples:
                indices = np.random.choice(len(self.data), max_samples, replace=False)
                self.data = self.data.select(indices)
                
            print(f"✅ Loaded ChartQA {split} split: {len(self.data)} samples")
            
        except Exception as e:
            print(f"❌ Failed to load ChartQA from HuggingFace: {e}")
            print("   Please ensure you have access to the dataset")
            # Fallback to empty dataset
            self.data = []
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # SHIRG-FIX: [2025-07-30] - Handle various image formats
        # ISSUE: Image data may be bytes or array causing errors
        # SOLUTION: Properly handle all possible image formats
        # LAVIDA IMPACT: None
        # SHIRG IMPACT: Robust image loading for all datasets
        image = item['image']
        if isinstance(image, str):
            # If image is a path, load it
            image = Image.open(image).convert('RGB')
        elif isinstance(image, bytes):
            # Handle bytes data
            import io
            image = Image.open(io.BytesIO(image)).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Handle numpy array
            image = Image.fromarray(image).convert('RGB')
        elif hasattr(image, 'convert'):
            # Already a PIL Image
            image = image.convert('RGB')
        else:
            # Try to convert whatever it is
            try:
                image = Image.fromarray(np.array(image)).convert('RGB')
            except:
                # Create a dummy image if all else fails
                image = Image.new('RGB', (self.image_size, self.image_size), (128, 128, 128))
        
        # Resize to target size while maintaining aspect ratio
        image.thumbnail((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        
        # Pad to square if needed
        if image.size != (self.image_size, self.image_size):
            new_image = Image.new('RGB', (self.image_size, self.image_size), (255, 255, 255))
            paste_x = (self.image_size - image.width) // 2
            paste_y = (self.image_size - image.height) // 2
            new_image.paste(image, (paste_x, paste_y))
            image = new_image
        
        return {
            'image': image,
            'question': item['question'],
            'answer': item['answer'],
            'id': f"chartqa_{idx}",
            'dataset': 'chartqa'
        }


class DocVQADataset(Dataset):
    """DocVQA dataset loader for document understanding"""
    
    def __init__(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        image_size: int = 672,
        cache_dir: str = "./data/docvqa",
    ):
        """
        Initialize DocVQA dataset
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            max_samples: Maximum number of samples to load
            image_size: Target image size
            cache_dir: Directory to cache dataset
        """
        # SHIRG-FIX: [2025-07-30] - Handle DocVQA split mapping
        # ISSUE: lmms-lab/DocVQA only has validation and test splits, no train
        # SOLUTION: For training, use TextVQA as alternative or skip DocVQA
        # LAVIDA IMPACT: None
        # SHIRG IMPACT: Training uses alternative datasets
        if split == "train":
            print("⚠️ DocVQA train split not available in lmms-lab/DocVQA")
            print("   This dataset is evaluation-only. Using empty dataset for training.")
            print("   Consider using TextVQA or ChartQA for training instead.")
            self.data = []
            self.split = split
            self.image_size = image_size
            self.cache_dir = cache_dir
            return
        
        self.split = split if split != "val" else "validation"
        self.image_size = image_size
        self.cache_dir = cache_dir
        
        try:
            # Load DocVQA for validation/test only
            dataset = load_dataset("lmms-lab/DocVQA", "DocVQA", split=self.split, cache_dir=cache_dir)
            self.data = dataset
            
            # Limit samples if requested
            if max_samples and len(self.data) > max_samples:
                indices = np.random.choice(len(self.data), max_samples, replace=False)
                self.data = self.data.select(indices)
                
            print(f"✅ Loaded DocVQA {split} split: {len(self.data)} samples")
            
        except Exception as e:
            print(f"❌ Failed to load DocVQA from HuggingFace: {e}")
            print("   Trying alternative source...")
            # Alternative: Load from local files if available
            self.data = []
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # DocVQA format
        image = item['image']
        # SHIRG-FIX: [2025-07-30] - Reuse robust image handling
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, bytes):
            import io
            image = Image.open(io.BytesIO(image)).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif hasattr(image, 'convert'):
            image = image.convert('RGB')
        else:
            try:
                image = Image.fromarray(np.array(image)).convert('RGB')
            except:
                image = Image.new('RGB', (self.image_size, self.image_size), (128, 128, 128))
        
        # Resize to target size
        image.thumbnail((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        
        # Pad to square
        if image.size != (self.image_size, self.image_size):
            new_image = Image.new('RGB', (self.image_size, self.image_size), (255, 255, 255))
            paste_x = (self.image_size - image.width) // 2
            paste_y = (self.image_size - image.height) // 2
            new_image.paste(image, (paste_x, paste_y))
            image = new_image
        
        # DocVQA may have multiple answers
        answers = item.get('answers', [])
        if isinstance(answers, list) and len(answers) > 0:
            answer = answers[0]  # Take first answer for training
        else:
            answer = str(answers)
        
        return {
            'image': image,
            'question': item['question'],
            'answer': answer,
            'id': f"docvqa_{idx}",
            'dataset': 'docvqa'
        }


class VQAv2Dataset(Dataset):
    """VQA v2 dataset loader for general visual question answering"""
    
    def __init__(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        image_size: int = 672,
        cache_dir: str = "./data/vqa_v2",
    ):
        """
        Initialize VQA v2 dataset
        
        Args:
            split: Dataset split ('train', 'validation')
            max_samples: Maximum number of samples
            image_size: Target image size
            cache_dir: Directory to cache dataset
        """
        self.split = split if split != "val" else "validation"
        self.image_size = image_size
        self.cache_dir = cache_dir
        
        try:
            # SHIRG-FIX: [2025-07-30] - Fix VQA v2 dataset loading
            # ISSUE: VQAv2 dataset requires specific loading approach
            # SOLUTION: Use lmms-lab/VQAv2 which is properly maintained
            # LAVIDA IMPACT: None
            # SHIRG IMPACT: Enables VQA v2 dataset for training
            # Use lmms-lab version which is maintained for VLM training
            dataset = load_dataset("lmms-lab/VQAv2", split=self.split, cache_dir=cache_dir)
            self.data = dataset
            
            # Limit samples if requested
            if max_samples and len(self.data) > max_samples:
                indices = np.random.choice(len(self.data), max_samples, replace=False)
                self.data = self.data.select(indices)
                
            print(f"✅ Loaded VQA v2 {split} split: {len(self.data)} samples")
            
        except Exception as e:
            print(f"❌ Failed to load VQA v2 from HuggingFace: {e}")
            self.data = []
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # VQA v2 format - handle various formats
        image = item.get('image', None)
        if image is None:
            # Create dummy image if missing
            image = Image.new('RGB', (self.image_size, self.image_size), (128, 128, 128))
        elif isinstance(image, str):
            # Path to image
            try:
                image = Image.open(image).convert('RGB')
            except:
                image = Image.new('RGB', (self.image_size, self.image_size), (128, 128, 128))
        elif isinstance(image, bytes):
            # Handle bytes data
            import io
            try:
                image = Image.open(io.BytesIO(image)).convert('RGB')
            except:
                image = Image.new('RGB', (self.image_size, self.image_size), (128, 128, 128))
        elif isinstance(image, np.ndarray):
            # Handle numpy array
            image = Image.fromarray(image).convert('RGB')
        elif hasattr(image, 'convert'):
            # Already a PIL Image
            image = image.convert('RGB')
        else:
            # Try to convert whatever it is
            try:
                image = Image.fromarray(np.array(image)).convert('RGB')
            except:
                image = Image.new('RGB', (self.image_size, self.image_size), (128, 128, 128))
        
        # Resize to target size
        image.thumbnail((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        
        # Pad to square
        if image.size != (self.image_size, self.image_size):
            new_image = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))
            paste_x = (self.image_size - image.width) // 2
            paste_y = (self.image_size - image.height) // 2
            new_image.paste(image, (paste_x, paste_y))
            image = new_image
        
        # VQA v2 has multiple answers with confidence scores
        answers = item.get('answers', [])
        if answers:
            # Get most common answer
            answer_counts = {}
            for ans in answers:
                answer_text = ans['answer']
                answer_counts[answer_text] = answer_counts.get(answer_text, 0) + 1
            answer = max(answer_counts, key=answer_counts.get)
        else:
            answer = "unknown"
        
        return {
            'image': image,
            'question': item['question'],
            'answer': answer,
            'id': f"vqa_v2_{idx}",
            'dataset': 'vqa_v2'
        }


class TextVQADataset(Dataset):
    """TextVQA dataset loader as alternative to DocVQA for training"""
    
    def __init__(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        image_size: int = 672,
        cache_dir: str = "./data/textvqa",
    ):
        """
        Initialize TextVQA dataset
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            max_samples: Maximum number of samples to load
            image_size: Target image size
            cache_dir: Directory to cache dataset
        """
        self.split = split if split != "val" else "validation"
        self.image_size = image_size
        self.cache_dir = cache_dir
        
        try:
            # SHIRG-FIX: [2025-07-30] - Add TextVQA as DocVQA alternative
            # ISSUE: Need text-heavy dataset for training since DocVQA lacks train split
            # SOLUTION: Use lmms-lab/textvqa which has proper train split
            # LAVIDA IMPACT: None
            # SHIRG IMPACT: Provides text-reading samples for training
            dataset = load_dataset("lmms-lab/textvqa", split=self.split, cache_dir=cache_dir)
            self.data = dataset
            
            # Limit samples if requested
            if max_samples and len(self.data) > max_samples:
                indices = np.random.choice(len(self.data), max_samples, replace=False)
                self.data = self.data.select(indices)
                
            print(f"✅ Loaded TextVQA {split} split: {len(self.data)} samples")
            
        except Exception as e:
            print(f"❌ Failed to load TextVQA: {e}")
            self.data = []
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # TextVQA format - similar handling to other datasets
        image = item.get('image', None)
        if image is None:
            image = Image.new('RGB', (self.image_size, self.image_size), (128, 128, 128))
        elif isinstance(image, str):
            try:
                image = Image.open(image).convert('RGB')
            except:
                image = Image.new('RGB', (self.image_size, self.image_size), (128, 128, 128))
        elif isinstance(image, bytes):
            import io
            try:
                image = Image.open(io.BytesIO(image)).convert('RGB')
            except:
                image = Image.new('RGB', (self.image_size, self.image_size), (128, 128, 128))
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif hasattr(image, 'convert'):
            image = image.convert('RGB')
        else:
            try:
                image = Image.fromarray(np.array(image)).convert('RGB')
            except:
                image = Image.new('RGB', (self.image_size, self.image_size), (128, 128, 128))
        
        # Resize to target size
        image.thumbnail((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        
        # Pad to square
        if image.size != (self.image_size, self.image_size):
            new_image = Image.new('RGB', (self.image_size, self.image_size), (255, 255, 255))
            paste_x = (self.image_size - image.width) // 2
            paste_y = (self.image_size - image.height) // 2
            new_image.paste(image, (paste_x, paste_y))
            image = new_image
        
        # TextVQA may have multiple answers
        answers = item.get('answers', [])
        if isinstance(answers, list) and len(answers) > 0:
            answer = answers[0]  # Take first answer
        else:
            answer = str(answers) if answers else "unknown"
        
        return {
            'image': image,
            'question': item.get('question', 'What text is shown in this image?'),
            'answer': answer,
            'id': f"textvqa_{idx}",
            'dataset': 'textvqa'
        }


class OCRVQADataset(Dataset):
    """OCR-VQA dataset loader for book cover text reading"""
    
    def __init__(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        image_size: int = 672,
        cache_dir: str = "./data/ocrvqa",
    ):
        """
        Initialize OCR-VQA dataset
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            max_samples: Maximum number of samples to load
            image_size: Target image size
            cache_dir: Directory to cache dataset
        """
        self.split = split if split != "val" else "validation"
        self.image_size = image_size
        self.cache_dir = cache_dir
        
        try:
            # SHIRG-FIX: [2025-07-30] - Add OCR-VQA for training
            # ISSUE: Need OCR-focused dataset for text reading capabilities
            # SOLUTION: Use howard-hou/OCR-VQA which has train/val/test splits
            # LAVIDA IMPACT: None
            # SHIRG IMPACT: Provides book cover OCR samples (1M QA pairs)
            dataset = load_dataset("howard-hou/OCR-VQA", split=self.split, cache_dir=cache_dir)
            self.data = dataset
            
            # Limit samples if requested
            if max_samples and len(self.data) > max_samples:
                indices = np.random.choice(len(self.data), max_samples, replace=False)
                self.data = self.data.select(indices)
                
            print(f"✅ Loaded OCR-VQA {split} split: {len(self.data)} samples")
            
        except Exception as e:
            print(f"❌ Failed to load OCR-VQA: {e}")
            self.data = []
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # OCR-VQA format - handle image data
        image = item.get('image', None)
        if image is None:
            image = Image.new('RGB', (self.image_size, self.image_size), (128, 128, 128))
        elif isinstance(image, str):
            try:
                image = Image.open(image).convert('RGB')
            except:
                image = Image.new('RGB', (self.image_size, self.image_size), (128, 128, 128))
        elif isinstance(image, bytes):
            import io
            try:
                image = Image.open(io.BytesIO(image)).convert('RGB')
            except:
                image = Image.new('RGB', (self.image_size, self.image_size), (128, 128, 128))
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif hasattr(image, 'convert'):
            image = image.convert('RGB')
        else:
            try:
                image = Image.fromarray(np.array(image)).convert('RGB')
            except:
                image = Image.new('RGB', (self.image_size, self.image_size), (128, 128, 128))
        
        # Resize to target size
        image.thumbnail((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        
        # Pad to square
        if image.size != (self.image_size, self.image_size):
            new_image = Image.new('RGB', (self.image_size, self.image_size), (255, 255, 255))
            paste_x = (self.image_size - image.width) // 2
            paste_y = (self.image_size - image.height) // 2
            new_image.paste(image, (paste_x, paste_y))
            image = new_image
        
        # OCR-VQA has questions about book metadata
        return {
            'image': image,
            'question': item.get('question', 'What text is visible in this image?'),
            'answer': item.get('answer', ''),
            'id': f"ocrvqa_{idx}",
            'dataset': 'ocrvqa'
        }


class InfoVQADataset(Dataset):
    """InfoVQA dataset loader for infographic understanding"""
    
    def __init__(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        image_size: int = 672,
        cache_dir: str = "./data/infovqa",
    ):
        """
        Initialize InfoVQA dataset
        
        Args:
            split: Dataset split (only 'train' available)
            max_samples: Maximum number of samples to load
            image_size: Target image size
            cache_dir: Directory to cache dataset
        """
        self.split = split
        self.image_size = image_size
        self.cache_dir = cache_dir
        
        try:
            # SHIRG-FIX: [2025-07-30] - Add InfoVQA for training
            # ISSUE: Need infographic dataset for dense visual reasoning
            # SOLUTION: Use vidore/infovqa_train which has 10.1k training samples
            # LAVIDA IMPACT: None
            # SHIRG IMPACT: Provides infographic samples requiring fine detail preservation
            if split == "train":
                dataset = load_dataset("vidore/infovqa_train", split="train", cache_dir=cache_dir)
                self.data = dataset
            else:
                # For validation/test, would need different source
                print(f"⚠️ InfoVQA {split} split not available in vidore/infovqa_train")
                self.data = []
            
            # Limit samples if requested
            if self.data and max_samples and len(self.data) > max_samples:
                indices = np.random.choice(len(self.data), max_samples, replace=False)
                self.data = self.data.select(indices)
                
            if self.data:
                print(f"✅ Loaded InfoVQA {split} split: {len(self.data)} samples")
            
        except Exception as e:
            print(f"❌ Failed to load InfoVQA: {e}")
            self.data = []
    
    def __len__(self):
        return len(self.data) if self.data else 0
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # InfoVQA format - handle image data
        image = item.get('image', None)
        if image is None:
            image = Image.new('RGB', (self.image_size, self.image_size), (128, 128, 128))
        elif isinstance(image, str):
            try:
                image = Image.open(image).convert('RGB')
            except:
                image = Image.new('RGB', (self.image_size, self.image_size), (128, 128, 128))
        elif isinstance(image, bytes):
            import io
            try:
                image = Image.open(io.BytesIO(image)).convert('RGB')
            except:
                image = Image.new('RGB', (self.image_size, self.image_size), (128, 128, 128))
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        elif hasattr(image, 'convert'):
            image = image.convert('RGB')
        else:
            try:
                image = Image.fromarray(np.array(image)).convert('RGB')
            except:
                image = Image.new('RGB', (self.image_size, self.image_size), (128, 128, 128))
        
        # Resize to target size
        image.thumbnail((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        
        # Pad to square
        if image.size != (self.image_size, self.image_size):
            new_image = Image.new('RGB', (self.image_size, self.image_size), (255, 255, 255))
            paste_x = (self.image_size - image.width) // 2
            paste_y = (self.image_size - image.height) // 2
            new_image.paste(image, (paste_x, paste_y))
            image = new_image
        
        return {
            'image': image,
            'question': item.get('question', 'What information is shown in this infographic?'),
            'answer': item.get('answer', ''),
            'id': f"infovqa_{idx}",
            'dataset': 'infovqa'
        }


class MixedVQADataset(Dataset):
    """Mixed dataset combining ChartQA, DocVQA, and VQA v2 with weighted sampling"""
    
    def __init__(
        self,
        split: str = "train",
        dataset_configs: Dict[str, Dict[str, Any]] = None,
        image_size: int = 672,
        cache_dir: str = "./data",
    ):
        """
        Initialize mixed dataset
        
        Args:
            split: Dataset split
            dataset_configs: Configuration for each dataset with weights and max_samples
            image_size: Target image size
            cache_dir: Base cache directory
        """
        self.split = split
        self.image_size = image_size
        
        # SHIRG-FIX: [2025-07-30] - Adjust dataset weights for training
        # ISSUE: DocVQA has no train split
        # SOLUTION: For training, use ChartQA and VQAv2 with adjusted weights
        # LAVIDA IMPACT: None
        # SHIRG IMPACT: Training focuses on available datasets
        if dataset_configs is None:
            if split == "train":
                # SHIRG-FIX: [2025-07-30] - Comprehensive training dataset mix
                # ISSUE: Need diverse datasets matching evaluation benchmarks
                # SOLUTION: Include 5 complementary datasets for robust training
                # LAVIDA IMPACT: None
                # SHIRG IMPACT: Better coverage of high-res OCR/VQA scenarios
                dataset_configs = {
                    "chartqa": {"weight": 0.2, "max_samples": 10000},   # Charts & graphs
                    "textvqa": {"weight": 0.2, "max_samples": 10000},   # Natural scene text
                    "ocrvqa": {"weight": 0.2, "max_samples": 10000},    # Book covers
                    "infovqa": {"weight": 0.2, "max_samples": 10000},   # Infographics
                    "vqa_v2": {"weight": 0.2, "max_samples": 10000},    # General VQA
                }
            else:
                # For validation/test, include all datasets
                dataset_configs = {
                    "chartqa": {"weight": 0.3, "max_samples": 10000},
                    "docvqa": {"weight": 0.3, "max_samples": 10000},
                    "vqa_v2": {"weight": 0.4, "max_samples": 10000},
                }
        
        self.datasets = {}
        self.weights = []
        self.dataset_names = []
        
        # Load individual datasets
        for name, config in dataset_configs.items():
            dataset_class = {
                "chartqa": ChartQADataset,
                "docvqa": DocVQADataset,
                "vqa_v2": VQAv2Dataset,
                "textvqa": TextVQADataset,
                "ocrvqa": OCRVQADataset,    # Book cover OCR
                "infovqa": InfoVQADataset,   # Infographic understanding
            }.get(name)
            
            if dataset_class:
                try:
                    dataset = dataset_class(
                        split=split,
                        max_samples=config.get("max_samples"),
                        image_size=image_size,
                        cache_dir=os.path.join(cache_dir, name),
                    )
                    if len(dataset) > 0:
                        self.datasets[name] = dataset
                        self.weights.append(config.get("weight", 1.0))
                        self.dataset_names.append(name)
                        print(f"   Added {name}: {len(dataset)} samples, weight={config.get('weight')}")
                except Exception as e:
                    print(f"   Failed to load {name}: {e}")
        
        # Normalize weights
        if self.weights:
            total_weight = sum(self.weights)
            self.weights = [w / total_weight for w in self.weights]
        
        # Calculate total samples
        self.total_samples = sum(len(d) for d in self.datasets.values())
        print(f"✅ Mixed dataset created: {self.total_samples} total samples")
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # Weighted random sampling
        dataset_idx = np.random.choice(len(self.dataset_names), p=self.weights)
        dataset_name = self.dataset_names[dataset_idx]
        dataset = self.datasets[dataset_name]
        
        # Random sample from selected dataset
        sample_idx = np.random.randint(0, len(dataset))
        return dataset[sample_idx]


def create_data_loaders(
    batch_size: int = 16,
    num_workers: int = 4,
    image_size: int = 672,
    max_samples_per_dataset: int = 10000,
    cache_dir: str = "./data",
):
    """
    Create training and validation data loaders for SHIRG training
    
    Args:
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        image_size: Target image size
        max_samples_per_dataset: Maximum samples per dataset
        cache_dir: Cache directory for datasets
    
    Returns:
        train_loader, val_loader: DataLoader objects
    """
    from torch.utils.data import DataLoader
    
    # Dataset configuration from research
    dataset_configs = {
        "chartqa": {"weight": 0.3, "max_samples": max_samples_per_dataset},
        "docvqa": {"weight": 0.3, "max_samples": max_samples_per_dataset},
        "vqa_v2": {"weight": 0.4, "max_samples": max_samples_per_dataset},
    }
    
    # Create train dataset
    train_dataset = MixedVQADataset(
        split="train",
        dataset_configs=dataset_configs,
        image_size=image_size,
        cache_dir=cache_dir,
    )
    
    # Create validation dataset with fewer samples
    val_configs = {k: {"weight": v["weight"], "max_samples": 1000} 
                   for k, v in dataset_configs.items()}
    val_dataset = MixedVQADataset(
        split="validation",
        dataset_configs=val_configs,
        image_size=image_size,
        cache_dir=cache_dir,
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset loading
    print("🧪 Testing SHIRG dataset loaders...")
    
    # Test individual datasets
    for dataset_class, name in [
        (ChartQADataset, "ChartQA"),
        (DocVQADataset, "DocVQA"),
        (VQAv2Dataset, "VQA v2"),
    ]:
        print(f"\nTesting {name}...")
        try:
            dataset = dataset_class(split="train", max_samples=10)
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"✅ {name} loaded successfully")
                print(f"   Sample keys: {list(sample.keys())}")
                print(f"   Image size: {sample['image'].size}")
                print(f"   Question: {sample['question'][:50]}...")
                print(f"   Answer: {sample['answer'][:50]}...")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
    
    # Test mixed dataset
    print("\nTesting mixed dataset...")
    try:
        train_loader, val_loader = create_data_loaders(
            batch_size=4,
            num_workers=0,  # Use 0 for testing
            max_samples_per_dataset=100,
        )
        print(f"✅ Data loaders created")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
    except Exception as e:
        print(f"❌ Failed to create data loaders: {e}")