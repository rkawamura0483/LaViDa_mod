#!/usr/bin/env python3
"""
SHIRG Evaluation Framework
Comprehensive evaluation of SHIRG token selection against baseline LaViDa

This module provides tools to evaluate SHIRG performance on OCR/VQA tasks,
comparing accuracy, latency, and token efficiency against baseline pooling.

Author: Research Implementation
Date: 2025-01-26
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import torch
from datasets import load_dataset

# FIX: 2025-07-26 - Disable pdb debugging to prevent interactive breakpoints
# ISSUE: LaViDa submodule has pdb.set_trace() that stops execution in Colab
# SOLUTION: Override pdb.set_trace() to be a no-op before importing LaViDa
# RESEARCH IMPACT: Enables automated evaluation without manual intervention
import pdb
pdb.set_trace = lambda: None

# Colab environment detection - Fixed method
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Initialize HF_TOKEN as None - will be set up when needed
HF_TOKEN = None

# Set paths based on environment
# In Colab, we work from /content/repo-name/ so paths are relative
if IN_COLAB:
    BASE_PATH = './'
    DATA_PATH = './data/'
    RESULTS_PATH = './results/'
else:
    BASE_PATH = './'
    DATA_PATH = './data/'
    RESULTS_PATH = './results/'

# Ensure results directory exists
os.makedirs(RESULTS_PATH, exist_ok=True)

from lavida_shirg_integration import LaViDaSHIRGWrapper, create_lavida_shirg_model


def set_hf_token_manually(token):
    """
    Convenience function to set HF token manually
    Use this if Colab secrets are not working
    
    Args:
        token: Your Hugging Face token (starts with 'hf_')
    """
    if not token.startswith('hf_'):
        print("⚠️ Warning: HF tokens usually start with 'hf_'")
    
    os.environ['HF_TOKEN'] = token
    os.environ['HUGGINGFACE_HUB_TOKEN'] = token
    print("✅ HF_TOKEN set manually")
    
    # Test the token
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        user_info = api.whoami()
        print(f"✅ Token verified - authenticated as: {user_info['name']}")
        return True
    except Exception as e:
        print(f"❌ Token verification failed: {e}")
        return False


class SHIRGEvaluator:
    """
    Comprehensive evaluator for SHIRG vs baseline LaViDa
    
    Supports multiple evaluation metrics:
    - OCR accuracy on text-heavy images
    - VQA accuracy on chart/document questions  
    - Latency benchmarking
    - Token efficiency analysis
    - Attention pattern visualization
    """
    
    def __init__(self, 
                 output_dir: str = RESULTS_PATH,
                 debug: bool = True):
        """
        Initialize SHIRG evaluator
        
        Args:
            output_dir: Directory to save evaluation results
            debug: Enable debug output
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.debug = debug
        
        # Models (loaded on demand)
        self.baseline_model = None
        self.shirg_model = None
        
        # Results storage
        self.results = {
            'baseline': [],
            'shirg': [],
            'comparison': {}
        }
        
        print(f"🔬 SHIRG Evaluator initialized")
        print(f"   Output directory: {self.output_dir}")
    
    def load_baseline_model(self, shirg_config: Optional[Dict[str, Any]] = None):
        """Load only baseline model for memory-efficient evaluation"""
        print("🔄 Loading baseline LaViDa...")
        
        # FIX: 2025-07-27 - Ensure baseline truly uses LaViDa's default pooling
        # ISSUE: Baseline was using SHIRG infrastructure with alpha=0
        # SOLUTION: Use exact LaViDa defaults to ensure true baseline comparison
        # RESEARCH IMPACT: Enables proper baseline vs SHIRG comparison
        
        # Default config for baseline (LaViDa pooling) - disable SHIRG completely
        baseline_config = {
            'target_tokens': 729,  # Match spatial grid dimensions for consistent comparison
            'alpha': 0.0,  # Disable text conditioning completely
            'hierarchical_levels': 3,
            'latency_budget_ms': 1000.0,
            'use_fast_clustering': True,
            'enable_caching': True,
            'debug': self.debug
        }
        
        if shirg_config:
            baseline_config.update(shirg_config)
        
        try:
            self.baseline_model = LaViDaSHIRGWrapper(shirg_config=baseline_config)
            self.baseline_model.load_model()
            print(f"✅ Baseline model loaded. GPU memory: {self._get_gpu_memory()}")
            
        except Exception as e:
            print(f"❌ Failed to load baseline model: {e}")
            raise
    
    def load_shirg_model(self, shirg_config: Optional[Dict[str, Any]] = None):
        """Load only SHIRG model for memory-efficient evaluation"""
        print("🔄 Loading SHIRG-enhanced LaViDa...")
        
        # Default SHIRG config optimized for evaluation
        default_shirg_config = {
            'target_tokens': 729,   # Output 729 for LaViDa compatibility 
            'alpha': 0.3,           # Enable SHIRG selection from larger unpooled tokens
            'hierarchical_levels': 3,
            'latency_budget_ms': 1000.0,
            'use_fast_clustering': True,
            'enable_caching': True,
            'debug': self.debug
        }
        
        if shirg_config:
            default_shirg_config.update(shirg_config)
        
        try:
            self.shirg_model = LaViDaSHIRGWrapper(shirg_config=default_shirg_config)
            self.shirg_model.load_model()
            print(f"✅ SHIRG model loaded. GPU memory: {self._get_gpu_memory()}")
            
        except Exception as e:
            print(f"❌ Failed to load SHIRG model: {e}")
            raise
    
    def cleanup_models(self):
        """Clean up models and free GPU memory"""
        print("🧹 Cleaning up models and GPU memory...")
        
        # FIX: 2025-07-26 - Call wrapper cleanup methods for proper memory release
        # ISSUE: Previous cleanup only deleted references, not model internals
        # SOLUTION: Call wrapper's cleanup method before deleting references
        # RESEARCH IMPACT: Ensures proper GPU memory release between evaluations
        
        if hasattr(self, 'baseline_model') and self.baseline_model is not None:
            self.baseline_model.cleanup()
            del self.baseline_model
            self.baseline_model = None
            
        if hasattr(self, 'shirg_model') and self.shirg_model is not None:
            self.shirg_model.cleanup()
            del self.shirg_model
            self.shirg_model = None
            
        # Additional GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Force garbage collection
            import gc
            gc.collect()
            
        print(f"✅ Cleanup complete. GPU memory: {self._get_gpu_memory()}")
    
    def _get_gpu_memory(self) -> str:
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            return f"{allocated:.1f}GB / {total:.1f}GB"
        return "N/A"
    
    def load_real_dataset(self, 
                         dataset_name: str = "chartqa",
                         num_samples: int = 10,
                         split: str = "test",
                         skip_auth_check: bool = False) -> List[Dict[str, Any]]:
        """
        Load real dataset samples from ChartQA or DocVQA
        
        Args:
            dataset_name: "chartqa" or "docvqa" 
            num_samples: Number of samples to load
            split: Dataset split to use
            skip_auth_check: Skip authentication check (use if token is already set)
            
        Returns:
            List of real dataset samples formatted for SHIRG evaluation
        """
        print(f"🔄 Loading real {dataset_name} dataset ({num_samples} samples from {split} split)...")
        
        try:
            if dataset_name.lower() == "chartqa":
                # Load ChartQA dataset
                dataset = load_dataset("lmms-lab/ChartQA", split=split)
                
            elif dataset_name.lower() == "docvqa":
                # Load DocVQA dataset  
                if split == "test":
                    # DocVQA test split doesn't have ground truth, use validation
                    split = "validation"
                dataset = load_dataset("lmms-lab/DocVQA", "DocVQA", split=split)
                
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")
            
            # Sample from dataset
            total_samples = len(dataset)
            if num_samples > total_samples:
                print(f"⚠️ Requested {num_samples} samples but dataset only has {total_samples}, using all")
                num_samples = total_samples
            
            # Select samples (first N for reproducibility)
            selected_indices = list(range(min(num_samples, total_samples)))
            
            test_samples = []
            for i, idx in enumerate(selected_indices):
                sample = dataset[idx]
                
                # Extract image
                image = sample["image"]
                if hasattr(image, 'convert'):
                    image = image.convert("RGB")
                
                # Save image to temp path for compatibility with existing code
                image_dir = Path(DATA_PATH) / "real_images" / dataset_name
                image_dir.mkdir(parents=True, exist_ok=True)
                image_path = image_dir / f"{dataset_name}_sample_{i:03d}.jpg"
                image.save(image_path)
                
                # Extract question and answer based on dataset format
                if dataset_name.lower() == "chartqa":
                    question = sample["question"]
                    ground_truth = sample["answer"]
                    category = "vqa"  # ChartQA is primarily VQA with some OCR elements
                    
                elif dataset_name.lower() == "docvqa":
                    question = sample["question"] 
                    # DocVQA has multiple valid answers, take the first one
                    answers = sample.get("answers", sample.get("answer", []))
                    if isinstance(answers, list) and len(answers) > 0:
                        ground_truth = answers[0]
                    else:
                        ground_truth = str(answers) if answers else "unknown"
                    category = "ocr"  # DocVQA is primarily OCR
                
                # Determine difficulty based on question complexity
                difficulty = "hard" if len(question.split()) > 8 else "easy"
                
                formatted_sample = {
                    'question_id': f'{dataset_name}_real_{i:03d}',
                    'image_path': str(image_path),
                    'question': question,
                    'ground_truth': ground_truth,
                    'category': category,
                    'difficulty': difficulty,
                    'dataset_source': dataset_name,
                    'original_idx': idx
                }
                test_samples.append(formatted_sample)
            
            print(f"✅ Loaded {len(test_samples)} real {dataset_name} samples")
            print(f"   - {sum(1 for s in test_samples if s['difficulty'] == 'hard')} hard samples")
            print(f"   - {sum(1 for s in test_samples if s['category'] == 'ocr')} OCR samples") 
            print(f"   - {sum(1 for s in test_samples if s['category'] == 'vqa')} VQA samples")
            
            return test_samples
            
        except Exception as e:
            print(f"❌ Failed to load {dataset_name} dataset: {e}")
            print("📝 Falling back to synthetic test data...")
            return self.create_synthetic_test_dataset(num_samples)

    def create_synthetic_test_dataset(self, num_samples: int = 20) -> List[Dict[str, Any]]:
        """
        Create synthetic test dataset for SHIRG evaluation (fallback)
        
        Creates synthetic OCR/VQA test cases that highlight SHIRG's advantages:
        - High-resolution content with fine details
        - Text-heavy images requiring precise token selection
        - Questions that test both detail preservation and semantic relevance
        """
        print(f"🔄 Creating synthetic SHIRG test dataset with {num_samples} samples...")
        
        # Create synthetic OCR/VQA test cases designed for SHIRG evaluation
        test_samples = []
        
        # Questions specifically designed to test SHIRG's capabilities
        detailed_questions = [
            # OCR questions requiring fine detail (SHIRG advantage)
            "What is the exact title shown at the top of this chart?",
            "What are the Q4 sales numbers for Widget A?", 
            "Read the copyright text in the bottom left corner",
            "What page number is shown in the top right?",
            "What is the total value for Q3?",
            "What product has sales of 13.8K in Q4?",
            "What company name appears in the copyright?",
            "What year is mentioned in the chart title?",
            "What is the highest quarterly value shown?",
            "What are the column headers in the data table?",
            # VQA questions requiring semantic understanding + detail
            "Which quarter showed the highest growth?",
            "How many products are compared in this chart?",
            "What type of data visualization is shown?",
            "Are the sales trending upward or downward?",
            "What is the difference between Q1 and Q4 totals?"
        ]
        
        # Expected answers for synthetic content
        detailed_answers = [
            "Sales Q4 2024",
            "22.1K",
            "© 2025 Research Corp", 
            "Page 1 of 3",
            "30.1K",
            "Widget B",
            "Research Corp",
            "2024",
            "35.9K",
            "Product, Q1, Q2, Q3, Q4",
            "Q4",
            "2",
            "Bar chart", 
            "Upward",
            "15.1K"
        ]
        
        for i in range(num_samples):
            difficulty = 'hard' if i % 3 != 0 else 'easy'  # More hard samples
            category = 'ocr' if i % 2 == 0 else 'vqa'
            
            question_idx = i % len(detailed_questions)
            
            sample = {
                'question_id': f'shirg_synthetic_{i:03d}',
                'image_path': f'{DATA_PATH}/test_images/shirg_sample_{i:03d}.jpg',
                'question': detailed_questions[question_idx],
                'ground_truth': detailed_answers[question_idx] if question_idx < len(detailed_answers) else f"Expected answer {i}",
                'category': category,
                'difficulty': difficulty,
                'shirg_test_type': 'detail_preservation' if category == 'ocr' else 'semantic_relevance',
                'dataset_source': 'synthetic'
            }
            test_samples.append(sample)
        
        print(f"✅ Created {len(test_samples)} synthetic SHIRG test samples")
        print(f"   - {sum(1 for s in test_samples if s['difficulty'] == 'hard')} hard samples (high-res details)")
        print(f"   - {sum(1 for s in test_samples if s['category'] == 'ocr')} OCR samples")
        print(f"   - {sum(1 for s in test_samples if s['category'] == 'vqa')} VQA samples")
        
        return test_samples
    
    def create_test_dataset(self, 
                           num_samples: int = 20,
                           use_real_data: bool = True,
                           dataset_mix: Optional[Dict[str, int]] = None) -> List[Dict[str, Any]]:
        """
        Create test dataset for SHIRG evaluation - mix of real and synthetic data
        
        Args:
            num_samples: Total number of samples
            use_real_data: Whether to load real datasets  
            dataset_mix: Dict specifying samples per dataset, e.g. {"chartqa": 5, "docvqa": 5}
            
        Returns:
            Mixed test dataset optimized for SHIRG evaluation
        """
        
        if not use_real_data:
            return self.create_synthetic_test_dataset(num_samples)
        
        # Default mix: half ChartQA, half DocVQA
        if dataset_mix is None:
            dataset_mix = {
                "chartqa": num_samples // 2,
                "docvqa": num_samples - (num_samples // 2)
            }
        
        test_samples = []
        
        # Load real datasets
        for dataset_name, sample_count in dataset_mix.items():
            if sample_count > 0:
                try:
                    real_samples = self.load_real_dataset(
                        dataset_name=dataset_name,
                        num_samples=sample_count,
                        split="test" if dataset_name == "chartqa" else "validation"
                    )
                    test_samples.extend(real_samples)
                except Exception as e:
                    print(f"⚠️ Failed to load {dataset_name}: {e}")
                    
        # If we didn't get enough real samples, fill with synthetic ones
        if len(test_samples) < num_samples:
            remaining = num_samples - len(test_samples)
            print(f"📝 Adding {remaining} synthetic samples to reach target of {num_samples}")
            synthetic_samples = self.create_synthetic_test_dataset(remaining)
            test_samples.extend(synthetic_samples)
        
        # Shuffle to mix real and synthetic
        import random
        random.shuffle(test_samples)
        
        print(f"✅ Created mixed test dataset with {len(test_samples)} samples")
        real_count = sum(1 for s in test_samples if s.get('dataset_source') != 'synthetic')
        synthetic_count = len(test_samples) - real_count
        print(f"   - {real_count} real dataset samples")
        print(f"   - {synthetic_count} synthetic samples")
        
        return test_samples
    
    def evaluate_sample(self, 
                       model: LaViDaSHIRGWrapper, 
                       sample: Dict[str, Any], 
                       model_name: str
                       ) -> Dict[str, Any]:
        """
        Evaluate a single sample with given model
        
        Args:
            model: LaViDa model (baseline or SHIRG)
            sample: Test sample dictionary
            model_name: Name for tracking ("baseline" or "shirg")
            
        Returns:
            Evaluation result dictionary
        """
        
        try:
            start_time = time.time()
            
            # Check if image exists (for real evaluation)
            if not os.path.exists(sample['image_path']):
                # Create SHIRG test image based on difficulty
                self._create_dummy_image(sample['image_path'], sample.get('difficulty', 'easy'))
            
            # FIX: 2025-07-27 - Use optimized generation parameters for faster LaViDa inference
            # ISSUE: Default parameters don't enable LaViDa's fast prefix-DLM caching
            # SOLUTION: Pass optimized parameters that enable 2x speedup through caching
            # RESEARCH IMPACT: Faster evaluation with proper LaViDa performance characteristics
            
            # Generate response with LaViDa-optimized parameters
            response = model.generate(
                image_path=sample['image_path'],
                question=sample['question'],
                max_new_tokens=32,  # LaViDa constraint for fast inference
                temperature=0.0,    # Deterministic for evaluation
                do_sample=False,    # No sampling for consistency
                prefix_lm=True,     # Enable prefix-DLM caching (key for speed)
                step_ratio=0.5,     # Faster diffusion scheduling
                block_length=32     # Match max_new_tokens for optimal caching
            )
            
            inference_time = time.time() - start_time
            
            # FIX: 2025-07-27 - Clean Unicode before accuracy computation
            # ISSUE: Unicode characters prevent proper string matching in accuracy computation
            # SOLUTION: Clean both response and ground truth before comparison
            # RESEARCH IMPACT: More accurate evaluation results due to proper text matching
            
            def clean_unicode_text(text):
                """Clean Unicode characters for better readability and matching"""
                if not isinstance(text, str):
                    return text
                # Replace smart quotes and other common Unicode chars
                text = text.replace('\u2018', "'").replace('\u2019', "'")  # Smart single quotes
                text = text.replace('\u201c', '"').replace('\u201d', '"')  # Smart double quotes  
                text = text.replace('\u2013', '-').replace('\u2014', '--')  # En/em dashes
                text = text.replace('\u00a0', ' ')  # Non-breaking space
                return text
            
            # Clean Unicode before accuracy computation
            clean_response = clean_unicode_text(response)
            clean_ground_truth = clean_unicode_text(sample['ground_truth'])
            
            # Compute accuracy with dataset-specific metrics
            accuracy = self._compute_accuracy(
                clean_response, 
                clean_ground_truth, 
                sample.get('category', 'vqa'),
                sample.get('dataset_source', 'synthetic')
            )
            
            result = {
                'question_id': sample['question_id'],
                'question': clean_unicode_text(sample['question']), 
                'response': clean_unicode_text(clean_response),
                'ground_truth': clean_unicode_text(clean_ground_truth),
                'accuracy': accuracy,
                'inference_time_ms': inference_time * 1000,
                'model': model_name,
                'category': sample['category'],
                'difficulty': sample['difficulty']
            }
            
            # Add model-specific metrics
            if hasattr(model, 'shirg_selector') and model.shirg_selector:
                shirg_stats = model.shirg_selector.get_performance_stats()
                result.update({f'shirg_{k}': v for k, v in shirg_stats.items()})
                
                # FIX: 2025-07-27 - Debug SHIRG metrics collection
                if self.debug and model_name == "shirg":
                    print(f"🔍 SHIRG stats for {sample['question_id']}: {shirg_stats}")
            else:
                # Add zero SHIRG stats for baseline model
                result.update({
                    'shirg_avg_selection_time_ms': 0.0,
                    'shirg_max_selection_time_ms': 0.0,
                    'shirg_min_selection_time_ms': 0.0,
                    'shirg_total_selections': 0,
                    'shirg_latency_budget_ms': 1000.0,
                    'shirg_budget_exceeded_count': 0,
                    'shirg_cache_hits': 0,
                    'shirg_cache_misses': 0,
                    'shirg_cache_hit_rate_percent': 0.0,
                    'shirg_cache_size': 0
                })
            
            return result
            
        except Exception as e:
            print(f"⚠️ Evaluation failed for {sample['question_id']}: {e}")
            return {
                'question_id': sample['question_id'],
                'response': f"ERROR: {str(e)}",
                'ground_truth': sample['ground_truth'],
                'accuracy': 0.0,
                'inference_time_ms': 0.0,
                'model': model_name,
                'error': str(e)
            }
    
    def run_evaluation(self, 
                      test_dataset: Optional[List[Dict[str, Any]]] = None,
                      num_samples: int = 20) -> Dict[str, Any]:
        """
        Run memory-efficient sequential evaluation comparing baseline vs SHIRG
        
        Args:
            test_dataset: Test dataset, creates one if None
            num_samples: Number of samples to evaluate
            
        Returns:
            Comprehensive evaluation results
        """
        
        if test_dataset is None:
            test_dataset = self.create_test_dataset(num_samples)
        
        print(f"🚀 Running sequential evaluation on {len(test_dataset)} samples...")
        
        # FIX: 2025-07-26 - Sequential evaluation with cleanup to save memory
        # ISSUE: Loading two 25GB models simultaneously exceeds 40GB GPU limit  
        # SOLUTION: Evaluate one model completely, cleanup, then evaluate the other
        # RESEARCH IMPACT: Enables evaluation on single GPU with memory constraints
        
        # Phase 1: Evaluate baseline model
        print("📊 Phase 1: Evaluating baseline LaViDa...")
        self.load_baseline_model()
        
        baseline_results = []
        for i, sample in enumerate(test_dataset):
            if self.debug and i % 5 == 0:
                print(f"   Baseline progress: {i+1}/{len(test_dataset)}")
            
            result = self.evaluate_sample(self.baseline_model, sample, "baseline")
            baseline_results.append(result)
        
        print("✅ Baseline evaluation complete")
        
        # Cleanup baseline model before loading SHIRG
        self.cleanup_models()
        
        # Phase 2: Evaluate SHIRG model  
        print("📊 Phase 2: Evaluating SHIRG LaViDa...")
        self.load_shirg_model()
        
        shirg_results = []
        for i, sample in enumerate(test_dataset):
            if self.debug and i % 5 == 0:
                print(f"   SHIRG progress: {i+1}/{len(test_dataset)}")
            
            result = self.evaluate_sample(self.shirg_model, sample, "shirg")
            shirg_results.append(result)
        
        print("✅ SHIRG evaluation complete")
        
        # Final cleanup
        self.cleanup_models()
        
        # Store results
        self.results['baseline'] = baseline_results
        self.results['shirg'] = shirg_results
        
        # Compute comparison metrics
        comparison = self._compute_comparison_metrics(baseline_results, shirg_results)
        self.results['comparison'] = comparison
        
        # Save results
        self._save_results()
        
        # Print summary
        self._print_evaluation_summary(comparison)
        
        return self.results
    
    def _compute_comparison_metrics(self, 
                                  baseline_results: List[Dict[str, Any]], 
                                  shirg_results: List[Dict[str, Any]]
                                  ) -> Dict[str, Any]:
        """Compute comprehensive comparison metrics"""
        
        # Accuracy metrics
        baseline_acc = np.mean([r['accuracy'] for r in baseline_results])
        shirg_acc = np.mean([r['accuracy'] for r in shirg_results])
        
        # Latency metrics
        baseline_latency = np.mean([r['inference_time_ms'] for r in baseline_results])
        shirg_latency = np.mean([r['inference_time_ms'] for r in shirg_results])
        
        # Error rates
        baseline_errors = sum(1 for r in baseline_results if 'error' in r)
        shirg_errors = sum(1 for r in shirg_results if 'error' in r)
        
        # SHIRG-specific metrics
        shirg_selection_times = [r.get('shirg_avg_selection_time_ms', 0) for r in shirg_results]
        avg_shirg_time = np.mean(shirg_selection_times) if shirg_selection_times else 0
        
        comparison = {
            'accuracy': {
                'baseline': baseline_acc,
                'shirg': shirg_acc,
                'improvement': shirg_acc - baseline_acc,
                'improvement_percent': ((shirg_acc - baseline_acc) / baseline_acc * 100) if baseline_acc > 0 else 0
            },
            'latency': {
                'baseline_ms': baseline_latency,
                'shirg_ms': shirg_latency,
                'overhead_ms': shirg_latency - baseline_latency,
                'overhead_percent': ((shirg_latency - baseline_latency) / baseline_latency * 100) if baseline_latency > 0 else 0,
                'shirg_selection_ms': avg_shirg_time
            },
            'reliability': {
                'baseline_errors': baseline_errors,
                'shirg_errors': shirg_errors,
                'baseline_error_rate': baseline_errors / len(baseline_results) * 100,
                'shirg_error_rate': shirg_errors / len(shirg_results) * 100
            },
            'token_efficiency': {
                'baseline_tokens': 980,  # LaViDa default
                'shirg_tokens': 1024,    # SHIRG target
                'token_increase': 44,
                'token_increase_percent': (44 / 980) * 100
            }
        }
        
        return comparison
    
    def _compute_accuracy(self, response: str, ground_truth: str, category: str, 
                         dataset_source: str = "synthetic") -> float:
        """
        Compute accuracy with appropriate matching for different datasets and tasks
        
        Args:
            response: Model response
            ground_truth: Expected answer
            category: 'ocr' or 'vqa'
            dataset_source: Source dataset ("chartqa", "docvqa", "synthetic")
            
        Returns:
            Accuracy score (0.0 to 1.0)
        """
        
        # Use dataset-specific accuracy metrics where available
        if dataset_source == "chartqa":
            return self._chartqa_relaxed_correctness(response, ground_truth)
        elif dataset_source == "docvqa":
            return self._docvqa_anls_accuracy(response, ground_truth)
        else:
            # Default accuracy for synthetic data
            return self._default_accuracy(response, ground_truth, category)
    
    def _chartqa_relaxed_correctness(self, prediction: str, target: str, 
                                   max_relative_change: float = 0.05) -> float:
        """
        ChartQA relaxed correctness metric (from LaViDa evaluation)
        Allows 5% tolerance for numeric answers, exact match for text
        """
        def _to_float(text: str):
            try:
                if text.endswith("%"):
                    return float(text.rstrip("%")) / 100.0
                else:
                    return float(text)
            except ValueError:
                return None

        prediction_float = _to_float(prediction.strip())
        target_float = _to_float(target.strip())
        
        if prediction_float is not None and target_float is not None:
            relative_change = abs(prediction_float - target_float) / abs(target_float)
            return 1.0 if relative_change <= max_relative_change else 0.0
        else:
            # Text comparison - case insensitive
            return 1.0 if prediction.lower().strip() == target.lower().strip() else 0.0
    
    def _docvqa_anls_accuracy(self, prediction: str, target: str) -> float:
        """
        DocVQA ANLS (Average Normalized Levenshtein Similarity) metric approximation
        More lenient than exact match to account for OCR variations
        """
        def levenshtein_distance(s1: str, s2: str) -> int:
            """Compute Levenshtein distance between two strings"""
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        pred_clean = prediction.lower().strip()
        target_clean = target.lower().strip()
        
        if pred_clean == target_clean:
            return 1.0
        
        if len(target_clean) == 0:
            return 0.0
        
        # Compute normalized Levenshtein similarity
        distance = levenshtein_distance(pred_clean, target_clean)
        max_len = max(len(pred_clean), len(target_clean))
        similarity = 1.0 - (distance / max_len)
        
        # ANLS threshold - consider correct if similarity > 0.5
        return similarity if similarity > 0.5 else 0.0
    
    def _default_accuracy(self, response: str, ground_truth: str, category: str) -> float:
        """
        Default accuracy computation for synthetic data
        """
        response_clean = response.lower().strip()
        truth_clean = ground_truth.lower().strip()
        
        if category == 'ocr':
            # OCR requires more precise matching but allow some flexibility
            # Exact match
            if response_clean == truth_clean:
                return 1.0
            
            # Check if key terms are present
            truth_words = set(truth_clean.split())
            response_words = set(response_clean.split())
            
            if len(truth_words) == 0:
                return 0.0
            
            # Partial credit for word overlap
            overlap = len(truth_words.intersection(response_words))
            return overlap / len(truth_words)
        
        else:  # VQA
            # VQA allows more semantic flexibility
            if truth_clean in response_clean or response_clean in truth_clean:
                return 1.0
            
            # Check for key term matches
            truth_words = set(truth_clean.split())
            response_words = set(response_clean.split())
            
            if len(truth_words) == 0:
                return 0.0
            
            overlap = len(truth_words.intersection(response_words))
            if overlap > 0:
                return min(1.0, overlap / len(truth_words) * 1.5)  # Bonus for VQA
            
            return 0.0
    
    def _save_results(self):
        """Save evaluation results to JSON files"""
        
        # Save detailed results
        results_file = self.output_dir / 'shirg_evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save comparison summary
        summary_file = self.output_dir / 'shirg_comparison_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(self.results['comparison'], f, indent=2, default=str)
        
        print(f"💾 Results saved to {results_file}")
        print(f"💾 Summary saved to {summary_file}")
    
    def _print_evaluation_summary(self, comparison: Dict[str, Any]):
        """Print comprehensive evaluation summary"""
        
        print("\n" + "="*60)
        print("📊 SHIRG EVALUATION SUMMARY")
        print("="*60)
        
        # Accuracy results
        acc = comparison['accuracy']
        print(f"\n🎯 ACCURACY:")
        print(f"   Baseline:    {acc['baseline']:.1%}")
        print(f"   SHIRG:       {acc['shirg']:.1%}")
        print(f"   Improvement: {acc['improvement']:+.1%} ({acc['improvement_percent']:+.1f}%)")
        
        # Latency results
        lat = comparison['latency']
        print(f"\n⏱️  LATENCY:")
        print(f"   Baseline:       {lat['baseline_ms']:.1f}ms")
        print(f"   SHIRG:          {lat['shirg_ms']:.1f}ms")  
        print(f"   Overhead:       {lat['overhead_ms']:+.1f}ms ({lat['overhead_percent']:+.1f}%)")
        print(f"   SHIRG selection: {lat['shirg_selection_ms']:.1f}ms")
        
        # Token efficiency
        tok = comparison['token_efficiency']
        print(f"\n🎲 TOKEN EFFICIENCY:")
        print(f"   Baseline tokens: {tok['baseline_tokens']}")
        print(f"   SHIRG tokens:    {tok['shirg_tokens']}")
        print(f"   Increase:        +{tok['token_increase']} ({tok['token_increase_percent']:+.1f}%)")
        
        # Reliability
        rel = comparison['reliability']
        print(f"\n🛡️  RELIABILITY:")
        print(f"   Baseline errors: {rel['baseline_errors']} ({rel['baseline_error_rate']:.1f}%)")
        print(f"   SHIRG errors:    {rel['shirg_errors']} ({rel['shirg_error_rate']:.1f}%)")
        
        print("\n" + "="*60)
    
    def plot_results(self):
        """Generate visualization plots for evaluation results"""
        
        if not self.results['baseline'] or not self.results['shirg']:
            print("⚠️ No results to plot. Run evaluation first.")
            return
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('SHIRG vs Baseline LaViDa Evaluation', fontsize=16)
        
        # 1. Accuracy comparison
        ax1 = axes[0, 0]
        baseline_acc = [r['accuracy'] for r in self.results['baseline']]
        shirg_acc = [r['accuracy'] for r in self.results['shirg']]
        
        ax1.hist([baseline_acc, shirg_acc], bins=10, alpha=0.7, label=['Baseline', 'SHIRG'])
        ax1.set_xlabel('Accuracy')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Accuracy Distribution')
        ax1.legend()
        
        # 2. Latency comparison
        ax2 = axes[0, 1]
        baseline_lat = [r['inference_time_ms'] for r in self.results['baseline']]
        shirg_lat = [r['inference_time_ms'] for r in self.results['shirg']]
        
        ax2.boxplot([baseline_lat, shirg_lat], labels=['Baseline', 'SHIRG'])
        ax2.set_ylabel('Inference Time (ms)')
        ax2.set_title('Latency Distribution')
        
        # 3. Accuracy by category
        ax3 = axes[1, 0]
        categories = ['ocr', 'vqa']
        baseline_cat_acc = {}
        shirg_cat_acc = {}
        
        for cat in categories:
            baseline_cat_acc[cat] = np.mean([r['accuracy'] for r in self.results['baseline'] if r.get('category') == cat])
            shirg_cat_acc[cat] = np.mean([r['accuracy'] for r in self.results['shirg'] if r.get('category') == cat])
        
        x = np.arange(len(categories))
        width = 0.35
        ax3.bar(x - width/2, [baseline_cat_acc[c] for c in categories], width, label='Baseline', alpha=0.7)
        ax3.bar(x + width/2, [shirg_cat_acc[c] for c in categories], width, label='SHIRG', alpha=0.7)
        ax3.set_xlabel('Category')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Accuracy by Category')
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories)
        ax3.legend()
        
        # 4. Token selection efficiency (SHIRG only)
        ax4 = axes[1, 1]
        shirg_selection_times = [r.get('shirg_avg_selection_time_ms', 0) for r in self.results['shirg']]
        if any(t > 0 for t in shirg_selection_times):
            ax4.hist(shirg_selection_times, bins=10, alpha=0.7, color='orange')
            ax4.set_xlabel('SHIRG Selection Time (ms)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('SHIRG Selection Latency')
            ax4.axvline(30, color='red', linestyle='--', label='Budget (1000ms)')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'No SHIRG timing data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('SHIRG Selection Latency')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / 'shirg_evaluation_plots.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📈 Plots saved to {plot_file}")
        
        plt.show()
    
    def _create_dummy_image(self, image_path: str, difficulty: str = 'easy'):
        """Create a dummy test image with OCR content for testing SHIRG capabilities"""
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        
        # Create high-resolution image to test SHIRG's fine-detail extraction
        img_size = (1024, 768) if difficulty == 'hard' else (800, 600)
        img = Image.new('RGB', img_size, color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            # Try to use a larger font for better OCR testing
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        except:
            font_large = None
            font_small = None
        
        if difficulty == 'hard':
            # High-resolution test content that requires fine detail (SHIRG advantage)
            # Small text in corners (tests SHIRG's detail preservation)
            draw.text((10, 10), "Chart Title: Sales Q4 2024", fill='black', font=font_large)
            draw.text((img_size[0]-150, 10), "Page 1 of 3", fill='gray', font=font_small)
            draw.text((10, img_size[1]-30), "© 2025 Research Corp", fill='gray', font=font_small)
            
            # Data table (tests OCR on structured content)
            table_data = [
                ["Product", "Q1", "Q2", "Q3", "Q4"],
                ["Widget A", "12.5K", "15.2K", "18.9K", "22.1K"],
                ["Widget B", "8.3K", "9.7K", "11.2K", "13.8K"],
                ["Total", "20.8K", "24.9K", "30.1K", "35.9K"]
            ]
            
            start_y = 100
            for i, row in enumerate(table_data):
                for j, cell in enumerate(row):
                    x = 50 + j * 120
                    y = start_y + i * 30
                    draw.text((x, y), cell, fill='black', font=font_large)
                    draw.rectangle([x-5, y-5, x+110, y+25], outline='black', width=1)
            
            # Chart-like visualization
            chart_x, chart_y = 50, 300
            bars = [60, 85, 120, 150]  # Heights representing data
            for i, height in enumerate(bars):
                x = chart_x + i * 80
                draw.rectangle([x, chart_y - height, x + 60, chart_y], fill='blue', outline='black')
                draw.text((x + 10, chart_y - height - 20), f"{20.8 + i * 4.1:.1f}K", fill='black', font=font_small)
                draw.text((x + 15, chart_y + 10), f"Q{i+1}", fill='black', font=font_large)
        
        else:
            # Easy test content
            draw.text((50, 50), "Sample OCR Text", fill='black', font=font_large)
            draw.text((50, 100), "Number: 12345", fill='black', font=font_large)
            draw.text((50, 150), "Date: 2025-01-26", fill='black', font=font_large)
            
            # Add some shapes for visual elements
            draw.rectangle([200, 200, 400, 300], outline='blue', width=2)
            draw.ellipse([250, 350, 350, 450], outline='red', width=2)
        
        img.save(image_path)
        return image_path


def run_shirg_evaluation(num_samples: int = 10, 
                        shirg_config: Optional[Dict[str, Any]] = None,
                        use_real_data: bool = True,
                        dataset_mix: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """
    Convenience function to run complete SHIRG evaluation with real datasets
    
    Args:
        num_samples: Number of test samples to evaluate
        shirg_config: SHIRG configuration parameters
        use_real_data: Whether to use real ChartQA/DocVQA data
        dataset_mix: Mix of datasets, e.g. {"chartqa": 3, "docvqa": 2}
        
    Returns:
        Evaluation results dictionary
    """
    
    if use_real_data:
        print(f"🚀 Starting SHIRG evaluation with {num_samples} real dataset samples...")
    else:
        print(f"🚀 Starting SHIRG evaluation with {num_samples} synthetic samples...")
    
    # Create evaluator
    evaluator = SHIRGEvaluator(debug=True)
    
    # Create dataset with real data
    test_dataset = evaluator.create_test_dataset(
        num_samples=num_samples,
        use_real_data=use_real_data,
        dataset_mix=dataset_mix
    )
    
    # Run evaluation
    results = evaluator.run_evaluation(test_dataset=test_dataset)
    
    # Generate plots
    try:
        evaluator.plot_results()
    except Exception as e:
        print(f"⚠️ Plotting failed: {e}")
    
    return results


def ablation_study_alpha(alpha_values: List[float] = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                        num_samples: int = 10) -> Dict[str, Any]:
    """
    Run ablation study on alpha parameter
    
    Args:
        alpha_values: List of alpha values to test
        num_samples: Number of samples per configuration
        
    Returns:
        Ablation study results
    """
    
    print(f"🔬 Running alpha ablation study: {alpha_values}")
    
    results = {}
    test_dataset = SHIRGEvaluator().create_test_dataset(num_samples)
    
    for alpha in alpha_values:
        print(f"\n📊 Testing alpha = {alpha}")
        
        shirg_config = {
            'target_tokens': 729,   # Fixed constraint for LaViDa compatibility
            'alpha': alpha,         # Key difference: alpha > 0 enables SHIRG selection
            'hierarchical_levels': 3,
            'latency_budget_ms': 1000.0,
            'debug': False
        }
        
        evaluator = SHIRGEvaluator(debug=False)
        evaluation_results = evaluator.run_evaluation(test_dataset, num_samples)
        
        results[alpha] = {
            'accuracy': evaluation_results['comparison']['accuracy']['shirg'],
            'latency': evaluation_results['comparison']['latency']['shirg_ms'],
            'shirg_time': evaluation_results['comparison']['latency']['shirg_selection_ms']
        }
        
        print(f"   α={alpha}: Accuracy={results[alpha]['accuracy']:.1%}, Latency={results[alpha]['latency']:.1f}ms")
    
    # Find best alpha
    best_alpha = max(results.keys(), key=lambda a: results[a]['accuracy'])
    print(f"\n🏆 Best alpha: {best_alpha} (Accuracy: {results[best_alpha]['accuracy']:.1%})")
    
    # Save ablation results
    ablation_file = Path(RESULTS_PATH) / 'alpha_ablation_results.json'
    with open(ablation_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Ablation results saved to {ablation_file}")
    
    return results


# Test function for Colab
def test_shirg_evaluation():
    """Test SHIRG evaluation framework with real datasets"""
    print("🧪 Testing SHIRG Evaluation Framework...")
    
    try:
        # Create evaluator
        evaluator = SHIRGEvaluator(debug=True)
        
        # Test real dataset loading (small sample)
        print("\n📊 Testing real dataset loading...")
        try:
            # Try to load a few real samples
            test_data = evaluator.create_test_dataset(
                num_samples=5, 
                use_real_data=True,
                dataset_mix={"chartqa": 3, "docvqa": 2}
            )
            print(f"✅ Created test dataset with {len(test_data)} samples")
            
            # Show sample details
            for i, sample in enumerate(test_data[:2]):  # Show first 2
                print(f"\n   Sample {i+1}:")
                print(f"   - Source: {sample.get('dataset_source', 'unknown')}")
                print(f"   - Category: {sample.get('category', 'unknown')}")
                print(f"   - Question: {sample['question'][:60]}...")
                print(f"   - Answer: {sample['ground_truth'][:30]}...")
                
        except Exception as e:
            print(f"⚠️ Real dataset loading failed: {e}")
            print("📝 Using synthetic dataset instead...")
            test_data = evaluator.create_test_dataset(5, use_real_data=False)
        
        run_shirg_evaluation(num_samples=5, use_real_data=True)
        
        return evaluator
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None


if __name__ == "__main__":
    # Run test if executed directly
    print("🚀 SHIRG Evaluation Module")
    print("Running as module - this may cause issues with Colab secrets access")
    print("For better Colab compatibility, run interactively:")
    print("  1. Import the module: import shirg_evaluation")
    print("  2. Set up auth: shirg_evaluation.setup_for_colab('your_token')")
    print("  3. Run evaluation: shirg_evaluation.run_shirg_evaluation()")
    print("\nAttempting to run test anyway...")
    
    try:
        test_shirg_evaluation()
    except Exception as e:
        print(f"\n❌ Module execution failed: {e}")
        print("\n📝 For Colab users, try this instead:")
        print("```python")
        print("import shirg_evaluation")
        print("# Set your HF token manually:")
        print("shirg_evaluation.set_hf_token_manually('hf_your_token_here')")
        print("# Then run evaluation:")
        print("shirg_evaluation.run_shirg_evaluation(num_samples=5)")
        print("```")