#!/usr/bin/env python3
"""
SHIRG Real OCR/VQA Image Validation - Main Orchestration
Test SHIRG token selection on actual OCR/VQA dataset images with questions
"""

import os
import sys
import time
import warnings
from typing import Dict, List, Tuple, Optional, Any



warnings.filterwarnings('ignore')

# Add paths for imports
sys.path.append('./shirg/')
sys.path.append('./llava/')
sys.path.append('./')

# Import the split components
from real_ocr_vqa_model_runner import LaViDaModelRunner, rank0_print
from real_ocr_vqa_dataset_loader import OCRVQADatasetLoader, OCRVQAResultAnalyzer

class RealOCRVQAValidator:
    """Main validator orchestrating dataset loading, model inference, and result analysis"""
    
    def __init__(self, selection_method='base', selection_params=None, prompt_style='extractive'):
        """
        Initialize validator with SHIRG selection method configuration
        
        Args:
            selection_method: Token selection method ('base', 'entropy', 'edge', 'full')
            selection_params: Method-specific parameters dict:
                - entropy_threshold: τ for noise filtering (default: 0.12)
                - edge_weight: Weight for edge prior (default: 0.25)
                - radial_sigma: σ for radial weighting (default: 0.65)
                - merge_threshold: Similarity threshold (default: 0.9)
                - merge_similar: Enable token merging (default: False)
            prompt_style: Prompt style for VQA evaluation ('extractive' or 'conversational')
                - 'extractive': Adds "\nAnswer the question using a single word or phrase."
                - 'conversational': Uses default LaViDa conversation style
        """
        self.selection_method = selection_method
        self.selection_params = selection_params or {}
        self.prompt_style = prompt_style
        
        # Initialize components with selection method and prompt style
        self.model_runner = LaViDaModelRunner(
            selection_method=selection_method,
            selection_params=selection_params,
            prompt_style=prompt_style
        )
        self.dataset_loader = OCRVQADatasetLoader()
        self.result_analyzer = OCRVQAResultAnalyzer()
    
    def run_real_ocr_vqa_validation(self):
        """Run validation on real OCR/VQA images with sequential model loading"""
        print("🔍 SHIRG REAL OCR/VQA IMAGE VALIDATION (SEQUENTIAL)")
        print("=" * 60)
        
        # SEQUENTIAL-FIX: 2025-07-28 - Load models sequentially to prevent GPU OOM
        # ISSUE: Loading both models simultaneously causes GPU memory exhaustion (39.5GB/39.56GB)
        # SOLUTION: Load baseline first, run all inferences, clear memory, then load SHIRG
        # MEMORY IMPACT: Reduces peak memory usage from 39.5GB to ~20GB per model
        # RESEARCH IMPACT: Enables proper SHIRG vs baseline comparison within GPU constraints
        
        # Get real OCR/VQA images with questions first
        ocr_vqa_samples = self.dataset_loader.get_real_ocr_vqa_samples()
        
        if not ocr_vqa_samples:
            print("❌ No samples loaded - validation cannot proceed")
            return {}
        
        # # Phase 1: Run all baseline inferences
        # print("\n🔄 PHASE 1: BASELINE LaViDa INFERENCE")
        # print("=" * 40)
        # baseline_results = self.model_runner._run_all_baseline_inferences(ocr_vqa_samples)
        
        # # Unload baseline model to free memory
        # self.model_runner._unload_baseline_model()
        
        # Phase 2: Run all SHIRG inferences
        print("\n🔄 PHASE 2: SHIRG LaViDa INFERENCE")
        print("=" * 40)
        shirg_results = self.model_runner._run_all_shirg_inferences(ocr_vqa_samples)
        
        # Phase 3: Combine and analyze results WITH visualization (BEFORE unloading models)
        print("\n🔄 PHASE 3: RESULT ANALYSIS AND VISUALIZATION")
        print("=" * 40)
        
        # VISUALIZATION-TIMING-FIX: 2025-07-29 - Create visualizations BEFORE unloading models
        # ISSUE: Models were unloaded before visualization, causing SHIRG tower unavailable errors
        # SOLUTION: Pass initialized model_runner to enable visualization, then unload
        # RESEARCH IMPACT: Enables actual SHIRG token selection visualization
        # LAVIDA IMPACT: Provides visual evidence of baseline vs SHIRG differences
        all_results = self.result_analyzer.combine_baseline_and_shirg_results(
            baseline_results, shirg_results, ocr_vqa_samples, model_runner=self.model_runner
        )
        
        # NOW unload SHIRG model after visualization is complete
        self.model_runner._unload_shirg_model()
        
        # Save consolidated results
        results_file, summary_file, simplified_file = self.result_analyzer.save_consolidated_results(all_results)
        
        print(f"\n✅ VALIDATION COMPLETE!")
        print(f"   📊 Processed {len(all_results)} samples")
        print(f"   💾 Detailed results: {results_file}")
        print(f"   📋 Simplified results: {simplified_file}")
        print(f"   📊 Summary: {summary_file}")
        print(f"   🖼️ Visualizations: /content/shirg_token_visualizations/")
        
        return all_results

def run_validation_with_method(method='base', params=None, prompt_style='extractive'):
    """Run validation with specific selection method"""
    print(f"\n{'='*60}")
    print(f"🧪 Running validation with method: {method.upper()}")
    print(f"   Parameters: {params}")
    print(f"   Prompt style: {prompt_style}")
    print(f"{'='*60}\n")
    
    validator = RealOCRVQAValidator(selection_method=method, selection_params=params, prompt_style=prompt_style)
    results = validator.run_real_ocr_vqa_validation()
    
    print(f"\n🎉 Baseline vs SHIRG-{method.upper()} comparison complete!")
    print(f"   📊 Results: /content/shirg_validation_results/")
    print(f"   🖼️ Visualizations: /content/shirg_token_visualizations/")
    return results

def main():
    """Run baseline vs SHIRG comparison validation with multiple methods"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SHIRG OCR/VQA Validation')
    parser.add_argument('--method', type=str, default='base',
                      choices=['base', 'entropy', 'edge', 'edge_only', 'custom', 'full', 'all'],
                      help='Token selection method to use')
    parser.add_argument('--entropy-threshold', type=float, default=0.12,
                      help='Entropy threshold for noise filtering')
    parser.add_argument('--edge-weight', type=float, default=0.25,
                      help='Weight for edge prior')
    parser.add_argument('--attention-weight', type=float, default=0.0,
                      help='Weight for attention scores (for custom method)')
    parser.add_argument('--similarity-weight', type=float, default=0.2,
                      help='Weight for text similarity scores (for custom method)')
    parser.add_argument('--radial-sigma', type=float, default=0.65,
                      help='Sigma for radial weighting')
    parser.add_argument('--merge-similar', action='store_true',
                      help='Enable token merging for similar tokens')
    parser.add_argument('--merge-threshold', type=float, default=0.9,
                      help='Similarity threshold for token merging')
    parser.add_argument('--prompt-style', type=str, default='extractive',
                      choices=['extractive', 'conversational'],
                      help='Prompt style: extractive (for VQA metrics) or conversational (LaViDa default)')
    
    args = parser.parse_args()
    
    if args.method == 'all':
        # Run all methods for comparison
        methods_configs = [
            ('base', {}),
            ('entropy', {'entropy_threshold': args.entropy_threshold}),
            ('edge', {'edge_weight': args.edge_weight}),
            ('full', {
                'entropy_threshold': args.entropy_threshold,
                'edge_weight': args.edge_weight,
                'radial_sigma': args.radial_sigma,
                'merge_similar': args.merge_similar,
                'merge_threshold': args.merge_threshold
            })
        ]
        
        all_results = {}
        for method, params in methods_configs:
            results = run_validation_with_method(method, params, prompt_style=args.prompt_style)
            all_results[method] = results
        
        # Compare results across methods
        print("\n" + "="*60)
        print("📊 COMPARISON ACROSS ALL METHODS")
        print("="*60)
        
        # TODO: Add comparison analysis here
        
        return all_results
    else:
        # Run single method
        params = {}
        if args.method == 'entropy':
            params = {'entropy_threshold': args.entropy_threshold}
        elif args.method == 'edge':
            params = {'edge_weight': args.edge_weight}
        elif args.method == 'edge_only':
            params = {'edge_weight': args.edge_weight}
        elif args.method == 'custom':
            params = {
                'attention_weight': args.attention_weight,
                'similarity_weight': args.similarity_weight,
                'edge_weight': args.edge_weight
            }
        elif args.method == 'full':
            params = {
                'entropy_threshold': args.entropy_threshold,
                'edge_weight': args.edge_weight,
                'radial_sigma': args.radial_sigma,
                'merge_similar': args.merge_similar,
                'merge_threshold': args.merge_threshold
            }
        
        return run_validation_with_method(args.method, params, prompt_style=args.prompt_style)

if __name__ == "__main__":
    main()