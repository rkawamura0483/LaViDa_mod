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
    
    def __init__(self):
        self.model_runner = LaViDaModelRunner()
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
        
        # Phase 1: Run all baseline inferences
        print("\n🔄 PHASE 1: BASELINE LaViDa INFERENCE")
        print("=" * 40)
        baseline_results = self.model_runner._run_all_baseline_inferences(ocr_vqa_samples)
        
        # Unload baseline model to free memory
        self.model_runner._unload_baseline_model()
        
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

def main():
    """Run baseline vs SHIRG comparison validation"""
    validator = RealOCRVQAValidator()
    results = validator.run_real_ocr_vqa_validation()
    
    print(f"\n🎉 Baseline vs SHIRG comparison complete!")
    print(f"   📊 Results: /content/shirg_validation_results/")
    print(f"   🖼️ Visualizations: /content/shirg_token_visualizations/")
    return results

if __name__ == "__main__":
    main()