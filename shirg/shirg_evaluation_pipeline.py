#!/usr/bin/env python3
"""
SHIRG Evaluation Pipeline
Extends the existing real_ocr_vqa_validation to support proper evaluation metrics
"""

import os
import sys
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

# Add paths for imports
sys.path.append('./shirg/')
sys.path.append('./llava/')
sys.path.append('./')

# Import evaluation metrics from lmms-eval
sys.path.append('./eval/')
from lmms_eval.api.metrics import anls, exact_match
from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor

class SHIRGEvaluationPipeline:
    """
    Evaluation pipeline for SHIRG that:
    1. Uses your existing inference code
    2. Applies proper evaluation metrics from lmms-eval
    3. Tests multiple configurations automatically
    """
    
    def __init__(self):
        self.processor = EvalAIAnswerProcessor()
        self.results_dir = Path("/content/shirg_evaluation_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Define parameter configurations to test
        self.parameter_configs = {
            "baseline": {
                "use_shirg": False,
                "description": "Baseline LaViDa (384x384, 729 tokens)"
            },
            "shirg_40_attn_sim": {
                "use_shirg": True,
                "keep_ratio": 0.40,
                "score_type": "attention_similarity",
                "score_weights": [0.7, 0.3],  # attention, similarity
                "description": "SHIRG 40% keep, 0.7*attn + 0.3*sim"
            },
            "shirg_45_attn_sim": {
                "use_shirg": True,
                "keep_ratio": 0.45,
                "score_type": "attention_similarity", 
                "score_weights": [0.7, 0.3],
                "description": "SHIRG 45% keep, 0.7*attn + 0.3*sim"
            },
            "shirg_50_attn_sim": {
                "use_shirg": True,
                "keep_ratio": 0.50,
                "score_type": "attention_similarity",
                "score_weights": [0.7, 0.3],
                "description": "SHIRG 50% keep, 0.7*attn + 0.3*sim"
            },
            "shirg_45_attn_only": {
                "use_shirg": True,
                "keep_ratio": 0.45,
                "score_type": "attention",
                "description": "SHIRG 45% keep, attention only"
            },
            "shirg_45_balanced": {
                "use_shirg": True,
                "keep_ratio": 0.45,
                "score_type": "attention_similarity",
                "score_weights": [0.5, 0.5],
                "description": "SHIRG 45% keep, 0.5*attn + 0.5*sim"
            },
            "shirg_45_distance": {
                "use_shirg": True,
                "keep_ratio": 0.45,
                "score_type": "attention_similarity_distance",
                "score_weights": [0.5, 0.3, 0.2],  # attention, similarity, distance
                "description": "SHIRG 45% keep with distance-aware scoring"
            }
        }
    
    def evaluate_single_sample(self, prediction: str, references: List[str]) -> Dict[str, float]:
        """Evaluate a single prediction against multiple reference answers"""
        
        # Process prediction and references
        pred_processed = self.processor(prediction)
        refs_processed = [self.processor(ref) for ref in references]
        
        # Calculate ANLS (Average Normalized Levenshtein Similarity)
        anls_result = anls(refs_processed, [pred_processed])
        
        # Calculate exact match
        em_score = float(any(pred_processed.lower() == ref.lower() for ref in refs_processed))
        
        # Calculate token F1 (for multi-word answers)
        f1_score = self._calculate_token_f1(pred_processed, refs_processed)
        
        return {
            'anls': anls_result['anls'],
            'exact_match': em_score,
            'token_f1': f1_score
        }
    
    def _calculate_token_f1(self, prediction: str, references: List[str]) -> float:
        """Calculate token-level F1 score"""
        pred_tokens = set(prediction.lower().split())
        
        if not pred_tokens:
            return 0.0
        
        f1_scores = []
        for ref in references:
            ref_tokens = set(ref.lower().split())
            if not ref_tokens:
                continue
                
            # Calculate precision and recall
            common = pred_tokens.intersection(ref_tokens)
            precision = len(common) / len(pred_tokens) if pred_tokens else 0
            recall = len(common) / len(ref_tokens) if ref_tokens else 0
            
            # Calculate F1
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            
            f1_scores.append(f1)
        
        return max(f1_scores) if f1_scores else 0.0
    
    def run_configuration_evaluation(self, 
                                   config_name: str,
                                   config: Dict,
                                   dataset_samples: List[Dict],
                                   model_runner) -> Dict:
        """Run evaluation for a single SHIRG configuration"""
        
        print(f"\n🔧 Evaluating configuration: {config_name}")
        print(f"   {config['description']}")
        
        # Set SHIRG configuration
        if model_runner.shirg_model:
            vision_tower = model_runner.shirg_model.get_vision_tower()
            if hasattr(vision_tower, 'set_shirg_config'):
                vision_tower.set_shirg_config(config)
        
        results = []
        start_time = time.time()
        
        for idx, sample in enumerate(dataset_samples):
            # Run inference
            if config.get('use_shirg', False):
                output = model_runner._run_shirg_inference(
                    sample['image_path'],
                    sample['question'],
                    sample.get('dataset_name', 'unknown')
                )
            else:
                output = model_runner._run_baseline_inference(
                    sample['image_path'],
                    sample['question'],
                    sample.get('dataset_name', 'unknown')
                )
            
            # Evaluate
            eval_scores = self.evaluate_single_sample(
                output['text'],
                sample.get('answers', [sample.get('answer', '')])
            )
            
            # Store result
            result = {
                'config': config_name,
                'sample_id': sample.get('question_id', idx),
                'dataset': sample.get('dataset_name', 'unknown'),
                'prediction': output['text'],
                'ground_truth': sample.get('answers', [sample.get('answer', '')]),
                **eval_scores
            }
            results.append(result)
            
            # Progress update
            if (idx + 1) % 10 == 0:
                print(f"   Processed {idx + 1}/{len(dataset_samples)} samples...")
        
        elapsed_time = time.time() - start_time
        
        # Aggregate metrics
        metrics_df = pd.DataFrame(results)
        aggregated = {
            'config_name': config_name,
            'description': config['description'],
            'num_samples': len(results),
            'elapsed_time': elapsed_time,
            'anls_mean': metrics_df['anls'].mean(),
            'anls_std': metrics_df['anls'].std(),
            'exact_match_mean': metrics_df['exact_match'].mean(),
            'token_f1_mean': metrics_df['token_f1'].mean(),
            'token_f1_std': metrics_df['token_f1'].std()
        }
        
        # Add per-dataset metrics
        for dataset in metrics_df['dataset'].unique():
            dataset_df = metrics_df[metrics_df['dataset'] == dataset]
            aggregated[f'{dataset}_anls'] = dataset_df['anls'].mean()
            aggregated[f'{dataset}_exact_match'] = dataset_df['exact_match'].mean()
            aggregated[f'{dataset}_token_f1'] = dataset_df['token_f1'].mean()
        
        # Save detailed results
        self._save_detailed_results(config_name, results, aggregated)
        
        return aggregated
    
    def run_full_evaluation(self, 
                          dataset_samples: List[Dict],
                          model_runner,
                          configs: Optional[Dict] = None) -> pd.DataFrame:
        """Run evaluation across all configurations"""
        
        configs = configs or self.parameter_configs
        
        print(f"\n🚀 Starting SHIRG Parameter Evaluation")
        print(f"   Configurations: {list(configs.keys())}")
        print(f"   Total samples: {len(dataset_samples)}")
        print("=" * 60)
        
        all_results = []
        
        for config_name, config in configs.items():
            result = self.run_configuration_evaluation(
                config_name, config, dataset_samples, model_runner
            )
            all_results.append(result)
            
            # Print intermediate results
            self._print_config_summary(result)
        
        # Create summary dataframe
        summary_df = pd.DataFrame(all_results)
        
        # Save and display final results
        self._save_summary_results(summary_df)
        self._print_final_summary(summary_df)
        
        return summary_df
    
    def _save_detailed_results(self, config_name: str, results: List[Dict], aggregated: Dict):
        """Save detailed results for a configuration"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save detailed predictions
        detail_path = self.results_dir / f"{config_name}_detailed_{timestamp}.json"
        with open(detail_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save aggregated metrics
        agg_path = self.results_dir / f"{config_name}_aggregated_{timestamp}.json"
        with open(agg_path, 'w') as f:
            json.dump(aggregated, f, indent=2)
    
    def _save_summary_results(self, summary_df: pd.DataFrame):
        """Save summary results"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save as CSV
        csv_path = self.results_dir / f"shirg_evaluation_summary_{timestamp}.csv"
        summary_df.to_csv(csv_path, index=False)
        
        # Save as JSON
        json_path = self.results_dir / f"shirg_evaluation_summary_{timestamp}.json"
        summary_df.to_json(json_path, orient='records', indent=2)
        
        print(f"\n💾 Results saved to:")
        print(f"   CSV: {csv_path}")
        print(f"   JSON: {json_path}")
    
    def _print_config_summary(self, result: Dict):
        """Print summary for a single configuration"""
        print(f"\n📊 {result['config_name']} Results:")
        print(f"   ANLS: {result['anls_mean']:.3f} (±{result['anls_std']:.3f})")
        print(f"   Exact Match: {result['exact_match_mean']:.3f}")
        print(f"   Token F1: {result['token_f1_mean']:.3f} (±{result['token_f1_std']:.3f})")
        print(f"   Time: {result['elapsed_time']:.1f}s")
    
    def _print_final_summary(self, summary_df: pd.DataFrame):
        """Print final comparison summary"""
        print("\n" + "=" * 80)
        print("🏆 SHIRG EVALUATION SUMMARY")
        print("=" * 80)
        
        # Select key columns for display
        display_cols = ['config_name', 'anls_mean', 'exact_match_mean', 'token_f1_mean', 'elapsed_time']
        display_df = summary_df[display_cols].copy()
        
        # Format numeric columns
        display_df['anls_mean'] = display_df['anls_mean'].map('{:.3f}'.format)
        display_df['exact_match_mean'] = display_df['exact_match_mean'].map('{:.3f}'.format)
        display_df['token_f1_mean'] = display_df['token_f1_mean'].map('{:.3f}'.format)
        display_df['elapsed_time'] = display_df['elapsed_time'].map('{:.1f}s'.format)
        
        # Rename columns for display
        display_df.columns = ['Configuration', 'ANLS', 'Exact Match', 'Token F1', 'Time']
        
        print(display_df.to_string(index=False))
        
        # Find best configuration
        best_anls = summary_df.loc[summary_df['anls_mean'].idxmax()]
        print(f"\n🥇 Best ANLS: {best_anls['config_name']} ({best_anls['anls_mean']:.3f})")
        
        # Calculate improvement over baseline
        baseline = summary_df[summary_df['config_name'] == 'baseline'].iloc[0]
        for _, row in summary_df.iterrows():
            if row['config_name'] != 'baseline':
                improvement = (row['anls_mean'] - baseline['anls_mean']) / baseline['anls_mean'] * 100
                print(f"   {row['config_name']}: {improvement:+.1f}% over baseline")
        
        print("\n" + "=" * 80)


def integrate_with_existing_evaluation():
    """
    Integration function to use with your existing real_ocr_vqa_validation.py
    """
    from real_ocr_vqa_model_runner import LaViDaModelRunner
    from real_ocr_vqa_dataset_loader import OCRVQADatasetLoader
    
    # Initialize components
    pipeline = SHIRGEvaluationPipeline()
    dataset_loader = OCRVQADatasetLoader()
    model_runner = LaViDaModelRunner()
    
    # Load dataset samples
    samples = dataset_loader.get_real_ocr_vqa_samples()
    
    # Run evaluation pipeline
    results_df = pipeline.run_full_evaluation(samples, model_runner)
    
    return results_df


if __name__ == "__main__":
    # Can be run standalone or integrated
    print("SHIRG Evaluation Pipeline")
    print("Use integrate_with_existing_evaluation() to run with your existing code")
    print("Or import SHIRGEvaluationPipeline to use in your own scripts")