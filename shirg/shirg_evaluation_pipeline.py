#!/usr/bin/env python3
"""
SHIRG-Fovea Evaluation Pipeline

This evaluation pipeline is updated to work with the new SHIRG-Fovea architecture
that uses 2-view processing (1 global + 1 foveal) and produces exactly 980 tokens.

Key Updates:
1. Supports new selection methods: 'base', 'entropy', 'edge', 'full'
2. Properly handles model initialization with selection_method and selection_params
3. Ensures exactly 980 tokens for LaViDa compatibility
4. Tests multiple configurations with proper GPU memory management

Usage Examples:
--------------
# Run all predefined configurations
python shirg_evaluation_pipeline.py --config all

# Run specific configuration
python shirg_evaluation_pipeline.py --config shirg_base

# Run custom method with parameters
python shirg_evaluation_pipeline.py --method full --entropy-threshold 0.10 --edge-weight 0.3 --merge-similar

# Quick test with limited samples
python shirg_evaluation_pipeline.py --config shirg_edge --samples 10

# In Python script
from shirg_evaluation_pipeline import integrate_with_existing_evaluation
results = integrate_with_existing_evaluation()
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
from lmms_eval.api.metrics import anls
from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor
from lavida_evaluation_metrics import LaViDaEvaluationMetrics

class SHIRGEvaluationPipeline:
    """
    Evaluation pipeline for SHIRG-Fovea that:
    1. Uses the new selection methods (base, entropy, edge, full)
    2. Applies proper evaluation metrics from lmms-eval
    3. Tests multiple configurations automatically
    """
    
    def __init__(self):
        self.processor = EvalAIAnswerProcessor()
        self.evaluator = LaViDaEvaluationMetrics()  # Use official LaViDa metrics
        self.results_dir = Path("/content/shirg_evaluation_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Define parameter configurations to test with new SHIRG-Fovea methods
        self.parameter_configs = {
            "baseline": {
                "use_shirg": False,
                "selection_method": None,
                "selection_params": None,
                "description": "Baseline LaViDa (5-view anyres, 980 tokens with pooling)"
            },
            "shirg_base": {
                "use_shirg": True,
                "selection_method": "base",
                "selection_params": {},
                "description": "SHIRG-Base: 0.7*attention + 0.3*similarity (980 tokens)"
            },
            "shirg_entropy": {
                "use_shirg": True,
                "selection_method": "entropy",
                "selection_params": {
                    "entropy_threshold": 0.12
                },
                "description": "SHIRG-Entropy: Base + noise filtering (τ=0.12)"
            },
            "shirg_entropy_aggressive": {
                "use_shirg": True,
                "selection_method": "entropy",
                "selection_params": {
                    "entropy_threshold": 0.10
                },
                "description": "SHIRG-Entropy: More aggressive filtering (τ=0.10)"
            },
            "shirg_edge": {
                "use_shirg": True,
                "selection_method": "edge",
                "selection_params": {
                    "edge_weight": 0.25
                },
                "description": "SHIRG-Edge: Base + edge/text priors (weight=0.25)"
            },
            "shirg_edge_strong": {
                "use_shirg": True,
                "selection_method": "edge",
                "selection_params": {
                    "edge_weight": 0.30
                },
                "description": "SHIRG-Edge: Stronger edge preference (weight=0.30)"
            },
            "shirg_full": {
                "use_shirg": True,
                "selection_method": "full",
                "selection_params": {
                    "entropy_threshold": 0.12,
                    "edge_weight": 0.25,
                    "radial_sigma": 0.65,
                    "merge_similar": False
                },
                "description": "SHIRG-Full: All enhancements without merging"
            },
            "shirg_full_merge": {
                "use_shirg": True,
                "selection_method": "full",
                "selection_params": {
                    "entropy_threshold": 0.12,
                    "edge_weight": 0.25,
                    "radial_sigma": 0.65,
                    "merge_similar": True,
                    "merge_threshold": 0.9
                },
                "description": "SHIRG-Full: All enhancements with token merging (θ=0.9)"
            }
        }
    
    def evaluate_single_sample(self, prediction: str, references: List[str], dataset_type: str = None) -> Dict[str, float]:
        """Evaluate a single prediction using dataset-specific metrics"""
        
        # Use LaViDa's official evaluation metrics based on dataset type
        if dataset_type:
            return self.evaluator.evaluate_sample(prediction, references, dataset_type)
        
        # Fallback to generic metrics if no dataset type specified
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
        
        # Note: With the new architecture, selection method and params are set
        # when initializing the model runner, not dynamically per configuration
        # For proper evaluation, we would need to reinitialize the model runner
        # with different selection parameters for each configuration
        
        results = []
        start_time = time.time()
        
        for idx, sample in enumerate(dataset_samples):
            try:
                # Extract image path and question from sample format
                if isinstance(sample.get('image_path'), str):
                    image_path = sample['image_path']
                else:
                    # Handle potential URL or PIL image
                    image_path = sample.get('image', sample.get('image_path'))
                
                question = sample.get('question', '')
                
                # Prepare input for model runner's inference methods
                # Model runner expects (image, input_ids, question, sample_id)
                # For LaViDa, let the model runner handle the proper tokenization
                # as it needs to use tokenizer_image_token for proper image token handling
                input_ids = None  # Let model runner handle tokenization
                
                # Run inference based on configuration
                if config.get('use_shirg', False):
                    output = model_runner._run_shirg_inference(
                        image_path,
                        input_ids,
                        question,
                        sample_id=f"{config_name}_sample_{idx}"
                    )
                else:
                    output = model_runner._run_baseline_inference(
                        image_path,
                        input_ids,
                        question,
                        sample_id=f"{config_name}_sample_{idx}"
                    )
                
                # Extract prediction text (key is 'response' not 'text')
                prediction = output.get('response', '') if isinstance(output, dict) else str(output)
                
                # Get ground truth answers
                # SHIRG-FIX: 2025-07-30 - Handle ground_truth key from dataset loader
                # ISSUE: Dataset loader stores ground truth as 'ground_truth' not 'answers'
                # SOLUTION: Check for 'ground_truth' key first, then fallback to 'answers'/'answer'
                # LAVIDA IMPACT: None - just accessing data correctly
                # SHIRG IMPACT: Enables proper evaluation metrics calculation
                ground_truth = sample.get('ground_truth', sample.get('answers', [sample.get('answer', '')]))
                if isinstance(ground_truth, str):
                    ground_truth = [ground_truth]
                elif ground_truth is None:
                    ground_truth = ['']
                
                # Print question, response, and ground truth for debugging
                print(f"\n   📝 Sample {idx+1}/{len(dataset_samples)}:")
                print(f"      Question: {question[:100]}...")
                print(f"      Response: {prediction[:100]}...")
                print(f"      Ground Truth: {ground_truth[0] if ground_truth else 'N/A'}")
                
                # Get dataset type for proper metric selection
                # SHIRG-FIX: 2025-07-30 - Extract dataset type from full dataset name
                # ISSUE: Dataset names include org prefix (e.g., "lmms-lab/DocVQA")
                # SOLUTION: Extract the actual dataset name after the slash
                # LAVIDA IMPACT: None - just data processing
                # SHIRG IMPACT: Enables proper metric selection for evaluation
                dataset_name = sample.get('dataset_type', sample.get('dataset_name', 'unknown'))
                # Extract dataset type from full name (e.g., "lmms-lab/DocVQA" -> "DocVQA")
                if '/' in dataset_name:
                    dataset_type = dataset_name.split('/')[-1]
                else:
                    dataset_type = dataset_name
                
                # Additional normalization for dataset type
                # Handle variations like "InfographicVQA" -> "InfoVQA"
                if dataset_type == "InfographicVQA":
                    dataset_type = "InfoVQA"
                
                # SHIRG-FIX: 2025-07-30 - Ensure ground_truth is always a list for evaluation
                # ISSUE: evaluate_single_sample expects references as list, but ground_truth might be string
                # SOLUTION: Convert ground_truth to list if it's a string
                # RESEARCH IMPACT: Ensures consistent evaluation metric calculation
                if isinstance(ground_truth, str):
                    ground_truth_list = [ground_truth] if ground_truth else []
                elif isinstance(ground_truth, list):
                    ground_truth_list = ground_truth
                else:
                    ground_truth_list = [str(ground_truth)] if ground_truth is not None else []
                
                # Evaluate using dataset-specific metrics
                eval_scores = self.evaluate_single_sample(prediction, ground_truth_list, dataset_type)
                
                # Store result (keep original ground_truth format for JSON output)
                result = {
                    'config': config_name,
                    'sample_id': sample.get('question_id', idx),
                    'dataset': sample.get('dataset_name', 'unknown'),
                    'question': question,
                    'prediction': prediction,
                    'ground_truth': ground_truth,  # Keep original format
                    **eval_scores
                }
                results.append(result)
                
            except Exception as e:
                # SHIRG-FIX: 2025-07-30 - Improved error handling and debugging
                # ISSUE: Error just shows "0" which doesn't help debugging
                # SOLUTION: Add detailed error information and traceback
                # RESEARCH IMPACT: Helps debug evaluation pipeline issues
                import traceback
                error_msg = f"{type(e).__name__}: {str(e)}"
                print(f"   ⚠️ Error processing sample {idx}: {error_msg}")
                traceback.print_exc()
                
                # Try to extract ground truth safely
                try:
                    gt = sample.get('ground_truth', None)
                    if gt is None:
                        gt = sample.get('answers', sample.get('answer', ''))
                except:
                    gt = ''
                
                # Add failed result with detailed error
                result = {
                    'config': config_name,
                    'sample_id': sample.get('question_id', idx),
                    'dataset': sample.get('dataset_name', 'unknown'),
                    'prediction': '',
                    'ground_truth': gt,
                    'anls': 0.0,
                    'exact_match': 0.0,
                    'token_f1': 0.0,
                    'error': error_msg
                }
                results.append(result)
            
            # Progress update
            if (idx + 1) % 10 == 0:
                print(f"   Processed {idx + 1}/{len(dataset_samples)} samples...")
        
        elapsed_time = time.time() - start_time
        
        # Aggregate metrics
        metrics_df = pd.DataFrame(results)
        
        # Get all metric columns (excluding metadata columns)
        metric_columns = [col for col in metrics_df.columns if col not in 
                         ['config', 'sample_id', 'dataset', 'question', 'prediction', 'ground_truth', 'error']]
        
        aggregated = {
            'config_name': config_name,
            'description': config['description'],
            'num_samples': len(results),
            'elapsed_time': elapsed_time
        }
        
        # Aggregate each metric found in the results
        for metric in metric_columns:
            if metric in metrics_df.columns:
                aggregated[f'{metric}_mean'] = metrics_df[metric].mean()
                aggregated[f'{metric}_std'] = metrics_df[metric].std()
        
        # Add per-dataset metrics
        for dataset in metrics_df['dataset'].unique():
            dataset_df = metrics_df[metrics_df['dataset'] == dataset]
            
            # Extract clean dataset name (e.g., "lmms-lab/DocVQA" -> "DocVQA")
            if '/' in dataset:
                clean_dataset_name = dataset.split('/')[-1]
            else:
                clean_dataset_name = dataset
            
            # Handle variations
            if clean_dataset_name == "InfographicVQA":
                clean_dataset_name = "InfoVQA"
            
            for metric in metric_columns:
                if metric in dataset_df.columns:
                    aggregated[f'{clean_dataset_name}_{metric}'] = dataset_df[metric].mean()
        
        # Save detailed results
        self._save_detailed_results(config_name, results, aggregated)
        
        return aggregated
    
    def run_full_evaluation(self, 
                          dataset_samples: List[Dict],
                          configs: Optional[Dict] = None) -> pd.DataFrame:
        """
        Run evaluation across all configurations
        
        This method now delegates to run_multi_config_evaluation which handles
        proper model initialization for each configuration.
        """
        return run_multi_config_evaluation(dataset_samples, configs)
    
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
        
        # SHIRG-FIX: 2025-07-30 - Handle missing metric fields gracefully
        # ISSUE: Some metrics might not be calculated for all datasets
        # SOLUTION: Check if fields exist before accessing them
        # RESEARCH IMPACT: Allows evaluation to complete even with partial metrics
        if 'anls_mean' in result:
            print(f"   ANLS: {result['anls_mean']:.3f} (±{result.get('anls_std', 0):.3f})")
        
        if 'exact_match_mean' in result:
            print(f"   Exact Match: {result['exact_match_mean']:.3f}")
        
        if 'token_f1_mean' in result:
            print(f"   Token F1: {result['token_f1_mean']:.3f} (±{result.get('token_f1_std', 0):.3f})")
        
        print(f"   Time: {result.get('elapsed_time', 0):.1f}s")
    
    def _print_per_dataset_summary(self, summary_df: pd.DataFrame):
        """Print per-dataset average metrics"""
        print("\n" + "=" * 80)
        print("📊 PER-DATASET AVERAGE METRICS")
        print("=" * 80)
        
        # Find all dataset-specific metric columns
        dataset_metrics = defaultdict(lambda: defaultdict(list))
        
        # Debug: print all columns to see what we have
        # print(f"DEBUG: Available columns: {list(summary_df.columns)}")
        
        # Extract dataset names dynamically from column names
        for col in summary_df.columns:
            # Skip non-metric columns
            if col in ['config_name', 'description', 'num_samples', 'elapsed_time']:
                continue
            
            # Skip aggregate metric columns (those ending with _mean or _std)
            if col.endswith('_mean') or col.endswith('_std'):
                continue
            
            # Parse dataset-specific columns
            # These have format: Dataset_metric or Dataset-Name_metric
            # Valid metrics we're looking for
            metric_names = ['relaxed_accuracy', 'exact_match', 'accuracy', 'anls', 'vqa_accuracy', 'token_f1']
            
            # Try to match against each metric type
            for metric in metric_names:
                if col.endswith('_' + metric):
                    # Extract dataset name by removing the metric suffix
                    dataset_name = col[:-len('_' + metric)]
                    dataset_metrics[dataset_name][metric] = col
                    break
        
        # Check if we found any dataset-specific metrics
        if not dataset_metrics:
            print("\nNo per-dataset metrics found. This might be because:")
            print("- Only aggregate metrics were calculated")
            print("- Dataset names in the data don't match expected patterns")
            print("- Consider checking the dataset names in your evaluation data")
        else:
            # Count datasets with valid data
            datasets_with_data = []
            
            # Print metrics for each dataset
            for dataset_name in sorted(dataset_metrics.keys()):
                # Create a table for this dataset
                dataset_data = []
                for _, row in summary_df.iterrows():
                    row_data = {'Configuration': row['config_name']}
                    
                    # Add each metric for this dataset
                    for metric_name, col_name in dataset_metrics[dataset_name].items():
                        if col_name in row and pd.notna(row[col_name]):
                            row_data[metric_name] = f"{row[col_name]:.3f}"
                        else:
                            row_data[metric_name] = "N/A"
                    
                    dataset_data.append(row_data)
                
                if dataset_data:
                    dataset_df = pd.DataFrame(dataset_data)
                    
                    # Check if this dataset has any non-zero/non-N/A values
                    has_valid_data = False
                    for _, row in dataset_df.iterrows():
                        for col in dataset_df.columns:
                            if col != 'Configuration':
                                val = row[col]
                                if val != "N/A" and val != "0.000":
                                    has_valid_data = True
                                    break
                        if has_valid_data:
                            break
                    
                    # Only print if there's valid data
                    if has_valid_data:
                        print(f"\n{dataset_name}:")
                        print("-" * 40)
                        datasets_with_data.append(dataset_name)
                        print(dataset_df.to_string(index=False))
                        
                        # Find best configuration for this dataset (if anls exists)
                        anls_col = dataset_metrics[dataset_name].get('anls')
                        if anls_col and anls_col in summary_df.columns:
                            valid_rows = summary_df[summary_df[anls_col].notna()]
                            if not valid_rows.empty:
                                # Only show best if ANLS > 0
                                best_idx = valid_rows[anls_col].idxmax()
                                best_config = summary_df.loc[best_idx]
                                if best_config[anls_col] > 0:
                                    print(f"\n  Best {dataset_name} ANLS: {best_config['config_name']} ({best_config[anls_col]:.3f})")
                                    
                                    # Calculate improvement over baseline for this dataset
                                    baseline_df = summary_df[summary_df['config_name'] == 'baseline']
                                    if not baseline_df.empty and anls_col in baseline_df.columns:
                                        baseline_val = baseline_df.iloc[0][anls_col]
                                        if pd.notna(baseline_val) and baseline_val > 0:
                                            for _, row in valid_rows.iterrows():
                                                if row['config_name'] != 'baseline' and row[anls_col] > 0:
                                                    improvement = (row[anls_col] - baseline_val) / baseline_val * 100
                                                    print(f"    {row['config_name']}: {improvement:+.1f}% over baseline")
            
            # Print summary of datasets
            if datasets_with_data:
                print(f"\n📊 Evaluated {len(datasets_with_data)} datasets with valid data: {', '.join(datasets_with_data)}")
            
            # List datasets that had no valid data (all zeros or N/A)
            all_datasets = list(dataset_metrics.keys())
            datasets_without_data = [d for d in all_datasets if d not in datasets_with_data]
            if datasets_without_data:
                print(f"⚠️  {len(datasets_without_data)} datasets had no valid data (all zeros or N/A): {', '.join(datasets_without_data)}")
        
        print("\n" + "=" * 80)

    def _print_final_summary(self, summary_df: pd.DataFrame):
        """Print final comparison summary"""
        print("\n" + "=" * 80)
        print("🏆 SHIRG EVALUATION SUMMARY")
        print("=" * 80)
        
        # SHIRG-FIX: 2025-07-30 - Handle missing columns in summary display
        # ISSUE: Some metric columns might not exist for all evaluations
        # SOLUTION: Only include columns that exist in the dataframe
        # RESEARCH IMPACT: Ensures summary display works with partial metrics
        
        # Select key columns for display (only if they exist)
        available_cols = ['config_name']
        format_funcs = {}
        
        if 'anls_mean' in summary_df.columns:
            available_cols.append('anls_mean')
            format_funcs['anls_mean'] = '{:.3f}'.format
            
        if 'exact_match_mean' in summary_df.columns:
            available_cols.append('exact_match_mean')
            format_funcs['exact_match_mean'] = '{:.3f}'.format
            
        if 'token_f1_mean' in summary_df.columns:
            available_cols.append('token_f1_mean')
            format_funcs['token_f1_mean'] = '{:.3f}'.format
            
        if 'elapsed_time' in summary_df.columns:
            available_cols.append('elapsed_time')
            format_funcs['elapsed_time'] = lambda x: f'{x:.1f}s'
        
        display_df = summary_df[available_cols].copy()
        
        # Format numeric columns
        for col, fmt_func in format_funcs.items():
            display_df[col] = display_df[col].map(fmt_func)
        
        # Rename columns for display
        col_names = {'config_name': 'Configuration'}
        if 'anls_mean' in display_df.columns:
            col_names['anls_mean'] = 'ANLS'
        if 'exact_match_mean' in display_df.columns:
            col_names['exact_match_mean'] = 'Exact Match'
        if 'token_f1_mean' in display_df.columns:
            col_names['token_f1_mean'] = 'Token F1'
        if 'elapsed_time' in display_df.columns:
            col_names['elapsed_time'] = 'Time'
        
        display_df = display_df.rename(columns=col_names)
        
        print(display_df.to_string(index=False))
        
        # Find best configuration (if ANLS metric exists)
        if 'anls_mean' in summary_df.columns:
            best_anls = summary_df.loc[summary_df['anls_mean'].idxmax()]
            print(f"\n🥇 Best ANLS: {best_anls['config_name']} ({best_anls['anls_mean']:.3f})")
            
            # Calculate improvement over baseline
            baseline_df = summary_df[summary_df['config_name'] == 'baseline']
            if not baseline_df.empty:
                baseline = baseline_df.iloc[0]
                for _, row in summary_df.iterrows():
                    if row['config_name'] != 'baseline':
                        improvement = (row['anls_mean'] - baseline['anls_mean']) / baseline['anls_mean'] * 100
                        print(f"   {row['config_name']}: {improvement:+.1f}% over baseline")
        
        # Print per-dataset summary
        self._print_per_dataset_summary(summary_df)
        
        print("\n" + "=" * 80)


def create_test_configs(methods: List[str] = None) -> Dict:
    """
    Create a subset of test configurations for quick evaluation
    
    Args:
        methods: List of methods to test. If None, tests ['base', 'entropy', 'edge', 'full']
    
    Returns:
        Dictionary of test configurations
    """
    methods = methods or ['base', 'entropy', 'edge', 'full']
    
    test_configs = {
        "baseline": {
            "use_shirg": False,
            "selection_method": None,
            "selection_params": None,
            "description": "Baseline LaViDa (980 tokens)"
        }
    }
    
    if 'base' in methods:
        test_configs["shirg_base"] = {
            "use_shirg": True,
            "selection_method": "base",
            "selection_params": {},
            "description": "SHIRG-Base: Original method"
        }
    
    if 'entropy' in methods:
        test_configs["shirg_entropy"] = {
            "use_shirg": True,
            "selection_method": "entropy",
            "selection_params": {"entropy_threshold": 0.12},
            "description": "SHIRG-Entropy: With noise filtering"
        }
    
    if 'edge' in methods:
        test_configs["shirg_edge"] = {
            "use_shirg": True,
            "selection_method": "edge",
            "selection_params": {"edge_weight": 0.25},
            "description": "SHIRG-Edge: With edge/text priors"
        }
    
    if 'full' in methods:
        test_configs["shirg_full"] = {
            "use_shirg": True,
            "selection_method": "full",
            "selection_params": {
                "entropy_threshold": 0.12,
                "edge_weight": 0.25,
                "radial_sigma": 0.65,
                "merge_similar": False
            },
            "description": "SHIRG-Full: All enhancements"
        }
    
    return test_configs


def run_multi_config_evaluation(dataset_samples: List[Dict], 
                               configs: Optional[Dict] = None) -> pd.DataFrame:
    """
    Run evaluation across multiple SHIRG configurations with proper model initialization
    
    This function properly handles the new architecture where selection method
    and params must be set when initializing the model runner.
    """
    from real_ocr_vqa_model_runner import LaViDaModelRunner
    
    pipeline = SHIRGEvaluationPipeline()
    configs = configs or pipeline.parameter_configs
    
    print(f"\n🚀 Starting SHIRG Multi-Configuration Evaluation")
    print(f"   Configurations: {list(configs.keys())}")
    print(f"   Total samples: {len(dataset_samples)}")
    print("=" * 60)
    
    all_results = []
    
    # Run baseline first (no SHIRG)
    if 'baseline' in configs:
        print("\n📊 Running BASELINE configuration...")
        baseline_runner = LaViDaModelRunner()
        
        # Load baseline model
        if baseline_runner._load_baseline_model():
            result = pipeline.run_configuration_evaluation(
                'baseline', configs['baseline'], dataset_samples, baseline_runner
            )
            all_results.append(result)
            pipeline._print_config_summary(result)
            
            # Unload baseline model
            baseline_runner._unload_baseline_model()
        else:
            print("❌ Failed to load baseline model")
    
    # Run each SHIRG configuration
    for config_name, config in configs.items():
        if config_name == 'baseline' or not config.get('use_shirg', False):
            continue
            
        print(f"\n📊 Running {config_name} configuration...")
        
        # Initialize model runner with specific selection method and params
        shirg_runner = LaViDaModelRunner(
            selection_method=config.get('selection_method', 'base'),
            selection_params=config.get('selection_params', {})
        )
        
        # Load SHIRG model
        if shirg_runner._load_shirg_model():
            result = pipeline.run_configuration_evaluation(
                config_name, config, dataset_samples, shirg_runner
            )
            all_results.append(result)
            pipeline._print_config_summary(result)
            
            # Unload SHIRG model
            shirg_runner._unload_shirg_model()
        else:
            print(f"❌ Failed to load SHIRG model for {config_name}")
    
    # Create summary dataframe
    if all_results:
        summary_df = pd.DataFrame(all_results)
        
        # Save and display final results
        pipeline._save_summary_results(summary_df)
        pipeline._print_final_summary(summary_df)
        
        return summary_df
    else:
        print("❌ No results collected")
        return pd.DataFrame()


def integrate_with_existing_evaluation(samples_per_dataset=10):
    """
    Integration function to use with your existing real_ocr_vqa_validation.py
    
    Updated to work with the new SHIRG-Fovea architecture where selection
    method and params are set at model initialization time.
    
    Args:
        samples_per_dataset: Number of samples to load from each dataset (default: 10)
    """
    from real_ocr_vqa_dataset_loader import OCRVQADatasetLoader
    
    # Initialize dataset loader
    dataset_loader = OCRVQADatasetLoader()
    
    # Load dataset samples
    samples_dict = dataset_loader.get_real_ocr_vqa_samples(samples_per_dataset=samples_per_dataset)
    
    if not samples_dict:
        print("❌ No dataset samples loaded")
        return None
    
    # Convert dictionary to list for processing
    samples = list(samples_dict.values())
    
    # Run evaluation with multiple configurations
    results_df = run_multi_config_evaluation(samples)
    
    return results_df


def main():
    """Main function with command-line argument support"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SHIRG Evaluation Pipeline')
    parser.add_argument('--config', type=str, default='all',
                      help='Configuration to run (all, baseline, shirg_base, shirg_entropy, shirg_edge, shirg_full, etc.)')
    parser.add_argument('--baseline-only', action='store_true',
                      help='Run only the baseline configuration')
    parser.add_argument('--method', type=str, default=None,
                      choices=['base', 'entropy', 'edge', 'full'],
                      help='Custom selection method (overrides config)')
    parser.add_argument('--entropy-threshold', type=float, default=0.12,
                      help='Entropy threshold for noise filtering')
    parser.add_argument('--edge-weight', type=float, default=0.25,
                      help='Weight for edge prior')
    parser.add_argument('--radial-sigma', type=float, default=0.65,
                      help='Sigma for radial weighting')
    parser.add_argument('--merge-similar', action='store_true',
                      help='Enable token merging for similar tokens')
    parser.add_argument('--merge-threshold', type=float, default=0.9,
                      help='Similarity threshold for token merging')
    parser.add_argument('--samples', type=int, default=None,
                      help='Number of samples to evaluate (default: all)')
    parser.add_argument('--samples-per-dataset', type=int, default=10,
                      help='Number of samples to load from each dataset (default: 10)')
    
    args = parser.parse_args()
    
    # Load dataset
    from real_ocr_vqa_dataset_loader import OCRVQADatasetLoader
    dataset_loader = OCRVQADatasetLoader()
    samples_dict = dataset_loader.get_real_ocr_vqa_samples(samples_per_dataset=args.samples_per_dataset)
    
    if not samples_dict:
        print("❌ No dataset samples loaded")
        return
    
    # Convert dictionary to list for processing
    samples = list(samples_dict.values())
    
    # Limit samples if requested
    if args.samples:
        samples = samples[:args.samples]
        print(f"📊 Using first {args.samples} samples")
    
    # Handle baseline-only flag
    if args.baseline_only:
        # Run only baseline configuration
        pipeline = SHIRGEvaluationPipeline()
        baseline_config = {"baseline": pipeline.parameter_configs["baseline"]}
        results_df = run_multi_config_evaluation(samples, baseline_config)
    
    # Handle custom configuration
    elif args.method:
        # Create custom configuration based on command-line args
        custom_config = {
            "custom": {
                "use_shirg": True,
                "selection_method": args.method,
                "selection_params": {},
                "description": f"Custom SHIRG-{args.method.upper()}"
            }
        }
        
        # Add parameters based on method
        if args.method == 'entropy':
            custom_config["custom"]["selection_params"]["entropy_threshold"] = args.entropy_threshold
        elif args.method == 'edge':
            custom_config["custom"]["selection_params"]["edge_weight"] = args.edge_weight
        elif args.method == 'full':
            custom_config["custom"]["selection_params"] = {
                "entropy_threshold": args.entropy_threshold,
                "edge_weight": args.edge_weight,
                "radial_sigma": args.radial_sigma,
                "merge_similar": args.merge_similar,
                "merge_threshold": args.merge_threshold
            }
        
        results_df = run_multi_config_evaluation(samples, custom_config)
    
    elif args.config == 'all':
        # Run all predefined configurations
        results_df = integrate_with_existing_evaluation(samples_per_dataset=args.samples_per_dataset)
    
    else:
        # Run specific configuration
        pipeline = SHIRGEvaluationPipeline()
        if args.config in pipeline.parameter_configs:
            single_config = {args.config: pipeline.parameter_configs[args.config]}
            results_df = run_multi_config_evaluation(samples, single_config)
        else:
            print(f"❌ Unknown configuration: {args.config}")
            print(f"   Available: {list(pipeline.parameter_configs.keys())}")
            return
    
    print("\n✅ Evaluation complete!")
    if results_df is not None and not results_df.empty:
        print(f"   Results saved to: /content/shirg_evaluation_results/")


if __name__ == "__main__":
    main()