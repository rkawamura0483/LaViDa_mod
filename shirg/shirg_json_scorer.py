#!/usr/bin/env python3
"""
SHIRG JSON Scorer
Simple utility to score your existing JSON outputs from real_ocr_vqa_validation.py
"""

import json
import sys
from typing import Dict, List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

# Add paths for imports
sys.path.append('./eval/')
from lmms_eval.api.metrics import anls
from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor


class SHIRGJSONScorer:
    """Score existing JSON outputs with proper evaluation metrics"""
    
    def __init__(self):
        self.processor = EvalAIAnswerProcessor()
    
    def score_json_file(self, json_path: str) -> Dict:
        """Score a single JSON file with model outputs"""
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        results = []
        
        for sample in data:
            # Handle both baseline and SHIRG output formats
            if 'baseline_output' in sample:
                baseline_scores = self._score_single_output(
                    sample['baseline_output']['text'],
                    sample.get('answers', [sample.get('answer', '')])
                )
                baseline_scores['type'] = 'baseline'
                baseline_scores['sample_id'] = sample.get('question_id', sample.get('index', ''))
                results.append(baseline_scores)
            
            if 'shirg_output' in sample:
                shirg_scores = self._score_single_output(
                    sample['shirg_output']['text'],
                    sample.get('answers', [sample.get('answer', '')])
                )
                shirg_scores['type'] = 'shirg'
                shirg_scores['sample_id'] = sample.get('question_id', sample.get('index', ''))
                results.append(shirg_scores)
            
            # Handle single output format
            if 'output' in sample and 'baseline_output' not in sample:
                scores = self._score_single_output(
                    sample['output']['text'],
                    sample.get('answers', [sample.get('answer', '')])
                )
                scores['sample_id'] = sample.get('question_id', sample.get('index', ''))
                results.append(scores)
        
        return self._aggregate_results(results)
    
    def _score_single_output(self, prediction: str, references: List[str]) -> Dict:
        """Score a single prediction"""
        
        # Ensure references is a list
        if isinstance(references, str):
            references = [references]
        
        # Process prediction and references
        pred_processed = self.processor(prediction)
        refs_processed = [self.processor(ref) for ref in references]
        
        # Calculate ANLS
        anls_result = anls(refs_processed, [pred_processed])
        
        # Calculate exact match
        em_score = float(any(pred_processed.lower() == ref.lower() for ref in refs_processed))
        
        # Calculate token F1
        f1_score = self._calculate_token_f1(pred_processed, refs_processed)
        
        return {
            'anls': anls_result['anls'],
            'exact_match': em_score,
            'token_f1': f1_score,
            'prediction': prediction,
            'references': references
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
                
            common = pred_tokens.intersection(ref_tokens)
            precision = len(common) / len(pred_tokens) if pred_tokens else 0
            recall = len(common) / len(ref_tokens) if ref_tokens else 0
            
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            
            f1_scores.append(f1)
        
        return max(f1_scores) if f1_scores else 0.0
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggregate results by type"""
        
        df = pd.DataFrame(results)
        
        aggregated = {
            'total_samples': len(results),
            'overall': {
                'anls_mean': df['anls'].mean(),
                'anls_std': df['anls'].std(),
                'exact_match_mean': df['exact_match'].mean(),
                'token_f1_mean': df['token_f1'].mean(),
                'token_f1_std': df['token_f1'].std()
            }
        }
        
        # Aggregate by type if available
        if 'type' in df.columns:
            for output_type in df['type'].unique():
                type_df = df[df['type'] == output_type]
                aggregated[output_type] = {
                    'anls_mean': type_df['anls'].mean(),
                    'anls_std': type_df['anls'].std(),
                    'exact_match_mean': type_df['exact_match'].mean(),
                    'token_f1_mean': type_df['token_f1'].mean(),
                    'token_f1_std': type_df['token_f1'].std(),
                    'num_samples': len(type_df)
                }
        
        return aggregated
    
    def compare_configurations(self, config_files: Dict[str, str]) -> pd.DataFrame:
        """Compare multiple configuration outputs"""
        
        comparison_results = []
        
        for config_name, file_path in config_files.items():
            print(f"Scoring {config_name}...")
            scores = self.score_json_file(file_path)
            
            result = {
                'configuration': config_name,
                'anls': scores['overall']['anls_mean'],
                'anls_std': scores['overall']['anls_std'],
                'exact_match': scores['overall']['exact_match_mean'],
                'token_f1': scores['overall']['token_f1_mean'],
                'token_f1_std': scores['overall']['token_f1_std'],
                'num_samples': scores['total_samples']
            }
            comparison_results.append(result)
        
        comparison_df = pd.DataFrame(comparison_results)
        
        # Print comparison
        print("\n" + "=" * 70)
        print("📊 CONFIGURATION COMPARISON")
        print("=" * 70)
        print(comparison_df.to_string(index=False))
        print("=" * 70)
        
        return comparison_df


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Score SHIRG evaluation outputs")
    parser.add_argument("json_files", nargs='+', help="JSON files to score")
    parser.add_argument("--compare", action='store_true', 
                       help="Compare multiple files as different configurations")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for comparison results")
    
    args = parser.parse_args()
    
    scorer = SHIRGJSONScorer()
    
    if args.compare and len(args.json_files) > 1:
        # Create configuration dictionary from file names
        config_files = {}
        for file_path in args.json_files:
            config_name = Path(file_path).stem
            config_files[config_name] = file_path
        
        comparison_df = scorer.compare_configurations(config_files)
        
        if args.output:
            comparison_df.to_csv(args.output, index=False)
            print(f"\nSaved comparison to: {args.output}")
    else:
        # Score individual files
        for file_path in args.json_files:
            print(f"\nScoring: {file_path}")
            scores = scorer.score_json_file(file_path)
            
            print(f"ANLS: {scores['overall']['anls_mean']:.3f} (±{scores['overall']['anls_std']:.3f})")
            print(f"Exact Match: {scores['overall']['exact_match_mean']:.3f}")
            print(f"Token F1: {scores['overall']['token_f1_mean']:.3f} (±{scores['overall']['token_f1_std']:.3f})")
            
            if 'baseline' in scores and 'shirg' in scores:
                print(f"\nBaseline ANLS: {scores['baseline']['anls_mean']:.3f}")
                print(f"SHIRG ANLS: {scores['shirg']['anls_mean']:.3f}")
                improvement = (scores['shirg']['anls_mean'] - scores['baseline']['anls_mean']) / scores['baseline']['anls_mean'] * 100
                print(f"Improvement: {improvement:+.1f}%")


if __name__ == "__main__":
    main()