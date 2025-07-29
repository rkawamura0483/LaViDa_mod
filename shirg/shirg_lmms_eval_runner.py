#!/usr/bin/env python3
"""
SHIRG Evaluation Runner using lmms-eval Framework
Evaluates SHIRG parameters on OCR and VQA benchmarks
"""

import os
import sys
import json
import subprocess
import argparse
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import torch
import pandas as pd
from datetime import datetime

# Add paths for imports
sys.path.append('./shirg/')
sys.path.append('./eval/')
sys.path.append('./')

class SHIRGLMMSEvalRunner:
    """Run SHIRG evaluation using the lmms-eval framework"""
    
    def __init__(self, model_path: str = "models/LaViDa"):
        self.model_path = model_path
        self.results_dir = Path("/content/shirg_lmms_eval_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Default SHIRG parameters to test
        self.default_shirg_configs = {
            "baseline": {
                "enable_shirg": False,
                "description": "Baseline LaViDa without SHIRG"
            },
            "shirg_40": {
                "enable_shirg": True,
                "keep_ratio": 0.40,
                "score_weights": {"attention": 0.7, "similarity": 0.3},
                "description": "SHIRG with 40% token retention"
            },
            "shirg_45": {
                "enable_shirg": True,
                "keep_ratio": 0.45,
                "score_weights": {"attention": 0.7, "similarity": 0.3},
                "description": "SHIRG with 45% token retention"
            },
            "shirg_50": {
                "enable_shirg": True,
                "keep_ratio": 0.50,
                "score_weights": {"attention": 0.7, "similarity": 0.3},
                "description": "SHIRG with 50% token retention"
            },
            "shirg_45_attn": {
                "enable_shirg": True,
                "keep_ratio": 0.45,
                "score_weights": {"attention": 1.0, "similarity": 0.0},
                "description": "SHIRG 45% with attention-only scoring"
            },
            "shirg_45_balanced": {
                "enable_shirg": True,
                "keep_ratio": 0.45,
                "score_weights": {"attention": 0.5, "similarity": 0.5},
                "description": "SHIRG 45% with balanced scoring"
            }
        }
        
        # OCR/VQA benchmarks to evaluate
        self.benchmarks = {
            "ocr": ["ocrbench", "textvqa_val_lite", "docvqa_val_lite"],
            "vqa": ["vqav2_val_lite", "ok_vqa_val2014_lite"],
            "full": ["ocrbench", "textvqa_val", "docvqa_val", "vqav2_val", "ok_vqa_val2014"]
        }
    
    def create_config_file(self, config_name: str, config: Dict) -> str:
        """Create a JSON config file for SHIRG parameters"""
        config_path = self.results_dir / f"shirg_config_{config_name}.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return str(config_path)
    
    def run_lmms_eval(self, 
                      config_name: str,
                      config: Dict,
                      tasks: List[str],
                      limit: Optional[int] = None) -> Dict:
        """Run lmms-eval with specific SHIRG configuration"""
        
        # Set SHIRG config via environment variable
        config_path = self.create_config_file(config_name, config)
        os.environ['SHIRG_CONFIG_PATH'] = config_path
        
        # Build lmms-eval command
        cmd = [
            "python", "-m", "lmms_eval",
            "--model", "llava_llada",
            "--model_args", f"pretrained={self.model_path}",
            "--tasks", ",".join(tasks),
            "--batch_size", "1",
            "--log_samples",
            "--output_path", str(self.results_dir / f"{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        ]
        
        if limit:
            cmd.extend(["--limit", str(limit)])
        
        print(f"\n🚀 Running evaluation for {config_name}:")
        print(f"   Config: {config}")
        print(f"   Tasks: {tasks}")
        print(f"   Command: {' '.join(cmd)}")
        
        # Run evaluation
        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            elapsed_time = time.time() - start_time
            
            # Parse results from output
            output_lines = result.stdout.split('\n')
            results = self._parse_lmms_output(output_lines)
            results['elapsed_time'] = elapsed_time
            results['config_name'] = config_name
            results['config'] = config
            
            print(f"✅ Completed {config_name} in {elapsed_time:.1f}s")
            return results
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running {config_name}: {e}")
            print(f"   STDOUT: {e.stdout}")
            print(f"   STDERR: {e.stderr}")
            return {
                'config_name': config_name,
                'config': config,
                'error': str(e),
                'elapsed_time': time.time() - start_time
            }
    
    def _parse_lmms_output(self, output_lines: List[str]) -> Dict:
        """Parse lmms-eval output to extract metrics"""
        results = {}
        
        # Look for task results in output
        in_results = False
        for line in output_lines:
            if "Task" in line and "Version" in line:
                in_results = True
                continue
            
            if in_results and "|" in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 3:
                    task_name = parts[0]
                    # Extract metrics (usually in the format metric_name: value)
                    for i in range(2, len(parts)):
                        if ":" in parts[i]:
                            metric, value = parts[i].split(":", 1)
                            try:
                                results[f"{task_name}_{metric.strip()}"] = float(value.strip())
                            except ValueError:
                                results[f"{task_name}_{metric.strip()}"] = value.strip()
        
        return results
    
    def run_parameter_sweep(self,
                           configs: Optional[Dict] = None,
                           benchmark_set: str = "ocr",
                           limit: Optional[int] = None) -> pd.DataFrame:
        """Run evaluation across multiple SHIRG configurations"""
        
        configs = configs or self.default_shirg_configs
        tasks = self.benchmarks.get(benchmark_set, self.benchmarks["ocr"])
        
        all_results = []
        
        print(f"\n🔍 SHIRG Parameter Sweep")
        print(f"   Configs: {list(configs.keys())}")
        print(f"   Benchmark set: {benchmark_set}")
        print(f"   Tasks: {tasks}")
        print(f"   Limit: {limit or 'None'}")
        print("=" * 60)
        
        for config_name, config in configs.items():
            results = self.run_lmms_eval(config_name, config, tasks, limit)
            all_results.append(results)
            
            # Save intermediate results
            self._save_results(all_results)
        
        # Create results dataframe
        df = pd.DataFrame(all_results)
        
        # Analyze and print summary
        self._print_summary(df)
        
        return df
    
    def _save_results(self, results: List[Dict]):
        """Save evaluation results to JSON"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f"shirg_sweep_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Saved results to: {results_file}")
    
    def _print_summary(self, df: pd.DataFrame):
        """Print summary of evaluation results"""
        print("\n" + "=" * 80)
        print("📊 SHIRG EVALUATION SUMMARY")
        print("=" * 80)
        
        # Group metrics by task
        metric_cols = [col for col in df.columns if any(task in col for task in self.benchmarks["ocr"] + self.benchmarks["vqa"])]
        
        if metric_cols:
            summary_data = []
            for _, row in df.iterrows():
                if 'error' not in row:
                    config_summary = {
                        'Config': row['config_name'],
                        'Keep Ratio': row['config'].get('keep_ratio', 'N/A'),
                        'Time (s)': f"{row['elapsed_time']:.1f}"
                    }
                    
                    # Add metric values
                    for metric in metric_cols:
                        if metric in row and pd.notna(row[metric]):
                            config_summary[metric] = f"{row[metric]:.3f}"
                    
                    summary_data.append(config_summary)
            
            summary_df = pd.DataFrame(summary_data)
            print(summary_df.to_string(index=False))
        
        print("\n" + "=" * 80)
    
    def run_custom_evaluation(self,
                             model_outputs_json: str,
                             ground_truth_json: str) -> Dict:
        """Evaluate pre-generated model outputs against ground truth"""
        
        # Load outputs and ground truth
        with open(model_outputs_json, 'r') as f:
            model_outputs = json.load(f)
        
        with open(ground_truth_json, 'r') as f:
            ground_truth = json.load(f)
        
        # Import evaluation metrics
        from lmms_eval.api.metrics import anls, exact_match
        from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor
        
        processor = EvalAIAnswerProcessor()
        
        results = {
            'total_samples': len(model_outputs),
            'anls_scores': [],
            'exact_match_scores': []
        }
        
        for sample_id, output in model_outputs.items():
            if sample_id in ground_truth:
                prediction = output.get('text', '')
                references = ground_truth[sample_id].get('answers', [])
                
                # Process answers
                pred_processed = processor(prediction)
                refs_processed = [processor(ref) for ref in references]
                
                # Calculate ANLS
                anls_result = anls(refs_processed, [pred_processed])
                results['anls_scores'].append(anls_result['anls'])
                
                # Calculate exact match
                em = int(any(pred_processed == ref for ref in refs_processed))
                results['exact_match_scores'].append(em)
        
        # Aggregate results
        results['anls_mean'] = sum(results['anls_scores']) / len(results['anls_scores'])
        results['exact_match_mean'] = sum(results['exact_match_scores']) / len(results['exact_match_scores'])
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Run SHIRG evaluation using lmms-eval")
    parser.add_argument("--model_path", type=str, default="models/LaViDa",
                       help="Path to LaViDa model")
    parser.add_argument("--benchmark_set", type=str, default="ocr",
                       choices=["ocr", "vqa", "full"],
                       help="Which benchmark set to run")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of samples per task (for testing)")
    parser.add_argument("--custom_configs", type=str, default=None,
                       help="Path to JSON file with custom SHIRG configs")
    parser.add_argument("--output_json", type=str, default=None,
                       help="Path to save evaluation outputs for custom scoring")
    
    args = parser.parse_args()
    
    runner = SHIRGLMMSEvalRunner(args.model_path)
    
    # Load custom configs if provided
    configs = None
    if args.custom_configs:
        with open(args.custom_configs, 'r') as f:
            configs = json.load(f)
    
    # Run parameter sweep
    results_df = runner.run_parameter_sweep(
        configs=configs,
        benchmark_set=args.benchmark_set,
        limit=args.limit
    )
    
    # Save final results
    if args.output_json:
        results_df.to_json(args.output_json, orient='records', indent=2)
        print(f"\n✅ Saved results to: {args.output_json}")
    
    print("\n🎉 SHIRG evaluation complete!")
    print(f"   Results directory: {runner.results_dir}")


if __name__ == "__main__":
    main()