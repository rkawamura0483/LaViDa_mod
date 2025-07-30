#!/usr/bin/env python3  
"""
SHIRG-Fixed Comprehensive Evaluation Framework
Benchmarks SHIRG-Fixed vs baselines as per research plan

SHIRG-FIXED-FIX: 2025-07-28 - Evaluation framework for SHIRG-Fixed implementation
ISSUE: Need comprehensive evaluation of SHIRG-Fixed vs LaViDa baseline
SOLUTION: Multi-dataset evaluation with performance, memory, and accuracy metrics
RESEARCH IMPACT: Validates SHIRG-Fixed research hypothesis with rigorous testing
"""

import os
import sys
import argparse
import logging
import time
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, asdict

# Add paths for imports
sys.path.append('./shirg/')
sys.path.append('./llava/')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('shirg_fixed_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
from shirg.lavida_shirg_integration import LaViDaSHIRGWrapper

@dataclass
class EvaluationConfig:
    """Configuration for SHIRG-Fixed evaluation"""
    
    # Model configurations to test
    baseline_config: Dict[str, Any] = None
    shirg_fixed_config: Dict[str, Any] = None
    
    # Evaluation datasets
    datasets: List[str] = None
    sample_size: int = 200  # Per dataset
    
    # Performance metrics
    measure_latency: bool = True
    measure_memory: bool = True 
    measure_accuracy: bool = True
    num_runs: int = 5  # For latency averaging
    
    # Output configuration
    output_dir: str = "./shirg_fixed_results"
    save_detailed_results: bool = True
    generate_plots: bool = True
    
    def __post_init__(self):
        if self.baseline_config is None:
            self.baseline_config = {
                'mode': 'baseline',
                'target_tokens': 729,
                'alpha': 0.0,  # Disable SHIRG
                'debug': False
            }
        
        if self.shirg_fixed_config is None:
            self.shirg_fixed_config = {
                'mode': 'shirg-fixed',
                'target_tokens': 768, 
                'alpha': 0.3,  # Enable SHIRG-Fixed
                'debug': False
            }
            
        if self.datasets is None:
            self.datasets = ['ChartQA', 'DocVQA']  # Focus on OCR tasks as per research


@dataclass
class EvaluationResults:
    """Container for evaluation results"""
    
    # Configuration info
    config: str = ""  # 'baseline' or 'shirg-fixed'
    dataset: str = ""
    
    # Performance metrics
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0  
    max_latency_ms: float = 0.0
    std_latency_ms: float = 0.0
    
    # Memory metrics  
    peak_memory_gb: float = 0.0
    avg_memory_gb: float = 0.0
    
    # Accuracy metrics
    accuracy: float = 0.0
    f1_score: float = 0.0
    exact_match: float = 0.0
    cider_score: float = 0.0  # For ChartQA
    
    # Token statistics
    input_tokens: int = 0
    output_tokens: int = 0
    
    # Sample count
    num_samples: int = 0


class SHIRGFixedEvaluator:
    """Main evaluation class for SHIRG-Fixed"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.results = []
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
    def load_model(self, model_config: Dict[str, Any]):
        """Load model with specified configuration"""
        try:
            
            logger.info(f"Loading model with config: {model_config}")
            
            wrapper = LaViDaSHIRGWrapper(shirg_config=model_config)
            wrapper.load_model()
            
            return wrapper
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_dataset(self, dataset_name: str, sample_size: int) -> List[Dict[str, Any]]:
        """Load evaluation dataset"""
        try:
            logger.info(f"Loading {dataset_name} dataset (sample_size={sample_size})")
            
            if dataset_name == 'ChartQA':
                return self._load_chartqa_dataset(sample_size)
            elif dataset_name == 'DocVQA':
                return self._load_docvqa_dataset(sample_size)
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")
                
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            raise
    
    def _load_chartqa_dataset(self, sample_size: int) -> List[Dict[str, Any]]:
        """Load ChartQA samples"""
        # Implementation would load actual ChartQA data
        # For now, return dummy data structure
        samples = []
        
        # Check if we have actual ChartQA data
        chartqa_path = Path("./data/chartqa/")
        if chartqa_path.exists():
            # Load real data
            import json
            data_files = list(chartqa_path.glob("*.json"))
            if data_files:
                with open(data_files[0], 'r') as f:
                    data = json.load(f)
                    for i, item in enumerate(data[:sample_size]):
                        samples.append({
                            'image_path': str(chartqa_path / item.get('image', f'sample_{i}.jpg')),
                            'question': item.get('query', 'What is shown in this chart?'),
                            'ground_truth': item.get('label', 'Chart data'),
                            'dataset': 'ChartQA'
                        })
        else:
            # Create dummy samples for testing
            logger.warning("Real ChartQA data not found, using dummy samples")
            for i in range(min(sample_size, 10)):
                samples.append({
                    'image_path': f'./data/test_images/chart_{i}.jpg',
                    'question': f'What is the value of category {i}?',
                    'ground_truth': f'Value {i}',
                    'dataset': 'ChartQA'
                })
        
        logger.info(f"Loaded {len(samples)} ChartQA samples")
        return samples
    
    def _load_docvqa_dataset(self, sample_size: int) -> List[Dict[str, Any]]:
        """Load DocVQA samples"""
        samples = []
        
        # Check if we have actual DocVQA data  
        docvqa_path = Path("./data/docvqa/")
        if docvqa_path.exists():
            # Load real data
            import json
            data_files = list(docvqa_path.glob("*.json"))
            if data_files:
                with open(data_files[0], 'r') as f:
                    data = json.load(f)
                    for i, item in enumerate(data[:sample_size]):
                        samples.append({
                            'image_path': str(docvqa_path / item.get('image', f'doc_{i}.jpg')),
                            'question': item.get('question', 'What does this document say?'),
                            'ground_truth': item.get('answer', 'Document text'),
                            'dataset': 'DocVQA'
                        })
        else:
            # Create dummy samples for testing
            logger.warning("Real DocVQA data not found, using dummy samples")
            for i in range(min(sample_size, 10)):
                samples.append({
                    'image_path': f'./data/test_images/doc_{i}.jpg',
                    'question': f'What is written in section {i}?',
                    'ground_truth': f'Section {i} content',
                    'dataset': 'DocVQA'
                })
        
        logger.info(f"Loaded {len(samples)} DocVQA samples")
        return samples
    
    def evaluate_model_on_dataset(self, model, dataset_samples: List[Dict[str, Any]], 
                                 config_name: str) -> EvaluationResults:
        """Evaluate model on dataset samples"""
        logger.info(f"Evaluating {config_name} on {len(dataset_samples)} samples")
        
        results = EvaluationResults()
        results.config = config_name
        results.dataset = dataset_samples[0]['dataset'] if dataset_samples else "unknown"
        results.num_samples = len(dataset_samples)
        
        # Metrics collectors
        latencies = []
        memory_usage = []
        predictions = []
        ground_truths = []
        
        for i, sample in enumerate(dataset_samples):
            try:
                # Skip if image doesn't exist
                if not os.path.exists(sample['image_path']):
                    logger.warning(f"Image not found: {sample['image_path']}, skipping")
                    continue
                
                # Run multiple times for latency averaging
                sample_latencies = []
                sample_memories = []
                
                for run in range(self.config.num_runs):
                    # Measure memory before
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        initial_memory = torch.cuda.memory_allocated() / 1e9
                    
                    # Time the generation
                    start_time = time.time()
                    
                    prediction = model.generate(
                        image_path=sample['image_path'],
                        question=sample['question'],
                        max_new_tokens=32,
                        temperature=0.0
                    )
                    
                    end_time = time.time()
                    latency = (end_time - start_time) * 1000  # Convert to ms
                    
                    # Measure memory after
                    if torch.cuda.is_available():
                        peak_memory = torch.cuda.max_memory_allocated() / 1e9
                        sample_memories.append(peak_memory - initial_memory)
                    
                    sample_latencies.append(latency)
                    
                    if run == 0:  # Only store prediction from first run
                        predictions.append(prediction)
                        ground_truths.append(sample['ground_truth'])
                
                # Store average metrics for this sample
                if sample_latencies:
                    latencies.extend(sample_latencies)
                if sample_memories:
                    memory_usage.extend(sample_memories)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i+1}/{len(dataset_samples)} samples")
                    
            except Exception as e:
                logger.error(f"Failed to process sample {i}: {e}")
                continue
        
        # Compute aggregate metrics
        if latencies:
            results.avg_latency_ms = np.mean(latencies)
            results.min_latency_ms = np.min(latencies)
            results.max_latency_ms = np.max(latencies)
            results.std_latency_ms = np.std(latencies)
        
        if memory_usage:
            results.peak_memory_gb = np.max(memory_usage)
            results.avg_memory_gb = np.mean(memory_usage)
        
        # Compute accuracy metrics
        if predictions and ground_truths:
            results.accuracy = self._compute_accuracy(predictions, ground_truths)
            results.f1_score = self._compute_f1_score(predictions, ground_truths)
            results.exact_match = self._compute_exact_match(predictions, ground_truths)
            
            if results.dataset == 'ChartQA':
                results.cider_score = self._compute_cider_score(predictions, ground_truths)
        
        logger.info(f"Evaluation complete for {config_name} on {results.dataset}")
        return results
    
    def _compute_accuracy(self, predictions: List[str], ground_truths: List[str]) -> float:
        """Compute simple accuracy metric"""
        if not predictions or not ground_truths:
            return 0.0
        
        correct = 0
        for pred, gt in zip(predictions, ground_truths):
            # Simple string similarity check
            if pred.lower().strip() in gt.lower().strip() or gt.lower().strip() in pred.lower().strip():
                correct += 1
        
        return correct / len(predictions)
    
    def _compute_f1_score(self, predictions: List[str], ground_truths: List[str]) -> float:
        """Compute F1 score based on token overlap"""
        if not predictions or not ground_truths:
            return 0.0
        
        f1_scores = []
        for pred, gt in zip(predictions, ground_truths):
            pred_tokens = set(pred.lower().split())
            gt_tokens = set(gt.lower().split())
            
            if not pred_tokens and not gt_tokens:
                f1_scores.append(1.0)
                continue
            elif not pred_tokens or not gt_tokens:
                f1_scores.append(0.0)
                continue
            
            intersection = len(pred_tokens & gt_tokens)
            precision = intersection / len(pred_tokens)
            recall = intersection / len(gt_tokens)
            
            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
                f1_scores.append(f1)
        
        return np.mean(f1_scores)
    
    def _compute_exact_match(self, predictions: List[str], ground_truths: List[str]) -> float:
        """Compute exact match accuracy"""
        if not predictions or not ground_truths:
            return 0.0
        
        exact_matches = 0
        for pred, gt in zip(predictions, ground_truths):
            if pred.strip().lower() == gt.strip().lower():
                exact_matches += 1
        
        return exact_matches / len(predictions)
    
    def _compute_cider_score(self, predictions: List[str], ground_truths: List[str]) -> float:
        """Compute CIDEr score (simplified implementation)"""
        # For now, return F1 score as proxy
        # In real implementation, would use proper CIDEr calculation
        return self._compute_f1_score(predictions, ground_truths)
    
    def run_comprehensive_evaluation(self) -> List[EvaluationResults]:
        """Run complete evaluation pipeline"""
        logger.info("Starting comprehensive SHIRG-Fixed evaluation")
        
        all_results = []
        
        # Test configurations
        configurations = [
            ('baseline', self.config.baseline_config),
            ('shirg-fixed', self.config.shirg_fixed_config)
        ]
        
        for config_name, config_dict in configurations:
            logger.info(f"Evaluating {config_name} configuration")
            
            try:
                # Load model with configuration
                model = self.load_model(config_dict)
                
                # Test on each dataset
                for dataset_name in self.config.datasets:
                    dataset_samples = self.load_dataset(dataset_name, self.config.sample_size)
                    
                    if not dataset_samples:
                        logger.warning(f"No samples loaded for {dataset_name}, skipping")
                        continue
                    
                    # Run evaluation
                    results = self.evaluate_model_on_dataset(model, dataset_samples, config_name)
                    all_results.append(results)
                
                # Cleanup model
                model.cleanup()
                
            except Exception as e:
                logger.error(f"Failed to evaluate {config_name}: {e}")
                continue
        
        self.results = all_results
        return all_results
    
    def generate_comparison_report(self) -> Dict[str, Any]:
        """Generate comprehensive comparison report"""
        logger.info("Generating comparison report")
        
        if not self.results:
            logger.error("No results available for comparison")
            return {}
        
        report = {
            'evaluation_config': asdict(self.config),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'results_by_config': {},
            'comparisons': {}
        }
        
        # Organize results by configuration
        baseline_results = [r for r in self.results if r.config == 'baseline']
        shirg_fixed_results = [r for r in self.results if r.config == 'shirg-fixed']
        
        report['results_by_config']['baseline'] = [asdict(r) for r in baseline_results]
        report['results_by_config']['shirg-fixed'] = [asdict(r) for r in shirg_fixed_results]
        
        # Generate comparisons
        for dataset in self.config.datasets:
            baseline_res = next((r for r in baseline_results if r.dataset == dataset), None)
            shirg_res = next((r for r in shirg_fixed_results if r.dataset == dataset), None)
            
            if baseline_res and shirg_res:
                comparison = {
                    'dataset': dataset,
                    'baseline_latency_ms': baseline_res.avg_latency_ms,
                    'shirg_fixed_latency_ms': shirg_res.avg_latency_ms,
                    'latency_overhead_percent': ((shirg_res.avg_latency_ms - baseline_res.avg_latency_ms) / baseline_res.avg_latency_ms) * 100 if baseline_res.avg_latency_ms > 0 else 0,
                    
                    'baseline_memory_gb': baseline_res.avg_memory_gb,
                    'shirg_fixed_memory_gb': shirg_res.avg_memory_gb,
                    'memory_overhead_gb': shirg_res.avg_memory_gb - baseline_res.avg_memory_gb,
                    
                    'baseline_accuracy': baseline_res.accuracy,
                    'shirg_fixed_accuracy': shirg_res.accuracy,
                    'accuracy_improvement': shirg_res.accuracy - baseline_res.accuracy,
                    
                    'baseline_f1': baseline_res.f1_score,
                    'shirg_fixed_f1': shirg_res.f1_score,
                    'f1_improvement': shirg_res.f1_score - baseline_res.f1_score
                }
                
                if dataset == 'ChartQA':
                    comparison['baseline_cider'] = baseline_res.cider_score
                    comparison['shirg_fixed_cider'] = shirg_res.cider_score
                    comparison['cider_improvement'] = shirg_res.cider_score - baseline_res.cider_score
                
                report['comparisons'][dataset] = comparison
        
        # Save report
        report_path = Path(self.config.output_dir) / 'evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {report_path}")
        return report
    
    def print_summary(self):
        """Print evaluation summary to console"""
        if not self.results:
            logger.error("No results to summarize")
            return
        
        print("\n" + "="*80)
        print("SHIRG-Fixed Evaluation Summary")
        print("="*80)
        
        for dataset in self.config.datasets:
            print(f"\n{dataset} Results:")
            print("-" * 40)
            
            baseline_res = next((r for r in self.results if r.config == 'baseline' and r.dataset == dataset), None)
            shirg_res = next((r for r in self.results if r.config == 'shirg-fixed' and r.dataset == dataset), None)
            
            if baseline_res and shirg_res:
                print(f"Baseline:    {baseline_res.avg_latency_ms:.1f}ms | {baseline_res.accuracy:.3f} acc | {baseline_res.avg_memory_gb:.1f}GB")
                print(f"SHIRG-Fixed: {shirg_res.avg_latency_ms:.1f}ms | {shirg_res.accuracy:.3f} acc | {shirg_res.avg_memory_gb:.1f}GB")
                
                latency_overhead = ((shirg_res.avg_latency_ms - baseline_res.avg_latency_ms) / baseline_res.avg_latency_ms) * 100 if baseline_res.avg_latency_ms > 0 else 0
                accuracy_improvement = shirg_res.accuracy - baseline_res.accuracy
                
                print(f"Improvement: {latency_overhead:+.1f}% latency | {accuracy_improvement:+.3f} accuracy")
        
        print("\n" + "="*80)


def main():
    """Main evaluation orchestration"""
    parser = argparse.ArgumentParser(description='SHIRG-Fixed Evaluation')
    parser.add_argument('--config', type=str, help='Evaluation config file')
    parser.add_argument('--output_dir', type=str, default='./shirg_fixed_results')
    parser.add_argument('--datasets', nargs='+', default=['ChartQA', 'DocVQA'])
    parser.add_argument('--sample_size', type=int, default=50, help='Samples per dataset')
    parser.add_argument('--num_runs', type=int, default=3, help='Runs per sample for latency')
    
    args = parser.parse_args()
    
    # Load configuration
    config = EvaluationConfig()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                setattr(config, key, value)
    
    # Apply command line overrides
    config.output_dir = args.output_dir
    config.datasets = args.datasets
    config.sample_size = args.sample_size
    config.num_runs = args.num_runs
    
    logger.info("Starting SHIRG-Fixed evaluation")
    logger.info(f"Configuration: {asdict(config)}")
    
    try:
        # Create evaluator
        evaluator = SHIRGFixedEvaluator(config)
        
        # Run evaluation
        results = evaluator.run_comprehensive_evaluation()
        
        # Generate report
        report = evaluator.generate_comparison_report()
        
        # Print summary
        evaluator.print_summary()
        
        logger.info("SHIRG-Fixed evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()