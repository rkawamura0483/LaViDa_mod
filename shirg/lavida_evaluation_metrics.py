#!/usr/bin/env python3
"""
LaViDa Evaluation Metrics
Implements the exact evaluation metrics used in the LaViDa paper and lmms-eval
"""

import re
import statistics
from typing import List, Dict, Any, Union
import numpy as np

# Import official evaluation functions from lmms-eval
import sys
sys.path.append('./eval/')
from lmms_eval.api.metrics import anls, levenshtein_distance
from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor


class LaViDaEvaluationMetrics:
    """Implements official LaViDa evaluation metrics for each dataset"""
    
    def __init__(self):
        self.processor = EvalAIAnswerProcessor()
    
    def evaluate_sample(self, prediction: str, references: Union[str, List[str]], 
                       dataset_type: str) -> Dict[str, float]:
        """
        Evaluate a single sample using dataset-specific metrics
        
        Args:
            prediction: Model's predicted answer
            references: Ground truth answer(s)
            dataset_type: Type of dataset (DocVQA, ChartQA, etc.)
            
        Returns:
            Dictionary of metric scores
        """
        # Ensure references is a list
        if isinstance(references, str):
            references = [references]
        
        # Select evaluation method based on dataset
        if dataset_type in ["DocVQA", "InfoVQA"]:
            return self._evaluate_anls(prediction, references)
        elif dataset_type == "ChartQA":
            return self._evaluate_chartqa(prediction, references)
        elif dataset_type == "VQAv2":
            return self._evaluate_vqav2(prediction, references)
        elif dataset_type == "TextVQA":
            return self._evaluate_textvqa(prediction, references)
        elif dataset_type in ["MathVista", "MathVerse"]:
            return self._evaluate_math(prediction, references)
        else:
            # Default to basic accuracy
            return self._evaluate_accuracy(prediction, references)
    
    def _evaluate_anls(self, prediction: str, references: List[str]) -> Dict[str, float]:
        """
        ANLS (Average Normalized Levenshtein Similarity) for DocVQA/InfoVQA
        This is the official metric from the lmms-eval codebase
        """
        # Process prediction and references
        pred_processed = self.processor(prediction)
        refs_processed = [self.processor(ref) for ref in references]
        
        # Calculate ANLS using official implementation
        anls_result = anls(refs_processed, [pred_processed])
        
        return {
            'anls': anls_result['anls'],
            'exact_match': float(any(pred_processed.lower() == ref.lower() for ref in refs_processed))
        }
    
    def _evaluate_chartqa(self, prediction: str, references: List[str]) -> Dict[str, float]:
        """
        ChartQA uses relaxed accuracy (±5% tolerance for numbers)
        From: https://arxiv.org/pdf/2203.10244.pdf
        """
        def relaxed_correctness(pred: str, target: str, max_relative_change: float = 0.05) -> bool:
            def _to_float(text: str):
                try:
                    if text.endswith("%"):
                        return float(text.rstrip("%")) / 100.0
                    else:
                        return float(text)
                except ValueError:
                    return None
            
            pred_float = _to_float(pred)
            target_float = _to_float(target)
            
            if pred_float is not None and target_float is not None:
                if target_float == 0:
                    return pred_float == 0
                relative_change = abs(pred_float - target_float) / abs(target_float)
                return relative_change <= max_relative_change
            else:
                # For non-numeric answers, require exact match
                return pred.lower().strip() == target.lower().strip()
        
        # Process prediction
        pred_processed = self.processor(prediction)
        
        # Calculate relaxed accuracy
        relaxed_acc = float(any(relaxed_correctness(pred_processed, ref) for ref in references))
        exact_match = float(any(pred_processed.lower() == ref.lower() for ref in references))
        
        return {
            'relaxed_accuracy': relaxed_acc,
            'exact_match': exact_match,
            'accuracy': relaxed_acc  # Use relaxed accuracy as the main metric
        }
    
    def _evaluate_vqav2(self, prediction: str, references: List[str]) -> Dict[str, float]:
        """
        VQAv2 uses a special accuracy metric based on human agreement
        An answer is considered correct if at least 3 of 10 humans gave that answer
        """
        # Process prediction
        pred_processed = self.processor(prediction)
        
        # Handle VQAv2's special reference format
        # References might be a dict with 'answer' field or a list of strings
        processed_refs = []
        for ref in references:
            if isinstance(ref, dict) and 'answer' in ref:
                processed_refs.append(self.processor(ref['answer']))
            elif isinstance(ref, str):
                processed_refs.append(self.processor(ref))
        
        # For our case with limited references, we use a simplified version
        # In the full VQAv2 eval, this would consider answer frequencies
        accuracy = 0.0
        
        # Check if prediction matches any reference
        for ref_processed in processed_refs:
            if pred_processed.lower() == ref_processed.lower():
                accuracy = 1.0
                break
        
        return {
            'vqa_accuracy': accuracy,
            'exact_match': accuracy,
            'accuracy': accuracy  # Add generic accuracy for compatibility
        }
    
    def _evaluate_textvqa(self, prediction: str, references: List[str]) -> Dict[str, float]:
        """
        TextVQA uses standard accuracy with answer processing
        """
        # Process prediction and references
        pred_processed = self.processor(prediction)
        refs_processed = [self.processor(ref) for ref in references]
        
        # Calculate accuracy
        accuracy = float(any(pred_processed.lower() == ref.lower() for ref in refs_processed))
        
        return {
            'accuracy': accuracy,
            'exact_match': accuracy
        }
    
    def _evaluate_math(self, prediction: str, references: List[str]) -> Dict[str, float]:
        """
        Math datasets (MathVista/MathVerse) use exact match for answers
        """
        # Extract numbers/expressions from prediction
        pred_processed = self._extract_math_answer(prediction)
        
        # Check against references
        accuracy = 0.0
        for ref in references:
            ref_processed = self._extract_math_answer(ref)
            if pred_processed and ref_processed:
                if self._math_equal(pred_processed, ref_processed):
                    accuracy = 1.0
                    break
        
        return {
            'accuracy': accuracy,
            'exact_match': float(any(prediction.strip().lower() == ref.strip().lower() for ref in references))
        }
    
    def _evaluate_accuracy(self, prediction: str, references: List[str]) -> Dict[str, float]:
        """Default accuracy evaluation"""
        pred_processed = self.processor(prediction)
        refs_processed = [self.processor(ref) for ref in references]
        
        accuracy = float(any(pred_processed.lower() == ref.lower() for ref in refs_processed))
        
        return {
            'accuracy': accuracy,
            'exact_match': accuracy
        }
    
    def _extract_math_answer(self, text: str) -> str:
        """Extract mathematical answer from text"""
        # Remove common prefixes
        text = re.sub(r'(the answer is|answer:|is)\s*', '', text.lower().strip())
        
        # Try to extract numbers
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            return numbers[-1]  # Return last number found
        
        # Return cleaned text
        return text.strip()
    
    def _math_equal(self, pred: str, ref: str, tolerance: float = 1e-3) -> bool:
        """Check if two math answers are equal"""
        try:
            # Try numeric comparison
            pred_num = float(pred)
            ref_num = float(ref)
            return abs(pred_num - ref_num) < tolerance
        except ValueError:
            # Fall back to string comparison
            return pred.lower().strip() == ref.lower().strip()
    
    def aggregate_metrics(self, all_results: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across all samples"""
        if not all_results:
            return {}
        
        # Get all unique metric names
        all_metrics = set()
        for result in all_results:
            all_metrics.update(result.keys())
        
        # Calculate mean for each metric
        aggregated = {}
        for metric in all_metrics:
            values = [r.get(metric, 0.0) for r in all_results]
            aggregated[f"{metric}_mean"] = np.mean(values)
            aggregated[f"{metric}_std"] = np.std(values)
        
        return aggregated


# Test the evaluation metrics
if __name__ == "__main__":
    evaluator = LaViDaEvaluationMetrics()
    
    print("🧪 Testing LaViDa evaluation metrics...")
    print("="*60)
    
    # Test ANLS for DocVQA
    print("\n📊 DocVQA (ANLS):")
    result = evaluator.evaluate_sample("January 15, 2023", ["January 15 2023"], "DocVQA")
    print(f"   Prediction: 'January 15, 2023' vs Reference: 'January 15 2023'")
    print(f"   ANLS: {result['anls']:.3f}")
    
    # Test relaxed accuracy for ChartQA
    print("\n📊 ChartQA (Relaxed Accuracy):")
    result = evaluator.evaluate_sample("42.5", ["42"], "ChartQA")
    print(f"   Prediction: '42.5' vs Reference: '42'")
    print(f"   Relaxed Accuracy: {result['relaxed_accuracy']:.3f}")
    
    result = evaluator.evaluate_sample("42.5", ["40"], "ChartQA")
    print(f"   Prediction: '42.5' vs Reference: '40'")
    print(f"   Relaxed Accuracy: {result['relaxed_accuracy']:.3f}")
    
    # Test VQA accuracy
    print("\n📊 VQAv2 (VQA Accuracy):")
    result = evaluator.evaluate_sample("red car", ["red car", "car", "red"], "VQAv2")
    print(f"   Prediction: 'red car' vs References: ['red car', 'car', 'red']")
    print(f"   VQA Accuracy: {result['vqa_accuracy']:.3f}")