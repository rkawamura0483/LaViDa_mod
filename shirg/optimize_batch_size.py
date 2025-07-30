#!/usr/bin/env python3
"""
SHIRG Batch Size Optimization
Finds optimal batch size for LoRA training on available GPU

This script helps determine the maximum batch size that fits in GPU memory
while leaving enough headroom for training stability.

Author: Research Implementation
Date: 2025-07-30
"""

import os
import sys
import torch
import torch.nn as nn
import gc
import time
import psutil
import GPUtil
from typing import Dict, List, Optional, Tuple
import numpy as np

# Add paths
sys.path.append('./')
sys.path.append('./shirg')

from shirg_lora_config import ShirgLoraConfig, create_lora_training_config


class BatchSizeOptimizer:
    """Optimize batch size for SHIRG LoRA training"""
    
    def __init__(
        self,
        config: ShirgLoraConfig,
        safety_margin: float = 0.9,
        min_batch_size: int = 1,
        max_batch_size: int = 64,
    ):
        """
        Initialize batch size optimizer
        
        Args:
            config: SHIRG LoRA configuration
            safety_margin: Use only this fraction of available memory (0.9 = 90%)
            min_batch_size: Minimum batch size to test
            max_batch_size: Maximum batch size to test
        """
        self.config = config
        self.safety_margin = safety_margin
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.results = {}
        
    def get_gpu_info(self) -> Dict[str, float]:
        """Get current GPU information"""
        if not torch.cuda.is_available():
            return {"available": False}
        
        gpu_id = torch.cuda.current_device()
        gpu = GPUtil.getGPUs()[gpu_id]
        
        return {
            "available": True,
            "name": gpu.name,
            "total_memory_gb": gpu.memoryTotal / 1024,
            "used_memory_gb": gpu.memoryUsed / 1024,
            "free_memory_gb": gpu.memoryFree / 1024,
            "temperature": gpu.temperature,
            "utilization": gpu.load * 100,
        }
    
    def estimate_memory_usage(
        self,
        batch_size: int,
        seq_length: int = 2048,
        image_size: int = 672,
    ) -> Dict[str, float]:
        """
        Estimate memory usage for given batch size
        
        Args:
            batch_size: Batch size to estimate
            seq_length: Maximum sequence length
            image_size: Input image size
            
        Returns:
            Memory estimates in GB
        """
        # Base model memory (LaViDa ~16GB in bf16)
        base_model_gb = 16.0
        
        # LoRA parameters (1.4% of model)
        lora_params_gb = 0.136 * 2  # bf16 = 2 bytes per param
        
        # Optimizer states (Adam needs 2x params)
        optimizer_gb = lora_params_gb * 2
        
        # Activations per sample
        # Image: 5 views * 3 channels * image_size^2 * 2 bytes
        image_memory_per_sample = 5 * 3 * image_size * image_size * 2 / 1e9
        
        # Text: seq_length * vocab_size * 2 bytes (rough estimate)
        text_memory_per_sample = seq_length * 32000 * 2 / 1e9
        
        # Vision tokens: 980 * hidden_dim * 2 bytes
        vision_tokens_per_sample = 980 * 1152 * 2 / 1e9
        
        # Total per sample
        per_sample_gb = (image_memory_per_sample + 
                        text_memory_per_sample + 
                        vision_tokens_per_sample)
        
        # Batch memory
        batch_memory_gb = per_sample_gb * batch_size
        
        # Gradient accumulation
        gradient_memory_gb = batch_memory_gb * 0.5  # Rough estimate
        
        # Total
        total_gb = (base_model_gb + lora_params_gb + optimizer_gb + 
                   batch_memory_gb + gradient_memory_gb)
        
        return {
            "base_model_gb": base_model_gb,
            "lora_params_gb": lora_params_gb,
            "optimizer_gb": optimizer_gb,
            "batch_memory_gb": batch_memory_gb,
            "gradient_memory_gb": gradient_memory_gb,
            "total_gb": total_gb,
            "per_sample_gb": per_sample_gb,
        }
    
    def test_batch_size(self, batch_size: int) -> Dict[str, any]:
        """
        Test if a specific batch size fits in memory
        
        Args:
            batch_size: Batch size to test
            
        Returns:
            Test results
        """
        print(f"\n🧪 Testing batch size: {batch_size}")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        result = {
            "batch_size": batch_size,
            "success": False,
            "error": None,
            "memory_used_gb": 0,
            "time_per_step_ms": 0,
        }
        
        try:
            # Get initial GPU state
            gpu_info_start = self.get_gpu_info()
            
            # Create dummy model components
            # Vision encoder output simulation
            vision_dim = 1152
            text_dim = 4096
            num_tokens = 980
            
            # Simulate vision tokens
            vision_tokens = torch.randn(
                batch_size, num_tokens, vision_dim,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                dtype=torch.bfloat16 if self.config.bf16 else torch.float32
            )
            
            # Simulate text tokens
            seq_length = 2048
            text_tokens = torch.randn(
                batch_size, seq_length, text_dim,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                dtype=torch.bfloat16 if self.config.bf16 else torch.float32
            )
            
            # Simulate LoRA layers
            lora_layers = []
            for _ in range(len(self.config.target_modules)):
                if "mm_projector" in self.config.target_modules[0]:
                    in_dim, out_dim = text_dim, vision_dim
                else:
                    in_dim, out_dim = vision_dim, vision_dim
                
                lora_A = nn.Parameter(torch.randn(self.config.rank, in_dim) * 0.01)
                lora_B = nn.Parameter(torch.zeros(out_dim, self.config.rank))
                
                if torch.cuda.is_available():
                    lora_A = lora_A.cuda()
                    lora_B = lora_B.cuda()
                
                lora_layers.append((lora_A, lora_B))
            
            # Simulate forward pass
            start_time = time.time()
            
            # Vision processing
            for lora_A, lora_B in lora_layers[:10]:  # First 10 are vision layers
                if lora_A.shape[1] == vision_dim:
                    lora_out = vision_tokens @ lora_A.T @ lora_B.T
            
            # Text processing
            for lora_A, lora_B in lora_layers[-2:]:  # Last 2 are projector layers
                if lora_A.shape[1] == text_dim:
                    proj_out = text_tokens @ lora_A.T @ lora_B.T
            
            # Simulate backward pass
            loss = vision_tokens.mean() + text_tokens.mean()
            loss.backward()
            
            # Synchronize and measure
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            forward_time = (time.time() - start_time) * 1000  # ms
            
            # Get memory usage
            gpu_info_end = self.get_gpu_info()
            memory_used = gpu_info_end["used_memory_gb"] - gpu_info_start["used_memory_gb"]
            
            result["success"] = True
            result["memory_used_gb"] = memory_used
            result["time_per_step_ms"] = forward_time
            
            print(f"   ✅ Success!")
            print(f"   Memory used: {memory_used:.2f}GB")
            print(f"   Time per step: {forward_time:.1f}ms")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                result["error"] = "OOM"
                print(f"   ❌ Out of memory!")
            else:
                result["error"] = str(e)
                print(f"   ❌ Error: {str(e)}")
        except Exception as e:
            result["error"] = str(e)
            print(f"   ❌ Error: {str(e)}")
        finally:
            # Cleanup
            if 'vision_tokens' in locals():
                del vision_tokens
            if 'text_tokens' in locals():
                del text_tokens
            if 'lora_layers' in locals():
                del lora_layers
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        return result
    
    def binary_search_optimal_batch(self) -> int:
        """
        Use binary search to find optimal batch size
        
        Returns:
            Optimal batch size
        """
        print(f"🔍 Binary search for optimal batch size")
        print(f"   Range: [{self.min_batch_size}, {self.max_batch_size}]")
        print(f"   Safety margin: {self.safety_margin * 100}%")
        
        left = self.min_batch_size
        right = self.max_batch_size
        optimal = self.min_batch_size
        
        while left <= right:
            mid = (left + right) // 2
            
            # Test this batch size
            result = self.test_batch_size(mid)
            self.results[mid] = result
            
            if result["success"]:
                # Check if we're within safety margin
                gpu_info = self.get_gpu_info()
                memory_usage_ratio = result["memory_used_gb"] / gpu_info["total_memory_gb"]
                
                if memory_usage_ratio <= self.safety_margin:
                    optimal = mid
                    left = mid + 1  # Try larger
                else:
                    right = mid - 1  # Too close to limit
            else:
                right = mid - 1  # Failed, try smaller
        
        return optimal
    
    def find_optimal_batch_size(self) -> Dict[str, any]:
        """
        Find optimal batch size with detailed analysis
        
        Returns:
            Optimization results
        """
        print("🚀 SHIRG Batch Size Optimization")
        print("=" * 60)
        
        # Get GPU info
        gpu_info = self.get_gpu_info()
        if not gpu_info["available"]:
            print("❌ No GPU available!")
            return {"error": "No GPU available"}
        
        print(f"GPU: {gpu_info['name']}")
        print(f"Total memory: {gpu_info['total_memory_gb']:.1f}GB")
        print(f"Free memory: {gpu_info['free_memory_gb']:.1f}GB")
        print(f"Temperature: {gpu_info['temperature']}°C")
        print(f"Utilization: {gpu_info['utilization']:.1f}%")
        
        # Find optimal batch size
        optimal_batch = self.binary_search_optimal_batch()
        
        # Get detailed results for optimal batch
        if optimal_batch in self.results:
            optimal_result = self.results[optimal_batch]
        else:
            optimal_result = self.test_batch_size(optimal_batch)
        
        # Test gradient accumulation options
        print(f"\n📊 Gradient Accumulation Analysis")
        print("-" * 40)
        
        accumulation_options = []
        for grad_accum in [1, 2, 4, 8]:
            effective_batch = optimal_batch * grad_accum
            if effective_batch <= 64:  # Reasonable limit
                accumulation_options.append({
                    "per_device_batch": optimal_batch,
                    "gradient_accumulation": grad_accum,
                    "effective_batch": effective_batch,
                    "estimated_time_per_update_ms": optimal_result["time_per_step_ms"] * grad_accum,
                })
                print(f"   Batch {optimal_batch} × Accum {grad_accum} = Effective {effective_batch}")
        
        # Memory estimates
        print(f"\n💾 Memory Breakdown (Batch={optimal_batch})")
        print("-" * 40)
        
        memory_est = self.estimate_memory_usage(optimal_batch)
        for key, value in memory_est.items():
            if key != "per_sample_gb":
                print(f"   {key}: {value:.2f}GB")
        
        # Summary
        results = {
            "gpu_info": gpu_info,
            "optimal_batch_size": optimal_batch,
            "optimal_result": optimal_result,
            "tested_sizes": self.results,
            "accumulation_options": accumulation_options,
            "memory_estimates": memory_est,
            "recommendations": self._generate_recommendations(optimal_batch, optimal_result),
        }
        
        print(f"\n✅ Optimization Complete!")
        print(f"Optimal batch size: {optimal_batch}")
        
        return results
    
    def _generate_recommendations(self, optimal_batch: int, result: Dict) -> Dict[str, any]:
        """Generate training recommendations"""
        recommendations = {
            "batch_size": optimal_batch,
            "gradient_accumulation": 1,
            "effective_batch_size": optimal_batch,
        }
        
        # Recommend gradient accumulation for small batches
        if optimal_batch < 8:
            target_effective = 16
            grad_accum = max(1, target_effective // optimal_batch)
            recommendations["gradient_accumulation"] = grad_accum
            recommendations["effective_batch_size"] = optimal_batch * grad_accum
            recommendations["note"] = f"Small batch size. Using gradient accumulation for effective batch of {optimal_batch * grad_accum}"
        elif optimal_batch >= 32:
            recommendations["note"] = "Large batch size. Consider reducing if training is unstable"
        else:
            recommendations["note"] = "Good batch size for stable training"
        
        # Memory usage recommendation
        gpu_info = self.get_gpu_info()
        memory_ratio = result["memory_used_gb"] / gpu_info["total_memory_gb"]
        if memory_ratio > 0.85:
            recommendations["memory_warning"] = "High memory usage. Monitor for OOM errors"
        elif memory_ratio < 0.5:
            recommendations["memory_note"] = "Low memory usage. Can increase batch size if needed"
        
        return recommendations


def main():
    """Run batch size optimization"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SHIRG Batch Size Optimization")
    parser.add_argument("--min-batch", type=int, default=1,
                       help="Minimum batch size to test")
    parser.add_argument("--max-batch", type=int, default=64,
                       help="Maximum batch size to test")
    parser.add_argument("--safety-margin", type=float, default=0.9,
                       help="Safety margin for memory usage (0.9 = 90%)")
    parser.add_argument("--selection-method", type=str, default="full",
                       choices=["base", "entropy", "edge", "full"],
                       help="SHIRG selection method")
    
    args = parser.parse_args()
    
    # Create config
    config = create_lora_training_config(selection_method=args.selection_method)
    
    # Create optimizer
    optimizer = BatchSizeOptimizer(
        config=config,
        safety_margin=args.safety_margin,
        min_batch_size=args.min_batch,
        max_batch_size=args.max_batch,
    )
    
    # Find optimal batch size
    results = optimizer.find_optimal_batch_size()
    
    # Print recommendations
    if "recommendations" in results:
        print("\n🎯 Training Recommendations:")
        print("=" * 40)
        recs = results["recommendations"]
        print(f"Batch size: {recs['batch_size']}")
        print(f"Gradient accumulation: {recs['gradient_accumulation']}")
        print(f"Effective batch size: {recs['effective_batch_size']}")
        if "note" in recs:
            print(f"Note: {recs['note']}")
        if "memory_warning" in recs:
            print(f"⚠️ {recs['memory_warning']}")
    
    # Save results
    import json
    with open("batch_size_optimization_results.json", "w") as f:
        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj
        
        json.dump(convert_types(results), f, indent=2)
    print(f"\n📄 Results saved to: batch_size_optimization_results.json")


if __name__ == "__main__":
    main()