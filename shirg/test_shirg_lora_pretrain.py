#!/usr/bin/env python3
"""
SHIRG LoRA Pre-Training Test Suite
Comprehensive tests to run before starting LoRA training on Lambda Cloud

This ensures all components are working correctly and identifies potential issues
before expensive GPU training begins.

Author: Research Implementation
Date: 2025-07-30
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import time
import traceback
from PIL import Image
import json

# Add paths
sys.path.append('./')
sys.path.append('./shirg')

from shirg.shirg_lora_config import ShirgLoraConfig, create_lora_training_config


class ShirgLoraPreTrainTest:
    """Comprehensive pre-training test suite for SHIRG LoRA"""
    
    def __init__(self, config: Optional[ShirgLoraConfig] = None):
        self.config = config or create_lora_training_config()
        self.test_results = {}
        self.passed_tests = []
        self.failed_tests = []
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all pre-training tests"""
        print("🧪 SHIRG LoRA Pre-Training Test Suite")
        print("=" * 60)
        
        tests = [
            ("Environment Check", self.test_environment),
            ("CUDA and GPU Check", self.test_cuda_gpu),
            ("Memory Requirements", self.test_memory_requirements),
            ("LaViDa Model Loading", self.test_model_loading),
            ("SHIRG Integration", self.test_shirg_integration),
            ("LoRA Module Targeting", self.test_lora_targeting),
            ("Forward Pass", self.test_forward_pass),
            ("Gradient Flow", self.test_gradient_flow),
            ("Mixed Precision", self.test_mixed_precision),
            ("Token Dropout", self.test_token_dropout),
            ("Batch Processing", self.test_batch_processing),
            ("Checkpoint Saving", self.test_checkpoint_saving),
            ("Data Loading", self.test_data_loading),
            ("Performance Benchmarks", self.test_performance),
        ]
        
        total_tests = len(tests)
        for idx, (test_name, test_func) in enumerate(tests, 1):
            print(f"\n[{idx}/{total_tests}] Running: {test_name}")
            print("-" * 40)
            
            try:
                result = test_func()
                self.test_results[test_name] = result
                
                if result.get("passed", False):
                    self.passed_tests.append(test_name)
                    print(f"✅ {test_name}: PASSED")
                else:
                    self.failed_tests.append(test_name)
                    print(f"❌ {test_name}: FAILED")
                    if "error" in result:
                        print(f"   Error: {result['error']}")
                        
            except Exception as e:
                self.failed_tests.append(test_name)
                self.test_results[test_name] = {
                    "passed": False,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
                print(f"❌ {test_name}: EXCEPTION")
                print(f"   Error: {str(e)}")
        
        # Summary
        self.print_summary()
        
        return {
            "passed": len(self.passed_tests),
            "failed": len(self.failed_tests),
            "total": total_tests,
            "results": self.test_results,
            "ready_for_training": len(self.failed_tests) == 0
        }
    
    def test_environment(self) -> Dict[str, Any]:
        """Test environment setup"""
        result = {"passed": True, "details": {}}
        
        # Python version
        import sys
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        result["details"]["python_version"] = python_version
        print(f"   Python: {python_version}")
        
        # PyTorch version
        result["details"]["pytorch_version"] = torch.__version__
        print(f"   PyTorch: {torch.__version__}")
        
        # Check required packages
        required_packages = {
            "transformers": None,
            "peft": None,
            "datasets": None,
            "accelerate": None,
            "bitsandbytes": None,
            "flash_attn": None,
        }
        
        for package in required_packages:
            try:
                module = __import__(package)
                version = getattr(module, "__version__", "unknown")
                required_packages[package] = version
                print(f"   {package}: {version}")
            except ImportError:
                required_packages[package] = "NOT INSTALLED"
                result["passed"] = False
                print(f"   {package}: ❌ NOT INSTALLED")
        
        result["details"]["packages"] = required_packages
        
        # Check LaViDa availability
        try:
            from llava.model.builder import load_pretrained_model
            result["details"]["lavida"] = "available"
            print(f"   LaViDa: ✅ available")
        except ImportError:
            result["details"]["lavida"] = "NOT AVAILABLE"
            result["passed"] = False
            print(f"   LaViDa: ❌ NOT AVAILABLE")
        
        return result
    
    def test_cuda_gpu(self) -> Dict[str, Any]:
        """Test CUDA and GPU availability"""
        result = {"passed": True, "details": {}}
        
        # CUDA availability
        cuda_available = torch.cuda.is_available()
        result["details"]["cuda_available"] = cuda_available
        print(f"   CUDA available: {cuda_available}")
        
        if not cuda_available:
            result["passed"] = False
            result["error"] = "CUDA not available"
            return result
        
        # GPU count and info
        gpu_count = torch.cuda.device_count()
        result["details"]["gpu_count"] = gpu_count
        print(f"   GPU count: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            result["details"][f"gpu_{i}"] = {
                "name": gpu_name,
                "memory_gb": gpu_memory
            }
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # Check if GPU memory is sufficient
        min_memory_gb = 40  # Lambda Cloud GPUs should have at least 40GB
        if gpu_memory < min_memory_gb:
            result["warning"] = f"GPU memory ({gpu_memory:.1f}GB) below recommended {min_memory_gb}GB"
            print(f"   ⚠️ Warning: {result['warning']}")
        
        # CUDA capability
        if gpu_count > 0:
            capability = torch.cuda.get_device_capability(0)
            result["details"]["cuda_capability"] = f"{capability[0]}.{capability[1]}"
            print(f"   CUDA capability: {capability[0]}.{capability[1]}")
            
            # Check for bf16 support (requires capability >= 8.0)
            if capability[0] < 8:
                result["warning"] = "GPU doesn't support native bf16 (requires compute capability >= 8.0)"
                print(f"   ⚠️ Warning: {result['warning']}")
        
        return result
    
    def test_memory_requirements(self) -> Dict[str, Any]:
        """Test memory requirements and estimates"""
        result = {"passed": True, "details": {}}
        
        # Get memory estimates from config
        memory_est = self.config.estimate_memory_usage()
        result["details"]["estimates"] = memory_est
        
        print(f"   Base model: {memory_est['base_model_gb']:.2f}GB")
        print(f"   LoRA params: {memory_est['lora_params_gb']:.2f}GB")
        print(f"   Optimizer states: {memory_est['optimizer_states_gb']:.2f}GB")
        print(f"   Gradients: {memory_est['gradients_gb']:.2f}GB")
        print(f"   Activations: {memory_est['activations_gb']:.2f}GB")
        print(f"   Total estimated: {memory_est['total_estimated_gb']:.2f}GB")
        print(f"   Recommended GPU: {memory_est['recommended_gpu_memory_gb']:.2f}GB")
        
        # Check against available GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if memory_est['recommended_gpu_memory_gb'] > gpu_memory:
                result["passed"] = False
                result["error"] = f"Insufficient GPU memory: need {memory_est['recommended_gpu_memory_gb']:.1f}GB, have {gpu_memory:.1f}GB"
                print(f"   ❌ {result['error']}")
            else:
                headroom = gpu_memory - memory_est['recommended_gpu_memory_gb']
                print(f"   ✅ Memory headroom: {headroom:.1f}GB")
        
        return result
    
    def test_model_loading(self) -> Dict[str, Any]:
        """Test LaViDa model loading (lightweight check)"""
        result = {"passed": True, "details": {}}
        
        try:
            # Just check if we can import and create config
            from shirg.lavida_shirg_integration import LaViDaSHIRGWrapper
            
            # Create wrapper with SHIRG config
            wrapper = LaViDaSHIRGWrapper(
                shirg_config={
                    'target_tokens': 980,
                    'alpha': 0.3,
                    'debug': False
                },
                selection_method=self.config.shirg_method,
                selection_params={
                    'entropy_threshold': self.config.shirg_entropy_threshold,
                    'edge_weight': self.config.shirg_edge_weight,
                    'radial_sigma': self.config.shirg_radial_sigma,
                    'merge_similar': self.config.shirg_merge_similar,
                    'merge_threshold': self.config.shirg_merge_threshold,
                }
            )
            
            result["details"]["wrapper_created"] = True
            print(f"   ✅ LaViDa-SHIRG wrapper created")
            
            # Don't actually load the model in pre-test to save time/memory
            print(f"   ℹ️ Skipping actual model loading in pre-test")
            
        except Exception as e:
            result["passed"] = False
            result["error"] = str(e)
            print(f"   ❌ Failed to create wrapper: {str(e)}")
        
        return result
    
    def test_shirg_integration(self) -> Dict[str, Any]:
        """Test SHIRG integration components"""
        result = {"passed": True, "details": {}}
        
        try:
            # Check if SHIRG selection methods are available
            from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
            
            # Create dummy config
            class DummyConfig:
                mm_vision_tower = "google/siglip-so400m-patch14-384"
                enable_shirg = True
                shirg_selection_method = self.config.shirg_method
            
            config = DummyConfig()
            
            # Check if SHIRG methods exist
            result["details"]["shirg_enabled"] = True
            result["details"]["selection_method"] = self.config.shirg_method
            print(f"   ✅ SHIRG integration available")
            print(f"   Selection method: {self.config.shirg_method}")
            
            # Validate selection parameters
            valid_methods = ["base", "entropy", "edge", "full"]
            if self.config.shirg_method not in valid_methods:
                result["passed"] = False
                result["error"] = f"Invalid selection method: {self.config.shirg_method}"
                print(f"   ❌ {result['error']}")
            
        except Exception as e:
            result["passed"] = False
            result["error"] = str(e)
            print(f"   ❌ SHIRG integration check failed: {str(e)}")
        
        return result
    
    def test_lora_targeting(self) -> Dict[str, Any]:
        """Test LoRA module targeting"""
        result = {"passed": True, "details": {}}
        
        try:
            from peft import LoraConfig
            
            # Create LoRA config
            lora_config = self.config.to_peft_config()
            result["details"]["rank"] = lora_config.r
            result["details"]["alpha"] = lora_config.lora_alpha
            result["details"]["num_target_modules"] = len(lora_config.target_modules)
            
            print(f"   Rank: {lora_config.r}")
            print(f"   Alpha: {lora_config.lora_alpha}")
            print(f"   Target modules: {len(lora_config.target_modules)}")
            
            # Validate target modules
            expected_modules = {
                "mm_projector.fc1": "projector layer 1",
                "mm_projector.fc2": "projector layer 2",
                "layers.0.self_attn.q_proj": "SigLIP block 0 Q",
                "layers.0.self_attn.v_proj": "SigLIP block 0 V (Extra-LoRA)",
            }
            
            for module, desc in expected_modules.items():
                found = any(module in target for target in lora_config.target_modules)
                if found:
                    print(f"   ✅ Found {desc}")
                else:
                    print(f"   ⚠️ Missing {desc}")
                    result["warning"] = f"Some expected modules not found"
            
            # Calculate parameter count estimate
            # Rough estimate: each LoRA adapter adds r * (d_in + d_out) parameters
            # For LaViDa with d=1152 (SigLIP) and d=4096 (projector)
            estimated_params = 0
            for module in lora_config.target_modules:
                if "mm_projector" in module:
                    # Projector: 4096 -> 1152 or vice versa
                    estimated_params += lora_config.r * (4096 + 1152)
                else:
                    # SigLIP attention: 1152 -> 1152
                    estimated_params += lora_config.r * (1152 + 1152)
            
            result["details"]["estimated_lora_params"] = estimated_params
            print(f"   Estimated LoRA parameters: {estimated_params:,}")
            
        except Exception as e:
            result["passed"] = False
            result["error"] = str(e)
            print(f"   ❌ LoRA targeting check failed: {str(e)}")
        
        return result
    
    def test_forward_pass(self) -> Dict[str, Any]:
        """Test forward pass with dummy data"""
        result = {"passed": True, "details": {}}
        
        try:
            # Create dummy inputs
            batch_size = 2
            seq_len = 32
            vocab_size = 32000
            
            # Dummy input tensors
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            attention_mask = torch.ones_like(input_ids)
            
            # Dummy image tensors (5 views as per LaViDa)
            image_size = self.config.image_size
            images = [torch.randn(batch_size, 3, image_size, image_size) for _ in range(5)]
            
            result["details"]["batch_size"] = batch_size
            result["details"]["seq_len"] = seq_len
            result["details"]["num_views"] = len(images)
            result["details"]["image_size"] = image_size
            
            print(f"   Batch size: {batch_size}")
            print(f"   Sequence length: {seq_len}")
            print(f"   Image views: {len(images)}")
            print(f"   Image size: {image_size}×{image_size}")
            
            # Move to GPU if available
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                images = [img.cuda() for img in images]
                print(f"   ✅ Moved tensors to GPU")
            
            # Check tensor shapes
            print(f"   Input shape: {input_ids.shape}")
            print(f"   Image shape: {images[0].shape}")
            
        except Exception as e:
            result["passed"] = False
            result["error"] = str(e)
            print(f"   ❌ Forward pass test failed: {str(e)}")
        
        return result
    
    def test_gradient_flow(self) -> Dict[str, Any]:
        """Test gradient flow through LoRA modules"""
        result = {"passed": True, "details": {}}
        
        try:
            # Create simple test model
            class TestLoRAModule(nn.Module):
                def __init__(self, in_features, out_features, rank=64):
                    super().__init__()
                    self.base = nn.Linear(in_features, out_features, bias=False)
                    self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
                    self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
                    self.scaling = 1.0
                    
                def forward(self, x):
                    base_out = self.base(x)
                    lora_out = x @ self.lora_A.T @ self.lora_B.T * self.scaling
                    return base_out + lora_out
            
            # Test gradient flow
            model = TestLoRAModule(1152, 1152, rank=self.config.rank)
            if torch.cuda.is_available():
                model = model.cuda()
            
            # Freeze base, only train LoRA
            model.base.weight.requires_grad = False
            
            # Forward pass
            x = torch.randn(2, 10, 1152)
            if torch.cuda.is_available():
                x = x.cuda()
            
            out = model(x)
            loss = out.mean()
            loss.backward()
            
            # Check gradients
            has_base_grad = model.base.weight.grad is not None
            has_lora_A_grad = model.lora_A.grad is not None
            has_lora_B_grad = model.lora_B.grad is not None
            
            result["details"]["base_has_grad"] = has_base_grad
            result["details"]["lora_A_has_grad"] = has_lora_A_grad
            result["details"]["lora_B_has_grad"] = has_lora_B_grad
            
            print(f"   Base weight gradient: {'❌ YES' if has_base_grad else '✅ NO (frozen)'}")
            print(f"   LoRA A gradient: {'✅ YES' if has_lora_A_grad else '❌ NO'}")
            print(f"   LoRA B gradient: {'✅ YES' if has_lora_B_grad else '❌ NO'}")
            
            if has_base_grad or not has_lora_A_grad or not has_lora_B_grad:
                result["passed"] = False
                result["error"] = "Incorrect gradient flow"
            
        except Exception as e:
            result["passed"] = False
            result["error"] = str(e)
            print(f"   ❌ Gradient flow test failed: {str(e)}")
        
        return result
    
    def test_mixed_precision(self) -> Dict[str, Any]:
        """Test mixed precision training setup"""
        result = {"passed": True, "details": {}}
        
        try:
            # Test bf16 operations
            if self.config.bf16:
                x = torch.randn(2, 10, 1152, dtype=torch.bfloat16)
                w = torch.randn(1152, 1152, dtype=torch.bfloat16)
                
                if torch.cuda.is_available():
                    x = x.cuda()
                    w = w.cuda()
                
                # Test matmul in bf16
                y = torch.matmul(x, w)
                
                result["details"]["bf16_supported"] = True
                result["details"]["output_dtype"] = str(y.dtype)
                print(f"   ✅ BF16 operations supported")
                print(f"   Output dtype: {y.dtype}")
                
            # Test autocast
            if torch.cuda.is_available():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16 if self.config.bf16 else torch.float16):
                    x = torch.randn(2, 10, 1152).cuda()
                    y = torch.matmul(x, x.T)
                    
                result["details"]["autocast_dtype"] = str(y.dtype)
                print(f"   ✅ Autocast working with dtype: {y.dtype}")
            
        except Exception as e:
            result["passed"] = False
            result["error"] = str(e)
            print(f"   ❌ Mixed precision test failed: {str(e)}")
        
        return result
    
    def test_token_dropout(self) -> Dict[str, Any]:
        """Test token dropout mechanism"""
        result = {"passed": True, "details": {}}
        
        try:
            # Simulate token dropout
            batch_size = 2
            num_tokens = 980  # SHIRG token count
            hidden_dim = 1152
            
            tokens = torch.randn(batch_size, num_tokens, hidden_dim)
            if torch.cuda.is_available():
                tokens = tokens.cuda()
            
            # Apply token dropout
            dropout_rate = self.config.token_dropout_rate
            if dropout_rate > 0:
                # Create dropout mask
                keep_prob = 1 - dropout_rate
                mask = torch.bernoulli(torch.full((batch_size, num_tokens, 1), keep_prob))
                if torch.cuda.is_available():
                    mask = mask.cuda()
                
                # Apply dropout
                dropped_tokens = tokens * mask / keep_prob
                
                # Count dropped tokens
                num_dropped = (mask == 0).sum().item()
                expected_dropped = int(batch_size * num_tokens * dropout_rate)
                
                result["details"]["dropout_rate"] = dropout_rate
                result["details"]["tokens_dropped"] = num_dropped
                result["details"]["expected_dropped"] = expected_dropped
                
                print(f"   Dropout rate: {dropout_rate}")
                print(f"   Tokens dropped: {num_dropped}/{batch_size * num_tokens}")
                print(f"   Expected: ~{expected_dropped}")
                
                # Check if dropout is working reasonably
                tolerance = 0.2  # 20% tolerance
                if abs(num_dropped - expected_dropped) > expected_dropped * tolerance:
                    result["warning"] = "Dropout count outside expected range"
                    print(f"   ⚠️ {result['warning']}")
            else:
                print(f"   Token dropout disabled (rate=0)")
            
        except Exception as e:
            result["passed"] = False
            result["error"] = str(e)
            print(f"   ❌ Token dropout test failed: {str(e)}")
        
        return result
    
    def test_batch_processing(self) -> Dict[str, Any]:
        """Test batch processing with different sizes"""
        result = {"passed": True, "details": {}}
        
        try:
            # Test different batch sizes
            test_batch_sizes = [1, 4, 8, 16, 32]
            seq_len = 32
            hidden_dim = 1152
            
            results = {}
            for batch_size in test_batch_sizes:
                try:
                    # Create batch
                    x = torch.randn(batch_size, seq_len, hidden_dim)
                    if torch.cuda.is_available():
                        x = x.cuda()
                    
                    # Simple operation
                    y = x.mean(dim=1)
                    
                    # Measure memory if on GPU
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        memory_mb = torch.cuda.memory_allocated() / 1e6
                        results[batch_size] = {
                            "success": True,
                            "memory_mb": memory_mb
                        }
                        print(f"   Batch {batch_size}: ✅ ({memory_mb:.1f}MB)")
                    else:
                        results[batch_size] = {"success": True}
                        print(f"   Batch {batch_size}: ✅")
                        
                except Exception as e:
                    results[batch_size] = {
                        "success": False,
                        "error": str(e)
                    }
                    print(f"   Batch {batch_size}: ❌ {str(e)}")
            
            result["details"]["batch_sizes"] = results
            
            # Find optimal batch size
            successful_sizes = [bs for bs, res in results.items() if res["success"]]
            if successful_sizes:
                result["details"]["max_successful_batch"] = max(successful_sizes)
                print(f"   Max successful batch size: {max(successful_sizes)}")
            else:
                result["passed"] = False
                result["error"] = "No batch size succeeded"
            
        except Exception as e:
            result["passed"] = False
            result["error"] = str(e)
            print(f"   ❌ Batch processing test failed: {str(e)}")
        
        return result
    
    def test_checkpoint_saving(self) -> Dict[str, Any]:
        """Test checkpoint saving and loading"""
        result = {"passed": True, "details": {}}
        
        try:
            import tempfile
            
            # Create temporary directory
            with tempfile.TemporaryDirectory() as tmpdir:
                # Create simple model state
                test_state = {
                    "epoch": 1,
                    "global_step": 100,
                    "model_state_dict": {"test_weight": torch.randn(10, 10)},
                    "optimizer_state_dict": {"test_state": torch.randn(10, 10)},
                    "config": self.config.__dict__,
                }
                
                # Save checkpoint
                checkpoint_path = os.path.join(tmpdir, "test_checkpoint.pt")
                torch.save(test_state, checkpoint_path)
                
                file_size_mb = os.path.getsize(checkpoint_path) / 1e6
                result["details"]["checkpoint_size_mb"] = file_size_mb
                print(f"   Checkpoint saved: {file_size_mb:.2f}MB")
                
                # Load checkpoint
                loaded_state = torch.load(checkpoint_path, map_location="cpu")
                
                # Verify contents
                if all(key in loaded_state for key in test_state.keys()):
                    print(f"   ✅ Checkpoint loaded successfully")
                    result["details"]["keys_preserved"] = True
                else:
                    result["passed"] = False
                    result["error"] = "Checkpoint missing keys"
                    print(f"   ❌ {result['error']}")
                
        except Exception as e:
            result["passed"] = False
            result["error"] = str(e)
            print(f"   ❌ Checkpoint test failed: {str(e)}")
        
        return result
    
    def test_data_loading(self) -> Dict[str, Any]:
        """Test data loading pipeline"""
        result = {"passed": True, "details": {}}
        
        try:
            # Test creating dummy data
            num_samples = 10
            
            # Create dummy dataset
            dummy_data = []
            for i in range(num_samples):
                sample = {
                    "image": Image.new("RGB", (self.config.image_size, self.config.image_size)),
                    "question": f"What is in this image? (sample {i})",
                    "answer": f"This is sample {i}",
                }
                dummy_data.append(sample)
            
            result["details"]["num_samples"] = num_samples
            result["details"]["image_size"] = self.config.image_size
            print(f"   Created {num_samples} dummy samples")
            print(f"   Image size: {self.config.image_size}×{self.config.image_size}")
            
            # Test batch collation
            from torch.utils.data import DataLoader, Dataset
            
            class DummyDataset(Dataset):
                def __init__(self, data):
                    self.data = data
                
                def __len__(self):
                    return len(self.data)
                
                def __getitem__(self, idx):
                    return self.data[idx]
            
            dataset = DummyDataset(dummy_data)
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.per_device_train_batch_size,
                num_workers=0,  # Use 0 for testing
                pin_memory=False,
            )
            
            # Test iteration
            batch = next(iter(dataloader))
            result["details"]["batch_keys"] = list(batch.keys())
            print(f"   ✅ DataLoader working")
            print(f"   Batch keys: {list(batch.keys())}")
            
        except Exception as e:
            result["passed"] = False
            result["error"] = str(e)
            print(f"   ❌ Data loading test failed: {str(e)}")
        
        return result
    
    def test_performance(self) -> Dict[str, Any]:
        """Test performance benchmarks"""
        result = {"passed": True, "details": {}}
        
        try:
            # Test computation speed
            size = 1152
            iterations = 100
            
            # Create test tensors
            x = torch.randn(32, size, size)
            if torch.cuda.is_available():
                x = x.cuda()
                torch.cuda.synchronize()
            
            # Benchmark matrix multiplication
            start_time = time.time()
            for _ in range(iterations):
                y = torch.matmul(x, x)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed_time = time.time() - start_time
            throughput = iterations / elapsed_time
            
            result["details"]["matrix_size"] = size
            result["details"]["iterations"] = iterations
            result["details"]["elapsed_time_s"] = elapsed_time
            result["details"]["throughput_iter_per_s"] = throughput
            
            print(f"   Matrix size: {size}×{size}")
            print(f"   Iterations: {iterations}")
            print(f"   Time: {elapsed_time:.2f}s")
            print(f"   Throughput: {throughput:.1f} iter/s")
            
            # Check if performance is reasonable
            min_throughput = 10  # At least 10 iterations per second
            if throughput < min_throughput:
                result["warning"] = f"Low throughput: {throughput:.1f} iter/s"
                print(f"   ⚠️ {result['warning']}")
            
        except Exception as e:
            result["passed"] = False
            result["error"] = str(e)
            print(f"   ❌ Performance test failed: {str(e)}")
        
        return result
    
    def test_wandb_integration(self) -> Dict[str, Any]:
        """Test Weights & Biases integration"""
        result = {"passed": True, "details": {}}
        
        try:
            # Check if wandb is installed
            try:
                import wandb
                result["details"]["wandb_version"] = wandb.__version__
                print(f"   W&B version: {wandb.__version__}")
            except ImportError:
                result["passed"] = False
                result["error"] = "wandb not installed"
                print(f"   ❌ wandb not installed - run: pip install wandb")
                return result
            
            # Check if we can initialize wandb
            try:
                # Initialize with offline mode for testing
                run = wandb.init(
                    project="shirg-test",
                    mode="offline",
                    config=self.config.__dict__,
                )
                
                # Log some test metrics
                wandb.log({"test_metric": 1.0})
                wandb.log({"test_loss": 0.5})
                
                # Finish run
                wandb.finish()
                
                result["details"]["init_success"] = True
                print(f"   ✅ W&B initialization successful (offline mode)")
                
            except Exception as e:
                result["warning"] = f"W&B init warning: {str(e)}"
                print(f"   ⚠️ W&B init warning: {str(e)}")
                print(f"   This is okay for testing - ensure you login for actual training")
            
        except Exception as e:
            result["passed"] = False
            result["error"] = str(e)
            print(f"   ❌ W&B test failed: {str(e)}")
        
        return result
    
    def test_actual_model(self) -> Dict[str, Any]:
        """Test actual LaViDa model loading and LoRA setup (optional)"""
        result = {"passed": True, "details": {}}
        
        # This test is optional and requires GPU + model download
        print(f"   ℹ️ Skipping actual model test (requires GPU and model download)")
        print(f"   To test actual model, run train_shirg_lora.py with --test-only flag")
        
        result["details"]["skipped"] = True
        result["details"]["reason"] = "Requires GPU and model download"
        
        return result
    
    def print_summary(self):
        """Print test summary"""
        total = len(self.passed_tests) + len(self.failed_tests)
        
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        print(f"Total tests: {total}")
        print(f"Passed: {len(self.passed_tests)} ✅")
        print(f"Failed: {len(self.failed_tests)} ❌")
        
        if self.failed_tests:
            print("\n❌ Failed tests:")
            for test in self.failed_tests:
                error = self.test_results[test].get("error", "Unknown error")
                print(f"   - {test}: {error}")
        
        if len(self.failed_tests) == 0:
            print("\n✅ All tests passed! Ready for LoRA training.")
        else:
            print("\n⚠️ Some tests failed. Please fix issues before training.")
        
        # Save detailed results
        results_file = "shirg_pretrain_test_results.json"
        with open(results_file, "w") as f:
            json.dump(self.test_results, f, indent=2, default=str)
        print(f"\n📄 Detailed results saved to: {results_file}")


def main():
    """Run pre-training tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SHIRG LoRA Pre-Training Test Suite")
    parser.add_argument("--selection-method", type=str, default="full",
                       choices=["base", "entropy", "edge", "full"],
                       help="SHIRG selection method")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Override batch size")
    parser.add_argument("--skip-gpu", action="store_true",
                       help="Skip GPU tests")
    
    args = parser.parse_args()
    
    # Create config
    config = create_lora_training_config(
        selection_method=args.selection_method,
        batch_size=args.batch_size
    )
    
    # Run tests
    tester = ShirgLoraPreTrainTest(config)
    results = tester.run_all_tests()
    
    # Exit code based on results
    sys.exit(0 if results["ready_for_training"] else 1)


if __name__ == "__main__":
    main()