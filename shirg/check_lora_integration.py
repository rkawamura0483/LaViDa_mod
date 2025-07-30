#!/usr/bin/env python3
"""
SHIRG LoRA Integration Checker
Comprehensive verification of LoRA training setup for Lambda Cloud

Author: Research Implementation
Date: 2025-07-30
"""

import os
import sys
import torch
from typing import Dict, List, Any
import json

# Add paths
sys.path.append('./')
sys.path.append('./shirg')


def check_integration():
    """Run comprehensive integration checks"""
    print("🔍 SHIRG LoRA Integration Check")
    print("=" * 60)
    
    results = {
        "passed": [],
        "failed": [],
        "warnings": []
    }
    
    # 1. Check dataset loading
    print("\n1️⃣ Checking Dataset Loading...")
    try:
        from dataset_loaders import ChartQADataset, DocVQADataset, VQAv2Dataset, MixedVQADataset
        
        # Test loading a small sample
        test_dataset = MixedVQADataset(
            split="train",
            dataset_configs={
                "chartqa": {"weight": 0.3, "max_samples": 10},
                "docvqa": {"weight": 0.3, "max_samples": 10},
                "vqa_v2": {"weight": 0.4, "max_samples": 10},
            },
            image_size=672,
            cache_dir="./temp_test_cache"
        )
        
        if len(test_dataset) > 0:
            sample = test_dataset[0]
            if all(k in sample for k in ['image', 'question', 'answer']):
                results["passed"].append("Dataset loading works correctly")
                print("   ✅ Dataset loading: PASSED")
            else:
                results["failed"].append("Dataset sample missing required keys")
                print("   ❌ Dataset sample missing keys")
        else:
            results["warnings"].append("No dataset samples loaded - check dataset access")
            print("   ⚠️ No samples loaded - you may need to download datasets")
            
    except Exception as e:
        results["failed"].append(f"Dataset loading failed: {str(e)}")
        print(f"   ❌ Dataset loading failed: {e}")
    
    # 2. Check LoRA configuration
    print("\n2️⃣ Checking LoRA Configuration...")
    try:
        from shirg_lora_config import ShirgLoraConfig, create_lora_training_config
        
        config = create_lora_training_config(selection_method="full")
        
        # Check target modules
        expected_modules = [
            "model.mm_projector.fc1",
            "model.mm_projector.fc2",
            "model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj",
            "model.vision_tower.vision_model.encoder.layers.0.self_attn.v_proj",
        ]
        
        missing_modules = [m for m in expected_modules if m not in config.target_modules]
        if missing_modules:
            results["failed"].append(f"Missing LoRA target modules: {missing_modules}")
            print(f"   ❌ Missing modules: {missing_modules}")
        else:
            results["passed"].append("LoRA target modules correctly configured")
            print("   ✅ LoRA modules: CORRECT")
            
        # Check memory estimates
        memory_est = config.estimate_memory_usage()
        if memory_est['recommended_gpu_memory_gb'] > 40:
            results["warnings"].append(f"Memory usage ({memory_est['recommended_gpu_memory_gb']:.1f}GB) may exceed 40GB GPU")
            print(f"   ⚠️ Memory estimate: {memory_est['recommended_gpu_memory_gb']:.1f}GB (may be tight on 40GB GPU)")
        else:
            print(f"   ✅ Memory estimate: {memory_est['recommended_gpu_memory_gb']:.1f}GB")
            
    except Exception as e:
        results["failed"].append(f"LoRA config check failed: {str(e)}")
        print(f"   ❌ LoRA config failed: {e}")
    
    # 3. Check SHIRG integration
    print("\n3️⃣ Checking SHIRG Integration...")
    try:
        from lavida_shirg_integration import LaViDaSHIRGWrapper
        
        # Check if wrapper can be created
        wrapper = LaViDaSHIRGWrapper(
            shirg_config={
                'target_tokens': 980,
                'alpha': 0.3,
                'debug': False
            },
            selection_method="full",
            selection_params={
                'entropy_threshold': 0.12,
                'edge_weight': 0.25,
                'radial_sigma': 0.65,
                'merge_similar': True,
                'merge_threshold': 0.9,
            }
        )
        
        results["passed"].append("SHIRG wrapper created successfully")
        print("   ✅ SHIRG wrapper: CREATED")
        
        # Check if model would have correct config
        if wrapper.shirg_config['alpha'] > 0:
            print("   ✅ SHIRG enabled (alpha > 0)")
        else:
            results["warnings"].append("SHIRG disabled (alpha = 0) - set alpha > 0 to enable")
            print("   ⚠️ SHIRG disabled - set alpha > 0")
            
    except Exception as e:
        results["failed"].append(f"SHIRG integration failed: {str(e)}")
        print(f"   ❌ SHIRG integration failed: {e}")
    
    # 4. Check W&B setup
    print("\n4️⃣ Checking W&B Integration...")
    try:
        import wandb
        results["passed"].append(f"W&B installed (version {wandb.__version__})")
        print(f"   ✅ W&B installed: {wandb.__version__}")
        
        # Check if logged in
        if wandb.api.api_key:
            print("   ✅ W&B logged in")
        else:
            results["warnings"].append("W&B not logged in - run 'wandb login'")
            print("   ⚠️ Not logged in - run: wandb login")
            
    except ImportError:
        results["failed"].append("W&B not installed")
        print("   ❌ W&B not installed - run: pip install wandb")
    
    # 5. Check environment
    print("\n5️⃣ Checking Environment...")
    
    # Python version
    import sys
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if sys.version_info.major == 3 and sys.version_info.minor >= 8:
        print(f"   ✅ Python {python_version}")
    else:
        results["warnings"].append(f"Python {python_version} - recommend 3.8+")
        print(f"   ⚠️ Python {python_version} - recommend 3.8+")
    
    # PyTorch
    print(f"   ✅ PyTorch {torch.__version__}")
    
    # CUDA
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   ✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        
        if gpu_memory < 40:
            results["warnings"].append(f"GPU memory ({gpu_memory:.1f}GB) below recommended 40GB")
    else:
        results["failed"].append("No GPU available")
        print("   ❌ No GPU available")
    
    # 6. Check for common issues
    print("\n6️⃣ Checking Common Issues...")
    
    # Check if LaViDa model files exist
    model_name = "KonstantinosKK/lavida-llada-v1.0-instruct-hf-transformers"
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_cache = os.path.join(cache_dir, f"models--{model_name.replace('/', '--')}")
    
    if os.path.exists(model_cache):
        print(f"   ✅ LaViDa model cached locally")
    else:
        results["warnings"].append("LaViDa model not cached - will download on first run (~16GB)")
        print(f"   ⚠️ Model not cached - will download on first run")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    total_checks = len(results["passed"]) + len(results["failed"]) + len(results["warnings"])
    print(f"Total checks: {total_checks}")
    print(f"✅ Passed: {len(results['passed'])}")
    print(f"⚠️ Warnings: {len(results['warnings'])}")
    print(f"❌ Failed: {len(results['failed'])}")
    
    if results["failed"]:
        print("\n❌ CRITICAL ISSUES TO FIX:")
        for issue in results["failed"]:
            print(f"   - {issue}")
    
    if results["warnings"]:
        print("\n⚠️ WARNINGS TO CONSIDER:")
        for warning in results["warnings"]:
            print(f"   - {warning}")
    
    if not results["failed"]:
        print("\n✅ All critical checks passed! Ready for LoRA training.")
        print("\n🚀 To start training, run:")
        print("   python train_shirg_lora.py --selection-method full")
    else:
        print("\n❌ Please fix critical issues before training.")
    
    # Save results
    with open("lora_integration_check_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📄 Detailed results saved to: lora_integration_check_results.json")


if __name__ == "__main__":
    check_integration()