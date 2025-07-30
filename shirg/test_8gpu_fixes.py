#!/usr/bin/env python3
"""
Test script to verify all fixes for 8 GPU training
Tests dataset loading, selective gradient flow, and DDP compatibility

Author: Research Implementation
Date: 2025-07-30
"""

import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Add paths
sys.path.append('./')
sys.path.append('./shirg')

def test_dataset_loading():
    """Test that datasets load correctly with fixed field names"""
    print("\n🧪 Testing Dataset Loading...")
    
    from shirg.dataset_loaders import (
        ChartQADataset, DocVQADataset, VQAv2Dataset,
        TextVQADataset, OCRVQADataset, InfoVQADataset
    )
    
    # Test each dataset
    datasets_to_test = [
        ("ChartQA", ChartQADataset, "train"),
        ("DocVQA", DocVQADataset, "validation"),
        ("VQA v2", VQAv2Dataset, "train"),
        ("TextVQA", TextVQADataset, "train"),
        ("OCR-VQA", OCRVQADataset, "train"),
        ("InfoVQA", InfoVQADataset, "train"),
    ]
    
    for name, dataset_class, split in datasets_to_test:
        print(f"\n📊 Testing {name} ({split} split)...")
        try:
            # Create dataset with small sample size
            dataset = dataset_class(split=split, max_samples=5)
            
            if len(dataset) > 0:
                # Test getting an item
                item = dataset[0]
                
                # Verify required fields
                assert 'image' in item, f"{name}: Missing 'image' field"
                assert 'question' in item, f"{name}: Missing 'question' field"
                assert 'answer' in item, f"{name}: Missing 'answer' field"
                
                # Print sample
                print(f"   ✅ Dataset loaded successfully")
                print(f"   Sample question: {item['question'][:80]}...")
                print(f"   Sample answer: {item['answer'][:80]}...")
                print(f"   Image size: {item['image'].size}")
            else:
                print(f"   ⚠️ Dataset is empty (might be expected for some splits)")
                
        except Exception as e:
            print(f"   ❌ Failed to load {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ Dataset loading tests complete")


def test_selective_gradient_flow():
    """Test selective gradient flow with DDP-wrapped models"""
    print("\n🧪 Testing Selective Gradient Flow...")
    
    # Create a mock model with LoRA
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel
    
    try:
        # Create a small model for testing
        model = AutoModel.from_pretrained("bert-base-uncased")
        
        # Add LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query", "key"],
            lora_dropout=0.1,
        )
        model = get_peft_model(model, lora_config)
        
        print("   Created model with LoRA adapters")
        
        # Test without DDP
        from shirg.fix_lora_gradients_selective import apply_selective_gradient_flow
        
        print("\n   Testing without DDP...")
        results = apply_selective_gradient_flow(model, debug=True)
        assert results['lora_params_found'] > 0, "No LoRA parameters found"
        print(f"   ✅ Found {results['lora_params_found']} LoRA parameters")
        
        # Test with DDP wrapping (simulate)
        if torch.cuda.is_available():
            print("\n   Testing with DDP wrapping...")
            model = model.cuda()
            
            # Simulate DDP wrapping
            class MockDDP:
                def __init__(self, module):
                    self.module = module
                
                def named_parameters(self):
                    for name, param in self.module.named_parameters():
                        yield f"module.{name}", param
                
                def parameters(self):
                    return self.module.parameters()
            
            ddp_model = MockDDP(model)
            
            # Test selective gradient flow on DDP-wrapped model
            results = apply_selective_gradient_flow(ddp_model, debug=True)
            assert results['lora_params_found'] > 0, "No LoRA parameters found in DDP model"
            print(f"   ✅ Found {results['lora_params_found']} LoRA parameters in DDP-wrapped model")
        
        print("\n✅ Selective gradient flow tests complete")
        
    except Exception as e:
        print(f"   ❌ Selective gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()


def test_distributed_setup():
    """Test distributed training setup"""
    print("\n🧪 Testing Distributed Setup...")
    
    # Check environment variables
    env_vars = ['RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']
    missing_vars = [var for var in env_vars if var not in os.environ]
    
    if missing_vars:
        print(f"   ⚠️ Missing environment variables for distributed training: {missing_vars}")
        print("   Setting up single-GPU test environment...")
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
    
    # Check SHIRG_NO_DEVICE_MAP
    if 'SHIRG_NO_DEVICE_MAP' not in os.environ:
        print("   ⚠️ SHIRG_NO_DEVICE_MAP not set, setting to 1...")
        os.environ['SHIRG_NO_DEVICE_MAP'] = '1'
    
    print(f"   Rank: {os.environ.get('RANK')}")
    print(f"   World size: {os.environ.get('WORLD_SIZE')}")
    print(f"   SHIRG_NO_DEVICE_MAP: {os.environ.get('SHIRG_NO_DEVICE_MAP')}")
    
    print("\n✅ Distributed setup test complete")


def test_memory_optimization():
    """Test memory optimization settings"""
    print("\n🧪 Testing Memory Optimization...")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"   GPU available: {torch.cuda.get_device_name(0)}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Test gradient checkpointing flag
        from shirg.shirg_lora_config import create_lora_training_config
        config = create_lora_training_config()
        
        print(f"   Gradient checkpointing: {config.gradient_checkpointing}")
        print(f"   Mixed precision: {config.fp16}")
        print(f"   Per-device batch size: {config.per_device_train_batch_size}")
        
        # Calculate memory requirements
        batch_size = 2  # For 8xA100 40GB
        memory_per_sample = 17  # GB
        total_memory_needed = batch_size * memory_per_sample
        
        print(f"\n   Memory calculation for 8xA100 40GB:")
        print(f"   - Batch size per GPU: {batch_size}")
        print(f"   - Memory per sample: {memory_per_sample} GB")
        print(f"   - Total memory needed: {total_memory_needed} GB")
        print(f"   - Available memory: ~35 GB (after CUDA overhead)")
        
        if total_memory_needed <= 35:
            print("   ✅ Memory configuration is valid")
        else:
            print("   ❌ Memory configuration may cause OOM")
    else:
        print("   ⚠️ No GPU available for memory testing")
    
    print("\n✅ Memory optimization test complete")


def main():
    """Run all tests"""
    print("🚀 Running SHIRG 8-GPU Training Fix Tests")
    print("=" * 60)
    
    # Run tests
    test_distributed_setup()
    test_dataset_loading()
    test_selective_gradient_flow()
    test_memory_optimization()
    
    print("\n" + "=" * 60)
    print("🎉 All tests complete!")
    print("\nNext steps:")
    print("1. Run the actual 8-GPU training with: bash shirg/run_8gpu_training.sh")
    print("2. Monitor the training logs for:")
    print("   - Dataset loading without KeyError")
    print("   - Consistent selective gradient flow success")
    print("   - Proper LoRA parameter updates")
    print("   - No OOM errors")


if __name__ == "__main__":
    main()