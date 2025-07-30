#!/usr/bin/env python3
"""
Test script to verify CUDA multiprocessing fixes for 8-GPU training

This script tests:
1. Multiprocessing start method is correctly set to 'spawn'
2. DataLoader with multiple workers doesn't crash
3. Collate function works without CUDA operations
4. Distributed training setup is compatible

Author: Research Implementation
Date: 2025-07-30
"""

import os
import sys
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Add paths
sys.path.append('./')
sys.path.append('./shirg')

print("="*80)
print("SHIRG CUDA Multiprocessing Fix Test")
print("="*80)

# Test 1: Check multiprocessing start method
print("\n1. Testing multiprocessing start method...")
try:
    current_method = mp.get_start_method()
    print(f"   Current start method: {current_method}")
    if current_method == 'spawn':
        print("   ✅ Spawn method is correctly set")
    else:
        print(f"   ⚠️ WARNING: Start method is '{current_method}', expected 'spawn'")
        print("   This may cause CUDA re-initialization errors!")
except RuntimeError as e:
    print(f"   ❌ Error getting start method: {e}")

# Test 2: Simple DataLoader test with workers
print("\n2. Testing DataLoader with multiprocessing workers...")

class DummyDataset(Dataset):
    """Simple dataset for testing"""
    def __init__(self, size=100):
        self.size = size
        self.data = torch.randn(size, 10)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            'input': self.data[idx],
            'label': torch.tensor(idx % 10)
        }

def test_collate_fn(batch):
    """Test collate function that doesn't use CUDA"""
    inputs = torch.stack([item['input'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    # IMPORTANT: Do not move to CUDA here!
    # This would cause the multiprocessing error
    
    return {
        'inputs': inputs,
        'labels': labels
    }

# Create test dataset and dataloader
test_dataset = DummyDataset(100)

# Test with different num_workers settings
for num_workers in [0, 2, 4]:
    print(f"\n   Testing with num_workers={num_workers}...")
    
    try:
        # Get multiprocessing context
        mp_context = mp.get_context('spawn') if num_workers > 0 else None
        
        # Create dataloader
        dataloader = DataLoader(
            test_dataset,
            batch_size=8,
            num_workers=num_workers,
            collate_fn=test_collate_fn,
            multiprocessing_context=mp_context,
            persistent_workers=(num_workers > 0) if num_workers > 0 else False,
            pin_memory=True
        )
        
        # Try to iterate through a few batches
        for i, batch in enumerate(dataloader):
            if i >= 3:  # Test first 3 batches
                break
            
            # Verify batch structure
            assert 'inputs' in batch
            assert 'labels' in batch
            assert batch['inputs'].shape == (8, 10)
            assert batch['labels'].shape == (8,)
            
            # Now it's safe to move to CUDA if available
            if torch.cuda.is_available():
                batch['inputs'] = batch['inputs'].cuda()
                batch['labels'] = batch['labels'].cuda()
        
        print(f"   ✅ DataLoader with {num_workers} workers works correctly")
        
    except Exception as e:
        print(f"   ❌ DataLoader with {num_workers} workers failed: {e}")

# Test 3: Import and test the actual training modules
print("\n3. Testing actual training module imports...")
try:
    from shirg.train_shirg_lora import ShirgLoraTrainer
    print("   ✅ Successfully imported ShirgLoraTrainer")
    
    # Check if spawn method is set after import
    method_after_import = mp.get_start_method()
    if method_after_import == 'spawn':
        print(f"   ✅ Spawn method still active after imports: {method_after_import}")
    else:
        print(f"   ⚠️ WARNING: Method changed to '{method_after_import}' after imports")
        
except ImportError as e:
    print(f"   ❌ Failed to import training modules: {e}")

# Test 4: Distributed training compatibility
print("\n4. Testing distributed training compatibility...")
if 'WORLD_SIZE' in os.environ:
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ.get('RANK', 0))
    print(f"   Distributed mode detected: World size={world_size}, Rank={rank}")
    
    # Check if SHIRG_NO_DEVICE_MAP is set
    if os.environ.get('SHIRG_NO_DEVICE_MAP') == '1':
        print("   ✅ SHIRG_NO_DEVICE_MAP=1 is correctly set")
    else:
        print("   ⚠️ WARNING: SHIRG_NO_DEVICE_MAP is not set to '1'")
        print("   This may cause gradient flow issues in LoRA training")
else:
    print("   Single GPU mode - distributed settings not applicable")

# Test 5: CUDA availability and memory
print("\n5. Testing CUDA environment...")
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"   ✅ CUDA is available with {num_gpus} GPU(s)")
    
    # Test basic CUDA operations
    try:
        test_tensor = torch.randn(100, 100)
        test_tensor_cuda = test_tensor.cuda()
        result = test_tensor_cuda @ test_tensor_cuda.T
        print("   ✅ Basic CUDA operations work correctly")
        
        # Check memory
        for i in range(min(num_gpus, 2)):  # Check first 2 GPUs
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / 1e9
            print(f"   GPU {i}: {props.name} with {total_memory:.1f}GB memory")
            
    except Exception as e:
        print(f"   ❌ CUDA operation failed: {e}")
else:
    print("   ⚠️ CUDA is not available - training will use CPU only")

# Summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)

# Provide recommendations
print("\nRecommendations for 8-GPU training:")
print("1. Ensure 'spawn' multiprocessing method is used (currently: {})".format(
    'YES' if mp.get_start_method() == 'spawn' else 'NO - NEEDS FIX!'
))
print("2. Use the updated training scripts with multiprocessing_context")
print("3. Keep SHIRG_NO_DEVICE_MAP=1 for proper LoRA gradient flow")
print("4. Monitor first few batches to ensure no worker crashes")
print("5. If crashes persist, try reducing num_workers or set to 0")

print("\n✅ Multiprocessing fix test completed!")
print("   If all tests passed, the training should work without CUDA re-initialization errors.")