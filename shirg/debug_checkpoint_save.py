#!/usr/bin/env python3
"""
Debug script to test checkpoint saving in multi-GPU environment
Run this to diagnose checkpoint saving issues

Usage:
    torchrun --nproc_per_node=8 shirg/debug_checkpoint_save.py
"""

import os
import sys
import torch
import torch.distributed as dist
import time
import shutil
from pathlib import Path

def test_checkpoint_save():
    """Test checkpoint saving with multi-GPU setup"""
    
    # Initialize distributed if available
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        print(f"[Rank {rank}] Initialized distributed training")
    else:
        rank = 0
        world_size = 1
        print("Single GPU mode")
    
    # Test directory
    test_dir = "./debug_checkpoints"
    
    # Test 1: Basic file I/O
    print(f"\n[Rank {rank}] Test 1: Basic file I/O")
    if rank == 0:
        os.makedirs(test_dir, exist_ok=True)
        test_file = os.path.join(test_dir, "test.txt")
        try:
            with open(test_file, "w") as f:
                f.write("Test checkpoint save")
            print(f"[Rank {rank}] ✅ File write successful")
        except Exception as e:
            print(f"[Rank {rank}] ❌ File write failed: {e}")
    
    if world_size > 1:
        dist.barrier()
    
    # Test 2: Directory operations
    print(f"\n[Rank {rank}] Test 2: Directory operations")
    if rank == 0:
        for i in range(5):
            checkpoint_dir = os.path.join(test_dir, f"checkpoint-{i*100}")
            try:
                os.makedirs(checkpoint_dir, exist_ok=True)
                # Write a dummy file
                with open(os.path.join(checkpoint_dir, "model.bin"), "w") as f:
                    f.write("dummy" * 1000)
                print(f"[Rank {rank}] ✅ Created {checkpoint_dir}")
            except Exception as e:
                print(f"[Rank {rank}] ❌ Failed to create {checkpoint_dir}: {e}")
    
    if world_size > 1:
        dist.barrier()
    
    # Test 3: Cleanup operations (similar to _cleanup_checkpoints)
    print(f"\n[Rank {rank}] Test 3: Cleanup operations")
    if rank == 0:
        try:
            # List checkpoints
            checkpoints = []
            for name in os.listdir(test_dir):
                if name.startswith("checkpoint-"):
                    path = os.path.join(test_dir, name)
                    if os.path.isdir(path):
                        step = int(name.split("-")[-1])
                        checkpoints.append((step, path))
            
            checkpoints.sort(key=lambda x: x[0])
            print(f"[Rank {rank}] Found {len(checkpoints)} checkpoints")
            
            # Remove oldest
            if len(checkpoints) > 3:
                _, path = checkpoints[0]
                print(f"[Rank {rank}] Removing {path}...")
                start_time = time.time()
                shutil.rmtree(path)
                elapsed = time.time() - start_time
                print(f"[Rank {rank}] ✅ Removed in {elapsed:.2f}s")
        except Exception as e:
            print(f"[Rank {rank}] ❌ Cleanup failed: {e}")
    
    if world_size > 1:
        dist.barrier()
    
    # Test 4: Synchronization with delays
    print(f"\n[Rank {rank}] Test 4: Synchronization test")
    if rank == 0:
        print(f"[Rank {rank}] Simulating slow checkpoint save...")
        time.sleep(2)  # Simulate slow save
    
    if world_size > 1:
        print(f"[Rank {rank}] Waiting at barrier...")
        start_time = time.time()
        dist.barrier()
        elapsed = time.time() - start_time
        print(f"[Rank {rank}] ✅ Barrier passed in {elapsed:.2f}s")
    
    # Test 5: PyTorch save
    print(f"\n[Rank {rank}] Test 5: PyTorch checkpoint save")
    if rank == 0:
        checkpoint = {
            "model_state": {"dummy": torch.randn(100, 100)},
            "optimizer_state": {"dummy": torch.randn(50, 50)},
            "step": 500,
        }
        checkpoint_path = os.path.join(test_dir, "checkpoint-500", "pytorch_model.bin")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        try:
            start_time = time.time()
            torch.save(checkpoint, checkpoint_path)
            elapsed = time.time() - start_time
            size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
            print(f"[Rank {rank}] ✅ PyTorch save successful: {size_mb:.2f}MB in {elapsed:.2f}s")
        except Exception as e:
            print(f"[Rank {rank}] ❌ PyTorch save failed: {e}")
    
    if world_size > 1:
        dist.barrier()
    
    # Cleanup
    if rank == 0:
        try:
            shutil.rmtree(test_dir)
            print(f"\n[Rank {rank}] ✅ Cleanup successful")
        except Exception as e:
            print(f"\n[Rank {rank}] ⚠️ Cleanup failed: {e}")
    
    # Clean shutdown
    if world_size > 1:
        dist.destroy_process_group()
    
    print(f"\n[Rank {rank}] All tests completed!")


if __name__ == "__main__":
    test_checkpoint_save()