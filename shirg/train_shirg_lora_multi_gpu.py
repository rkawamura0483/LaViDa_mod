#!/usr/bin/env python3
"""
SHIRG LoRA Multi-GPU Training Script for Lambda Cloud
Optimized for 8x A100 GPUs with distributed training

Author: Research Implementation
Date: 2025-07-30
"""

import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse

# SHIRG-FIX: 2025-07-30 - Set multiprocessing start method early for CUDA compatibility
# ISSUE: Default 'fork' method causes "Cannot re-initialize CUDA in forked subprocess" error
# SOLUTION: Use 'spawn' method before any CUDA operations or imports
# LAVIDA IMPACT: Prevents DataLoader worker crashes in multi-GPU training
# SHIRG IMPACT: Fixes the fatal multiprocessing error during first training step
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass

# Add paths
sys.path.append('./')
sys.path.append('./shirg')

from shirg.train_shirg_lora import ShirgLoraTrainer

# SHIRG-FIX: 2025-07-30 - Import rank0_print locally
# ISSUE: Need rank0_print for distributed logging
# SOLUTION: Define it here to avoid import issues
# LAVIDA IMPACT: None - just utility function
# SHIRG IMPACT: Allows proper distributed logging
def rank0_print(msg):
    """Print only on rank 0 for distributed training"""
    try:
        from torch.distributed import get_rank, is_initialized
        if is_initialized() and get_rank() != 0:
            return
    except:
        pass
    print(msg)
from shirg.shirg_lora_config import ShirgLoraConfig, create_lora_training_config


class MultiGPUShirgTrainer(ShirgLoraTrainer):
    """Extended trainer for multi-GPU training on Lambda Cloud"""
    
    def __init__(self, *args, data_dir=None, skip_validation=False, **kwargs):
        # Store data directory
        self.data_dir = data_dir
        
        # Initialize distributed training
        self.setup_distributed()
        
        # Pass skip_validation to parent
        kwargs['skip_validation'] = skip_validation
        
        # Call parent init
        super().__init__(*args, **kwargs)
        
        # Override batch size for multi-GPU
        if dist.is_initialized():
            world_size = dist.get_world_size()
            # Scale effective batch size with number of GPUs
            self.config.per_device_train_batch_size = max(1, self.config.per_device_train_batch_size // world_size)
            print(f"📊 Adjusted batch size per GPU: {self.config.per_device_train_batch_size}")
            print(f"📊 Effective batch size: {self.config.per_device_train_batch_size * world_size}")
    
    def setup_distributed(self):
        """Setup distributed training environment"""
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            
            # Initialize distributed backend
            dist.init_process_group(backend='nccl')
            
            # Set device
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            torch.cuda.set_device(local_rank)
            
            print(f"🌐 Distributed training: Rank {rank}/{world_size}, Local rank: {local_rank}")
            
            # SHIRG-FIX: 2025-07-30 - Enable NO_DEVICE_MAP for distributed LoRA training
            # ISSUE: device_map breaks gradient flow in distributed LoRA training
            # SOLUTION: Automatically set SHIRG_NO_DEVICE_MAP=1 for distributed training
            # LAVIDA IMPACT: Each GPU loads full model for proper gradient flow
            # SHIRG IMPACT: Fixes zero gradient issue in multi-GPU training
            os.environ['SHIRG_NO_DEVICE_MAP'] = '1'
            if rank == 0:
                print("📍 Setting SHIRG_NO_DEVICE_MAP=1 for distributed LoRA training")
        else:
            print("📍 Single GPU training mode")
    
    def prepare_datasets(self):
        """Prepare datasets with distributed sampler"""
        # Call parent method
        super().prepare_datasets()
        
        # Replace samplers with distributed versions
        if dist.is_initialized():
            # Training sampler
            train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=True,
                drop_last=True,
            )
            
            # Validation sampler (only if validation is enabled)
            if not self.skip_validation:
                val_sampler = DistributedSampler(
                    self.val_dataset,
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=False,
                    drop_last=False,
                )
            
            # Recreate dataloaders with distributed samplers
            from torch.utils.data import DataLoader
            
            # SHIRG-FIX: 2025-07-30 - Configure multiprocessing context for distributed training
            # ISSUE: Distributed training needs spawn context for CUDA operations in workers
            # SOLUTION: Explicitly set multiprocessing context to spawn
            # LAVIDA IMPACT: Prevents worker process crashes in 8-GPU training
            # SHIRG IMPACT: Enables stable data loading across all GPU ranks
            mp_context = mp.get_context('spawn') if self.config.dataloader_num_workers > 0 else None
            
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.config.per_device_train_batch_size,
                sampler=train_sampler,
                num_workers=self.config.dataloader_num_workers,
                pin_memory=self.config.dataloader_pin_memory,
                collate_fn=self.collate_fn,
                drop_last=True,
                multiprocessing_context=mp_context,
                persistent_workers=(self.config.dataloader_num_workers > 0),
            )
            
            if not self.skip_validation:
                self.val_dataloader = DataLoader(
                    self.val_dataset,
                    batch_size=self.config.per_device_train_batch_size,
                    sampler=val_sampler,
                    num_workers=self.config.dataloader_num_workers,
                    pin_memory=self.config.dataloader_pin_memory,
                    collate_fn=self.collate_fn,
                    multiprocessing_context=mp_context,
                    persistent_workers=(self.config.dataloader_num_workers > 0),
                )
            
            print(f"✅ Distributed samplers configured")
        
        return len(self.train_dataloader)
    
    def setup_model(self):
        """Setup model with DDP wrapper"""
        # Call parent method (which includes the LoRA gradient fix)
        super().setup_model()
        
        # SHIRG-FIX: 2025-07-30 - Override optimizer to use only LoRA parameters
        # ISSUE: Parent optimizer may include base parameters with grad=True
        # SOLUTION: Recreate optimizer with only LoRA parameters
        # LAVIDA IMPACT: Only LoRA weights are updated during training
        # SHIRG IMPACT: Enables selective gradient flow pattern
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            from shirg.fix_lora_gradients_selective import get_lora_parameters_only
            
            # Get only LoRA parameters
            lora_params = get_lora_parameters_only(self.model)
            
            rank0_print(f"🔧 Recreating optimizer with {len(lora_params)} LoRA parameters only")
            
            # Get current optimizer settings
            old_optimizer = self.optimizer
            lr = old_optimizer.param_groups[0]['lr']
            betas = old_optimizer.param_groups[0].get('betas', (0.9, 0.999))
            weight_decay = old_optimizer.param_groups[0].get('weight_decay', 0.0)
            
            # Create new optimizer with only LoRA params
            self.optimizer = torch.optim.AdamW(
                lora_params,
                lr=lr,
                betas=betas,
                weight_decay=weight_decay
            )
            
            rank0_print(f"✅ Optimizer recreated with only LoRA parameters")
        
        # SHIRG-FIX: 2025-07-30 - Handle multi-GPU setup for 8 GPU training
        # ISSUE: Need proper handling for distributed model with device_map
        # SOLUTION: Check if model uses device_map and handle appropriately
        # LAVIDA IMPACT: Supports both single and multi-GPU training modes
        # SHIRG IMPACT: Enables 8 GPU distributed training
        
        # Wrap model with DDP
        if dist.is_initialized():
            # Find local rank
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            
            # SHIRG-FIX: 2025-07-30 - Force DDP for LoRA training
            # ISSUE: device_map model parallelism breaks LoRA gradient flow
            # SOLUTION: Always use DDP for LoRA training, never device_map
            # LAVIDA IMPACT: Each GPU loads full model for proper gradient flow
            # SHIRG IMPACT: Ensures all LoRA parameters receive gradients
            
            # Check if model has device_map (model parallelism)
            if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map is not None:
                # This shouldn't happen if SHIRG_NO_DEVICE_MAP=1 is set correctly
                rank0_print(f"⚠️ WARNING: Model has device_map but we need DDP for LoRA!")
                rank0_print(f"   Device map detected: {list(self.model.hf_device_map.keys())[:5]}...")
                rank0_print(f"   This will likely cause gradient flow issues!")
                rank0_print(f"   Please ensure SHIRG_NO_DEVICE_MAP=1 is set before model loading.")
                
                # Try to move model to single device anyway
                try:
                    # Move entire model to local GPU
                    self.model = self.model.cuda(local_rank)
                    rank0_print(f"   Attempted to move model to GPU {local_rank}")
                except Exception as e:
                    rank0_print(f"   ❌ Failed to move model: {e}")
                    rank0_print(f"   LoRA training will likely fail with zero gradients!")
            
            # Standard DDP setup for data parallelism
            if not hasattr(self.model, 'module'):  # Not already wrapped
                # Ensure model is on local GPU
                if not next(self.model.parameters()).is_cuda:
                    self.model = self.model.cuda(local_rank)
                
                # Wrap with DDP
                # SHIRG-FIX: 2025-07-30 - Set find_unused_parameters=False for LoRA training
                # ISSUE: find_unused_parameters=True causes performance overhead
                # SOLUTION: With LoRA, all parameters in forward pass are used
                # LAVIDA IMPACT: Better training performance
                # SHIRG IMPACT: Removes unnecessary autograd graph traversal
                self.model = DDP(
                    self.model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=False,  # LoRA uses all params in forward
                )
                
                rank0_print(f"✅ Model wrapped with DistributedDataParallel on GPU {local_rank}")
            else:
                rank0_print(f"✅ Model already wrapped with DDP")
        
        # SHIRG-FIX: 2025-07-30 - DISABLE selective gradient flow for DDP compatibility
        # ISSUE: Selective gradient flow causes "parameter marked ready twice" error
        # SOLUTION: Disable the fix and rely on PEFT's default gradient handling
        # LAVIDA IMPACT: May have lower gradients but avoids DDP conflicts
        # SHIRG IMPACT: Training should proceed without DDP errors
        
        disable_gradient_fix = True  # Set to False to re-enable
        
        if dist.is_initialized() and not disable_gradient_fix:
            try:
                from shirg.fix_lora_gradients_selective import (
                    apply_selective_gradient_flow,
                    get_lora_parameters_only,
                    verify_selective_gradient_flow,
                    apply_memory_optimizations
                )
                
                # Apply selective gradient flow fix AFTER DDP wrapping
                rank0_print(f"\n🔧 Applying selective gradient flow fix (post-DDP)...")
                results = apply_selective_gradient_flow(self.model, debug=(dist.get_rank() == 0))
                
                if results['success']:
                    rank0_print(f"✅ Selective gradient flow enabled")
                    rank0_print(f"   Base params with grad=True: {results['base_params_enabled']}")
                    rank0_print(f"   LoRA params found: {results['lora_params_found']}")
                    rank0_print(f"   Vision tower fixed: {results['vision_tower_fixed']}")
                else:
                    rank0_print(f"⚠️ Selective gradient flow fix failed")
                
                # Apply memory optimizations for multi-GPU
                apply_memory_optimizations(self.model, self.config.__dict__)
                
                # Verify the setup
                verify_results = verify_selective_gradient_flow(self.model, debug=(dist.get_rank() == 0))
                if not verify_results['setup_correct']:
                    rank0_print("⚠️ Setup verification failed - check optimizer configuration")
                
            except ImportError:
                rank0_print("⚠️ Selective gradient fix not available")
            except Exception as e:
                rank0_print(f"⚠️ Error applying selective gradient flow: {e}")
        elif dist.is_initialized():
            # Gradient fix disabled
            rank0_print("⚠️ Gradient flow fix disabled for DDP compatibility")
            rank0_print("   Using PEFT's default gradient handling")
    
    def save_checkpoint(self, *args, **kwargs):
        """Only save checkpoints from rank 0"""
        if not dist.is_initialized() or dist.get_rank() == 0:
            super().save_checkpoint(*args, **kwargs)
    
    def log_metrics(self, metrics, prefix=""):
        """Only log metrics from rank 0"""
        if not dist.is_initialized() or dist.get_rank() == 0:
            super().log_metrics(metrics, prefix)
    
    def cleanup(self):
        """Clean up distributed training"""
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    """Main entry point for multi-GPU training"""
    parser = argparse.ArgumentParser(description="SHIRG LoRA Multi-GPU Training")
    
    # Model arguments
    parser.add_argument("--model-path", type=str, 
                       default="KonstantinosKK/lavida-llada-v1.0-instruct-hf-transformers",
                       help="LaViDa model path")
    parser.add_argument("--output-dir", type=str, default="./shirg_lora_checkpoints",
                       help="Output directory for checkpoints")
    
    # Training arguments
    parser.add_argument("--selection-method", type=str, default="full",
                       choices=["base", "entropy", "edge", "full"],
                       help="SHIRG selection method")
    parser.add_argument("--batch-size", type=int, default=128,  # Larger for 8 GPUs
                       help="Total batch size (will be divided by number of GPUs)")
    parser.add_argument("--learning-rate", type=float, default=1.8e-5,
                       help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=3,
                       help="Number of epochs")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                       help="Gradient accumulation steps")
    
    # Distributed training arguments
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="Local rank for distributed training")
    
    # Other arguments
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable Weights & Biases logging")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of dataloader workers per GPU")
    parser.add_argument("--data-dir", type=str, default="./data/vqa_datasets",
                       help="Directory containing downloaded VQA datasets")
    parser.add_argument("--skip-validation", action="store_true",
                       help="Skip validation to save memory and prevent OOM")
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_lora_training_config(
        selection_method=args.selection_method,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
    )
    
    # Update config with multi-GPU settings
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.dataloader_num_workers = args.num_workers
    
    # Adjust learning rate for multi-GPU (linear scaling)
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
        if world_size > 1:
            # Linear LR scaling
            config.learning_rate = config.learning_rate * world_size
            print(f"📈 Scaled learning rate for {world_size} GPUs: {config.learning_rate}")
    
    # SHIRG-FIX: 2025-07-30 - Set environment variable before creating trainer
    # ISSUE: device_map needs to be disabled before model loading
    # SOLUTION: Set SHIRG_NO_DEVICE_MAP=1 before trainer initialization
    # LAVIDA IMPACT: Full model loaded per GPU for gradient flow
    # SHIRG IMPACT: Fixes zero gradient issue in distributed training
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        os.environ['SHIRG_NO_DEVICE_MAP'] = '1'
        if 'RANK' in os.environ and int(os.environ['RANK']) == 0:
            print("🔧 Set SHIRG_NO_DEVICE_MAP=1 for multi-GPU LoRA training")
    
    # Create trainer
    trainer = MultiGPUShirgTrainer(
        config=config,
        model_path=args.model_path,
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb,
        data_dir=args.data_dir,
        skip_validation=args.skip_validation,
    )
    
    try:
        # Start training
        trainer.train()
    finally:
        # Cleanup
        trainer.cleanup()


if __name__ == "__main__":
    main()