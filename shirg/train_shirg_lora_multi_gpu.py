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
    
    def __init__(self, *args, **kwargs):
        # Initialize distributed training
        self.setup_distributed()
        
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
            
            # Validation sampler
            val_sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=False,
                drop_last=False,
            )
            
            # Recreate dataloaders with distributed samplers
            from torch.utils.data import DataLoader
            
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.config.per_device_train_batch_size,
                sampler=train_sampler,
                num_workers=self.config.dataloader_num_workers,
                pin_memory=self.config.dataloader_pin_memory,
                collate_fn=self.collate_fn,
                drop_last=True,
            )
            
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.config.per_device_train_batch_size,
                sampler=val_sampler,
                num_workers=self.config.dataloader_num_workers,
                pin_memory=self.config.dataloader_pin_memory,
                collate_fn=self.collate_fn,
            )
            
            print(f"✅ Distributed samplers configured")
        
        return len(self.train_dataloader)
    
    def setup_model(self):
        """Setup model with DDP wrapper"""
        # Call parent method (which includes the LoRA gradient fix)
        super().setup_model()
        
        # SHIRG-FIX: 2025-07-30 - Apply enhanced gradient fix for multi-GPU
        # ISSUE: Device mismatches more common in multi-GPU setups
        # SOLUTION: Ensure comprehensive device and gradient fixes
        # LAVIDA IMPACT: Consistent device placement across GPUs
        # SHIRG IMPACT: Enables proper gradient flow in distributed training
        try:
            from shirg.fix_lora_gradients_enhanced import apply_comprehensive_fix
            
            # Apply comprehensive fix before DDP wrapping
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            device_str = f"cuda:{local_rank}"
            
            rank0_print(f"🔧 Applying comprehensive LoRA fix for GPU {local_rank}...")
            apply_comprehensive_fix(self, force_device=device_str)
            
        except ImportError:
            rank0_print("⚠️ Enhanced gradient fix not available")
        
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
                self.model = DDP(
                    self.model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=True,  # Required for LoRA
                )
                
                rank0_print(f"✅ Model wrapped with DistributedDataParallel on GPU {local_rank}")
            else:
                rank0_print(f"✅ Model already wrapped with DDP")
    
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
    )
    
    try:
        # Start training
        trainer.train()
    finally:
        # Cleanup
        trainer.cleanup()


if __name__ == "__main__":
    main()