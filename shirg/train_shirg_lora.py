#!/usr/bin/env python3
"""
SHIRG LoRA Training Script for Lambda Cloud
Main training script for SHIRG Extra-LoRA adaptation on LaViDa

This script handles the complete training pipeline including:
- Model loading with SHIRG integration
- LoRA setup with Extra-LoRA footprint
- Dataset loading and preprocessing
- Training with token dropout
- Checkpointing and evaluation
- Performance monitoring

Author: Research Implementation
Date: 2025-07-30
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import time
import json
import argparse
from tqdm import tqdm
import wandb
from datetime import datetime
import gc

# Add paths
sys.path.append('./')
sys.path.append('./shirg')

# Import SHIRG components
from shirg.shirg_lora_config import ShirgLoraConfig, create_lora_training_config
from shirg.shirg_token_dropout import ShirgTokenDropout, ShirgDropoutScheduler
from shirg.lavida_shirg_integration import LaViDaSHIRGWrapper
from shirg.dataset_loaders import create_data_loaders, MixedVQADataset

# Import HuggingFace components
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator
from datasets import load_dataset


# Dataset loading is now handled by dataset_loaders.py


class ShirgLoraTrainer:
    """Main trainer class for SHIRG LoRA"""
    
    def __init__(
        self,
        config: ShirgLoraConfig,
        model_path: str = "KonstantinosKK/lavida-llada-v1.0-instruct-hf-transformers",
        output_dir: str = "./shirg_lora_checkpoints",
        use_wandb: bool = True,
    ):
        """
        Initialize trainer
        
        Args:
            config: Training configuration
            model_path: LaViDa model path
            output_dir: Output directory for checkpoints
            use_wandb: Whether to use Weights & Biases logging
        """
        self.config = config
        self.model_path = model_path
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize accelerator for distributed training
        self.accelerator = Accelerator(
            mixed_precision="bf16" if config.bf16 else "fp16" if config.fp16 else "no",
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.dropout_scheduler = None
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_loss = float('inf')
        
        # Initialize wandb if requested
        if use_wandb and self.accelerator.is_main_process:
            self._init_wandb()
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging"""
        wandb.init(
            project="shirg-lora-training",
            name=f"shirg_{self.config.shirg_method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=self.config.__dict__,
        )
    
    def setup_model(self):
        """Setup LaViDa model with SHIRG and LoRA"""
        print("🔧 Setting up LaViDa-SHIRG model with LoRA...")
        
        # Create SHIRG wrapper
        self.wrapper = LaViDaSHIRGWrapper(
            model_path=self.model_path,
            shirg_config={
                'target_tokens': 980,
                'alpha': 0.3,  # Enable SHIRG
                'debug': False,
            },
            selection_method=self.config.shirg_method,
            selection_params={
                'entropy_threshold': self.config.shirg_entropy_threshold,
                'edge_weight': self.config.shirg_edge_weight,
                'radial_sigma': self.config.shirg_radial_sigma,
                'merge_similar': self.config.shirg_merge_similar,
                'merge_threshold': self.config.shirg_merge_threshold,
            },
        )
        
        # Load model
        self.wrapper.load_model()
        self.model = self.wrapper.model
        self.tokenizer = self.wrapper.tokenizer
        
        # Ensure vision tower has SHIRG enabled
        vision_tower = self.model.get_model().get_vision_tower()
        if hasattr(vision_tower, 'config'):
            vision_tower.config.enable_shirg = True
            vision_tower.config.shirg_selection_method = self.config.shirg_method
            vision_tower.config.shirg_selection_params = {
                'entropy_threshold': self.config.shirg_entropy_threshold,
                'edge_weight': self.config.shirg_edge_weight,
                'radial_sigma': self.config.shirg_radial_sigma,
                'merge_similar': self.config.shirg_merge_similar,
                'merge_threshold': self.config.shirg_merge_threshold,
            }
        
        # Debug: Find actual module paths in the model
        print("🔍 Discovering model structure for LoRA targeting...")
        base_model = self.model.get_model() if hasattr(self.model, 'get_model') else self.model
        
        # Find projector and vision modules
        projector_modules = []
        vision_modules = []
        
        for name, module in base_model.named_modules():
            if 'mm_projector' in name and isinstance(module, nn.Linear):
                projector_modules.append(name)
                print(f"   Found projector: {name}")
            elif 'vision_tower' in name and 'self_attn' in name and any(suffix in name for suffix in ['q_proj', 'k_proj', 'v_proj']):
                vision_modules.append(name)
        
        # Create corrected target modules
        corrected_target_modules = []
        
        # Add projector modules
        for module_name in projector_modules:
            if 'fc1' in module_name or 'fc2' in module_name:
                corrected_target_modules.append(module_name)
        
        # Add vision tower modules (blocks 0-5)
        for i in range(6):
            for proj in ['q_proj', 'k_proj', 'v_proj'] if i < 4 else ['q_proj', 'k_proj']:
                pattern = f"layers.{i}.self_attn.{proj}"
                matching = [m for m in vision_modules if pattern in m]
                if matching:
                    corrected_target_modules.extend(matching)
        
        # Update config if we found modules
        if corrected_target_modules:
            print(f"✅ Found {len(corrected_target_modules)} target modules")
            self.config.target_modules = corrected_target_modules
        else:
            # Try without model prefix
            print("⚠️ Trying alternative module paths without 'model.' prefix...")
            self.config.target_modules = [m.replace('model.', '') for m in self.config.target_modules]
        
        # Apply LoRA with Extra-LoRA footprint
        lora_config = self.config.to_peft_config()
        try:
            self.model = get_peft_model(self.model, lora_config)
        except ValueError as e:
            print(f"❌ LoRA application failed: {e}")
            # Last resort: try with most common patterns
            fallback_modules = [
                "mm_projector.fc1",
                "mm_projector.fc2",
                "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj",
                "vision_tower.vision_model.encoder.layers.0.self_attn.k_proj",
                "vision_tower.vision_model.encoder.layers.0.self_attn.v_proj",
            ]
            print(f"   Trying fallback modules: {fallback_modules[:3]}...")
            lora_config.target_modules = fallback_modules
            self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        # Setup token dropout
        self.token_dropout = ShirgTokenDropout(
            dropout_rate=self.config.token_dropout_rate,
            structured_dropout=True,
            spatial_consistency=True,
        )
        
        # Setup dropout scheduler
        total_steps = self.config.num_train_epochs * 1000  # Estimate
        self.dropout_scheduler = ShirgDropoutScheduler(
            initial_dropout=self.config.token_dropout_rate,
            final_dropout=0.0,
            warmup_steps=self.config.warmup_steps,
            total_steps=total_steps,
            schedule_type="cosine",
        )
        
        print("✅ Model setup complete")
    
    def setup_optimizer_scheduler(self, num_training_steps: int):
        """Setup optimizer and learning rate scheduler"""
        print("🔧 Setting up optimizer and scheduler...")
        
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay,
        )
        
        # Setup scheduler
        if self.config.lr_scheduler_type == "linear":
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=num_training_steps,
            )
        elif self.config.lr_scheduler_type == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=num_training_steps,
            )
        else:
            self.scheduler = None
        
        print(f"✅ Optimizer: AdamW (lr={self.config.learning_rate})")
        print(f"✅ Scheduler: {self.config.lr_scheduler_type}")
    
    def prepare_datasets(self):
        """Prepare training and validation datasets"""
        print("📊 Preparing datasets...")
        
        # Dataset configuration from research
        dataset_configs = {
            "chartqa": {"weight": 0.3, "max_samples": 10000},
            "docvqa": {"weight": 0.3, "max_samples": 10000},
            "vqa_v2": {"weight": 0.4, "max_samples": 10000},
        }
        
        # Create mixed datasets using proper loaders
        self.train_dataset = MixedVQADataset(
            split="train",
            dataset_configs=dataset_configs,
            image_size=self.config.image_size,
            cache_dir=os.path.join(self.output_dir, "data_cache"),
        )
        
        # Validation dataset with fewer samples
        val_configs = {k: {"weight": v["weight"], "max_samples": 1000} 
                      for k, v in dataset_configs.items()}
        self.val_dataset = MixedVQADataset(
            split="validation",
            dataset_configs=val_configs,
            image_size=self.config.image_size,
            cache_dir=os.path.join(self.output_dir, "data_cache"),
        )
        
        print(f"✅ Training samples: {len(self.train_dataset)}")
        print(f"✅ Validation samples: {len(self.val_dataset)}")
        
        # Create dataloaders
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.dataloader_pin_memory,
            collate_fn=self.collate_fn,
        )
        
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=False,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.dataloader_pin_memory,
            collate_fn=self.collate_fn,
        )
        
        return len(self.train_dataloader)
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """Custom collate function for batching"""
        # Extract components
        images = [item["image"] for item in batch]
        questions = [item["question"] for item in batch]
        answers = [item["answer"] for item in batch]
        ids = [item["id"] for item in batch]
        
        # Process images
        try:
            from llava.mm_utils import process_images
            image_tensors = process_images(images, self.wrapper.image_processor, self.model.config)
        except Exception as e:
            # Fallback if LaViDa processing fails
            print(f"⚠️ LaViDa image processing failed: {e}")
            # Convert PIL images to tensors manually
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image_tensors = torch.stack([transform(img) for img in images])
        
        # Process text
        conversations = []
        for question, answer in zip(questions, answers):
            conv = f"{question}\n{answer}"
            conversations.append(conv)
        
        # Tokenize
        encodings = self.tokenizer(
            conversations,
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt",
        )
        
        return {
            "images": image_tensors,
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": encodings["input_ids"].clone(),  # For language modeling
            "ids": ids,
        }
    
    def training_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single training step"""
        # Move batch to device
        batch = {k: v.to(self.accelerator.device) if torch.is_tensor(v) else v 
                for k, v in batch.items()}
        
        # Apply token dropout if in training mode
        if self.model.training:
            dropout_rate = self.dropout_scheduler.get_dropout_rate(self.global_step)
            self.token_dropout.dropout_rate = dropout_rate
        
        # Forward pass
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            images=batch["images"],
            labels=batch["labels"],
        )
        
        loss = outputs.loss
        
        # Backward pass
        self.accelerator.backward(loss)
        
        # Gradient clipping
        if self.config.max_grad_norm > 0:
            self.accelerator.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
        
        # Optimizer step
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad()
        
        # Update global step
        self.global_step += 1
        
        # Collect metrics
        metrics = {
            "loss": loss.item(),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "dropout_rate": self.token_dropout.dropout_rate,
        }
        
        return metrics
    
    def validation_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single validation step"""
        # Move batch to device
        batch = {k: v.to(self.accelerator.device) if torch.is_tensor(v) else v 
                for k, v in batch.items()}
        
        # Forward pass (no dropout in eval mode)
        with torch.no_grad():
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                images=batch["images"],
                labels=batch["labels"],
            )
        
        loss = outputs.loss
        
        return {"val_loss": loss.item()}
    
    def train(self):
        """Main training loop"""
        print("🚀 Starting SHIRG LoRA training...")
        
        # Setup model and optimizer
        self.setup_model()
        num_training_steps = self.prepare_datasets() * self.config.num_train_epochs
        self.setup_optimizer_scheduler(num_training_steps)
        
        # Prepare for distributed training
        self.model, self.optimizer, self.train_dataloader, self.val_dataloader = \
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader, self.val_dataloader
            )
        
        # Training loop
        for epoch in range(self.config.num_train_epochs):
            self.current_epoch = epoch
            print(f"\n📅 Epoch {epoch + 1}/{self.config.num_train_epochs}")
            
            # Training
            self.model.train()
            train_metrics = []
            
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Training Epoch {epoch + 1}",
                disable=not self.accelerator.is_local_main_process,
            )
            
            for step, batch in enumerate(progress_bar):
                metrics = self.training_step(batch)
                train_metrics.append(metrics)
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "lr": f"{metrics['learning_rate']:.2e}",
                    "dropout": f"{metrics['dropout_rate']:.2f}",
                })
                
                # Log metrics
                if self.global_step % self.config.logging_steps == 0:
                    self.log_metrics(metrics, prefix="train")
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
                
                # Evaluate
                if self.global_step % self.config.eval_steps == 0:
                    self.evaluate()
            
            # End of epoch evaluation
            self.evaluate()
            
            # Save epoch checkpoint
            self.save_checkpoint(is_epoch_end=True)
        
        print("✅ Training complete!")
        
        # Save final model
        self.save_checkpoint(is_final=True)
        
        # Cleanup
        if self.use_wandb and self.accelerator.is_main_process:
            wandb.finish()
    
    def evaluate(self):
        """Run evaluation on validation set"""
        print("\n📊 Running evaluation...")
        
        self.model.eval()
        val_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(
                self.val_dataloader,
                desc="Evaluating",
                disable=not self.accelerator.is_local_main_process,
            ):
                metrics = self.validation_step(batch)
                val_metrics.append(metrics)
        
        # Aggregate metrics
        avg_val_loss = np.mean([m["val_loss"] for m in val_metrics])
        
        # Log metrics
        eval_metrics = {"val_loss": avg_val_loss}
        self.log_metrics(eval_metrics, prefix="eval")
        
        print(f"📊 Validation loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < self.best_eval_loss:
            self.best_eval_loss = avg_val_loss
            self.save_checkpoint(is_best=True)
            print(f"🏆 New best validation loss: {avg_val_loss:.4f}")
        
        self.model.train()
    
    def save_checkpoint(
        self,
        is_epoch_end: bool = False,
        is_best: bool = False,
        is_final: bool = False,
    ):
        """Save training checkpoint"""
        if not self.accelerator.is_main_process:
            return
        
        # Determine checkpoint name
        if is_final:
            checkpoint_name = "checkpoint-final"
        elif is_best:
            checkpoint_name = "checkpoint-best"
        elif is_epoch_end:
            checkpoint_name = f"checkpoint-epoch-{self.current_epoch + 1}"
        else:
            checkpoint_name = f"checkpoint-{self.global_step}"
        
        checkpoint_dir = os.path.join(self.output_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print(f"💾 Saving checkpoint to {checkpoint_dir}")
        
        # Save model state
        self.accelerator.wait_for_everyone()
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        # Save LoRA weights
        unwrapped_model.save_pretrained(checkpoint_dir)
        
        # Save training state
        training_state = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_eval_loss": self.best_eval_loss,
            "config": self.config.__dict__,
        }
        
        state_path = os.path.join(checkpoint_dir, "training_state.json")
        with open(state_path, "w") as f:
            json.dump(training_state, f, indent=2)
        
        # Clean up old checkpoints if needed
        if not (is_best or is_final) and self.config.save_total_limit > 0:
            self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to save space"""
        # Get all checkpoint directories
        checkpoints = []
        for name in os.listdir(self.output_dir):
            if name.startswith("checkpoint-") and name not in ["checkpoint-best", "checkpoint-final"]:
                path = os.path.join(self.output_dir, name)
                if os.path.isdir(path):
                    # Extract step number
                    try:
                        if "epoch" in name:
                            continue  # Keep epoch checkpoints
                        step = int(name.split("-")[-1])
                        checkpoints.append((step, path))
                    except:
                        pass
        
        # Sort by step number
        checkpoints.sort(key=lambda x: x[0])
        
        # Remove oldest checkpoints
        while len(checkpoints) > self.config.save_total_limit:
            _, path = checkpoints.pop(0)
            print(f"🗑️ Removing old checkpoint: {path}")
            import shutil
            shutil.rmtree(path)
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """Log metrics to console and wandb"""
        # Add prefix
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # Log to wandb
        if self.use_wandb and self.accelerator.is_main_process:
            wandb.log(metrics, step=self.global_step)
        
        # Log to console (only first process)
        if self.accelerator.is_local_main_process and self.config.logging_first_step:
            if self.global_step == 1 or self.global_step % 100 == 0:
                print(f"Step {self.global_step}: {metrics}")


def main():
    """Main training entry point"""
    parser = argparse.ArgumentParser(description="SHIRG LoRA Training")
    
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
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Override batch size")
    parser.add_argument("--learning-rate", type=float, default=None,
                       help="Override learning rate")
    parser.add_argument("--num-epochs", type=int, default=None,
                       help="Override number of epochs")
    
    # Other arguments
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable Weights & Biases logging")
    parser.add_argument("--resume-from", type=str, default=None,
                       help="Resume from checkpoint")
    parser.add_argument("--save-samples-interval", type=int, default=None,
                       help="Save checkpoint every N samples (e.g., 5000)")
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_lora_training_config(
        selection_method=args.selection_method,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        save_samples_interval=args.save_samples_interval,
    )
    
    # Create trainer
    trainer = ShirgLoraTrainer(
        config=config,
        model_path=args.model_path,
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb,
    )
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()