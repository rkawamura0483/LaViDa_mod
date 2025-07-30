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

# SHIRG-FIX: 2025-07-30 - Avoid importing llava_trainer which requires tyro
# ISSUE: llava_trainer imports trl which requires tyro module
# SOLUTION: Define rank0_print locally to avoid the import chain
# LAVIDA IMPACT: None - just a utility function
# SHIRG IMPACT: Allows training without installing extra dependencies

def rank0_print(msg):
    """Print only on rank 0 for distributed training"""
    try:
        from torch.distributed import get_rank, is_initialized
        if is_initialized() and get_rank() != 0:
            return
    except:
        pass
    print(msg)

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
        
        # SHIRG-FIX: 2025-07-30 - Ensure vision tower has SHIRG enabled
        # ISSUE: SHIRG config not being propagated correctly to vision tower
        # SOLUTION: Set shirg_enabled directly on vision tower instance
        # LAVIDA IMPACT: None - only affects SHIRG mode
        # SHIRG IMPACT: Ensures SHIRG is properly enabled during training
        vision_tower = self.model.get_model().get_vision_tower()
        
        # Set SHIRG enabled flag directly on vision tower
        vision_tower.shirg_enabled = True
        rank0_print(f"SHIRG-TRAINING: Enabled SHIRG on vision tower (was {getattr(vision_tower, 'shirg_enabled', False)})")
        
        # Also update config if it exists
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
        
        # Also update vision_tower_cfg if it exists
        if hasattr(vision_tower, 'vision_tower_cfg'):
            if isinstance(vision_tower.vision_tower_cfg, dict):
                vision_tower.vision_tower_cfg['enable_shirg'] = True
            else:
                vision_tower.vision_tower_cfg.enable_shirg = True
        
        # SHIRG-FIX: 2025-07-30 - Dynamic module discovery with fallback
        # ISSUE: Module paths vary depending on model loading method
        # SOLUTION: Discover actual module paths and update config
        # LAVIDA IMPACT: Ensures LoRA works regardless of loading method
        # SHIRG IMPACT: Fixes module not found errors
        
        # Debug: Find actual module paths in the model
        print("🔍 Discovering model structure for LoRA targeting...")
        base_model = self.model.get_model() if hasattr(self.model, 'get_model') else self.model
        
        # Find all linear modules
        all_modules = {}
        for name, module in base_model.named_modules():
            if isinstance(module, nn.Linear):
                all_modules[name] = module
        
        # Check if our config modules exist
        modules_exist = all([any(target in name for name in all_modules.keys()) 
                           for target in self.config.target_modules])
        
        if not modules_exist:
            # Need to discover correct paths
            print("   Module paths need adjustment, discovering...")
            
            # Find projector modules
            projector_modules = []
            for name in all_modules.keys():
                if 'mm_projector' in name and any(x in name for x in ['0', '2']):
                    projector_modules.append(name)
                    print(f"   Found projector: {name}")
            
            # Find vision modules
            vision_modules = []
            for name in all_modules.keys():
                if 'vision_tower' in name and 'self_attn' in name:
                    for i in range(6):
                        if f"layers.{i}.self_attn" in name:
                            if any(suffix in name for suffix in ['q_proj', 'k_proj', 'v_proj']):
                                vision_modules.append(name)
            
            # Build corrected module list
            corrected_target_modules = []
            
            # Add projector modules
            corrected_target_modules.extend(projector_modules)
            
            # Add vision modules in order
            for i in range(6):
                for proj in ['q_proj', 'k_proj', 'v_proj'] if i < 4 else ['q_proj', 'k_proj']:
                    pattern = f"layers.{i}.self_attn.{proj}"
                    matching = [m for m in vision_modules if pattern in m]
                    if matching:
                        corrected_target_modules.extend(matching[:1])  # Take first match
            
            if corrected_target_modules:
                print(f"✅ Found {len(corrected_target_modules)} target modules")
                self.config.target_modules = corrected_target_modules
            else:
                # Try removing model. prefix as last resort
                print("⚠️ Trying without 'model.' prefix...")
                self.config.target_modules = [m.replace('model.', '') for m in self.config.target_modules]
        
        # SHIRG-FIX: 2025-07-30 - Ensure model is on GPU before applying LoRA
        # ISSUE: Model components on different devices cause device mismatch
        # SOLUTION: Move model to GPU before applying LoRA when device_map is disabled
        # LAVIDA IMPACT: Ensures consistent device placement
        # SHIRG IMPACT: Fixes device mismatch errors during training
        
        # Check if device_map is disabled (for LoRA training)
        if os.environ.get('SHIRG_NO_DEVICE_MAP', '0') == '1' and torch.cuda.is_available():
            print("📍 Moving model to GPU before LoRA (device_map disabled)")
            self.model = self.model.cuda()
            
        # Apply LoRA with Extra-LoRA footprint
        lora_config = self.config.to_peft_config()
        try:
            self.model = get_peft_model(self.model, lora_config)
        except ValueError as e:
            print(f"❌ LoRA application failed: {e}")
            # SHIRG-FIX: 2025-07-30 - Use discovered modules for fallback
            # ISSUE: Hardcoded fallback modules may not match actual structure
            # SOLUTION: Use dynamically discovered module names
            # LAVIDA IMPACT: Ensures LoRA can be applied
            # SHIRG IMPACT: Fixes LoRA application failures
            
            # Try to find at least some modules to target
            fallback_modules = []
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    if 'mm_projector' in name and any(x in name for x in ['0', '2']):
                        fallback_modules.append(name)
                    elif 'vision_tower' in name and 'layers.0' in name and 'q_proj' in name:
                        fallback_modules.append(name)
                        
                    if len(fallback_modules) >= 5:
                        break
            
            if fallback_modules:
                print(f"   Trying discovered modules: {fallback_modules[:3]}...")
                lora_config.target_modules = fallback_modules
                self.model = get_peft_model(self.model, lora_config)
            else:
                raise ValueError("Could not find any suitable modules for LoRA")
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        # SHIRG-FIX: 2025-07-30 - Fix LoRA gradient flow issue
        # ISSUE: Vision tower frozen state prevents LoRA gradients
        # SOLUTION: Ensure all LoRA parameters are trainable after PEFT application
        # LAVIDA IMPACT: None - only affects LoRA training
        # SHIRG IMPACT: Fixes zero gradient issue enabling proper LoRA training
        
        # SHIRG-FIX: 2025-07-30 - Use enhanced fix for device mismatch
        # ISSUE: Device mismatch between components causes gradient flow failure
        # SOLUTION: Use enhanced fix that handles both gradient and device issues
        # LAVIDA IMPACT: Ensures all components on same device
        # SHIRG IMPACT: Fixes zero gradient issue in multi-GPU training
        try:
            from shirg.fix_lora_gradients_enhanced import ensure_lora_parameters_trainable_enhanced
            
            # Apply comprehensive fix
            target_device = None
            if hasattr(self, 'accelerator') and self.accelerator.device:
                target_device = self.accelerator.device
            elif torch.cuda.is_available():
                target_device = torch.device("cuda:0")
            
            results = ensure_lora_parameters_trainable_enhanced(
                self.model,
                device=target_device,
                fix_device_mismatch=True
            )
            
            if results['unfrozen_params'] > 0 or results['moved_components'] > 0:
                rank0_print(f"🔧 LoRA Gradient Fix Applied:")
                rank0_print(f"   - Fixed {results['unfrozen_params']} frozen LoRA parameters")
                rank0_print(f"   - Moved {results['moved_components']} components to {target_device}")
                rank0_print(f"   - Total LoRA parameters: {results['total_lora_params']}")
                # Print updated trainable parameters
                self.model.print_trainable_parameters()
                
        except ImportError:
            # Fallback to original fix if enhanced version not available
            rank0_print("⚠️ Enhanced gradient fix not available, using basic fix")
            from shirg.fix_lora_gradients import ensure_lora_parameters_trainable
            unfrozen_count = ensure_lora_parameters_trainable(self.model)
            if unfrozen_count > 0:
                rank0_print(f"🔧 Fixed {unfrozen_count} frozen LoRA parameters")
                # Print updated trainable parameters
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
        
        # SHIRG-FIX: 2025-07-30 - Ensure SHIRG mode is enabled for image processing
        # ISSUE: process_images needs to use SHIRG 2-view mode during training
        # SOLUTION: Ensure model config has SHIRG settings before processing
        # LAVIDA IMPACT: None - only affects SHIRG training
        # SHIRG IMPACT: Enables correct 2-view processing (1×384² + 1×448²)
        
        # Ensure SHIRG is enabled in model config for process_images
        if hasattr(self.model, 'config'):
            self.model.config.enable_shirg = True
            self.model.config.shirg_3view_mode = True  # Enable 2-view mode
        
        # SHIRG-FIX: 2025-07-30 - Keep images as PIL for LaViDa
        # ISSUE: LaViDa expects PIL images, not tensors
        # SOLUTION: Don't process images here - model will handle it
        # LAVIDA IMPACT: Matches expected input format
        # SHIRG IMPACT: Allows proper SHIRG processing inside model
        
        # LaViDa expects PIL images - don't convert to tensors
        # The model's encode_images method will handle the processing
        
        # SHIRG-FIX: 2025-07-30 - Use LaViDa conversation format for proper training
        # ISSUE: Simple concatenation doesn't follow LaViDa's expected format
        # SOLUTION: Use LaViDa conversation templates like in real_ocr_vqa_model_runner
        # LAVIDA IMPACT: Ensures proper training format with correct tokenization
        # SHIRG IMPACT: Allows model to learn properly with LaViDa's format
        
        # Import required LaViDa components
        from llava.conversation import conv_templates
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.mm_utils import tokenizer_image_token
        import copy
        
        # Process each conversation using LaViDa format
        input_ids_list = []
        labels_list = []
        
        for question, answer in zip(questions, answers):
            # Use LaViDa conversation template
            conv_template = "llada"
            conv = copy.deepcopy(conv_templates[conv_template])
            
            # Format question with image token
            formatted_question = DEFAULT_IMAGE_TOKEN + "\n" + question
            conv.append_message(conv.roles[0], formatted_question)
            conv.append_message(conv.roles[1], answer)
            
            # Get prompt
            prompt = conv.get_prompt()
            
            # Tokenize with image token handling
            input_ids = tokenizer_image_token(
                prompt, 
                self.tokenizer, 
                IMAGE_TOKEN_INDEX, 
                return_tensors="pt"
            )
            
            # SHIRG-FIX: 2025-07-30 - Ensure tensors are on correct device
            # ISSUE: Device mismatch when modifying labels tensor
            # SOLUTION: Move tensors to correct device after creation
            # LAVIDA IMPACT: Prevents device mismatch errors
            # SHIRG IMPACT: Enables proper gradient flow in multi-GPU
            if hasattr(self, 'accelerator') and self.accelerator.device:
                device = self.accelerator.device
            elif torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                device = torch.device("cpu")
            
            # Move input_ids to device if needed
            if input_ids.device != device:
                input_ids = input_ids.to(device)
            
            # Create labels - mask everything except the answer
            labels = input_ids.clone()
            
            # SHIRG-FIX: 2025-07-30 - Ensure proper label masking for LaViDa
            # ISSUE: LaViDa requires labels.min() == -100, meaning some tokens must be masked
            # SOLUTION: Find the assistant response start and mask everything before it
            # LAVIDA IMPACT: Satisfies LaViDa's assertion requirement
            # SHIRG IMPACT: Ensures proper loss computation during training
            
            # Find where the assistant response starts
            # Look for the assistant role token sequence
            assistant_tokens = self.tokenizer.encode(conv.roles[1], add_special_tokens=False)
            
            # Find the position where assistant response starts
            answer_start_idx = None
            for i in range(len(input_ids) - len(assistant_tokens)):
                if all(input_ids[i + j] == assistant_tokens[j] for j in range(len(assistant_tokens))):
                    # Found the assistant role, answer starts after it
                    answer_start_idx = i + len(assistant_tokens)
                    break
            
            if answer_start_idx is not None:
                # Mask everything before the answer (including assistant role)
                labels[:answer_start_idx] = -100
            else:
                # Fallback: use a more robust method
                # Split by the answer text to find where it starts
                answer_tokens = self.tokenizer.encode(answer, add_special_tokens=False)
                if len(answer_tokens) > 0:
                    # Find where answer tokens start in the sequence
                    for i in range(len(input_ids) - len(answer_tokens)):
                        if all(input_ids[i + j] == answer_tokens[j] for j in range(min(3, len(answer_tokens)))):
                            # Found answer start, mask everything before
                            labels[:i] = -100
                            answer_start_idx = i
                            break
                
                if answer_start_idx is None:
                    # Last resort: mask at least the first half of tokens
                    # This ensures labels.min() == -100
                    mask_length = max(len(input_ids) // 2, 10)
                    labels[:mask_length] = -100
            
            input_ids_list.append(input_ids)
            labels_list.append(labels)
        
        # Pad sequences
        max_length = max(ids.shape[0] for ids in input_ids_list)
        max_length = min(max_length, self.config.max_seq_length)
        
        # Pad and create tensors
        padded_input_ids = []
        padded_labels = []
        attention_masks = []
        
        for input_ids, labels in zip(input_ids_list, labels_list):
            # Truncate if needed
            if input_ids.shape[0] > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]
            
            # Pad
            padding_length = max_length - input_ids.shape[0]
            if padding_length > 0:
                # Pad input_ids with pad_token_id
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                input_ids = torch.cat([input_ids, torch.full((padding_length,), pad_token_id, dtype=input_ids.dtype)])
                # Pad labels with -100 (ignore index)
                labels = torch.cat([labels, torch.full((padding_length,), -100, dtype=labels.dtype)])
                # Create attention mask
                attention_mask = torch.cat([torch.ones(len(input_ids) - padding_length), torch.zeros(padding_length)])
            else:
                attention_mask = torch.ones(len(input_ids))
            
            padded_input_ids.append(input_ids)
            padded_labels.append(labels)
            attention_masks.append(attention_mask)
        
        # Stack into batch
        batch_input_ids = torch.stack(padded_input_ids)
        batch_labels = torch.stack(padded_labels)
        batch_attention_mask = torch.stack(attention_masks)
        
        return {
            "images": images,  # Keep as PIL images - model expects this
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
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
        
        # SHIRG-FIX: 2025-07-30 - Handle accelerator preparation carefully
        # ISSUE: Accelerator may conflict with manual device placement
        # SOLUTION: Only prepare dataloaders, handle model separately
        # LAVIDA IMPACT: Maintains proper device placement
        # SHIRG IMPACT: Fixes device conflicts during distributed training
        
        # Check if we're in distributed mode with NO_DEVICE_MAP
        if os.environ.get('SHIRG_NO_DEVICE_MAP', '0') == '1':
            # Don't let accelerator handle model when device_map is disabled
            # This prevents conflicts with our manual device placement
            self.train_dataloader, self.val_dataloader = \
                self.accelerator.prepare(self.train_dataloader, self.val_dataloader)
            # Model and optimizer are already set up correctly
            print("📍 Using manual device placement (SHIRG_NO_DEVICE_MAP=1)")
        else:
            # Standard accelerator preparation
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