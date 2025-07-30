#!/usr/bin/env python3
"""
SHIRG Extra-LoRA Configuration for LaViDa Training
Following the research methodology for enhanced token selection adaptation

Author: Research Implementation
Date: 2025-07-30
"""

import os
import torch
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from peft import LoraConfig, TaskType

@dataclass
class ShirgLoraConfig:
    """
    SHIRG Extra-LoRA configuration following research methodology
    
    Target: ~1.4% parameters (136M out of 8B) with enhanced footprint
    - SigLIP blocks 0-3: q, k, v (not just q, k)
    - SigLIP blocks 4-5: q, k only
    - mm_projector.fc2: full weight
    """
    
    # LoRA hyperparameters from research
    rank: int = 64
    alpha: int = 128
    dropout: float = 0.05
    
    # Learning rate from research (1.8e-5)
    learning_rate: float = 1.8e-5
    warmup_steps: int = 500
    
    # Training schedule
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    
    # Token dropout for stabilization
    token_dropout_rate: float = 0.1  # 10% random token dropout during training
    
    # Mixed precision settings
    fp16: bool = False
    bf16: bool = True  # Use bf16 as specified
    tf32: bool = True  # Enable TF32 for A100/H100
    
    # Optimizer settings
    optim: str = "adamw_torch"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.01
    
    # Scheduler
    lr_scheduler_type: str = "cosine"
    
    # Gradient settings
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    
    # Save settings
    save_steps: int = 500
    save_total_limit: int = 3
    save_samples_interval: Optional[int] = None  # Save checkpoint every N samples (e.g., 5000)
    
    # Logging
    logging_steps: int = 10
    logging_first_step: bool = True
    
    # Evaluation
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    
    # Target modules following Extra-LoRA footprint
    target_modules: List[str] = field(default_factory=lambda: [
        # MM Projector layers (both fc1 and fc2)
        "model.mm_projector.fc1",
        "model.mm_projector.fc2",  # NEW: fc2 as per Extra-LoRA
        
        # SigLIP blocks 0-3 with q, k, v (Extra-LoRA enhancement)
        # Note: LaViDa uses nested model structure, so we need model. prefix
        "model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj",
        "model.vision_tower.vision_model.encoder.layers.0.self_attn.k_proj",
        "model.vision_tower.vision_model.encoder.layers.0.self_attn.v_proj",  # NEW: v_proj
        
        "model.vision_tower.vision_model.encoder.layers.1.self_attn.q_proj",
        "model.vision_tower.vision_model.encoder.layers.1.self_attn.k_proj",
        "model.vision_tower.vision_model.encoder.layers.1.self_attn.v_proj",  # NEW: v_proj
        
        "model.vision_tower.vision_model.encoder.layers.2.self_attn.q_proj",
        "model.vision_tower.vision_model.encoder.layers.2.self_attn.k_proj",
        "model.vision_tower.vision_model.encoder.layers.2.self_attn.v_proj",  # NEW: v_proj
        
        "model.vision_tower.vision_model.encoder.layers.3.self_attn.q_proj",
        "model.vision_tower.vision_model.encoder.layers.3.self_attn.k_proj",
        "model.vision_tower.vision_model.encoder.layers.3.self_attn.v_proj",  # NEW: v_proj
        
        # SigLIP blocks 4-5 with q, k only (NEW mid-layer adaptation)
        "model.vision_tower.vision_model.encoder.layers.4.self_attn.q_proj",
        "model.vision_tower.vision_model.encoder.layers.4.self_attn.k_proj",
        
        "model.vision_tower.vision_model.encoder.layers.5.self_attn.q_proj",
        "model.vision_tower.vision_model.encoder.layers.5.self_attn.k_proj",
    ])
    
    # Dataset configuration for 672² training
    image_size: int = 672  # High-resolution training
    use_shirg: bool = True
    shirg_method: str = "full"  # Use full enhancement method
    
    # SHIRG-specific parameters
    shirg_entropy_threshold: float = 0.12
    shirg_edge_weight: float = 0.25
    shirg_radial_sigma: float = 0.65
    shirg_merge_similar: bool = True
    shirg_merge_threshold: float = 0.9
    
    # Memory optimization
    use_cache: bool = True
    max_seq_length: int = 2048
    
    # Hardware settings
    dataloader_num_workers: int = 8
    dataloader_pin_memory: bool = True
    
    def to_peft_config(self) -> LoraConfig:
        """Convert to PEFT LoraConfig"""
        return LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            target_modules=self.target_modules,
            lora_dropout=self.dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
        )
    
    def get_training_args(self) -> Dict[str, Any]:
        """Get HuggingFace TrainingArguments compatible dict"""
        return {
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "optim": self.optim,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_epsilon": self.adam_epsilon,
            "weight_decay": self.weight_decay,
            "lr_scheduler_type": self.lr_scheduler_type,
            "max_grad_norm": self.max_grad_norm,
            "gradient_checkpointing": self.gradient_checkpointing,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "tf32": self.tf32,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "logging_steps": self.logging_steps,
            "logging_first_step": self.logging_first_step,
            "evaluation_strategy": self.evaluation_strategy,
            "eval_steps": self.eval_steps,
            "dataloader_num_workers": self.dataloader_num_workers,
            "dataloader_pin_memory": self.dataloader_pin_memory,
        }
    
    def estimate_memory_usage(self, model_size_gb: float = 16.0) -> Dict[str, float]:
        """
        Estimate memory usage for training
        
        Args:
            model_size_gb: Base model size in GB (LaViDa ~16GB in bf16)
        
        Returns:
            Memory estimates
        """
        # Base model memory
        base_memory = model_size_gb
        
        # LoRA parameters (roughly 1.4% of 8B params)
        num_lora_params = 136_000_000  # 136M parameters
        lora_memory_gb = (num_lora_params * 2) / 1e9  # bf16 = 2 bytes per param
        
        # Optimizer states (Adam needs 2x params for momentum/variance)
        optimizer_memory_gb = lora_memory_gb * 2
        
        # Gradients
        gradient_memory_gb = lora_memory_gb
        
        # Activations (rough estimate based on batch size and sequence length)
        activation_memory_gb = (self.per_device_train_batch_size * 
                               self.max_seq_length * 8192 * 2) / 1e9
        
        # Total
        total_memory_gb = (base_memory + lora_memory_gb + 
                          optimizer_memory_gb + gradient_memory_gb + 
                          activation_memory_gb)
        
        return {
            "base_model_gb": base_memory,
            "lora_params_gb": lora_memory_gb,
            "optimizer_states_gb": optimizer_memory_gb,
            "gradients_gb": gradient_memory_gb,
            "activations_gb": activation_memory_gb,
            "total_estimated_gb": total_memory_gb,
            "recommended_gpu_memory_gb": total_memory_gb * 1.2,  # 20% safety margin
        }
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return any warnings"""
        warnings = []
        
        # Check rank
        if self.rank < 32:
            warnings.append(f"Low LoRA rank ({self.rank}) may limit adaptation capacity")
        elif self.rank > 128:
            warnings.append(f"High LoRA rank ({self.rank}) increases memory usage significantly")
        
        # Check learning rate
        if self.learning_rate > 5e-5:
            warnings.append(f"High learning rate ({self.learning_rate}) may cause instability")
        elif self.learning_rate < 5e-6:
            warnings.append(f"Low learning rate ({self.learning_rate}) may slow convergence")
        
        # Check batch size
        effective_batch = self.per_device_train_batch_size * self.gradient_accumulation_steps
        if effective_batch < 8:
            warnings.append(f"Small effective batch size ({effective_batch}) may hurt training stability")
        
        # Memory estimate
        memory_est = self.estimate_memory_usage()
        if memory_est["recommended_gpu_memory_gb"] > 40:
            warnings.append(f"Estimated memory ({memory_est['recommended_gpu_memory_gb']:.1f}GB) exceeds 40GB GPU")
        
        return warnings


def get_optimal_batch_size(gpu_memory_gb: int = 40) -> int:
    """
    Calculate optimal batch size for given GPU memory
    
    Args:
        gpu_memory_gb: Available GPU memory
    
    Returns:
        Recommended batch size
    """
    # Conservative estimates for LaViDa + SHIRG
    base_model_memory = 16  # LaViDa in bf16
    per_sample_memory = 1.5  # Per sample overhead with 672² images
    
    available_for_batch = gpu_memory_gb - base_model_memory - 8  # 8GB safety margin
    max_batch_size = int(available_for_batch / per_sample_memory)
    
    # Round down to power of 2 for efficiency
    batch_size = 1
    while batch_size * 2 <= max_batch_size:
        batch_size *= 2
    
    return min(batch_size, 32)  # Cap at 32 for stability


def create_lora_training_config(
    selection_method: str = "full",
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    num_epochs: Optional[int] = None,
    save_samples_interval: Optional[int] = None,
) -> ShirgLoraConfig:
    """
    Create LoRA training configuration
    
    Args:
        selection_method: SHIRG selection method
        batch_size: Override batch size
        learning_rate: Override learning rate
        num_epochs: Override number of epochs
    
    Returns:
        Configuration object
    """
    config = ShirgLoraConfig()
    
    # Set SHIRG method
    config.shirg_method = selection_method
    
    # Override parameters if provided
    if batch_size is not None:
        config.per_device_train_batch_size = batch_size
    if learning_rate is not None:
        config.learning_rate = learning_rate
    if num_epochs is not None:
        config.num_train_epochs = num_epochs
    if save_samples_interval is not None:
        config.save_samples_interval = save_samples_interval
    
    # Validate and print warnings
    warnings = config.validate_config()
    if warnings:
        print("⚠️ Configuration warnings:")
        for warning in warnings:
            print(f"   - {warning}")
    
    return config


def print_lora_config_summary(config: ShirgLoraConfig):
    """Print configuration summary"""
    print("🔧 SHIRG Extra-LoRA Configuration Summary")
    print("=" * 50)
    
    print(f"\n📊 LoRA Parameters:")
    print(f"   Rank: {config.rank}")
    print(f"   Alpha: {config.alpha}")
    print(f"   Dropout: {config.dropout}")
    print(f"   Target modules: {len(config.target_modules)} layers")
    
    print(f"\n🎯 Training Settings:")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Epochs: {config.num_train_epochs}")
    print(f"   Batch size: {config.per_device_train_batch_size}")
    print(f"   Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"   Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    
    print(f"\n🖼️ SHIRG Settings:")
    print(f"   Method: {config.shirg_method}")
    print(f"   Image size: {config.image_size}×{config.image_size}")
    print(f"   Token dropout: {config.token_dropout_rate * 100}%")
    
    print(f"\n💾 Memory Estimates:")
    memory = config.estimate_memory_usage()
    for key, value in memory.items():
        print(f"   {key}: {value:.2f} GB")
    
    print("\n" + "=" * 50)


if __name__ == "__main__":
    # Test configuration creation
    config = create_lora_training_config()
    print_lora_config_summary(config)
    
    # Test batch size optimization
    print(f"\n🔄 Optimal batch size for 40GB GPU: {get_optimal_batch_size(40)}")
    print(f"🔄 Optimal batch size for 80GB GPU: {get_optimal_batch_size(80)}")