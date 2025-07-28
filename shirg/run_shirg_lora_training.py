#!/usr/bin/env python3
"""
SHIRG-Fixed LoRA Training Script
Implements rank-64 LoRA training for SHIRG-Fixed implementation as per research plan

SHIRG-FIXED-FIX: 2025-07-28 - Training pipeline for SHIRG-Fixed implementation
ISSUE: Need training script for rank-64 LoRA (mm_projector + early SigLIP layers)
SOLUTION: Mixed-resolution training with fixed K=768 token selection
RESEARCH IMPACT: Enables cross-resolution alignment with minimal parameter overhead
"""

import os
import sys
import argparse
import logging
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Add paths for imports
sys.path.append('./shirg/')
sys.path.append('./llava/')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('shirg_fixed_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SHIRGFixedTrainingConfig:
    """SHIRG-Fixed training configuration as per research plan"""
    
    # LoRA Configuration (rank-64 as specified)
    lora_rank: int = 64
    lora_alpha: int = 32  # α/r = 0.5 scaling
    lora_dropout: float = 0.05
    
    # Training Parameters
    batch_size: int = 16
    gradient_accumulation: int = 8
    effective_batch_size: int = 128  # 16 * 8
    learning_rate: float = 7e-5  # Higher than original due to rank-64
    weight_decay: float = 0.01
    epochs: int = 2
    warmup_ratio: float = 0.1
    
    # Dataset Configuration  
    dataset_size: int = 558000  # LCS-558K
    max_length: int = 2048
    mixed_resolution: bool = True  # Essential for generalization
    
    # SHIRG-Fixed Specific
    fixed_token_budget: int = 768  # Fixed K=768
    high_res_size: int = 672  # 672p processing
    
    # System Configuration
    num_gpus: int = 8
    output_dir: str = "./shirg_fixed_checkpoints"
    save_steps: int = 500
    logging_steps: int = 100


def setup_shirg_fixed_model():
    """
    Setup LaViDa model with SHIRG-Fixed LoRA configuration
    
    Returns:
        model: LaViDa model with SHIRG-Fixed LoRA applied
        tokenizer: Model tokenizer
        image_processor: Image processor
    """
    logger.info("Setting up LaViDa model with SHIRG-Fixed LoRA...")
    
    try:
        from shirg.lavida_shirg_integration import LaViDaSHIRGWrapper, setup_shirg_fixed_lora, verify_shirg_fixed_setup
        
        # Create LaViDa-SHIRG wrapper in SHIRG-Fixed mode
        shirg_config = {
            'mode': 'shirg-fixed',
            'target_tokens': 768,
            'alpha': 0.3,  # Enable SHIRG
            'debug': True
        }
        
        wrapper = LaViDaSHIRGWrapper(shirg_config=shirg_config)
        wrapper.load_model()
        
        # Apply SHIRG-Fixed LoRA configuration
        model = setup_shirg_fixed_lora(wrapper.model)
        
        # Verify LoRA setup
        verify_shirg_fixed_setup(model)
        
        return model, wrapper.tokenizer, wrapper.image_processor
        
    except Exception as e:
        logger.error(f"Failed to setup SHIRG-Fixed model: {e}")
        raise


def prepare_shirg_fixed_dataset(config: SHIRGFixedTrainingConfig):
    """
    Prepare mixed-resolution dataset for SHIRG-Fixed training
    
    Args:
        config: Training configuration
        
    Returns:
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
    """
    logger.info("Preparing SHIRG-Fixed mixed-resolution dataset...")
    
    try:
        from shirg.shirg_dataset_preparation import create_mixed_resolution_dataset
        
        # Dataset configuration for SHIRG-Fixed
        dataset_config = {
            'source': 'LCS-558K', 
            'size': config.dataset_size,
            'mixed_resolution': config.mixed_resolution,
            'fixed_token_budget': config.fixed_token_budget,
            'high_res_size': config.high_res_size,
            'batch_size': config.batch_size,
            'max_length': config.max_length
        }
        
        train_dataloader, val_dataloader = create_mixed_resolution_dataset(dataset_config)
        
        logger.info(f"Dataset prepared: {len(train_dataloader)} training batches")
        return train_dataloader, val_dataloader
        
    except Exception as e:
        logger.error(f"Failed to prepare dataset: {e}")
        raise


def train_shirg_fixed_lora(model, train_dataloader, val_dataloader, config: SHIRGFixedTrainingConfig):
    """
    Train SHIRG-Fixed LoRA with mixed-resolution data
    
    Args:
        model: LaViDa model with SHIRG-Fixed LoRA
        train_dataloader: Training data
        val_dataloader: Validation data  
        config: Training configuration
        
    Returns:
        trained_model: Model after training
    """
    logger.info("Starting SHIRG-Fixed LoRA training...")
    
    # Setup optimizer for SHIRG-Fixed parameters
    trainable_params = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Setup scheduler
    total_steps = len(train_dataloader) * config.epochs // config.gradient_accumulation
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=config.learning_rate * 0.1
    )
    
    # Training loop
    model.train()
    global_step = 0
    
    for epoch in range(config.epochs):
        logger.info(f"Starting epoch {epoch+1}/{config.epochs}")
        
        for step, batch in enumerate(train_dataloader):
            # Move batch to device
            images = batch['images'].cuda()
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['labels'].cuda()
            
            # Forward pass with SHIRG-Fixed
            outputs = model(
                input_ids=input_ids,
                images=images,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss / config.gradient_accumulation
            loss.backward()
            
            if (step + 1) % config.gradient_accumulation == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Logging
                if global_step % config.logging_steps == 0:
                    logger.info(f"Step {global_step}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
                
                # Save checkpoint
                if global_step % config.save_steps == 0:
                    save_checkpoint(model, optimizer, scheduler, global_step, config)
    
    logger.info("SHIRG-Fixed LoRA training completed!")
    return model


def save_checkpoint(model, optimizer, scheduler, step: int, config: SHIRGFixedTrainingConfig):
    """Save training checkpoint"""
    checkpoint_dir = Path(config.output_dir) / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model.save_pretrained(checkpoint_dir)
    
    # Save training state
    torch.save({
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step,
        'config': config.__dict__
    }, checkpoint_dir / 'training_state.pt')
    
    logger.info(f"Checkpoint saved at step {step}")


def evaluate_shirg_fixed(model, val_dataloader, config: SHIRGFixedTrainingConfig):
    """
    Evaluate SHIRG-Fixed model on validation set
    
    Args:
        model: Trained SHIRG-Fixed model
        val_dataloader: Validation data
        config: Training configuration
        
    Returns:
        eval_results: Evaluation metrics
    """
    logger.info("Evaluating SHIRG-Fixed model...")
    
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            images = batch['images'].cuda()
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['labels'].cuda()
            
            outputs = model(
                input_ids=input_ids,
                images=images,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    eval_results = {
        'validation_loss': avg_loss,
        'perplexity': torch.exp(torch.tensor(avg_loss)).item()
    }
    
    logger.info(f"Validation results: Loss={avg_loss:.4f}, Perplexity={eval_results['perplexity']:.2f}")
    return eval_results


def main():
    """Main training orchestration"""
    parser = argparse.ArgumentParser(description='SHIRG-Fixed LoRA Training')
    parser.add_argument('--config', type=str, help='Training config file')
    parser.add_argument('--output_dir', type=str, default='./shirg_fixed_checkpoints')
    parser.add_argument('--resume_from_checkpoint', type=str, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load configuration
    config = SHIRGFixedTrainingConfig()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                setattr(config, key, value)
    
    config.output_dir = args.output_dir
    
    logger.info("Starting SHIRG-Fixed LoRA training pipeline")
    logger.info(f"Configuration: {config.__dict__}")
    
    try:
        # Setup model
        model, tokenizer, image_processor = setup_shirg_fixed_model()
        
        # Prepare dataset
        train_dataloader, val_dataloader = prepare_shirg_fixed_dataset(config)
        
        # Train model
        trained_model = train_shirg_fixed_lora(model, train_dataloader, val_dataloader, config)
        
        # Final evaluation
        eval_results = evaluate_shirg_fixed(trained_model, val_dataloader, config)
        
        # Save final model
        final_checkpoint_dir = Path(config.output_dir) / "final"
        final_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trained_model.save_pretrained(final_checkpoint_dir)
        
        # Save evaluation results
        with open(final_checkpoint_dir / 'eval_results.json', 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        logger.info("SHIRG-Fixed training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()