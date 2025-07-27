#!/usr/bin/env python3
"""
SHIRG LoRA Training Implementation
Mixed-ratio LoRA adaptation of mm_projector for high-resolution token processing

SHIRG-FIX: 2025-07-27 - Complete LoRA training pipeline for SHIRG-v2
ISSUE: Need lightweight projector adaptation for 3,645 → 2,304 high-res tokens
SOLUTION: Mixed-ratio LoRA training following proven HiRes-LLaVA methodology
LAVIDA IMPACT: Maintains LaViDa performance while enabling high-res processing
SHIRG IMPACT: Enables production-ready SHIRG with 512-1024 token flexibility
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import warnings
from tqdm import tqdm
import logging

# Add paths for imports
sys.path.append('./shirg/')
sys.path.append('./llava/')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SHIRGLoRAConfig:
    """SHIRG LoRA training configuration following research specifications"""
    
    # LoRA Architecture (HiRes-LLaVA proven parameters)
    lora_rank: int = 16                    # Rank: 16 (primary), 32 (comparison)
    lora_alpha: int = 32                   # Alpha: 32 (α/r = 2.0 scaling)
    lora_dropout: float = 0.05             # Low dropout for stable adaptation
    target_modules: List[str] = field(default_factory=lambda: ["mm_projector.0", "mm_projector.2"])
    bias: str = "lora"                     # LoRA bias for better convergence
    
    # Training Configuration (3.5h on 8×A100 target)
    batch_size_per_gpu: int = 16           # Memory-optimized for A100
    gradient_accumulation_steps: int = 4   # Effective batch size = 128
    num_epochs: int = 3                    # 3 epochs proven optimal
    learning_rate: float = 1e-4            # Proven optimal for projector LoRA
    weight_decay: float = 0.01             # L2 regularization
    warmup_ratio: float = 0.1              # 10% warmup
    lr_scheduler_type: str = "cosine"      # Cosine annealing
    
    # Mixed-Ratio Training (SHIRG-v2 key innovation)
    token_budgets: List[int] = field(default_factory=lambda: [512, 768, 1024])
    include_pooled: bool = True            # Include original 980 pooled tokens
    ratio_sampling: str = "uniform"        # uniform, weighted, or curriculum
    
    # Dataset Configuration
    dataset_size: int = 558000             # LCS-558K base dataset
    ocr_enhancement_size: int = 50000      # Additional OCR samples
    max_length: int = 2048                 # Max sequence length
    image_resolution: int = 672            # High-res input for token extraction
    
    # Hardware Configuration
    device: str = "cuda"
    mixed_precision: str = "fp16"          # Memory optimization
    dataloader_num_workers: int = 8
    pin_memory: bool = True
    
    # Monitoring & Validation
    eval_steps: int = 500                  # Validation frequency
    save_steps: int = 1000                 # Checkpoint frequency
    logging_steps: int = 100               # Log frequency
    max_grad_norm: float = 1.0             # Gradient clipping
    
    # Output Configuration
    output_dir: str = "./shirg_lora_output"
    run_name: str = "shirg_mixed_ratio_lora"
    save_safetensors: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters"""
        assert self.lora_rank in [16, 32, 64], "LoRA rank must be 16, 32, or 64"
        assert self.learning_rate <= 2e-4, "Learning rate too high for LoRA stability"
        assert len(self.token_budgets) > 0, "Must specify at least one token budget"
        
        # Calculate derived parameters
        self.effective_batch_size = self.batch_size_per_gpu * self.gradient_accumulation_steps
        self.total_samples = self.dataset_size + self.ocr_enhancement_size
        self.total_steps = (self.total_samples * self.num_epochs) // self.effective_batch_size
        self.warmup_steps = int(self.total_steps * self.warmup_ratio)
        
        logger.info(f"SHIRG LoRA Config: {self.total_steps} steps, {self.warmup_steps} warmup")

class SHIRGLoRATrainer:
    """Mixed-ratio LoRA trainer for SHIRG mm_projector adaptation"""
    
    def __init__(self, config: SHIRGLoRAConfig, model, tokenizer=None):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        
        # Setup device and precision
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision == "fp16" else None
        
        # Initialize components
        self.setup_lora()
        self.setup_optimizer()
        self.setup_scheduler()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        
        logger.info(f"SHIRG LoRA Trainer initialized with {self.count_trainable_parameters()} trainable parameters")
    
    def setup_lora(self):
        """Setup LoRA adaptation for mm_projector following research specifications"""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError("Please install peft: pip install peft")
        
        # LoRA configuration optimized for mm_projector
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.bias,
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False
        )
        
        # Apply LoRA to the model
        self.model = get_peft_model(self.model, lora_config)
        
        # Freeze everything except LoRA parameters
        for name, param in self.model.named_parameters():
            if "lora_" not in name.lower():
                param.requires_grad = False
            else:
                param.requires_grad = True
                logger.info(f"✓ LoRA parameter enabled: {name}")
        
        # Verify setup
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"LoRA Setup: {trainable_params:,} / {total_params:,} params trainable ({trainable_params/total_params:.4f})")
        
        # Ensure ratio is reasonable (should be < 1%)
        assert trainable_params / total_params < 0.01, "Too many trainable parameters for LoRA"
    
    def setup_optimizer(self):
        """Setup AdamW optimizer for LoRA parameters"""
        # Only optimize LoRA parameters
        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = torch.optim.AdamW(
            lora_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        logger.info(f"Optimizer setup: {len(lora_params)} LoRA parameter groups")
    
    def setup_scheduler(self):
        """Setup learning rate scheduler"""
        if self.config.lr_scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.total_steps,
                eta_min=self.config.learning_rate * 0.1
            )
        else:
            # Linear warmup + cosine decay
            from transformers import get_cosine_schedule_with_warmup
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=self.config.total_steps
            )
    
    def count_trainable_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_mixed_ratio_batch(self, batch_data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, int]:
        """
        Apply mixed-ratio token selection following SHIRG-v2 specifications
        
        Args:
            batch_data: Dictionary containing 'images', 'texts', and other data
            
        Returns:
            vision_features: Processed vision tokens with selected ratio
            selected_ratio: The token budget used for this batch
        """
        # Sample token ratio for this batch
        if self.config.ratio_sampling == "uniform":
            ratios = self.config.token_budgets.copy()
            if self.config.include_pooled:
                ratios.append("pooled")
            selected_ratio = np.random.choice(ratios)
        elif self.config.ratio_sampling == "weighted":
            # Weight higher ratios more during early training
            progress = self.global_step / self.config.total_steps
            weights = [1.0 + progress] * len(self.config.token_budgets)
            if self.config.include_pooled:
                weights.append(0.5 - progress * 0.3)  # Reduce pooled weight over time
                ratios = self.config.token_budgets + ["pooled"]
            else:
                ratios = self.config.token_budgets
            selected_ratio = np.random.choice(ratios, p=np.array(weights)/sum(weights))
        else:  # curriculum
            # Start with pooled, gradually move to higher ratios
            progress = self.global_step / self.config.total_steps
            if progress < 0.3:
                selected_ratio = "pooled"
            elif progress < 0.7:
                selected_ratio = np.random.choice([512, 768])
            else:
                selected_ratio = np.random.choice([768, 1024])
        
        # Get vision tower
        vision_tower = self.model.get_vision_tower() if hasattr(self.model, 'get_vision_tower') else None
        if vision_tower is None:
            # Fallback: assume model has vision processing capabilities
            logger.warning("Cannot access vision tower, using fallback processing")
            return batch_data.get('image_features'), selected_ratio
        
        images = batch_data['images'].to(self.device)
        
        if selected_ratio == "pooled":
            # Use original LaViDa pooling (729 → 980 tokens)
            vision_features = vision_tower.forward(images)
        else:
            # Apply SHIRG-v2 selection from high-resolution tokens
            try:
                # Get high-resolution tokens (2304 from 672×672)
                high_res_tokens = vision_tower.get_multiview_tokens_for_shirg(images)
                
                # Apply SHIRG selection
                text_embeddings = batch_data.get('text_embeddings', None)
                selected_tokens = vision_tower.shirg_token_selection(
                    high_res_tokens, 
                    target_count=selected_ratio,
                    text_embeddings=text_embeddings
                )
                vision_features = selected_tokens
                
            except Exception as e:
                logger.warning(f"SHIRG selection failed: {e}, falling back to standard processing")
                vision_features = vision_tower.forward(images)
                selected_ratio = "pooled"
        
        return vision_features, selected_ratio
    
    def compute_alignment_loss(self, vision_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Compute vision-text alignment loss for projector training
        Following InfoNCE loss for multimodal alignment
        """
        # L2 normalize features
        vision_norm = F.normalize(vision_features.mean(dim=1), p=2, dim=-1)  # Pool over tokens
        text_norm = F.normalize(text_features.mean(dim=1), p=2, dim=-1)     # Pool over tokens
        
        # Compute similarity matrix
        batch_size = vision_norm.shape[0]
        similarity = torch.matmul(vision_norm, text_norm.transpose(0, 1))
        
        # InfoNCE loss with temperature scaling
        temperature = 0.07
        labels = torch.arange(batch_size, device=similarity.device)
        
        # Symmetric loss (vision→text + text→vision)
        loss_v2t = F.cross_entropy(similarity / temperature, labels)
        loss_t2v = F.cross_entropy(similarity.transpose(0, 1) / temperature, labels)
        
        return (loss_v2t + loss_t2v) / 2
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with mixed-ratio processing"""
        
        # Apply mixed-ratio token selection
        vision_features, selected_ratio = self.get_mixed_ratio_batch(batch)
        
        # Move to device
        vision_features = vision_features.to(self.device)
        text_features = batch.get('text_embeddings', batch.get('text_features')).to(self.device)
        
        # Forward pass through LoRA-adapted mm_projector
        if self.config.mixed_precision == "fp16":
            with torch.cuda.amp.autocast():
                # Apply mm_projector with LoRA adaptation
                projected_vision = self.model.mm_projector(vision_features)
                
                # Compute alignment loss
                loss = self.compute_alignment_loss(projected_vision, text_features)
        else:
            projected_vision = self.model.mm_projector(vision_features)
            loss = self.compute_alignment_loss(projected_vision, text_features)
        
        # Backward pass
        if self.config.mixed_precision == "fp16":
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return {
            "loss": loss.item(),
            "selected_ratio": selected_ratio,
            "vision_tokens": vision_features.shape[1],
            "batch_size": vision_features.shape[0]
        }
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = []
        ratio_counts = {}
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")
        
        for step, batch in enumerate(progress_bar):
            # Training step
            step_outputs = self.training_step(batch)
            
            # Accumulate gradients
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.mixed_precision == "fp16":
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Track metrics
            epoch_losses.append(step_outputs["loss"])
            ratio = step_outputs["selected_ratio"]
            ratio_counts[ratio] = ratio_counts.get(ratio, 0) + 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{step_outputs['loss']:.4f}",
                "ratio": str(ratio),
                "lr": f"{self.scheduler.get_last_lr()[0]:.6f}"
            })
            
            # Logging
            if self.global_step % self.config.logging_steps == 0:
                logger.info(
                    f"Step {self.global_step}: loss={step_outputs['loss']:.4f}, "
                    f"ratio={ratio}, tokens={step_outputs['vision_tokens']}, "
                    f"lr={self.scheduler.get_last_lr()[0]:.6f}"
                )
            
            # Save checkpoint
            if self.global_step % self.config.save_steps == 0:
                self.save_checkpoint()
        
        return {
            "avg_loss": np.mean(epoch_losses),
            "ratio_distribution": ratio_counts
        }
    
    def validate(self, val_dataloader) -> Dict[str, float]:
        """Validation pass"""
        self.model.eval()
        
        val_losses = []
        ratio_performance = {}
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                # Test each ratio separately for detailed analysis
                for ratio in self.config.token_budgets + (["pooled"] if self.config.include_pooled else []):
                    # Force specific ratio for validation
                    original_sampling = self.config.ratio_sampling
                    self.config.ratio_sampling = "forced"
                    self._forced_ratio = ratio
                    
                    try:
                        step_outputs = self.training_step(batch)
                        val_losses.append(step_outputs["loss"])
                        
                        if ratio not in ratio_performance:
                            ratio_performance[ratio] = []
                        ratio_performance[ratio].append(step_outputs["loss"])
                        
                    except Exception as e:
                        logger.warning(f"Validation failed for ratio {ratio}: {e}")
                    
                    # Restore original sampling
                    self.config.ratio_sampling = original_sampling
                    if hasattr(self, '_forced_ratio'):
                        delattr(self, '_forced_ratio')
        
        # Compute per-ratio averages
        ratio_avg_performance = {
            ratio: np.mean(losses) for ratio, losses in ratio_performance.items()
        }
        
        return {
            "avg_val_loss": np.mean(val_losses),
            "ratio_performance": ratio_avg_performance
        }
    
    def save_checkpoint(self):
        """Save training checkpoint"""
        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{self.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA adapter
        self.model.save_pretrained(checkpoint_dir)
        
        # Save training state
        training_state = {
            "global_step": self.global_step,
            "current_epoch": self.current_epoch,
            "best_loss": self.best_loss,
            "config": self.config.__dict__,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
        }
        
        torch.save(training_state, checkpoint_dir / "training_state.pt")
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def train(self, train_dataloader, val_dataloader=None):
        """Main training loop"""
        logger.info("Starting SHIRG LoRA training...")
        logger.info(f"Configuration: {self.config}")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_dataloader)
            logger.info(f"Epoch {epoch} training: {train_metrics}")
            
            # Validation
            if val_dataloader is not None:
                val_metrics = self.validate(val_dataloader)
                logger.info(f"Epoch {epoch} validation: {val_metrics}")
                
                # Save best model
                if val_metrics["avg_val_loss"] < self.best_loss:
                    self.best_loss = val_metrics["avg_val_loss"]
                    best_model_dir = Path(self.config.output_dir) / "best_model"
                    best_model_dir.mkdir(parents=True, exist_ok=True)
                    self.model.save_pretrained(best_model_dir)
                    logger.info(f"New best model saved with loss {self.best_loss:.4f}")
        
        # Final save
        final_model_dir = Path(self.config.output_dir) / "final_model"
        final_model_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(final_model_dir)
        
        logger.info("SHIRG LoRA training completed!")
        return {
            "final_loss": self.best_loss,
            "total_steps": self.global_step,
            "trainable_parameters": self.count_trainable_parameters()
        }

def create_dummy_dataset(config: SHIRGLoRAConfig, size: int = 1000):
    """Create dummy dataset for testing"""
    
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return {
                'images': torch.randn(3, config.image_resolution, config.image_resolution),
                'text_embeddings': torch.randn(20, 1152),  # Assuming SigLIP hidden size
                'texts': f"Sample text {idx}"
            }
    
    return DummyDataset(size)

def setup_model_for_lora():
    """Setup LaViDa model for LoRA training"""
    try:
        # Try to load the actual LaViDa model
        from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
        
        # Create a minimal model wrapper for testing
        class MinimalLaViDaForLoRA(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_tower = SigLipVisionTower(
                    vision_tower="google/siglip-so400m-patch14-384",
                    vision_tower_cfg=None,
                    delay_load=False
                )
                
                # mm_projector following LaViDa architecture
                self.mm_projector = nn.Sequential(
                    nn.Linear(1152, 4096),  # SigLIP hidden_size to LLM hidden_size
                    nn.GELU(),
                    nn.Linear(4096, 4096)   # Project to final LLM embedding space
                )
            
            def get_vision_tower(self):
                return self.vision_tower
            
            def forward(self, images):
                vision_features = self.vision_tower(images)
                return self.mm_projector(vision_features)
        
        model = MinimalLaViDaForLoRA()
        logger.info("LaViDa model setup complete")
        return model
        
    except Exception as e:
        logger.error(f"Failed to setup LaViDa model: {e}")
        raise

def main():
    """Main function for SHIRG LoRA training"""
    
    # Configuration
    config = SHIRGLoRAConfig(
        # Training configuration optimized for testing
        batch_size_per_gpu=4,           # Reduced for testing
        gradient_accumulation_steps=2,   # Smaller effective batch
        num_epochs=1,                   # Single epoch for testing
        dataset_size=1000,              # Small dataset for testing
        ocr_enhancement_size=100,
        eval_steps=50,
        save_steps=100,
        logging_steps=10
    )
    
    # Setup model
    model = setup_model_for_lora()
    
    # Create datasets
    train_dataset = create_dummy_dataset(config, config.dataset_size)
    val_dataset = create_dummy_dataset(config, 200)
    
    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size_per_gpu,
        shuffle=True,
        num_workers=config.dataloader_num_workers,
        pin_memory=config.pin_memory
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size_per_gpu,
        shuffle=False,
        num_workers=config.dataloader_num_workers,
        pin_memory=config.pin_memory
    )
    
    # Initialize trainer
    trainer = SHIRGLoRATrainer(config, model)
    
    # Start training
    results = trainer.train(train_dataloader, val_dataloader)
    
    logger.info(f"Training completed: {results}")

if __name__ == "__main__":
    main()