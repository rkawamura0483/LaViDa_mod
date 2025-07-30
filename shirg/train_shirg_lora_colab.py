#!/usr/bin/env python3
"""
SHIRG LoRA Training Script for Google Colab
Adapted version of the main training script optimized for Colab environment

This version includes:
- Automatic Colab detection
- Memory-efficient defaults for free Colab GPU
- Simplified dataset handling
- Progress visualization for notebooks

Author: Research Implementation
Date: 2025-07-30
"""

import os
import sys
import torch
import gc
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Detect Colab environment
try:
    import google.colab
    IN_COLAB = True
    print("🎯 Running in Google Colab")
    
    # Mount Google Drive if needed
    from google.colab import drive
    if input("Mount Google Drive? (y/n): ").lower() == 'y':
        drive.mount('/content/drive')
except ImportError:
    IN_COLAB = False
    print("⚠️ Not running in Colab")

# Add paths
sys.path.append('./')
sys.path.append('./shirg')

# Quick setup function for Colab
def setup_colab_environment():
    """Quick setup for Colab environment"""
    if not IN_COLAB:
        return
    
    print("🔧 Setting up Colab environment...")
    
    # Install dependencies if not already installed
    try:
        import transformers
        import peft
        import accelerate
        import wandb
        print("✅ Core packages already installed")
    except ImportError:
        print("📦 Installing dependencies...")
        os.system("python install_dependencies.py")
    
    # Set environment variables for better performance
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")


def get_colab_optimized_config():
    """Get training configuration optimized for Colab"""
    from shirg.shirg_lora_config import ShirgLoraConfig
    
    # Detect available GPU memory
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    else:
        gpu_memory_gb = 0
    
    # Adjust batch size based on GPU
    if gpu_memory_gb >= 40:  # A100
        batch_size = 8
        gradient_accumulation = 2
    elif gpu_memory_gb >= 16:  # V100 or T4
        batch_size = 4
        gradient_accumulation = 4
    else:  # Free Colab T4 (15GB)
        batch_size = 2
        gradient_accumulation = 8
    
    config = ShirgLoraConfig(
        # Reduced parameters for Colab
        rank=32,  # Reduced from 64
        alpha=64,  # 2x rank
        dropout=0.05,
        
        # Training parameters
        learning_rate=2e-5,  # Slightly higher for smaller rank
        warmup_steps=200,  # Reduced
        num_train_epochs=1,  # Quick training for demo
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        
        # Memory optimization
        gradient_checkpointing=True,
        fp16=True,  # Use FP16 instead of BF16 for older GPUs
        bf16=False,
        
        # Reduced token dropout for smaller model
        token_dropout_rate=0.05,
        
        # Simplified target modules for faster training
        target_modules=[
            "mm_projector.fc1",
            "mm_projector.fc2",
            # Only first 2 SigLIP blocks for demo
            "model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj",
            "model.vision_tower.vision_model.encoder.layers.0.self_attn.k_proj",
            "model.vision_tower.vision_model.encoder.layers.1.self_attn.q_proj",
            "model.vision_tower.vision_model.encoder.layers.1.self_attn.k_proj",
        ],
        
        # Logging
        logging_steps=5,
        save_steps=100,
        eval_steps=100,
    )
    
    print(f"📊 Colab Configuration:")
    print(f"   GPU Memory: {gpu_memory_gb:.1f}GB")
    print(f"   Batch Size: {batch_size}")
    print(f"   Gradient Accumulation: {gradient_accumulation}")
    print(f"   Effective Batch Size: {batch_size * gradient_accumulation}")
    print(f"   LoRA Rank: {config.rank}")
    
    return config


def create_demo_dataset(num_samples=100):
    """Create a small demo dataset for Colab"""
    from PIL import Image
    import numpy as np
    
    print(f"🎨 Creating demo dataset with {num_samples} samples...")
    
    data = []
    for i in range(num_samples):
        # Create synthetic image
        img_array = np.random.randint(0, 255, (336, 336, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Create question-answer pairs
        questions = [
            "What is shown in this image?",
            "Describe the main elements.",
            "What colors are present?",
            "Is there text in the image?",
        ]
        
        q_idx = i % len(questions)
        
        data.append({
            "image": img,
            "question": questions[q_idx],
            "answer": f"This is demo sample {i} showing synthetic data.",
            "id": f"demo_{i}",
        })
    
    return data


def train_in_colab(
    config: Optional[Dict] = None,
    use_wandb: bool = False,
    num_samples: int = 100,
    checkpoint_dir: str = "./colab_checkpoints",
):
    """
    Simplified training function for Colab
    
    Args:
        config: Training configuration (uses optimized defaults if None)
        use_wandb: Whether to use Weights & Biases logging
        num_samples: Number of demo samples to create
        checkpoint_dir: Directory for saving checkpoints
    """
    # Setup environment
    setup_colab_environment()
    
    # Get configuration
    if config is None:
        config = get_colab_optimized_config()
    
    # Import trainer
    from shirg.train_shirg_lora import ShirgLoraTrainer
    from torch.utils.data import Dataset, DataLoader
    
    # Create demo dataset
    class DemoDataset(Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    # Initialize trainer with memory-efficient settings
    trainer = ShirgLoraTrainer(
        config=config,
        output_dir=checkpoint_dir,
        use_wandb=use_wandb and not IN_COLAB,  # Disable W&B in free Colab
    )
    
    # Override dataset creation with demo data
    print("📊 Preparing demo dataset...")
    demo_data = create_demo_dataset(num_samples)
    
    # Split into train/val
    split_idx = int(0.9 * len(demo_data))
    train_data = demo_data[:split_idx]
    val_data = demo_data[split_idx:]
    
    trainer.train_dataset = DemoDataset(train_data)
    trainer.val_dataset = DemoDataset(val_data)
    
    # Create dataloaders with Colab-friendly settings
    trainer.train_dataloader = DataLoader(
        trainer.train_dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=True,
        num_workers=0,  # Important for Colab
        pin_memory=False,  # Disable for stability
        collate_fn=trainer.collate_fn,
    )
    
    trainer.val_dataloader = DataLoader(
        trainer.val_dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=trainer.collate_fn,
    )
    
    # Setup model (this is the memory-intensive part)
    print("🚀 Setting up model...")
    try:
        trainer.setup_model()
        
        # SHIRG-FIX: 2025-07-30 - Apply selective gradient flow for Colab
        # ISSUE: PEFT LoRA needs gradient flow through base modules
        # SOLUTION: Enable selective gradient flow and memory optimizations
        # LAVIDA IMPACT: Base modules allow gradient flow but stay frozen
        # SHIRG IMPACT: Enables LoRA training in memory-constrained Colab
        try:
            from shirg.fix_lora_gradients_selective import (
                apply_selective_gradient_flow,
                verify_selective_gradient_flow,
                apply_memory_optimizations,
                fix_trainer_optimizer
            )
            
            print("🔧 Applying selective gradient flow fix...")
            results = apply_selective_gradient_flow(trainer.model, debug=True)
            
            if results['success']:
                print(f"✅ Selective gradient flow enabled")
                
                # Fix the trainer's optimizer to use only LoRA params
                fix_trainer_optimizer(trainer, debug=True)
                
                # Apply memory optimizations for Colab
                apply_memory_optimizations(trainer.model, config.__dict__)
                
                # Verify the setup
                verify_results = verify_selective_gradient_flow(
                    trainer.model, 
                    trainer.optimizer,
                    debug=True
                )
                
                if not verify_results['setup_correct']:
                    print("⚠️ Warning: Gradient flow setup may have issues")
            else:
                print("⚠️ Selective gradient flow fix failed")
                
        except ImportError:
            print("⚠️ Selective gradient fix not available, using default setup")
        
        num_training_steps = len(trainer.train_dataloader) * config.num_train_epochs
        trainer.setup_optimizer_scheduler(num_training_steps)
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("❌ Out of memory! Try reducing batch size or LoRA rank.")
            print("💡 Suggestions:")
            print("   - Reduce batch_size to 1")
            print("   - Reduce LoRA rank to 16")
            print("   - Enable gradient checkpointing")
            print("   - Use smaller selection method (base instead of full)")
            return
        else:
            raise e
    
    # Quick training loop with progress bar
    from tqdm.notebook import tqdm
    import time
    
    print("🏃 Starting training...")
    trainer.model.train()
    
    # Training with notebook-friendly progress
    epoch_progress = tqdm(range(config.num_train_epochs), desc="Epochs")
    
    for epoch in epoch_progress:
        # Training steps
        batch_progress = tqdm(
            trainer.train_dataloader,
            desc=f"Training Epoch {epoch + 1}",
            leave=False
        )
        
        train_losses = []
        for step, batch in enumerate(batch_progress):
            try:
                metrics = trainer.training_step(batch)
                train_losses.append(metrics["loss"])
                
                # Update progress
                batch_progress.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "lr": f"{metrics['learning_rate']:.2e}",
                })
                
                # Periodic memory cleanup
                if step % 10 == 0:
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"⚠️ OOM at step {step}, clearing cache...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e
        
        # Simple evaluation
        print(f"📊 Epoch {epoch + 1} - Avg Loss: {sum(train_losses) / len(train_losses):.4f}")
        
        # Save checkpoint
        if IN_COLAB:
            checkpoint_path = f"{checkpoint_dir}/epoch_{epoch + 1}"
            print(f"💾 Saving to {checkpoint_path}")
            trainer.save_checkpoint(is_epoch_end=True)
    
    print("✅ Training complete!")
    
    # Save final model
    trainer.save_checkpoint(is_final=True)
    
    # Cleanup
    del trainer
    torch.cuda.empty_cache()
    gc.collect()
    
    print("🎉 Done! Model saved to:", checkpoint_dir)


# Interactive notebook interface
def interactive_train():
    """Interactive training interface for Colab notebooks"""
    if not IN_COLAB:
        print("This function is designed for Colab notebooks")
        return
    
    from IPython.display import display, HTML
    import ipywidgets as widgets
    
    # Create UI widgets
    method_dropdown = widgets.Dropdown(
        options=['base', 'entropy', 'edge', 'full'],
        value='base',
        description='Method:',
    )
    
    rank_slider = widgets.IntSlider(
        value=32,
        min=8,
        max=64,
        step=8,
        description='LoRA Rank:',
    )
    
    batch_slider = widgets.IntSlider(
        value=2,
        min=1,
        max=8,
        description='Batch Size:',
    )
    
    samples_slider = widgets.IntSlider(
        value=100,
        min=50,
        max=500,
        step=50,
        description='Samples:',
    )
    
    train_button = widgets.Button(
        description='Start Training',
        button_style='success',
        icon='play'
    )
    
    output = widgets.Output()
    
    def on_train_click(b):
        with output:
            output.clear_output()
            
            # Create config
            config = get_colab_optimized_config()
            config.shirg_method = method_dropdown.value
            config.rank = rank_slider.value
            config.alpha = rank_slider.value * 2
            config.per_device_train_batch_size = batch_slider.value
            
            # Start training
            train_in_colab(
                config=config,
                num_samples=samples_slider.value,
                use_wandb=False
            )
    
    train_button.on_click(on_train_click)
    
    # Display UI
    display(HTML("<h3>🎯 SHIRG LoRA Training Configuration</h3>"))
    display(widgets.VBox([
        method_dropdown,
        rank_slider,
        batch_slider,
        samples_slider,
        train_button,
        output
    ]))


# Main entry point
if __name__ == "__main__":
    if IN_COLAB:
        print("🎯 SHIRG LoRA Training for Colab")
        print("=" * 50)
        print("\nOptions:")
        print("1. Quick demo: train_in_colab()")
        print("2. Interactive: interactive_train()")
        print("3. Custom config: train_in_colab(config=your_config)")
        
        # Auto-run demo if in Colab
        if input("\nRun quick demo? (y/n): ").lower() == 'y':
            train_in_colab()
    else:
        # Run standard training
        from shirg.train_shirg_lora import main
        main()