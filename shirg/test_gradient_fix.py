#!/usr/bin/env python3
"""
Test script to verify the LoRA gradient fix works
"""

import os
import sys
import torch

# Add paths
sys.path.append('./')
sys.path.append('./shirg')

# Set environment variable for single GPU testing
os.environ['SHIRG_NO_DEVICE_MAP'] = '1'

def test_gradient_fix():
    """Test that the gradient fix resolves the zero gradient issue"""
    print("🧪 Testing LoRA Gradient Fix")
    print("=" * 60)
    
    try:
        # Import required components
        from shirg.shirg_lora_config import create_lora_training_config
        from shirg.train_shirg_lora import ShirgLoraTrainer
        from PIL import Image
        
        # Create minimal config
        config = create_lora_training_config(
            selection_method='full',
            batch_size=1,
            num_epochs=1
        )
        
        # Create trainer
        print("\n1. Creating trainer with gradient fix...")
        trainer = ShirgLoraTrainer(
            config=config,
            output_dir="./test_gradient_fix",
            use_wandb=False
        )
        
        # Setup model (this applies the gradient fix)
        print("\n2. Setting up model with LoRA...")
        trainer.setup_model()
        
        # Setup optimizer
        print("\n3. Setting up optimizer...")
        trainer.setup_optimizer_scheduler(1)
        
        # Create dummy batch
        print("\n4. Creating test batch...")
        dummy_sample = {
            "image": Image.new('RGB', (672, 672)),
            "question": "What text is shown in this image?",
            "answer": "Test answer",
            "id": "test_1"
        }
        
        # Process batch
        batch = trainer.collate_fn([dummy_sample])
        
        # Move to device
        if torch.cuda.is_available():
            batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
        
        # Test gradient computation
        print("\n5. Testing gradient computation...")
        trainer.model.train()
        trainer.model.zero_grad()
        
        # Forward pass
        outputs = trainer.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            images=batch["images"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        print(f"   Loss: {loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        print("\n6. Checking LoRA gradients...")
        lora_params_with_grad = 0
        lora_params_without_grad = 0
        sample_grads = []
        
        for name, param in trainer.model.named_parameters():
            if 'lora' in name.lower() and param.requires_grad:
                if param.grad is not None and param.grad.norm().item() > 0:
                    lora_params_with_grad += 1
                    if len(sample_grads) < 5:
                        sample_grads.append((name, param.grad.norm().item()))
                else:
                    lora_params_without_grad += 1
        
        print(f"\n   LoRA parameters with gradients: {lora_params_with_grad}")
        print(f"   LoRA parameters without gradients: {lora_params_without_grad}")
        
        if sample_grads:
            print(f"\n   Sample gradient norms:")
            for name, grad_norm in sample_grads:
                print(f"      {name}: {grad_norm:.6f}")
        
        # Check specific components
        vision_lora_grads = 0
        projector_lora_grads = 0
        
        for name, param in trainer.model.named_parameters():
            if 'lora' in name.lower() and param.requires_grad and param.grad is not None:
                if 'vision_tower' in name:
                    vision_lora_grads += 1
                elif 'mm_projector' in name:
                    projector_lora_grads += 1
        
        print(f"\n   Vision tower LoRA gradients: {vision_lora_grads}")
        print(f"   Projector LoRA gradients: {projector_lora_grads}")
        
        # Success criteria
        success = lora_params_with_grad > 0
        
        print("\n" + "=" * 60)
        if success:
            print("✅ GRADIENT FIX SUCCESSFUL!")
            print(f"   {lora_params_with_grad} LoRA parameters are receiving gradients")
        else:
            print("❌ GRADIENT FIX FAILED!")
            print("   No LoRA parameters are receiving gradients")
        
        # Cleanup
        del trainer
        torch.cuda.empty_cache()
        
        return success
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gradient_fix()
    sys.exit(0 if success else 1)