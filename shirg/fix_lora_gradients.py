#!/usr/bin/env python3
"""
Fix for LoRA gradient flow issue in SHIRG training

The problem: Vision tower is frozen with requires_grad_(False) BEFORE LoRA is applied,
and LoRA parameters inherit this frozen state, resulting in zero gradients.

The solution: After applying LoRA with PEFT, ensure all LoRA parameters have requires_grad=True
"""

import torch
import torch.nn as nn
from typing import Any

def ensure_lora_parameters_trainable(model: nn.Module) -> int:
    """
    Ensure all LoRA parameters in the model are trainable
    
    This fixes the issue where LoRA parameters inherit frozen state from base model
    
    Args:
        model: The PEFT model with LoRA adapters
        
    Returns:
        int: Number of LoRA parameters that were unfrozen
    """
    unfrozen_count = 0
    lora_params_info = []
    
    # Find all LoRA parameters and ensure they're trainable
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            lora_params_info.append({
                'name': name,
                'shape': param.shape,
                'was_frozen': not param.requires_grad,
                'device': param.device
            })
            
            if not param.requires_grad:
                param.requires_grad = True
                unfrozen_count += 1
    
    # Print summary
    print(f"\n🔧 LoRA Gradient Fix Summary:")
    print(f"   Total LoRA parameters found: {len(lora_params_info)}")
    print(f"   Parameters that were frozen: {unfrozen_count}")
    print(f"   Parameters now trainable: {len(lora_params_info)}")
    
    if unfrozen_count > 0:
        print(f"\n   ✅ Fixed {unfrozen_count} frozen LoRA parameters!")
        print(f"\n   Sample fixed parameters:")
        for info in lora_params_info[:5]:  # Show first 5
            if info['was_frozen']:
                print(f"      - {info['name']} (shape: {info['shape']})")
    
    # Also ensure base model components that LoRA attaches to aren't frozen
    # This is important for gradient flow through the adapter
    vision_tower_unfrozen = 0
    projector_unfrozen = 0
    
    for name, module in model.named_modules():
        if 'vision_tower' in name and hasattr(module, 'weight'):
            # Check if this module has LoRA adapters attached
            has_lora = any('lora' in child_name for child_name, _ in module.named_parameters())
            if has_lora and hasattr(module, 'weight') and not module.weight.requires_grad:
                # Don't unfreeze the base weight, but ensure LoRA can compute gradients
                # by enabling gradient computation on the module itself
                for lora_name, lora_param in module.named_parameters():
                    if 'lora' in lora_name and not lora_param.requires_grad:
                        lora_param.requires_grad = True
                        vision_tower_unfrozen += 1
                        
        elif 'mm_projector' in name and hasattr(module, 'weight'):
            # Similar check for projector
            has_lora = any('lora' in child_name for child_name, _ in module.named_parameters())
            if has_lora:
                for lora_name, lora_param in module.named_parameters():
                    if 'lora' in lora_name and not lora_param.requires_grad:
                        lora_param.requires_grad = True
                        projector_unfrozen += 1
    
    if vision_tower_unfrozen > 0:
        print(f"\n   ✅ Fixed {vision_tower_unfrozen} frozen LoRA parameters in vision tower")
    if projector_unfrozen > 0:
        print(f"   ✅ Fixed {projector_unfrozen} frozen LoRA parameters in projector")
    
    # Verify gradient flow is possible
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n   📊 Model Statistics:")
    print(f"      Total parameters: {total_params:,}")
    print(f"      Trainable parameters: {total_trainable:,}")
    print(f"      Trainable percentage: {total_trainable/total_params*100:.4f}%")
    
    return unfrozen_count

def verify_lora_gradients(model: nn.Module, dummy_batch: dict) -> bool:
    """
    Verify that LoRA parameters receive gradients
    
    Args:
        model: The model to test
        dummy_batch: A dummy batch to run through the model
        
    Returns:
        bool: True if gradients flow to LoRA parameters
    """
    model.train()
    model.zero_grad()
    
    # Forward pass
    outputs = model(**dummy_batch)
    loss = outputs.loss if hasattr(outputs, 'loss') else outputs.mean()
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    lora_grads = []
    zero_grad_params = []
    
    for name, param in model.named_parameters():
        if 'lora' in name.lower() and param.requires_grad:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                lora_grads.append((name, grad_norm))
                if grad_norm == 0:
                    zero_grad_params.append(name)
            else:
                zero_grad_params.append(name + " (None)")
    
    print(f"\n🔍 Gradient Verification:")
    print(f"   LoRA parameters with gradients: {len(lora_grads)}")
    print(f"   LoRA parameters with zero/no gradients: {len(zero_grad_params)}")
    
    if lora_grads:
        print(f"\n   Sample gradients (first 5):")
        for name, grad_norm in lora_grads[:5]:
            print(f"      - {name}: {grad_norm:.6f}")
    
    if zero_grad_params:
        print(f"\n   ⚠️ Parameters with zero/no gradients (first 5):")
        for name in zero_grad_params[:5]:
            print(f"      - {name}")
    
    success = len(lora_grads) > 0 and len(zero_grad_params) < len(lora_grads)
    
    if success:
        print(f"\n   ✅ Gradient flow verified!")
    else:
        print(f"\n   ❌ Gradient flow issue detected!")
    
    return success

def apply_lora_gradient_fix_to_trainer(trainer):
    """
    Apply the LoRA gradient fix to a ShirgLoraTrainer instance
    
    This should be called after model setup but before training starts
    """
    if hasattr(trainer, 'model') and trainer.model is not None:
        print("\n🔧 Applying LoRA gradient fix to trainer...")
        unfrozen = ensure_lora_parameters_trainable(trainer.model)
        
        if unfrozen > 0:
            print(f"   ✅ Successfully fixed {unfrozen} frozen LoRA parameters")
        else:
            print(f"   ℹ️ All LoRA parameters were already trainable")
        
        return True
    else:
        print("   ⚠️ Trainer model not initialized yet")
        return False

# For direct import and use
if __name__ == "__main__":
    print("LoRA Gradient Fix Module")
    print("========================")
    print("This module provides fixes for LoRA gradient flow issues in SHIRG training")
    print("\nUsage:")
    print("  from shirg.fix_lora_gradients import ensure_lora_parameters_trainable")
    print("  ensure_lora_parameters_trainable(model)")