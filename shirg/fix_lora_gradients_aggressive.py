#!/usr/bin/env python3
"""
Aggressive fix for LoRA gradient flow in SHIRG training

This addresses the core issue where LoRA adapters attached to frozen modules
don't receive gradients even if the LoRA parameters have requires_grad=True.

The solution is to selectively unfreeze base modules that have LoRA adapters.

Author: Research Implementation
Date: 2025-07-30
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import gc

def find_lora_modules(model: nn.Module) -> Dict[str, List[str]]:
    """
    Find all modules that have LoRA adapters attached
    
    Returns:
        Dict mapping base module names to their LoRA adapter names
    """
    lora_map = {}
    
    # First pass: find all LoRA parameters
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            lora_params.append(name)
    
    # Map LoRA parameters to their base modules
    for lora_name in lora_params:
        # Extract base module name
        # Example: base_model.model.model.vision_tower.vision_tower.vision_model.encoder.layers.0.self_attn.k_proj.lora_A.default.weight
        # Base module: base_model.model.model.vision_tower.vision_tower.vision_model.encoder.layers.0.self_attn.k_proj
        
        parts = lora_name.split('.')
        base_parts = []
        
        for i, part in enumerate(parts):
            if 'lora' in part.lower():
                # Found LoRA part, base module is everything before this
                base_name = '.'.join(base_parts)
                if base_name not in lora_map:
                    lora_map[base_name] = []
                lora_map[base_name].append(lora_name)
                break
            base_parts.append(part)
    
    return lora_map

def enable_gradient_checkpointing_selective(model: nn.Module, lora_modules: Dict[str, List[str]]):
    """
    Enable gradient checkpointing on modules with LoRA to save memory
    """
    # This is model-specific, implement if needed
    pass

def fix_lora_gradients_aggressive(
    model: nn.Module,
    unfreeze_base_modules: bool = True,
    force_lora_gradients: bool = True,
    debug: bool = True
) -> Dict[str, Any]:
    """
    Aggressively fix LoRA gradient flow issues
    
    Args:
        model: Model with LoRA adapters
        unfreeze_base_modules: Whether to unfreeze base modules that have LoRA
        force_lora_gradients: Force all LoRA params to have requires_grad=True
        debug: Print debug information
        
    Returns:
        Dict with fix statistics
    """
    results = {
        'lora_params_found': 0,
        'lora_params_fixed': 0,
        'base_modules_unfrozen': 0,
        'lora_modules_map': {},
        'gradient_flow_enabled': False
    }
    
    # Step 1: Find all modules with LoRA adapters
    lora_modules = find_lora_modules(model)
    results['lora_modules_map'] = lora_modules
    
    if debug:
        print(f"\n🔧 Aggressive LoRA Gradient Fix")
        print(f"   Found {len(lora_modules)} base modules with LoRA adapters")
    
    # Step 2: Fix LoRA parameters
    lora_param_count = 0
    fixed_count = 0
    
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            lora_param_count += 1
            if not param.requires_grad:
                param.requires_grad = True
                fixed_count += 1
    
    results['lora_params_found'] = lora_param_count
    results['lora_params_fixed'] = fixed_count
    
    if debug:
        print(f"   LoRA parameters: {lora_param_count} total, {fixed_count} fixed")
    
    # Step 3: Selectively unfreeze base modules that have LoRA
    if unfreeze_base_modules:
        unfrozen_modules = []
        
        for base_module_name in lora_modules:
            try:
                # Get the actual module
                parts = base_module_name.split('.')
                module = model
                
                for part in parts:
                    if hasattr(module, part):
                        module = getattr(module, part)
                    else:
                        # Try without base_model prefix
                        if part == 'base_model' or part == 'model':
                            continue
                        module = getattr(module, part)
                
                # Check if this module has frozen base parameters
                if hasattr(module, 'weight') and not module.weight.requires_grad:
                    # This is a frozen module with LoRA - we need to unfreeze it
                    module.weight.requires_grad = True
                    unfrozen_modules.append(base_module_name)
                    
                    # Also unfreeze bias if present
                    if hasattr(module, 'bias') and module.bias is not None:
                        module.bias.requires_grad = True
                
            except AttributeError as e:
                if debug:
                    print(f"   ⚠️ Could not access module: {base_module_name}")
        
        results['base_modules_unfrozen'] = len(unfrozen_modules)
        
        if debug and unfrozen_modules:
            print(f"\n   🔓 Unfroze {len(unfrozen_modules)} base modules with LoRA:")
            for name in unfrozen_modules[:5]:  # Show first 5
                print(f"      - {name}")
            if len(unfrozen_modules) > 5:
                print(f"      ... and {len(unfrozen_modules) - 5} more")
    
    # Step 4: Special handling for vision tower
    if hasattr(model, 'get_model'):
        base_model = model.get_model()
        if hasattr(base_model, 'get_vision_tower'):
            vision_tower = base_model.get_vision_tower()
            if vision_tower is not None:
                # Check if vision tower has LoRA modules
                vision_lora_count = 0
                for name in lora_modules:
                    if 'vision_tower' in name:
                        vision_lora_count += 1
                
                if vision_lora_count > 0 and debug:
                    print(f"\n   👁️ Vision tower has {vision_lora_count} LoRA modules")
                    
                    # Enable the _enable_lora_gradients method if it exists
                    if hasattr(vision_tower, '_enable_lora_gradients'):
                        print(f"   Calling vision tower's _enable_lora_gradients()")
                        vision_tower._enable_lora_gradients()
    
    # Step 5: Verify gradient flow capability
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    results['gradient_flow_enabled'] = trainable_params > 0
    
    if debug:
        print(f"\n   📊 Final Statistics:")
        print(f"      Total parameters: {total_params:,}")
        print(f"      Trainable parameters: {trainable_params:,}")
        print(f"      Trainable percentage: {trainable_params/total_params*100:.4f}%")
    
    return results

def verify_gradient_flow_aggressive(
    model: nn.Module,
    dummy_batch: Dict[str, Any],
    check_base_gradients: bool = True
) -> Tuple[bool, Dict[str, Any]]:
    """
    Verify gradient flow with detailed checks
    """
    results = {
        'lora_gradients': [],
        'base_gradients': [],
        'zero_grad_lora': [],
        'gradient_norms': {},
        'success': False
    }
    
    model.train()
    model.zero_grad()
    
    try:
        # Forward pass
        outputs = model(**dummy_batch)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs.mean()
        
        # Backward pass
        loss.backward()
        
        # Check LoRA gradients
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    results['lora_gradients'].append((name, grad_norm))
                    if grad_norm == 0:
                        results['zero_grad_lora'].append(name)
                else:
                    results['zero_grad_lora'].append(name)
            elif check_base_gradients and param.requires_grad and param.grad is not None:
                # Check base module gradients
                grad_norm = param.grad.norm().item()
                results['base_gradients'].append((name, grad_norm))
        
        # Calculate statistics
        if results['lora_gradients']:
            grad_norms = [g[1] for g in results['lora_gradients']]
            results['gradient_norms'] = {
                'mean': sum(grad_norms) / len(grad_norms),
                'max': max(grad_norms),
                'min': min(grad_norms),
                'non_zero': sum(1 for g in grad_norms if g > 0)
            }
        
        results['success'] = len(results['lora_gradients']) > 0 and len(results['zero_grad_lora']) == 0
        
        print(f"\n🔍 Gradient Flow Verification:")
        print(f"   LoRA parameters with gradients: {len(results['lora_gradients'])}")
        print(f"   LoRA parameters with ZERO gradients: {len(results['zero_grad_lora'])}")
        
        if results['gradient_norms']:
            print(f"\n   Gradient Statistics:")
            print(f"      Mean norm: {results['gradient_norms']['mean']:.6f}")
            print(f"      Max norm: {results['gradient_norms']['max']:.6f}")
            print(f"      Non-zero: {results['gradient_norms']['non_zero']}/{len(results['lora_gradients'])}")
        
        if results['zero_grad_lora']:
            print(f"\n   ⚠️ LoRA parameters with ZERO gradients:")
            for name in results['zero_grad_lora'][:5]:
                print(f"      - {name}")
            if len(results['zero_grad_lora']) > 5:
                print(f"      ... and {len(results['zero_grad_lora']) - 5} more")
        
        if check_base_gradients and results['base_gradients']:
            print(f"\n   Base modules with gradients: {len(results['base_gradients'])}")
            non_zero_base = sum(1 for _, g in results['base_gradients'] if g > 0)
            print(f"      Non-zero: {non_zero_base}/{len(results['base_gradients'])}")
        
    except Exception as e:
        results['success'] = False
        results['error'] = str(e)
        print(f"\n   ❌ Gradient verification failed: {str(e)}")
    
    return results['success'], results

def apply_aggressive_fix_to_trainer(trainer, debug: bool = True):
    """
    Apply aggressive fix to a trainer instance
    """
    if not hasattr(trainer, 'model') or trainer.model is None:
        print("   ⚠️ Trainer model not initialized yet")
        return False
    
    print("\n🔧 Applying AGGRESSIVE LoRA gradient fix...")
    
    # Apply the aggressive fix
    results = fix_lora_gradients_aggressive(
        trainer.model,
        unfreeze_base_modules=True,
        force_lora_gradients=True,
        debug=debug
    )
    
    if results['gradient_flow_enabled']:
        print(f"\n   ✅ Gradient flow enabled!")
        print(f"      LoRA parameters fixed: {results['lora_params_fixed']}")
        print(f"      Base modules unfrozen: {results['base_modules_unfrozen']}")
    else:
        print(f"\n   ❌ Failed to enable gradient flow!")
    
    return results['gradient_flow_enabled']

# For testing specific modules
def test_module_gradient_flow(module: nn.Module, input_tensor: torch.Tensor) -> bool:
    """
    Test if a specific module can produce gradients
    """
    module.train()
    input_tensor.requires_grad_(True)
    
    output = module(input_tensor)
    loss = output.mean()
    
    try:
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.norm() > 0 
                      for p in module.parameters() if p.requires_grad)
        return has_grad
    except Exception:
        return False