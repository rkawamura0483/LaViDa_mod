#!/usr/bin/env python3
"""
Enhanced fix for LoRA gradient flow and device mismatch issues in SHIRG training

This addresses two critical issues:
1. Zero gradients for LoRA parameters 
2. Device mismatch causing gradient flow failures

Author: Research Implementation
Date: 2025-07-30
"""

import torch
import torch.nn as nn
from typing import Any, Optional, Dict, List, Tuple
import gc

def diagnose_device_mismatch(model: nn.Module) -> Dict[str, List[Tuple[str, torch.device]]]:
    """
    Diagnose device placement issues in the model
    
    Returns:
        Dict with lists of parameters on each device
    """
    device_map = {}
    
    for name, param in model.named_parameters():
        device_str = str(param.device)
        if device_str not in device_map:
            device_map[device_str] = []
        device_map[device_str].append((name, param.device))
    
    # Also check buffers
    for name, buffer in model.named_buffers():
        device_str = str(buffer.device)
        if device_str not in device_map:
            device_map[device_str] = []
        device_map[device_str].append((f"buffer:{name}", buffer.device))
    
    return device_map

def ensure_model_on_single_device(model: nn.Module, device: Optional[torch.device] = None) -> int:
    """
    Ensure all model components are on the same device
    
    Args:
        model: The model to fix
        device: Target device (if None, uses cuda:0 if available)
        
    Returns:
        int: Number of components moved
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    moved_count = 0
    device_issues = []
    
    # First diagnose current state
    print(f"\n🔍 Diagnosing device placement...")
    device_map = diagnose_device_mismatch(model)
    
    if len(device_map) > 1:
        print(f"   ⚠️ Model components on multiple devices:")
        for dev, components in device_map.items():
            print(f"      {dev}: {len(components)} components")
            # Show sample components
            for comp_name, _ in components[:3]:
                print(f"         - {comp_name}")
    
    # Move all parameters to target device
    for name, param in model.named_parameters():
        if param.device != device:
            try:
                # Use .data to avoid in-place operation issues
                param.data = param.data.to(device)
                if param.grad is not None:
                    param.grad.data = param.grad.data.to(device)
                moved_count += 1
                device_issues.append(f"param:{name}")
            except Exception as e:
                print(f"   ❌ Failed to move {name}: {e}")
    
    # Move all buffers to target device
    for name, buffer in model.named_buffers():
        if buffer.device != device:
            try:
                # For buffers, we need to handle them differently
                # Find the parent module and set the buffer
                parts = name.split('.')
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], buffer.to(device))
                moved_count += 1
                device_issues.append(f"buffer:{name}")
            except Exception as e:
                print(f"   ❌ Failed to move buffer {name}: {e}")
    
    # Special handling for vision tower components
    if hasattr(model, 'get_model'):
        base_model = model.get_model()
        if hasattr(base_model, 'get_vision_tower'):
            vision_tower = base_model.get_vision_tower()
            if vision_tower is not None and hasattr(vision_tower, 'to'):
                # Move entire vision tower
                vision_tower.to(device)
                print(f"   ✅ Moved vision tower to {device}")
    
    if moved_count > 0:
        print(f"\n   ✅ Moved {moved_count} components to {device}")
        if len(device_issues) <= 10:
            print(f"   Components moved:")
            for comp in device_issues:
                print(f"      - {comp}")
    else:
        print(f"   ✅ All components already on {device}")
    
    return moved_count

def ensure_lora_parameters_trainable_enhanced(
    model: nn.Module, 
    device: Optional[torch.device] = None,
    fix_device_mismatch: bool = True
) -> Dict[str, int]:
    """
    Enhanced version that fixes both gradient flow and device placement
    
    Args:
        model: The PEFT model with LoRA adapters
        device: Target device for all components
        fix_device_mismatch: Whether to fix device placement issues
        
    Returns:
        Dict with counts of fixed issues
    """
    results = {
        'unfrozen_params': 0,
        'moved_components': 0,
        'total_lora_params': 0,
        'device_mismatches_found': 0
    }
    
    # Step 1: Fix device placement if requested
    if fix_device_mismatch:
        print("\n🔧 Step 1: Fixing device placement...")
        
        # Diagnose initial state
        initial_devices = diagnose_device_mismatch(model)
        results['device_mismatches_found'] = len(initial_devices) - 1 if len(initial_devices) > 1 else 0
        
        if results['device_mismatches_found'] > 0:
            results['moved_components'] = ensure_model_on_single_device(model, device)
            
            # Clear GPU cache after moving
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
    
    # Step 2: Fix LoRA parameter gradients
    print("\n🔧 Step 2: Fixing LoRA parameter gradients...")
    
    lora_params_info = []
    device_check = {}
    
    # Find all LoRA parameters
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            lora_params_info.append({
                'name': name,
                'shape': param.shape,
                'was_frozen': not param.requires_grad,
                'device': param.device,
                'dtype': param.dtype
            })
            
            # Track device for each param
            device_str = str(param.device)
            if device_str not in device_check:
                device_check[device_str] = 0
            device_check[device_str] += 1
            
            if not param.requires_grad:
                param.requires_grad = True
                results['unfrozen_params'] += 1
    
    results['total_lora_params'] = len(lora_params_info)
    
    # Print summary
    print(f"\n   📊 LoRA Parameter Summary:")
    print(f"      Total LoRA parameters: {results['total_lora_params']}")
    print(f"      Parameters unfrozen: {results['unfrozen_params']}")
    
    # Check device consistency
    if len(device_check) > 1:
        print(f"\n   ⚠️ LoRA parameters on multiple devices:")
        for dev, count in device_check.items():
            print(f"      {dev}: {count} parameters")
    else:
        single_device = list(device_check.keys())[0] if device_check else "none"
        print(f"   ✅ All LoRA parameters on {single_device}")
    
    # Step 3: Additional checks for common issues
    print("\n🔧 Step 3: Additional checks...")
    
    # Check for meta tensors
    meta_tensors_found = 0
    for name, param in model.named_parameters():
        if param.is_meta:
            meta_tensors_found += 1
            print(f"   ⚠️ Meta tensor found: {name}")
    
    if meta_tensors_found > 0:
        print(f"   ❌ Found {meta_tensors_found} meta tensors - these need to be materialized!")
    
    # Check vision tower specific issues
    if hasattr(model, 'get_model'):
        base_model = model.get_model()
        if hasattr(base_model, 'get_vision_tower'):
            vision_tower = base_model.get_vision_tower()
            if vision_tower is not None:
                # Check if vision tower has proper device
                try:
                    vision_device = next(vision_tower.parameters()).device
                except StopIteration:
                    vision_device = None
                if vision_device:
                    print(f"   Vision tower on: {vision_device}")
                    
                    # Check SHIRG configuration
                    if hasattr(vision_tower, 'shirg_enabled'):
                        print(f"   SHIRG enabled: {vision_tower.shirg_enabled}")
    
    # Final statistics
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n   📊 Final Model Statistics:")
    print(f"      Total parameters: {total_params:,}")
    print(f"      Trainable parameters: {total_trainable:,}")
    print(f"      Trainable percentage: {total_trainable/total_params*100:.4f}%")
    
    return results

def verify_gradient_flow_enhanced(
    model: nn.Module, 
    dummy_batch: dict,
    check_device_consistency: bool = True
) -> Tuple[bool, Dict[str, Any]]:
    """
    Enhanced gradient verification with device consistency checks
    
    Returns:
        Tuple of (success, detailed_results)
    """
    results = {
        'has_gradients': False,
        'lora_params_with_grads': 0,
        'lora_params_zero_grads': 0,
        'device_errors': [],
        'gradient_stats': {}
    }
    
    # First check device consistency of batch
    if check_device_consistency:
        batch_devices = set()
        for key, value in dummy_batch.items():
            if torch.is_tensor(value):
                batch_devices.add(str(value.device))
        
        if len(batch_devices) > 1:
            results['device_errors'].append(f"Batch tensors on multiple devices: {batch_devices}")
            print(f"   ⚠️ Batch device inconsistency detected!")
    
    model.train()
    model.zero_grad()
    
    try:
        # Forward pass
        outputs = model(**dummy_batch)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs.mean()
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        lora_grads = []
        zero_grad_params = []
        grad_norms = []
        
        for name, param in model.named_parameters():
            if 'lora' in name.lower() and param.requires_grad:
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    lora_grads.append((name, grad_norm))
                    grad_norms.append(grad_norm)
                    if grad_norm == 0:
                        zero_grad_params.append(name)
                else:
                    zero_grad_params.append(name + " (None)")
        
        results['lora_params_with_grads'] = len(lora_grads)
        results['lora_params_zero_grads'] = len(zero_grad_params)
        results['has_gradients'] = len(lora_grads) > 0 and len(zero_grad_params) < len(lora_grads)
        
        if grad_norms:
            import numpy as np
            results['gradient_stats'] = {
                'mean': np.mean(grad_norms),
                'max': np.max(grad_norms),
                'min': np.min(grad_norms),
                'non_zero_count': sum(1 for g in grad_norms if g > 0)
            }
        
        print(f"\n🔍 Enhanced Gradient Verification:")
        print(f"   LoRA parameters with gradients: {results['lora_params_with_grads']}")
        print(f"   LoRA parameters with zero/no gradients: {results['lora_params_zero_grads']}")
        
        if results['gradient_stats']:
            print(f"\n   Gradient Statistics:")
            print(f"      Mean norm: {results['gradient_stats']['mean']:.6f}")
            print(f"      Max norm: {results['gradient_stats']['max']:.6f}")
            print(f"      Non-zero: {results['gradient_stats']['non_zero_count']}/{len(grad_norms)}")
        
        success = results['has_gradients']
        
    except Exception as e:
        success = False
        results['device_errors'].append(f"Gradient computation error: {str(e)}")
        print(f"\n   ❌ Gradient computation failed: {str(e)}")
        
        # Try to identify the source of device mismatch
        if "two devices" in str(e):
            print(f"\n   🔍 Attempting to identify device mismatch source...")
            device_map = diagnose_device_mismatch(model)
            for dev, components in device_map.items():
                if len(components) > 0:
                    print(f"      {dev}: {len(components)} components")
    
    return success, results

def apply_comprehensive_fix(trainer, force_device: Optional[str] = None):
    """
    Apply comprehensive fix to a trainer instance
    
    Args:
        trainer: ShirgLoraTrainer instance
        force_device: Force specific device (e.g., "cuda:0")
    """
    if not hasattr(trainer, 'model') or trainer.model is None:
        print("   ⚠️ Trainer model not initialized yet")
        return False
    
    print("\n🔧 Applying comprehensive LoRA fix...")
    
    # Determine target device
    if force_device:
        device = torch.device(force_device)
    elif hasattr(trainer, 'accelerator') and trainer.accelerator.device:
        device = trainer.accelerator.device
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f"   Target device: {device}")
    
    # Apply the enhanced fix
    results = ensure_lora_parameters_trainable_enhanced(
        trainer.model,
        device=device,
        fix_device_mismatch=True
    )
    
    print(f"\n   📊 Fix Results:")
    print(f"      Device mismatches found: {results['device_mismatches_found']}")
    print(f"      Components moved: {results['moved_components']}")
    print(f"      LoRA parameters unfrozen: {results['unfrozen_params']}")
    print(f"      Total LoRA parameters: {results['total_lora_params']}")
    
    # Also ensure data collation happens on the right device
    if hasattr(trainer, 'collate_fn'):
        original_collate = trainer.collate_fn
        
        def device_aware_collate(batch):
            """Wrapper to ensure batch is on correct device"""
            result = original_collate(batch)
            
            # Move tensor outputs to correct device
            for key, value in result.items():
                if torch.is_tensor(value) and value.device != device:
                    result[key] = value.to(device)
            
            return result
        
        trainer.collate_fn = device_aware_collate
        print(f"   ✅ Updated collate_fn to ensure device consistency")
    
    return True

# For backward compatibility
ensure_lora_parameters_trainable = ensure_lora_parameters_trainable_enhanced
verify_lora_gradients = lambda model, batch: verify_gradient_flow_enhanced(model, batch)[0]