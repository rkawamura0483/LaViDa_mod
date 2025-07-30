#!/usr/bin/env python3
"""
Selective gradient flow fix for SHIRG LoRA training

The key insight: PEFT's LoRA implementation requires gradient flow through
base modules to reach LoRA adapters, but we don't want to update base weights.

Solution: Enable requires_grad=True on base modules with LoRA, but ensure
only LoRA parameters are in the optimizer.

Author: Research Implementation  
Date: 2025-07-30
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Set
import gc


def find_modules_with_lora(model: nn.Module) -> Dict[str, Any]:
    """
    Find all modules that have LoRA adapters and their base modules
    """
    modules_info = {
        'lora_modules': {},  # module_path -> lora_params
        'base_modules': {},  # module_path -> base_module
        'lora_param_names': set(),
        'base_param_names': set()
    }
    
    # First, find all LoRA parameters
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            modules_info['lora_param_names'].add(name)
            
            # Extract base module path
            parts = name.split('.')
            base_path = []
            for i, part in enumerate(parts):
                if 'lora' in part.lower():
                    # Found LoRA component, base is everything before
                    base_name = '.'.join(base_path)
                    if base_name not in modules_info['lora_modules']:
                        modules_info['lora_modules'][base_name] = []
                    modules_info['lora_modules'][base_name].append(name)
                    break
                base_path.append(part)
    
    # Now find the actual base modules
    for base_path in modules_info['lora_modules']:
        try:
            # Navigate to the module
            parts = base_path.split('.')
            module = model
            for part in parts:
                if hasattr(module, part):
                    module = getattr(module, part)
                elif part == 'base_model' or part == 'model':
                    # Skip these wrapping layers
                    continue
                else:
                    break
            
            if isinstance(module, nn.Module):
                modules_info['base_modules'][base_path] = module
                
                # Find base parameters
                for param_name, param in module.named_parameters():
                    if 'lora' not in param_name.lower():
                        full_name = f"{base_path}.{param_name}"
                        modules_info['base_param_names'].add(full_name)
        except:
            pass
    
    return modules_info


def apply_selective_gradient_flow(
    model: nn.Module,
    debug: bool = True
) -> Dict[str, Any]:
    """
    Enable gradient flow through base modules that have LoRA adapters
    while keeping other modules frozen
    """
    results = {
        'modules_with_lora': 0,
        'base_params_enabled': 0,
        'lora_params_found': 0,
        'vision_tower_fixed': False,
        'success': False
    }
    
    if debug:
        print("\n🔧 Applying Selective Gradient Flow Fix")
        print("   Enabling gradients on base modules with LoRA adapters")
    
    # Step 1: Find modules with LoRA
    modules_info = find_modules_with_lora(model)
    results['modules_with_lora'] = len(modules_info['lora_modules'])
    results['lora_params_found'] = len(modules_info['lora_param_names'])
    
    if debug:
        print(f"   Found {results['modules_with_lora']} modules with LoRA adapters")
        print(f"   Found {results['lora_params_found']} LoRA parameters")
    
    # Step 2: Enable gradients on base modules that have LoRA
    enabled_count = 0
    for base_path, base_module in modules_info['base_modules'].items():
        if hasattr(base_module, 'weight') and base_module.weight is not None:
            if not base_module.weight.requires_grad:
                base_module.weight.requires_grad = True
                enabled_count += 1
                
        if hasattr(base_module, 'bias') and base_module.bias is not None:
            if not base_module.bias.requires_grad:
                base_module.bias.requires_grad = True
                enabled_count += 1
    
    results['base_params_enabled'] = enabled_count
    
    # Step 3: Special handling for vision tower
    if hasattr(model, 'get_model'):
        base_model = model.get_model()
        if hasattr(base_model, 'get_vision_tower'):
            vision_tower = base_model.get_vision_tower()
            if vision_tower is not None:
                # Check if vision tower has LoRA modules
                vision_lora_count = sum(1 for path in modules_info['lora_modules'] 
                                      if 'vision_tower' in path)
                
                if vision_lora_count > 0:
                    # Enable gradient flow through vision tower
                    # But don't globally unfreeze - just enable critical paths
                    
                    # Enable embeddings for gradient flow
                    if hasattr(vision_tower, 'vision_tower'):
                        inner_tower = vision_tower.vision_tower
                        if hasattr(inner_tower, 'vision_model'):
                            vision_model = inner_tower.vision_model
                            
                            # Enable embeddings
                            if hasattr(vision_model, 'embeddings'):
                                for param in vision_model.embeddings.parameters():
                                    if not param.requires_grad:
                                        param.requires_grad = True
                                        results['vision_tower_fixed'] = True
                            
                            # Enable encoder layers that have LoRA
                            if hasattr(vision_model, 'encoder'):
                                encoder = vision_model.encoder
                                
                                # Enable layer norm
                                if hasattr(encoder, 'layers'):
                                    for i, layer in enumerate(encoder.layers):
                                        # Check if this layer has LoRA
                                        layer_has_lora = any(f"layers.{i}." in path 
                                                           for path in modules_info['lora_modules'])
                                        if layer_has_lora:
                                            # Enable self-attention parameters
                                            if hasattr(layer, 'self_attn'):
                                                attn = layer.self_attn
                                                for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                                                    if hasattr(attn, name):
                                                        proj = getattr(attn, name)
                                                        if hasattr(proj, 'weight'):
                                                            proj.weight.requires_grad = True
                    
                    if debug and results['vision_tower_fixed']:
                        print(f"   ✅ Enabled gradient flow in vision tower ({vision_lora_count} LoRA modules)")
    
    # Step 4: Ensure all LoRA parameters have requires_grad=True
    lora_fixed = 0
    for name, param in model.named_parameters():
        if 'lora' in name.lower() and not param.requires_grad:
            param.requires_grad = True
            lora_fixed += 1
    
    if lora_fixed > 0 and debug:
        print(f"   Fixed {lora_fixed} LoRA parameters with requires_grad=False")
    
    # Calculate success
    results['success'] = results['base_params_enabled'] > 0 or results['vision_tower_fixed']
    
    if debug:
        print(f"\n   📊 Results:")
        print(f"      Base parameters enabled: {results['base_params_enabled']}")
        print(f"      Vision tower fixed: {results['vision_tower_fixed']}")
        print(f"      Success: {results['success']}")
        
        # Important note
        print(f"\n   ⚠️ IMPORTANT: Only add LoRA parameters to optimizer!")
        print(f"      Base module weights have requires_grad=True for gradient flow")
        print(f"      But they should NOT be in the optimizer")
    
    return results


def get_lora_parameters_only(model: nn.Module) -> List[nn.Parameter]:
    """
    Get only LoRA parameters for the optimizer
    """
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name.lower() and param.requires_grad:
            lora_params.append(param)
    return lora_params


def verify_selective_gradient_flow(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    debug: bool = True
) -> Dict[str, Any]:
    """
    Verify the selective gradient flow setup
    """
    results = {
        'total_params': 0,
        'params_with_grad': 0,
        'lora_params': 0,
        'base_params_with_grad': 0,
        'optimizer_params': 0,
        'setup_correct': False
    }
    
    # Count parameters
    lora_names = set()
    base_with_grad = set()
    
    for name, param in model.named_parameters():
        results['total_params'] += 1
        
        if param.requires_grad:
            results['params_with_grad'] += 1
            
            if 'lora' in name.lower():
                results['lora_params'] += 1
                lora_names.add(name)
            else:
                results['base_params_with_grad'] += 1
                base_with_grad.add(name)
    
    # Check optimizer
    if optimizer is not None:
        optimizer_params = set()
        for group in optimizer.param_groups:
            for param in group['params']:
                # Find parameter name
                for name, p in model.named_parameters():
                    if p is param:
                        optimizer_params.add(name)
                        break
        
        results['optimizer_params'] = len(optimizer_params)
        
        # Check if only LoRA params are in optimizer
        non_lora_in_optimizer = optimizer_params - lora_names
        if non_lora_in_optimizer:
            results['non_lora_in_optimizer'] = list(non_lora_in_optimizer)[:5]
    
    # Determine if setup is correct
    results['setup_correct'] = (
        results['lora_params'] > 0 and
        results['params_with_grad'] > results['lora_params'] and
        (optimizer is None or results['optimizer_params'] == results['lora_params'])
    )
    
    if debug:
        print(f"\n🔍 Selective Gradient Flow Verification:")
        print(f"   Total parameters: {results['total_params']}")
        print(f"   Parameters with requires_grad=True: {results['params_with_grad']}")
        print(f"   - LoRA parameters: {results['lora_params']}")
        print(f"   - Base parameters: {results['base_params_with_grad']}")
        
        if optimizer is not None:
            print(f"\n   Optimizer parameters: {results['optimizer_params']}")
            if 'non_lora_in_optimizer' in results:
                print(f"   ⚠️ Non-LoRA parameters in optimizer:")
                for name in results['non_lora_in_optimizer']:
                    print(f"      - {name}")
        
        print(f"\n   Setup correct: {results['setup_correct']}")
        
        if results['base_params_with_grad'] > 0:
            print(f"\n   ℹ️ Base parameters with grad=True (for gradient flow):")
            for i, name in enumerate(list(base_with_grad)[:3]):
                print(f"      - {name}")
            if len(base_with_grad) > 3:
                print(f"      ... and {len(base_with_grad) - 3} more")
    
    return results


def apply_memory_optimizations(model: nn.Module, config: Dict[str, Any]) -> None:
    """
    Apply memory optimizations to prevent OOM
    """
    # 1. Enable gradient checkpointing if available
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("   ✅ Enabled gradient checkpointing")
    
    # 2. Empty cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("   ✅ Cleared GPU cache")
    
    # 3. Set memory efficient attention if available
    if hasattr(model, 'enable_xformers_memory_efficient_attention'):
        try:
            model.enable_xformers_memory_efficient_attention()
            print("   ✅ Enabled memory efficient attention")
        except:
            pass


# For integration with trainer
def fix_trainer_optimizer(trainer, debug: bool = True):
    """
    Fix trainer to use only LoRA parameters in optimizer
    """
    if not hasattr(trainer, 'model') or trainer.model is None:
        return False
    
    # Get only LoRA parameters
    lora_params = get_lora_parameters_only(trainer.model)
    
    if debug:
        print(f"\n🔧 Fixing trainer optimizer")
        print(f"   Found {len(lora_params)} LoRA parameters")
    
    # Recreate optimizer with only LoRA parameters
    if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
        # Get optimizer settings
        old_optimizer = trainer.optimizer
        lr = old_optimizer.param_groups[0]['lr']
        betas = old_optimizer.param_groups[0].get('betas', (0.9, 0.999))
        weight_decay = old_optimizer.param_groups[0].get('weight_decay', 0.0)
        
        # Create new optimizer with only LoRA params
        trainer.optimizer = torch.optim.AdamW(
            lora_params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay
        )
        
        if debug:
            print(f"   ✅ Recreated optimizer with only LoRA parameters")
            print(f"   Optimizer now has {len(lora_params)} parameters")
    
    return True


if __name__ == "__main__":
    print("SHIRG LoRA Selective Gradient Flow Fix")
    print("=" * 60)
    print()
    print("Key insights:")
    print("1. PEFT LoRA needs gradient flow through base modules")
    print("2. Base modules need requires_grad=True for gradient flow")
    print("3. But only LoRA parameters should be in the optimizer")
    print("4. This gives us gradient flow without updating base weights")