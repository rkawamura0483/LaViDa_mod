# SHIRG LoRA Gradient Fix Documentation

## Problem Description

When running SHIRG LoRA training with 8 GPUs, there are two critical issues:

1. **Zero Gradients**: All LoRA parameters had zero gradients, preventing the model from learning
2. **Device Mismatch**: Components on different devices (CPU/GPU) causing gradient computation failures

### Root Causes

#### Issue 1: Frozen Parameters
The vision tower is frozen BEFORE LoRA adapters are applied:

1. In `siglip_encoder.py`, the vision tower is frozen: `self.vision_tower.requires_grad_(False)`
2. When PEFT applies LoRA adapters, the new LoRA parameters inherit the frozen state
3. This results in all LoRA parameters having `requires_grad=False`, preventing gradient computation

#### Issue 2: Device Mismatch
In multi-GPU setups, components can end up on different devices:

1. Model loaded with `device_map="auto"` distributes layers across GPUs
2. LoRA training requires all components on same device for gradient computation
3. Tokenizer creates tensors on CPU by default
4. This causes "Expected all tensors to be on the same device" errors

### Why This Happens

- LaViDa freezes the vision tower to save memory and computation
- The `_enable_lora_gradients()` method tries to unfreeze specific layers
- However, this method runs BEFORE LoRA is applied via PEFT
- PEFT LoRA parameters are created AFTER model initialization, inheriting the frozen state
- Multi-GPU setups with device_map cause additional device placement issues

## Solution

We created an enhanced fix that addresses both issues:

### 1. Created Enhanced Fix: `fix_lora_gradients_enhanced.py`

This enhanced module provides:
- `diagnose_device_mismatch()`: Identifies components on different devices
- `ensure_model_on_single_device()`: Moves all components to same device
- `ensure_lora_parameters_trainable_enhanced()`: Fixes both gradient and device issues
- `verify_gradient_flow_enhanced()`: Enhanced verification with device checks
- `apply_comprehensive_fix()`: One-step fix for trainer instances

Key features:
- Diagnoses and fixes device placement issues
- Ensures all LoRA parameters are trainable
- Handles meta tensors and buffer placement
- Provides detailed debugging information

### 2. Original Fix: `fix_lora_gradients.py`

The original module (kept for compatibility) provides:
- `ensure_lora_parameters_trainable()`: Finds all LoRA parameters and ensures `requires_grad=True`
- `verify_lora_gradients()`: Tests that gradients flow properly
- `apply_lora_gradient_fix_to_trainer()`: Applies fix to trainer instances

### 3. Updated Training Scripts

Updated both single and multi-GPU training scripts to use enhanced fix:

**train_shirg_lora.py**:
```python
# After applying LoRA with PEFT
from shirg.fix_lora_gradients_enhanced import ensure_lora_parameters_trainable_enhanced

results = ensure_lora_parameters_trainable_enhanced(
    self.model,
    device=target_device,
    fix_device_mismatch=True
)
```

**train_shirg_lora_multi_gpu.py**:
```python
# Apply comprehensive fix for multi-GPU
from shirg.fix_lora_gradients_enhanced import apply_comprehensive_fix

apply_comprehensive_fix(self, force_device=f"cuda:{local_rank}")
```

Also fixed collate_fn to ensure tensors are created on correct device.

### 4. Testing

Created test scripts to verify the fixes:
- `test_gradient_fix.py`: Tests original gradient fix
- `test_gradient_fix.sh`: Bash script for original test
- `test_enhanced_gradient_fix.py`: Tests enhanced fix with device checks
- `test_enhanced_gradient_fix.sh`: Bash script for enhanced test

## How to Use

1. **For Training**: The fix is automatically applied in `train_shirg_lora.py` and `train_shirg_lora_multi_gpu.py`

2. **For Testing**: Run the enhanced gradient fix test:
   ```bash
   # Test enhanced fix (recommended)
   bash shirg/test_enhanced_gradient_fix.sh
   
   # Test original fix
   bash shirg/test_gradient_fix.sh
   ```

3. **For Custom Code**: Apply the enhanced fix after PEFT:
   ```python
   from peft import get_peft_model
   from shirg.fix_lora_gradients_enhanced import ensure_lora_parameters_trainable_enhanced
   
   # Apply LoRA
   model = get_peft_model(model, lora_config)
   
   # Fix both gradient and device issues
   results = ensure_lora_parameters_trainable_enhanced(
       model,
       fix_device_mismatch=True
   )
   
   # Or use the comprehensive fix for trainers
   from shirg.fix_lora_gradients_enhanced import apply_comprehensive_fix
   apply_comprehensive_fix(trainer)
   ```

## Technical Details

### What the Enhanced Fix Does

1. **Device Diagnosis & Fix**:
   - Diagnoses components on different devices
   - Moves all parameters and buffers to target device
   - Handles special cases like vision tower and projector
   - Clears GPU cache after moving components

2. **Gradient Fix**:
   - Iterates through all model parameters
   - Identifies LoRA parameters (containing "lora" in name)
   - Ensures `param.requires_grad = True` for all LoRA parameters
   - Reports detailed statistics

3. **Additional Checks**:
   - Detects meta tensors that need materialization
   - Verifies vision tower configuration
   - Checks SHIRG settings
   - Provides comprehensive debugging output

### Why It Works

- Addresses both frozen parameters AND device placement issues
- LoRA parameters need to be trainable even if base model is frozen
- All components must be on same device for gradient computation
- The fix runs AFTER PEFT creates LoRA parameters
- Only affects LoRA parameters, not base model weights
- Maintains memory efficiency while enabling training

## Verification

After applying the fix, you should see:

1. Non-zero gradient norms for LoRA parameters
2. Successful backward pass without errors
3. Model loss decreasing during training

## Impact

- **Training**: Enables successful LoRA fine-tuning
- **Memory**: No additional memory usage (only LoRA parameters are trainable)
- **Performance**: No impact on inference speed
- **Compatibility**: Works with both single and multi-GPU setups

## Troubleshooting

### Common Issues and Solutions

1. **Still getting zero gradients after fix**:
   - Ensure `SHIRG_NO_DEVICE_MAP=1` is set before model loading
   - Check that enhanced fix is being used (not just basic fix)
   - Verify all batch tensors are on same device as model

2. **Device mismatch errors persist**:
   - Use `apply_comprehensive_fix()` which handles collate_fn
   - Check tokenizer output device (may need manual .to(device))
   - Ensure image processing happens on correct device

3. **Out of Memory (OOM) errors**:
   - Reduce batch size
   - Use gradient accumulation
   - Enable gradient checkpointing
   - Use mixed precision training (bf16)

4. **Multi-GPU specific issues**:
   - Ensure DDP is applied AFTER the fix
   - Each GPU should fix its local model copy
   - Use force_device parameter in multi-GPU setups

### Debugging Commands

```python
# Check device placement
from shirg.fix_lora_gradients_enhanced import diagnose_device_mismatch
device_map = diagnose_device_mismatch(model)
print(f"Components on devices: {list(device_map.keys())}")

# Verify gradient flow
from shirg.fix_lora_gradients_enhanced import verify_gradient_flow_enhanced
success, results = verify_gradient_flow_enhanced(model, dummy_batch)
print(f"Gradient flow: {success}")
print(f"Stats: {results['gradient_stats']}")
```