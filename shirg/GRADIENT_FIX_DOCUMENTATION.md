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

### 1. Created `fix_lora_gradients.py`

This module provides:
- `ensure_lora_parameters_trainable()`: Finds all LoRA parameters and ensures `requires_grad=True`
- `verify_lora_gradients()`: Tests that gradients flow properly
- `apply_lora_gradient_fix_to_trainer()`: Applies fix to trainer instances

### 2. Updated Training Scripts

Added the fix to `train_shirg_lora.py` after LoRA application:

```python
# After applying LoRA with PEFT
from shirg.fix_lora_gradients import ensure_lora_parameters_trainable
unfrozen_count = ensure_lora_parameters_trainable(self.model)
```

### 3. Testing

Created test scripts to verify the fix:
- `test_gradient_fix.py`: Comprehensive test of gradient flow
- `test_gradient_fix.sh`: Simple bash script to run the test

## How to Use

1. **For Training**: The fix is automatically applied in `train_shirg_lora.py` and `train_shirg_lora_multi_gpu.py`

2. **For Testing**: Run the gradient fix test:
   ```bash
   bash shirg/test_gradient_fix.sh
   ```

3. **For Custom Code**: Apply the fix after PEFT:
   ```python
   from peft import get_peft_model
   from shirg.fix_lora_gradients import ensure_lora_parameters_trainable
   
   # Apply LoRA
   model = get_peft_model(model, lora_config)
   
   # Fix frozen LoRA parameters
   ensure_lora_parameters_trainable(model)
   ```

## Technical Details

### What the Fix Does

1. Iterates through all model parameters
2. Identifies LoRA parameters (containing "lora" in name)
3. Ensures `param.requires_grad = True` for all LoRA parameters
4. Reports how many parameters were fixed

### Why It Works

- LoRA parameters need to be trainable even if base model is frozen
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