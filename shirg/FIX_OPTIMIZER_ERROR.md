# SHIRG LoRA Training Test Fix

## Problem
When running `test_8gpu.sh`, the test suite failed at test 15/16 with the error:
```
Training Step Simulation: 'NoneType' object has no attribute 'step'
```

## Root Cause
The error occurred in `train_shirg_lora.py` at line 543:
```python
self.optimizer.step()
AttributeError: 'NoneType' object has no attribute 'step'
```

The `test_training_step` function in `test_shirg_lora_pretrain.py` was creating a trainer and setting up the model, but not initializing the optimizer before calling `trainer.training_step()`.

## Solution
Added optimizer setup in the test before calling the training step:

```python
# Setup model (this tests the full model loading pipeline)
print(f"   Setting up model...")
trainer.setup_model()

# SHIRG-FIX: 2025-07-30 - Setup optimizer before training step
# ISSUE: test_training_step calls trainer.training_step without setting up optimizer
# SOLUTION: Call setup_optimizer_scheduler to initialize self.optimizer
# LAVIDA IMPACT: None - just test infrastructure
# SHIRG IMPACT: Allows training step test to complete successfully
print(f"   Setting up optimizer...")
# Estimate number of training steps (just 1 for test)
num_training_steps = 1
trainer.setup_optimizer_scheduler(num_training_steps)
```

## Files Modified
- `shirg/test_shirg_lora_pretrain.py` - Added optimizer setup in `test_training_step` function

## Testing
Created `test_optimizer_fix.py` to quickly verify the fix works:
```bash
cd ~/LaViDa_mod
python shirg/test_optimizer_fix.py
```

## Impact
- **LaViDa Impact**: None - this only affects test infrastructure
- **SHIRG Impact**: Allows the complete test suite to pass, enabling 8-GPU training

## Next Steps
After applying this fix, re-run the full test suite:
```bash
bash shirg/test_8gpu.sh
```

All 16 tests should now pass successfully.