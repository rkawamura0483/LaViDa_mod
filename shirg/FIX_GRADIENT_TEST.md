# SHIRG LoRA Training Gradient Test Fix

## Problem
When running `test_8gpu.sh`, the test suite failed at test 15/16 with:
```
Training Step Simulation: No gradients computed
```

## Root Cause Analysis
1. The `training_step` method in `train_shirg_lora.py` calls `optimizer.zero_grad()` after `optimizer.step()` (line 546)
2. The test was checking for gradients AFTER the training step completed, but by then gradients were already cleared
3. This made it impossible to verify if gradients were actually computed

## Solution
Modified `test_shirg_lora_pretrain.py` with a two-part approach:

### Part 1: Separate Gradient Test
Added a separate forward/backward pass BEFORE the actual training step to verify gradients:
```python
# Test gradient computation with a simple forward/backward
trainer.model.train()
trainer.model.zero_grad()

# Forward pass
outputs = trainer.model(...)
test_loss = outputs.loss

# Backward pass  
test_loss.backward()

# Check gradients NOW before they're cleared
grad_check_passed = False
for name, param in trainer.model.named_parameters():
    if param.requires_grad and param.grad is not None:
        if "lora" in name.lower():
            grad_check_passed = True
```

### Part 2: Validate Training Metrics
Instead of checking gradients after training_step (which are cleared), validate the training worked by checking:
1. Valid loss was returned (loss > 0)
2. Model has trainable parameters
3. Training step completed without errors

## Files Modified
- `shirg/test_shirg_lora_pretrain.py` - Added gradient pre-test and improved validation

## Impact
- **LaViDa Impact**: None - only affects test infrastructure
- **SHIRG Impact**: Ensures LoRA training tests pass correctly

## Testing
The test now:
1. Verifies gradients flow through LoRA parameters
2. Confirms training step produces valid loss
3. Validates the model has trainable parameters
4. Properly handles the optimizer.zero_grad() behavior

## Next Steps
Re-run the full test suite:
```bash
bash shirg/test_8gpu.sh
```

All 16 tests should now pass successfully.