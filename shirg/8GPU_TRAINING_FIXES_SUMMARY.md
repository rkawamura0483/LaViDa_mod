# SHIRG 8-GPU Training Fixes Summary

## Date: 2025-07-30

This document summarizes all fixes implemented to resolve the 8-GPU training errors.

## 1. Dataset Loading Fixes

### Problem: KeyError: 'question'
The training failed with KeyError because different datasets use different field names for questions and answers.

### Solution: Robust Field Handling
Modified `dataset_loaders.py` to handle various field names:

#### ChartQA
- Uses `query` instead of `question`
- Uses `label` instead of `answer`
- Fixed in `ChartQADataset.__getitem__()`

#### InfoVQA
- Uses `query` instead of `question`
- Uses `answer` field (correct)
- Fixed in `InfoVQADataset.__getitem__()`

#### OCR-VQA
- Has multiple questions/answers per image
- Implemented flattening logic to create individual Q&A samples
- Fixed in `OCRVQADataset.__init__()` and `__getitem__()`

#### All Datasets
- Added defensive checking for missing fields
- Added fallback values for missing data
- Added robust image format handling

## 2. Selective Gradient Flow Fixes

### Problem: Inconsistent Success Rates
The selective gradient flow fix showed inconsistent behavior in distributed training:
- Sometimes: "Base parameters enabled: 36, Vision tower fixed: True"
- Sometimes: "Base parameters enabled: 0, Vision tower fixed: False"

### Root Cause
1. The model gets wrapped with DistributedDataParallel (DDP) which adds 'module.' prefix to all parameter names
2. The selective gradient flow fix was being applied BEFORE DDP wrapping
3. After DDP wrapping, the module paths change and the fix couldn't find LoRA modules

### Solution: DDP-Aware Gradient Flow

#### Fix 1: Updated `fix_lora_gradients_selective.py`
```python
# Handle DDP-wrapped models
actual_model = model
is_ddp = False
if hasattr(model, 'module'):
    # DDP wrapped model
    actual_model = model.module
    is_ddp = True

# Remove DDP prefix when analyzing module names
clean_name = name
if is_ddp and name.startswith('module.'):
    clean_name = name[7:]  # Remove 'module.' prefix
```

#### Fix 2: Reordered Operations in `train_shirg_lora_multi_gpu.py`
- Moved selective gradient flow fix to AFTER DDP wrapping
- This ensures the fix works with the final model structure
- Added proper error handling and debug logging

## 3. Additional Fixes

### ChartQA Validation Split
- ChartQA uses "val" not "validation" for the validation split
- This was already handled correctly in the dataset loader

### Memory Optimizations
- Confirmed gradient checkpointing is enabled
- Verified batch size calculations for 8xA100 40GB setup
- Ensured SHIRG_NO_DEVICE_MAP=1 is set for proper DDP training

## 4. Testing

Created comprehensive test script `test_8gpu_fixes.py` that verifies:
1. All datasets load correctly with proper field names
2. Selective gradient flow works with both regular and DDP-wrapped models
3. Distributed setup is configured correctly
4. Memory settings are appropriate for 8xA100 configuration

## 5. Key Environment Variables

For successful 8-GPU training, ensure these are set:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WORLD_SIZE=8
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export SHIRG_NO_DEVICE_MAP=1  # Critical for LoRA gradient flow
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
```

## 6. Verification Steps

Before running training:
1. Run the test script: `python shirg/test_8gpu_fixes.py`
2. Verify all tests pass
3. Check that datasets load without errors
4. Confirm selective gradient flow succeeds

During training, monitor for:
1. No KeyError during dataset loading
2. Consistent "Selective gradient flow enabled" messages
3. Non-zero LoRA gradients during training
4. No OOM errors with batch size 2 per GPU

## 7. Expected Training Configuration

For 8xA100 40GB GPUs:
- Per-GPU batch size: 2
- Gradient accumulation: 32
- Effective batch size: 512 (2 × 8 × 32)
- Learning rate: 1.8e-5
- Mixed precision: bf16
- Gradient checkpointing: enabled
- Estimated training time: 7-8 hours for 100K samples

## Summary

All critical issues have been addressed:
1. ✅ Dataset KeyError fixed with robust field handling
2. ✅ Selective gradient flow fixed for DDP-wrapped models
3. ✅ Memory and distributed settings optimized
4. ✅ Comprehensive testing implemented

The training should now run successfully on 8 GPUs without the previous errors.