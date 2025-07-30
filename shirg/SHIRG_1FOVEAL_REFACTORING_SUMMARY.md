# SHIRG 1-Foveal View Refactoring Summary

## Overview
This document summarizes all changes made to refactor the SHIRG implementation from using 2 foveal views to 1 foveal view, as per the updated research proposal.

## Configuration Changes

### Before (2 foveal views):
- Global: 384² → 196 tokens (pooled)
- Foveal: 2×448² crops → 2×784 raw tokens each
- Selection: Keep exactly 50% from each crop = 2×392 = 784 tokens
- Total: 196 + 784 = 980 tokens

### After (1 foveal view):
- Global: 384² → 196 tokens (pooled)
- Foveal: 1×448² crop → 1024 raw tokens
- Selection: Keep 76.6% from single crop = 784 tokens
- Total: 196 + 784 = 980 tokens

## Files Modified

### 1. `/llava/model/multimodal_encoder/siglip_shirg.py`
**Key Changes:**
- Updated module docstring to reflect 2-view processing instead of 3-view
- Modified `forward_with_shirg()` to process 2 views instead of 3
- Changed token selection from 50% keep rate to 76.6% keep rate for the single foveal view
- Updated `extract_multiview_tokens()` to process 1 foveal view instead of 2
- Adjusted validation logic to expect 980 tokens (196 global + 784 foveal)

### 2. `/llava/model/multimodal_encoder/siglip_encoder.py`
**Key Changes:**
- Updated references from 5-view anyres processing to 2-view format
- Changed expected token counts from ~1832 to 980
- Updated documentation and comments to reflect the new architecture

### 3. `/llava/mm_utils.py`
**Key Changes:**
- Renamed `process_shirg_3view_image()` to `process_shirg_2view_image()`
- Modified function to create only 1 foveal view instead of 2
- Updated all references and comments to reflect 2-view processing

### 4. `/shirg/real_ocr_vqa_model_runner.py`
**Key Changes:**
- Updated configuration comments from 3-view to 2-view mode
- Modified print statements to reflect "2-view processing" instead of "3-view processing"

### 5. `/shirg/lavida_shirg_integration.py`
**Key Changes:**
- Updated `target_tokens` from 729 to 980 in default configuration
- Updated all references to maintain consistency with new token count

## Key Implementation Details

### Token Selection Algorithm
The new implementation uses a single Top-K operation with 76.6% keep rate:
```python
if actual_tokens == 1024:  # 448² with patch_size=14 → 32×32
    K = 784  # Exactly 76.6% of 1024
else:
    # Fallback: maintain 76.6% ratio for other resolutions
    K = int(actual_tokens * 0.766)
```

### Benefits of the New Architecture
1. **Simpler Implementation**: No need for complex deduplication between overlapping crops
2. **Computational Efficiency**: Process only 1 high-res view instead of 2
3. **Higher Keep Ratio**: Keep 76.6% of tokens instead of 50%, preserving more information
4. **Same Output**: Still produces exactly 980 tokens for cache compatibility

## Testing Recommendations
1. Verify that the image preprocessing correctly creates 2 views (1 global + 1 foveal)
2. Confirm that token selection produces exactly 980 tokens
3. Test that LaViDa's cache compatibility is maintained
4. Validate OCR/VQA performance with the new configuration

## Date
Refactoring completed: 2025-07-30