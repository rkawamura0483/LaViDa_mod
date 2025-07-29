# SHIRG Implementation Fixes - 2025-07-29

## Overview
This document summarizes the fixes implemented to resolve SHIRG-Fovea processing errors and ensure proper integration with LaViDa's anyres image processing pipeline.

## Key Issues Resolved

### 1. Input Format Mismatch
**Problem**: SHIRG expected a list of 5 views but received a stacked tensor `[5, C, H, W]` from LaViDa's `process_anyres_image`.

**Solution**: Added input format conversion in `siglip_shirg.py`:
- Detect stacked tensor format and convert to list of views
- Handle both 4D `[num_views, C, H, W]` and 5D `[B, num_views, C, H, W]` tensors
- Maintain compatibility with list inputs

### 2. Resolution Adaptation
**Problem**: SHIRG research specified 1×384² + 4×512² views, but LaViDa uses 5×384² from 768×768 grid.

**Solution**: Adapted SHIRG to work with LaViDa's existing format:
- Use LaViDa's 5×384² patches (first as global, rest as peripheral)
- Pool global view to 196 tokens as per research spec
- Apply per-view Top-K selection to 729-token peripheral views
- Adjusted K value: 45% of 729 = ~328 tokens per view

### 3. Token Flow Correction
**Original Research Target**: 196 global + 4×460 peripheral = ~2,036 tokens
**Adapted Implementation**: 196 global + 4×328 peripheral = ~1,508 tokens

This maintains the research principle of per-view selection while working within LaViDa's constraints.

### 4. Vision Tower 5D Tensor Handling
**Problem**: Fallback processing failed with "Expected 3D or 4D input to conv2d, but got 5D".

**Solution**: Added 5D tensor handling in `siglip_encoder.py`:
- Detect 5D input `[B, num_views, C, H, W]`
- Squeeze batch dimension when B=1
- Process as anyres patches

### 5. Pooling Behavior Verification
**Baseline**: Environment variable `NOT_ALWASY_DO_2DPOOL=0` enables pooling (3,645→980 tokens)
**SHIRG**: Environment variable `NOT_ALWASY_DO_2DPOOL=1` disables pooling (preserves all 3,645 tokens for selection)

## Token Processing Flow

### Baseline LaViDa
1. Image → 768×768 anyres → 5×384² patches
2. SigLIP encoder → 5×729 = 3,645 tokens
3. 2×2 pooling → 5×196 = 980 tokens
4. Projector → Language model

### SHIRG-Fovea (Adapted)
1. Image → 768×768 anyres → 5×384² patches
2. SigLIP encoder → 5×729 = 3,645 tokens
3. SHIRG selection:
   - Global view: 729 → 196 tokens (pooled)
   - 4 peripheral views: 4×729 → 4×328 tokens (Top-K)
   - Total: 196 + 1,312 = 1,508 tokens
4. Projector → Language model

## Implementation Details

### Key Files Modified
1. `llava/model/multimodal_encoder/siglip_shirg.py` - Core SHIRG processing
2. `llava/model/multimodal_encoder/siglip_encoder.py` - 5D tensor handling
3. `shirg/real_ocr_vqa_model_runner.py` - Configuration updates

### Research Methodology Adaptations
- Maintained per-view Top-K selection principle
- Adapted to LaViDa's 384² resolution constraints
- Preserved cache compatibility through static selection
- Keep ratio: 45% (within research-specified 40-50% range)

## Next Steps
1. Validate token selection quality on OCR/VQA tasks
2. Measure latency overhead (target: <30ms for selection)
3. Compare performance metrics vs baseline
4. Fine-tune LoRA weights if needed