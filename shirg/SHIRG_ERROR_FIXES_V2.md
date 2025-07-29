# SHIRG Integration Error Fixes - Version 2

## Date: 2025-07-29

## Error Summary
The SHIRG integration was encountering multiple errors:
1. Tensor indexing error in diversity selection
2. Split_sizes error when processing SHIRG tokens
3. Pooling error when trying to reshape 1508 tokens to 27×27 grid

## Root Causes
1. **Diversity Selection**: Complex iterative selection with tensor indexing issues
2. **Split Logic**: LaViDa expects to split 5 views but SHIRG returns concatenated tokens
3. **Pooling Logic**: LaViDa tries to apply 2D pooling to all tokens, but SHIRG tokens aren't in grid format
4. **Config Detection**: Model config missing `enable_shirg` flag

## Implemented Fixes

### 1. Simplified Diversity Selection (siglip_shirg.py)
- **Issue**: Complex iterative diversity selection caused tensor indexing errors
- **Fix**: Simplified to use top-K selection with random noise for diversity
- **Code**: Lines 342-373 in siglip_shirg.py

### 2. Robust Split Detection (llava_arch.py)
- **Issue**: Split logic failed when SHIRG returned different tensor shapes
- **Fix**: Added robust detection based on actual vs expected image counts
- **Code**: Lines 392-429 in llava_arch.py

### 3. Pooling Bypass for SHIRG (llava_arch.py)
- **Issue**: get_2dPool tried to reshape 1508 tokens to 27×27 grid (729 tokens)
- **Fix**: Added detection for non-grid token counts and bypass pooling
- **Code**: Lines 197-213 in llava_arch.py

### 4. Model Config Fix (real_ocr_vqa_model_runner.py)
- **Issue**: Model config missing enable_shirg flag
- **Fix**: Set enable_shirg on both model config and vision tower
- **Code**: Lines 570-577 in real_ocr_vqa_model_runner.py

## Token Flow
1. **Input**: 5 views (1×384² + 4×384²) from LaViDa anyres
2. **SHIRG Processing**:
   - Global view: 384² → 729 tokens → pooled to 196 tokens
   - Peripheral views: 4×384² → 4×729 tokens → 4×328 selected tokens
   - Total: 196 + 1312 = 1508 tokens
3. **Output**: Single tensor [1, 1508, D] treated as one "super-view"

## Key Design Decisions
1. **Simplified Scoring**: 0.7*attention + 0.3*similarity + small noise
2. **Single-Pass Selection**: Avoid complex iterative selection
3. **Automatic Detection**: Use token count to detect SHIRG vs standard tokens
4. **Config Propagation**: Ensure enable_shirg is set at all levels

## Testing Checklist
- [x] SHIRG processes 5 views without errors
- [x] Token selection produces 1508 tokens
- [x] No split_sizes errors
- [x] No pooling reshape errors
- [ ] OCR/VQA inference produces valid responses
- [ ] Performance comparison with baseline

## Research Adherence
- ✅ Per-view Top-K selection (45% keep rate)
- ✅ Two-scale processing (global + peripheral)
- ✅ Static token selection for cache compatibility
- ✅ Target token count achieved (1508 tokens)
- ⚠️ Simplified diversity scoring (noise-based instead of iterative)