# SHIRG-X Implementation Modification Checklist

## Overview
This checklist tracks the implementation of SHIRG-X (dual-scale, spatially aware) modifications to the LaViDa fork. SHIRG-X preserves global spatial information while selecting high-detail tokens through distance-aware scoring and coordinate embedding.

---

## Core SHIRG-X Components

### 1. Dual-Scale Token Extraction
- [ ] **Modify SigLIP encoder** (`llava/model/multimodal_encoder/siglip_encoder.py`)
  - [ ] Comment out layer removal: `# del self.vision_tower.vision_model.encoder.layers[-1:]`
  - [ ] Add SHIRG-X configuration flags
  - [ ] Implement `extract_shirg_x_tokens()` method
  - [ ] Add lo-res scaffold generation (4×4 avg pooling → 144 tokens)
  - [ ] Add patch centroid coordinate computation

- [ ] **Dual-scale token output**
  - [ ] Hi-detail tokens: 2,304 from 672² image (48×48 patches)
  - [ ] Lo-res scaffold: 144 from 12×12 avg pooling (always kept)
  - [ ] Coordinate features: (x, y, h, w) for each patch

### 2. Distance-Aware Token Selection
- [ ] **Implement TopV-style scoring** (`shirg/shirg_selector.py`)
  - [ ] Distance-aware importance: `s_i = 0.7*Sim_i - 0.2*||p_i-p_j||_2 - 0.1*||p_i-c||_2`
  - [ ] Text-image similarity computation
  - [ ] Distance to image center calculation
  - [ ] Neighbor-aware scoring (simplified pairwise distance)

- [ ] **Token merge instead of drop** (ToMe-style)
  - [ ] Identify neighboring tokens with score difference < ε = 0.05
  - [ ] Merge tokens with area-weighted centroid preservation
  - [ ] Update coordinate information for merged tokens

### 3. Centroid Coordinate Embedding
- [ ] **Add coordinate embedding layer** (`lavida_shirg_integration.py`)
  - [ ] Create `CoordinateEmbedding` class with 4→128 linear layer
  - [ ] Add to mm_projector LoRA target modules
  - [ ] Implement coordinate feature addition to hi-detail tokens
  - [ ] Ensure gradient flow for LoRA training

### 4. Instance-Adaptive Keep-Rate
- [ ] **Implement adaptive-K gating head**
  - [ ] 2-layer MLP: patch_entropy → hidden(32) → 3 budget options
  - [ ] Predict optimal K ∈ {512, 768, 1024} from patch entropy
  - [ ] Add to trainable parameters (~32k params)
  - [ ] Implement patch entropy computation

### 5. LoRA Scope Expansion
- [ ] **Update LoRA configuration**
  - [ ] Projector LoRA: rank=32, target `["mm_projector.fc1", "mm_projector.fc2"]`
  - [ ] Coordinate LoRA: rank=8, target `["coord_linear"]`
  - [ ] Total trainable: ~65M params (~0.8% of model)
  - [ ] Training time: ~5h on 8×A100

---

## Integration Points

### 6. SHIRG-X Selection Pipeline
- [ ] **Update `forward_with_shirg_x()` method**
  - [ ] Extract dual-scale tokens
  - [ ] Apply distance-aware scoring to hi-detail tokens
  - [ ] Perform neighbor merging
  - [ ] Select top-K hi-detail tokens
  - [ ] Combine with lo-res scaffold (K + 144 tokens)
  - [ ] Add coordinate embeddings

### 7. Modified encode_images
- [ ] **Update integration patch** (`lavida_shirg_integration.py`)
  - [ ] Replace `forward_with_high_res()` with `forward_with_shirg_x()`
  - [ ] Handle dual-scale token processing
  - [ ] Add coordinate embedding to selected tokens
  - [ ] Maintain backward compatibility

### 8. Training Pipeline Updates
- [ ] **Mixed-budget training**
  - [ ] Train with budgets [512, 768, 1024] randomly sampled
  - [ ] Add coordinate embedding loss component
  - [ ] Add adaptive-K prediction loss (weight: 0.1)
  - [ ] Separate learning rates: projector (2e-4), coord (1e-3)

---

## Performance Targets

### 9. Latency & Memory
- [ ] **Performance benchmarks**
  - [ ] SHIRG-X-768: 48ms for 30 steps (target)
  - [ ] SHIRG-X-512: 46ms for 30 steps (target)
  - [ ] Memory usage: <18GB per A100
  - [ ] Selection overhead: <2ms (well within 30ms budget)

### 10. Quality Metrics
- [ ] **Spatial reasoning improvements**
  - [ ] ChartQA: +8 CIDEr improvement (target)
  - [ ] EntityGrid-QA: +12 F1 improvement (target)
  - [ ] DocVQA: +3 EM improvement (target)
  - [ ] Maintain baseline performance on non-spatial tasks

---

## Implementation Order

### Phase 1: Core Infrastructure (Day 1)
1. [ ] Modify SigLIP encoder for dual-scale extraction
2. [ ] Implement distance-aware scoring algorithm
3. [ ] Add coordinate embedding layer
4. [ ] Update LoRA configuration

### Phase 2: Integration (Day 1-2)
5. [ ] Implement SHIRG-X selection pipeline
6. [ ] Update encode_images integration
7. [ ] Add adaptive-K gating head
8. [ ] Test end-to-end pipeline

### Phase 3: Training (Day 2)
9. [ ] Setup mixed-budget training loop
10. [ ] Launch LoRA training (5h on 8×A100)
11. [ ] Monitor convergence and losses
12. [ ] Validate on development sets

### Phase 4: Evaluation (Day 2-3)
13. [ ] Run ablation studies
14. [ ] Performance profiling and optimization
15. [ ] Generate qualitative examples
16. [ ] Document results and create paper

---

## Validation Checkpoints

### Technical Validation
- [ ] **Token extraction**: Verify [B, 2304, D] hi-detail + [B, 144, D] lo-res
- [ ] **Coordinate embedding**: Check (x,y,h,w) → 128-d projection
- [ ] **Selection quality**: Ensure spatial diversity in selected tokens
- [ ] **Memory efficiency**: <18GB GPU memory during inference
- [ ] **Cache compatibility**: Static token set after step 0

### Performance Validation
- [ ] **Spatial reasoning**: Improvement on layout-aware benchmarks
- [ ] **Speed maintenance**: <50ms latency for dual-scale processing
- [ ] **Quality preservation**: No degradation on standard VQA tasks
- [ ] **Adaptive budgeting**: Appropriate K selection based on image complexity

---

## Risk Mitigation

### Fallback Strategies
- [ ] **LoRA convergence issues**: Reduce rank to 16 if training unstable
- [ ] **Memory constraints**: Reduce batch size or use gradient checkpointing
- [ ] **Selection quality**: Fall back to attention-based scoring if distance-aware fails
- [ ] **Integration errors**: Maintain original SHIRG path as backup

### Debugging Tools
- [ ] **Debug flags**: Enable detailed logging for each component
- [ ] **Visualization**: Generate token selection heatmaps
- [ ] **Profiling**: Track latency for each SHIRG-X component
- [ ] **Memory monitoring**: GPU memory usage tracking

---

## Success Criteria

### Must-Have (MVP)
- [ ] ✅ Dual-scale token extraction working
- [ ] ✅ Distance-aware selection functional
- [ ] ✅ Coordinate embedding integrated
- [ ] ✅ LoRA training converging
- [ ] ✅ Spatial reasoning improvement >+5%

### Nice-to-Have (Full Implementation)
- [ ] ✅ Adaptive-K working optimally
- [ ] ✅ Token merging implemented
- [ ] ✅ Full ablation study complete
- [ ] ✅ Performance optimizations applied
- [ ] ✅ Paper-ready results generated

---

## Completion Status

**Overall Progress**: ⬜ 0% Complete

**Phase Breakdown**:
- Phase 1 (Infrastructure): ⬜ 0/4 tasks
- Phase 2 (Integration): ⬜ 0/4 tasks  
- Phase 3 (Training): ⬜ 0/4 tasks
- Phase 4 (Evaluation): ⬜ 0/4 tasks

**Timeline**: 72-hour crash schedule
- **Day 1**: Infrastructure + Integration (16 tasks)
- **Day 2**: Training + Early Evaluation (8 tasks) 
- **Day 3**: Final Evaluation + Paper (4 tasks)

---

*Last Updated: Initial Creation*
*Next Milestone: Begin Phase 1 Implementation*