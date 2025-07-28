# SHIRG Implementation Modification Checklist

## Overview
This checklist tracks the implementation of SHIRG (Static Hierarchical Relevance Gate) modifications to the LaViDa fork. SHIRG enables high-resolution processing (672×672, 2,304 tokens) while maintaining cache compatibility through static token selection with minimal LoRA adaptation (1.4% parameters).

---

## Core SHIRG Components

### 1. Dual-Scale Token Extraction
- [ ] **Modify SigLIP encoder** (`llava/model/multimodal_encoder/siglip_encoder.py`)
  - [ ] Add SHIRG configuration flags and methods
  - [ ] Implement `extract_shirg_tokens()` method for 672×672 processing
  - [ ] Add lo-res scaffold generation: 4×4 avg pooling → 144 tokens (always kept)
  - [ ] Add hi-detail token extraction: 48×48 patches → 2,304 tokens
  - [ ] Implement patch coordinate computation: (x,y,width,height) per token
  - [ ] Add positional embedding interpolation: 24×24 → 48×48 for 672p inputs

### 2. Distance-Aware Token Selection
- [ ] **Implement SHIRG selector** (`shirg/shirg_selector.py`)
  - [ ] Distance-aware importance scoring: `s_i = 0.7×Sim_i - 0.2×||p_i-p_neighbors|| - 0.1×||p_i-center||`
  - [ ] Text-image similarity computation using cosine similarity
  - [ ] Neighbor distance calculation (8-connected adjacency, averaged)
  - [ ] Center distance calculation (Euclidean from patch to image center)
  - [ ] Fixed K=768 token selection (eliminate adaptive gating variance)
  - [ ] SAINT-style coverage guarantee ensuring each 4×4 region keeps ≥1 token

### 3. Neighbor-Aware Token Merging
- [ ] **Implement token merging** (ToMe-style approach)
  - [ ] Identify neighboring tokens with score difference < ε=0.05
  - [ ] Merge tokens using area-weighted centroid preservation
  - [ ] Update coordinate information for merged tokens
  - [ ] Maintain spatial relationships during pruning

### 4. Coordinate Embedding Integration
- [ ] **Add coordinate embedding layer** (`shirg/lavida_shirg_integration.py`)
  - [ ] Create `CoordinateEmbedding` class: ℝ⁴ → ℝ¹²⁸ linear layer
  - [ ] Integrate with LoRA training pipeline (rank-8 LoRA)
  - [ ] Add coordinate features to selected hi-detail tokens before projection
  - [ ] Ensure gradient flow for coordinate embedding training

### 5. LoRA Adaptation Pipeline
- [ ] **Configure minimal LoRA training**
  - [ ] **Projector LoRA**: rank-64 on `["mm_projector.fc1", "mm_projector.fc2"]`
  - [ ] **SigLIP LoRA**: rank-64 on `["blocks.0.attn.qkv", "blocks.1.attn.qkv", "blocks.2.attn.qkv", "blocks.3.attn.qkv"]`
  - [ ] **Coordinate LoRA**: rank-8 on `["coord_linear"]`
  - [ ] Total trainable: ~120M params (1.4% of 8B model)
  - [ ] Training configuration: LR 7e-5, batch 16×8 GPUs, 2-3 epochs, <8h training time

---

## Integration Architecture

### 6. SHIRG Selection Pipeline
- [ ] **Update integration layer** (`shirg/lavida_shirg_integration.py`)
  - [ ] Implement `forward_with_shirg()` method replacing high-res processing
  - [ ] Extract dual-scale tokens (hi-detail + scaffold)
  - [ ] Apply distance-aware scoring to hi-detail tokens
  - [ ] Perform neighbor-aware merging
  - [ ] Select top-768 hi-detail tokens
  - [ ] Combine with 144 scaffold tokens → total 912 tokens
  - [ ] Add coordinate embeddings to selected tokens
  - [ ] Process through LoRA-adapted mm_projector

### 7. Modified LaViDa Pipeline
- [ ] **Update encode_images integration**
  - [ ] Replace existing high-res processing with SHIRG pipeline
  - [ ] Handle dual-scale token processing seamlessly
  - [ ] Maintain backward compatibility with 384×384 baseline
  - [ ] Ensure static token set for prefix KV-cache compatibility

### 8. Cache Optimization
- [ ] **Integrate PrefixKV for memory efficiency**
  - [ ] Add 16-bit KV compression for visual prefix tokens
  - [ ] Reduce memory footprint by ~40% with <2ms overhead
  - [ ] Visual prefix: 912 tokens → ~3GB (vs 5GB full precision)
  - [ ] Total inference memory target: ~8.9GB (vs 20GB full high-res)

---

## Training and Evaluation

### 9. LoRA Training Pipeline
- [ ] **Setup mixed-resolution training**
  - [ ] Random sampling: 384², 512², 672² images during training
  - [ ] PEFT 0.10 integration for LoRA management
  - [ ] Separate learning rates: LoRA (7e-5), base weights (2e-5)
  - [ ] Cosine decay scheduler with 500 warmup steps
  - [ ] Monitor convergence and coordinate embedding loss component

### 10. Performance Benchmarking
- [ ] **Target performance metrics**
  - [ ] ChartQA CIDEr: +3.3 improvement (baseline 45.2 → 48.5)
  - [ ] DocVQA EM: +1.8 improvement (baseline 76.3 → 78.1)
  - [ ] EntityGrid F1: +2.2 improvement (baseline 68.1 → 70.3)
  - [ ] Latency: 50ms for 30 steps (1.35× baseline, 0.66× full high-res)
  - [ ] GPU memory: 8.9GB (1.17× baseline, 0.45× full high-res)

### 11. Comparison Baselines
- [ ] **Benchmark against key comparisons**
  - [ ] LaViDa-384 (current baseline performance)
  - [ ] LaViDa-672 (full high-res upper bound)
  - [ ] SAINT pruning (zero-shot token selection)
  - [ ] HiRes-LLaVA (full fine-tuning comparison)

---

## Implementation Schedule (3-Day Plan)

### Day 1: Core Infrastructure (8h)
**Phase 1A: Token Selection Stabilization (4h)**
- [ ] Replace adaptive-K gating with fixed K=768
- [ ] Implement SAINT-style coverage guarantee for 4×4 regions
- [ ] Fix token selector variance issues
- [ ] Basic distance-aware scoring implementation

**Phase 1B: LoRA Setup (4h)**
- [ ] Configure rank-64 LoRA on mm_projector + SigLIP blocks 0-3
- [ ] Setup training pipeline with PEFT 0.10
- [ ] Launch training job: 8 GPUs, LR 7e-5, 2 epochs
- [ ] Monitor initial convergence

### Day 2: Integration and Optimization (10h)
**Phase 2A: Technical Fixes (2h)**
- [ ] Implement bicubic interpolation: 24×24 → 48×48 grid
- [ ] One-line positional embedding fix for 672p compatibility

**Phase 2B: Cache Integration (6h)**
- [ ] Integrate PrefixKV for 16-bit KV compression
- [ ] Test memory efficiency and latency impact
- [ ] Ensure <2ms overhead target met
- [ ] Validate cache compatibility

**Phase 2C: Benchmark Setup (4h)**
- [ ] Prepare evaluation datasets: ChartQA, DocVQA, EntityGrid
- [ ] Setup 4-configuration comparison pipeline
- [ ] Run initial performance sweep

### Day 3: Results and Analysis (6h+)
**Phase 3A: Results Generation (4h)**
- [ ] Complete benchmark sweep across all configurations
- [ ] Document ΔCIDEr, latency, memory usage metrics
- [ ] Generate nvprof memory usage graphs
- [ ] Create performance trade-off analysis

**Phase 3B: Documentation (2h+)**
- [ ] Compile implementation summary
- [ ] Document key findings and limitations
- [ ] Prepare ablation study results
- [ ] Create final research documentation

---

## Technical Validation Checkpoints

### Infrastructure Validation
- [ ] **Token extraction**: Verify [B, 2304, D] hi-detail + [B, 144, D] scaffold output
- [ ] **Coordinate embedding**: Check (x,y,h,w) → 128-d projection functionality
- [ ] **Selection quality**: Ensure spatial diversity in selected 768 tokens
- [ ] **Memory efficiency**: Confirm <8.9GB GPU memory during inference
- [ ] **Cache compatibility**: Validate static token set after step 0

### Performance Validation
- [ ] **Speed maintenance**: <50ms latency for 30-step diffusion with dual-scale processing
- [ ] **Quality preservation**: No degradation on standard VQA tasks (±1% baseline)
- [ ] **Spatial reasoning**: Measurable improvement on layout-aware benchmarks (+3-5%)
- [ ] **Training convergence**: LoRA adaptation completes within 8h target

### Integration Validation
- [ ] **Pipeline compatibility**: SHIRG integrates seamlessly with LaViDa inference
- [ ] **Backward compatibility**: Can fallback to 384×384 baseline without issues
- [ ] **Error handling**: Graceful degradation when high-res processing fails
- [ ] **Resource monitoring**: Proper GPU memory tracking and cleanup

---

## Risk Mitigation Strategies

### Technical Risks
- [ ] **LoRA convergence failure**: Backup plan to reduce rank to 32, increase LR
- [ ] **Memory constraints**: Gradient checkpointing and batch size reduction ready
- [ ] **Cache compatibility issues**: Extensive prefix validation test suite
- [ ] **Selection quality degradation**: Fallback to attention-based scoring mechanism
- [ ] **Integration complexity**: Maintain original processing path as backup

### Timeline Risks
- [ ] **Training delays**: Parallel rank-32 and rank-64 jobs for redundancy
- [ ] **PrefixKV integration issues**: 32-bit cache fallback implementation ready
- [ ] **Benchmark infrastructure**: Pre-validated evaluation setup and datasets
- [ ] **Performance target shortfall**: Conservative +3 CIDEr target vs ambitious +8

---

## Success Criteria

### Must-Have (MVP)
- [ ] ✅ Dual-scale token extraction (2,304 hi-detail + 144 scaffold) functional
- [ ] ✅ Distance-aware selection algorithm working with fixed K=768
- [ ] ✅ Coordinate embedding integrated and trainable
- [ ] ✅ LoRA training converging within 8h budget
- [ ] ✅ Measurable spatial reasoning improvement >+3% on target benchmarks
- [ ] ✅ Memory usage within 8.9GB target
- [ ] ✅ Latency within 50ms target for 30-step inference

### Nice-to-Have (Full Implementation)
- [ ] ✅ Neighbor-aware token merging optimally implemented
- [ ] ✅ PrefixKV cache compression achieving 40% memory reduction
- [ ] ✅ Complete ablation study across all components
- [ ] ✅ Performance optimizations reducing overhead below 2ms
- [ ] ✅ Comprehensive comparison against HiRes-LLaVA and SAINT baselines

---

## Progress Tracking

### Overall Implementation Status
**Current Progress**: ⬜ 0% Complete

### Phase Breakdown
- **Phase 1 (Infrastructure)**: ⬜ 0/8 core tasks
- **Phase 2 (Integration)**: ⬜ 0/6 integration tasks  
- **Phase 3 (Training)**: ⬜ 0/4 training tasks
- **Phase 4 (Evaluation)**: ⬜ 0/4 evaluation tasks

### Component Status
- **Token Extraction**: ⬜ Not Started
- **Selection Algorithm**: ⬜ Not Started
- **LoRA Integration**: ⬜ Not Started
- **Cache Optimization**: ⬜ Not Started
- **Evaluation Pipeline**: ⬜ Not Started

### Timeline Adherence
- **72-hour crash schedule**: On track for intensive 3-day implementation
- **Day 1 Target**: Complete infrastructure + launch training
- **Day 2 Target**: Integration + optimization + initial benchmarks
- **Day 3 Target**: Results generation + documentation + analysis

### Critical Path Items
- [ ] LoRA training launch (Day 1 boundary)
- [ ] PrefixKV integration success (Day 2 mid-point)
- [ ] Benchmark infrastructure validation (Day 2 evening)
- [ ] Performance target achievement (Day 3 morning)

---

**Next Milestone**: Begin Phase 1 Implementation - Token Selection Stabilization
**Timeline**: 72-hour intensive development schedule
**Success Metric**: +3-5 CIDEr improvement with <50ms latency and <8.9GB memory