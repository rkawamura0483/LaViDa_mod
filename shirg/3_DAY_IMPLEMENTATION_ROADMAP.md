# SHIRG-Fixed: 3-Day Implementation Roadmap

## Overview

This roadmap provides a **practical, risk-minimized approach** to implementing SHIRG token selection with LaViDa. The plan focuses on **stabilizing existing components** and **demonstrating measurable improvements** rather than building complex new features.

### Key Strategy Changes
- **Eliminate variance sources**: Replace adaptive-K gating with fixed K=768
- **Minimal but effective LoRA**: Target 1.4% parameters (rank-64) for cross-resolution alignment  
- **Proven techniques**: Use established methods (PrefixKV, bicubic interpolation)
- **Conservative targets**: Aim for +3-5 CIDEr improvement with 1.35× latency

---

## Day 1: Stabilize Token Selector & Setup LoRA Training (8 hours)

### Phase 1A: Replace Adaptive-K with Fixed Selection (4 hours)

**Objective**: Eliminate the two largest sources of variance (gate mis-predictions and over-merging)

#### Tasks:
1. **Remove adaptive gating MLP** from existing SHIRG implementation
   - Delete `AdaptiveKGating` class and related prediction logic
   - Set fixed `K = 768` throughout the codebase
   - Update configuration files to use fixed budget

2. **Implement SAINT-style coverage rule**
   - Add `ensure_coverage_4x4()` method to guarantee each 4×4 region keeps ≥1 token
   - This prevents spatial clustering artifacts that hurt OCR performance
   - Use existing SAINT code patterns for hierarchical region coverage

3. **Test stability of fixed selector**
   - Run basic validation on 10 test images
   - Verify consistent 768 token output across different images
   - Check coverage guarantee is working (no empty 4×4 regions)

**Success Criteria**: Fixed selector produces exactly 768 tokens per image with guaranteed spatial coverage

### Phase 1B: Setup Rank-64 LoRA Training (4 hours)

**Objective**: Prepare LoRA training for both mm_projector and early SigLIP layers

#### Tasks:
1. **Install PEFT 0.10** with proper dependencies
   ```bash
   pip install peft==0.10 accelerate bitsandbytes
   ```

2. **Configure dual LoRA setup**:
   - **Projector LoRA**: rank-64 on `mm_projector.fc1` and `mm_projector.fc2`
   - **SigLIP LoRA**: rank-64 on blocks 0-3 QKV matrices only (~1.8% params)
   - Target total: ~120M trainable parameters (1.4% of 8B model)

3. **Prepare training data**:
   - Use existing LCS-558K dataset (558K image-text pairs)
   - Setup data loading pipeline for mixed-resolution training
   - Configure batch size=16, gradient accumulation=8 for effective batch=128

4. **Launch training job**:
   - Learning rate: 7e-5 (higher than original due to increased rank)
   - Scheduler: cosine with 10% warmup
   - Target: 2 epochs, ~8 hours wall clock time
   - Monitor convergence every 500 steps

**Success Criteria**: Training job launches successfully and shows decreasing loss within first 2 hours

---

## Day 2: Cache Optimization & Benchmarking (10 hours)

### Phase 2A: Fix 672p Positional Embeddings (2 hours)

**Objective**: Enable proper 672p processing without positional drift

#### Tasks:
1. **Implement bicubic interpolation** for SigLIP positional embeddings
   - One-line change: interpolate 24×24 learned grid to 48×48
   - Standard ViT technique, zero additional parameters
   - Add `interpolate_pos_embeddings()` method to SigLIP encoder

2. **Test high-resolution processing**:
   - Verify 672p images produce 2304 tokens (48×48 grid)
   - Check for positional encoding artifacts in corner patches
   - Compare feature quality against 384p baseline

**Success Criteria**: 672p images process without errors and maintain feature quality

### Phase 2B: Integrate PrefixKV Cache Compression (6 hours)

**Objective**: Manage memory pressure from increased token count (729→912)

#### Tasks:
1. **Install PrefixKV** dependency:
   ```bash
   pip install prefixkv
   ```

2. **Wrap diffusion model** with PrefixKV compression:
   - Enable 16-bit key/value compression for visual prefix only
   - Measured overhead <2ms for 1K tokens (acceptable)
   - Plug-and-play integration, no CUDA coding required

3. **Memory validation**:
   - Measure GPU VRAM usage: baseline vs SHIRG-Fixed vs full 2304
   - Target: SHIRG-Fixed ≤ 8.9 GB (vs 20 GB for full sequence)
   - Profile memory allocation patterns during inference

**Success Criteria**: SHIRG-Fixed uses ≤9 GB VRAM while maintaining <2ms cache overhead

### Phase 2C: Run Benchmark Sweep (4 hours)

**Objective**: Generate comprehensive performance comparison across 4 configurations

#### Benchmark Configurations:
1. **LaViDa-384 baseline**: Original performance reference
2. **LaViDa-672 full**: All 2304 tokens (upper bound performance)
3. **SHIRG-original**: Existing adaptive-K implementation
4. **SHIRG-Fixed**: New implementation with fixed K=768

#### Evaluation Tasks:
1. **Datasets**: ChartQA (200 samples) and DocVQA (200 samples)
2. **Metrics**: CIDEr score, 30-step latency, GPU memory usage
3. **Run in parallel**: Use 2×A100 nodes to complete in 4 hours
4. **Generate results table**: Clear performance comparison matrix

**Success Criteria**: Complete benchmark results showing SHIRG-Fixed achieves +3-5 CIDEr with ~50ms latency

---

## Day 3: Results Analysis & Documentation (Variable time)

### Phase 3A: Results Analysis (6 hours)

**Objective**: Document performance improvements and generate publishable evidence

#### Tasks:
1. **Performance analysis**:
   - Calculate ΔCIDEr for each configuration vs baseline
   - Generate latency vs accuracy trade-off curves
   - Analyze memory usage patterns across configurations

2. **Ablation studies**:
   - Fixed-K vs adaptive-K comparison
   - Coverage guarantee impact (with/without 4×4 rule)
   - LoRA rank analysis (if time permits: test rank-32 vs rank-64)

3. **Generate visualizations**:
   - Memory usage graphs (nvprof output)
   - Performance comparison charts
   - Example images showing selected vs dropped tokens

**Success Criteria**: Clear evidence of +3-5 CIDEr improvement with 1.35× latency (50ms vs 37ms baseline)

### Phase 3B: Technical Documentation (Remaining time)

**Objective**: Document implementation details and results for reproducibility

#### Tasks:
1. **Update research documentation**:
   - Revise SHIRG_RESEARCH_IDEA.md with actual results
   - Document any implementation challenges and solutions
   - Update performance expectations based on empirical results

2. **Create implementation guide**:
   - Step-by-step setup instructions
   - Configuration files and scripts
   - Troubleshooting common issues

3. **Generate technical report** (if time allows):
   - 2-3 page technical summary
   - Performance comparison table
   - Architecture diagrams showing SHIRG-Fixed pipeline

**Success Criteria**: Complete documentation enabling reproduction of results

---

## Expected Results (Conservative Estimates)

| Variant | CIDEr Δ (ChartQA) | 30-step latency | GPU VRAM (16-bit) | Success Probability |
|---------|-------------------|-----------------|-------------------|-------------------|
| Baseline 384p | — | 37 ms | 7.6 GB | — |
| 672p full seq | **+7** | 76 ms | 20 GB | Reference |
| Original SHIRG-X | +1 ↔ +3 | 48 ms | 10 GB | Current |
| **SHIRG-Fixed-R64-PKV** | **+3 ↔ +5** | **50 ms (≈1.35×)** | **8.9 GB** | **85%** |

### Risk Mitigation

**High-Risk Items**:
1. **LoRA training convergence**: Monitor loss closely, ready to adjust learning rate
2. **Memory constraints**: Use gradient checkpointing if VRAM insufficient
3. **PrefixKV compatibility**: Fallback to standard caching if integration issues

**Fallback Plans**:
- If rank-64 LoRA fails: Drop to rank-32 (still better than original)
- If PrefixKV fails: Use standard 16-bit precision without compression
- If 672p fails: Use 512p (still higher resolution than 384p baseline)

**Success Indicators**:
- Day 1: Training loss decreasing within 2 hours
- Day 2: Memory usage <9 GB, latency <55ms  
- Day 3: CIDEr improvement >+2 on evaluation datasets

---

## Resource Requirements

**Compute**: 2×8-GPU A100-80GB nodes (16 total GPUs)
- Day 1: 8 GPUs for LoRA training
- Day 2: 16 GPUs for parallel evaluation
- Day 3: 2-4 GPUs for analysis and ablations

**Storage**: ~100 GB for datasets, checkpoints, and results
**Time**: 72 hours total (8h + 10h + variable)
**Cost**: ~$250-350 for complete implementation

This roadmap provides a **realistic path to demonstrable results** while minimizing technical risk and computational requirements.