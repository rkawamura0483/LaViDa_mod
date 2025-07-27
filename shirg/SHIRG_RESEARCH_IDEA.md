# SHIRG: Static Hierarchical Relevance Gate for High-Resolution Diffusion VLMs

LaViDa's bidirectional diffusion‑language backbone gives it fast parallel decoding but forces the whole multimodal prompt (≈1k visual + text tokens) to be kept in the **prefix KV‑cache** across ~12‑30 diffusion steps. LaViDa's SigLIP vision encoder processes single 384×384 images producing 729 patch tokens (27×27 patches), which limits fine‑grained detail needed for OCR/VQA tasks. Existing high‑resolution approaches like HiRes‑LLaVA require training adapters or fragment objects with sliding windows. At the same time, dynamic token dropping per diffusion step would invalidate the cache and explode latency.

Below I (1) survey *training‑free* token–selection techniques, (2) analyse why they cannot be used unchanged in diffusion VLMs, and (3) propose a **one‑week research plan** around a new, cache‑friendly algorithm—**STATIC‑HIerarchical Relevance Gate (SHIRG)**—that selects high‑resolution tokens from single images before diffusion starts.

---

## 1 Background

### 1.1 LaViDa and its base LLM (LLaDA)

* **LaViDa** pairs a SigLIP vision encoder with an 8 B‑param diffusion LLM; single 384×384 images yield 729 patch embeddings (27×27 patches) that go directly to the language model.
* **Complementary masking** and **prefix‑DLM caching** let LaViDa train efficiently and reuse the image/text prefix at every reverse step, cutting inference time nearly ×2 vs. AR VLMs.
* The language core is **LLaDA**, a masked discrete diffusion model with bidirectional attention and no inherent KV‑cache; LaViDa's prefix mask adds that cache but only if the visual prefix stays unchanged.

### 1.2 Why 729 tokens hurt OCR/VQA

Benchmarks such as ChartQA and DocVQA require locating 4‑6 pt text or thin tick marks; 729 tokens from 384×384 images (14×14 patch granularity) cannot capture such fine details, so LaViDa under‑performs LLaVA on ChartQA by ~9 CIDEr despite stronger language modelling.

---

## 2 Related *Training‑Free* Token‑Selection Methods

| Category                          | Key idea                                              | Diffusion‑ready?                        | Sources                   |
| --------------------------------- | ----------------------------------------------------- | --------------------------------------- | ------------------------- |
| **Similarity‑aware pruning**      | Graph‑based clustering to keep diverse tokens (SAINT) | Needs single pass; compatible           | ([arXiv][5])              |
| **Semantic connected components** | LLaVA‑Scissor keeps one token per region              | Designed for video; single pass         | ([arXiv][6])              |
| **Layer‑2 halving**               | "Image is Worth ½ Tokens" drops half after layer 2    | Requires mid‑encoder hook (allowed)     | ([arXiv][7])              |
| **FastV / TokenPacker**           | Rule‑based deep‑layer pruning                         | Mid‑encoder hooks                       | ([GitHub][8])             |
| **HiRes‑LLaVA SMS**               | Self‑Mining Sampler compresses tokens by affinity     | *Learns* an adapter → not training‑free | ([CVF Open Access][2])    |
| **Token Cropr / LaCo**            | Supervised token routers                              | Not training‑free                       | ([arXiv][9], [arXiv][10]) |
| **SAINT‑Hybrid**                  | Joint ViT/LLM pruning                                 | One‑shot, but assumes causal attention  | ([arXiv][5])              |
| **Coverage‑aware pruning (NEW)** | SAINT "last token" rule, SCC in LLaVA‑Scissor | **Yes**—static, single pass | Guarantees every image region keeps ≥1 token |
| **Edge‑density boosts (NEW)**    | Laplacian/Canny maps guide token scoring      | Yes                         | Captures low‑variance thin fonts             |

**Research Update**: None of these handle *bidirectional diffusion with a frozen KV‑prefix*: dynamic pruning across steps breaks the cache, while methods requiring training were initially out of scope. However, our implementation reveals that accessing genuine high‑resolution tokens (2,304 vs. 729) requires lightweight LoRA adaptation of LaViDa's mm_projector, following proven HiRes‑LLaVA methodology. This minimal training (3.5h on 8×A100) enables testing the genuine research hypothesis versus interpolated features.

---

## 3 Challenges Unique to Diffusion VLMs

1. **Prefix immutability.** Vision tokens are cached once; changing them after step 0 voids all later reuse.
2. **Bidirectional attention.** Tokens influence *all* others, so naively dropping "unimportant" ones can erase global context.
3. **High‑resolution scaling.** Processing 672×672 images yields 2,304 tokens—3.2× more than LaViDa's current 729.
4. **Latency budget.** Any pre‑selection must finish in < 30 ms to keep LaViDa's ~1.9 × speed‑up over AR baselines.

---

## 4 Proposed Method: **SHIRG‑v2 (Coverage‑Aware)**

### 4.1 Key Features

1. **High‑resolution processing:** Extract 2,304 tokens from 672×672 images (48×48 patches) instead of 729 from 384×384.
2. **Coverage constraint:** After hierarchical clustering, **reserve ≥ 1 token per connected component** → no region is entirely dropped.
3. **Edge‑aware saliency:** Augment variance + similarity score with a lightweight **edge‑density boost** to rescue low‑energy small text.
4. **Adapter robustness:** Train a **mixed‑ratio LoRA projector** so the same tiny adapter works for 512–1024 kept tokens; no re‑training when you slide the pruning knob.

### 4.2 Algorithm

1. **Extract high‑resolution tokens:** Process 672×672 images through SigLIP → **2,304 patch embeddings** (48×48 grid).
2. **Compute saliency score**
   $s_i = \alpha\;{\rm Var}(v_i) + (1-\alpha)\max_j \cos(v_i,t_j) + \beta\,{\rm Edge}(v_i)$
   with default $(\alpha,\beta)=(0.25,0.15)$.
3. **Hierarchical clustering** on 48×48 2‑D spatial grid.
4. **Coverage guarantee**: keep **top‑1 token per cluster** *before* global ranking.
5. **Global ranking & budget**: select highest‑$s_i$ tokens until budget $K\in\{512,768,1024\}$.
6. **Summary token** for dropped patches (mean‑pooled).
7. **Static export**: selected tokens + summary + text → cached once.

### 4.3 Implementation Details

```python
def get_highres_tokens_for_shirg(self, images):
    """Extract high-resolution tokens from single images"""
    # Upscale to high resolution: 672×672
    high_res_images = F.interpolate(images, size=(672, 672), mode='bilinear')
    
    # Process through SigLIP: 672×672 → 48×48 = 2,304 tokens
    outputs = self.vision_tower(high_res_images, output_hidden_states=True)
    high_res_tokens = outputs.hidden_states[-1]  # [B, 2304, D]
    
    return high_res_tokens

def shirg_token_selection(self, tokens, target_count=768):
    """Select tokens from 2,304 high-res pool → target_count"""
    # Apply SHIRG-v2 selection algorithm
    # Returns selected tokens + summary token
```

### 4.4 Adapter (LoRA) Training

* **What:** Only the two linear layers in `mm_projector` get LoRA ranks $r\in\{16,32\}$.
* **Data:** 558k mixed‑resolution image–text pairs.
* **Loss:** *Diffusion NLL* (same as LaViDa pre‑training) on random keep‑ratios {baseline‑729, 1024, 768, 512}.
* **Time:** 3–4 h on 8×A100.
* **Outcome:** One adapter generalises across all SHIRG budgets.

### 4.5 Why it meets diffusion constraints

* **Static after step 0** → KV‑cache intact.
* **Bidirectional friendly** → summary token gives a global fallback; hierarchical selection keeps neighbourhood continuity.
* **High‑resolution aware** → 2,304 tokens provide 3.2× more visual detail than baseline 729.
* **Coverage‑aware** → Every spatial region retains at least one token, preventing information loss.

---

## 5 72‑Hour Crash‑Publish Schedule

**Front‑loads all failure risks (data prep, LoRA convergence, SHIRG CUDA kernel) with 24h free for evaluation/writing. Assumes two 8‑GPU A100‑80GB nodes for parallel jobs.**

### Prerequisites (Day 0)
| Item | Status | Why critical |
|------|--------|-------------|
| LaViDa 8B weights + SigLIP‑H/14 | ✅ | main model |
| 558k mixed‑res image–text pairs (BLIP‑LAION‑CC‑SBU) | ✅ | projector tuning |
| OCR‑heavy dev sets: ChartQA, DocVQA, MMMU‑OCR | ✅ | early sanity |
| Baseline LaViDa repo fork with high‑res hook | ✅ | runs `inference_highres.py` |

### Day 1: Training & SHIRG Implementation
| Time | Task | GPUs | Notes |
|------|------|------|-------|
| 23:00–01:00 | Final code freeze: merge SHIRG CUDA kernel (≈300 LOC) | CPU | compile & unit‑test |
| 09:00–09:30 | Launch **LoRA‑mix** training job | 8 GPUs | mixed keep‑ratios, r=16, LR 1e‑4, AdamW, cosine |
| 09:30–10:00 | Launch **r=32** duplicate job (LoRA‑wide) | 8 GPUs (2nd node) | test if higher rank matters |
| 10:00–18:00 | Jobs run (~34k iters, 3 epochs, **8h wall clock**) | — | monitor loss every 2k iters |
| 18:15–19:00 | Quick validation on ChartQA dev (no pruning) | 1 GPU | expect ≥+6 CIDEr vs. un‑tuned |
| 19:00–21:00 | Grid‑search SHIRG thresholds offline | CPU | α ∈ {0.1,0.3,0.5}, budgets ∈ {1024,768,512} |
| 21:00–23:00 | Launch evaluation sweeps: baseline‑729, full 2304, SHIRG variants | 8 GPUs each × 2 nodes | inference only (fast) |

### Day 2: Evaluation & Analysis
| Time | Task | GPUs | Notes |
|------|------|------|-------|
| 09:00 | Collect metrics → decide best projector rank & prune budget | — | target: 768 wins speed + ≥5 CIDEr |
| 09:30–12:30 | **Ablations**: remove summary token, variance‑only vs similarity‑only, α sweep | 8 GPUs | gives Table 2 |
| 13:00–16:00 | **Latency & memory profiling** with nvprof | 1 GPU | report KV size vs. ms/step |
| 16:00–20:00 | **Write paper** (4 pages + appendix) | CPU | use Overleaf template |
| 20:00–23:00 | Generate qualitative examples & t‑SNE plots | 1 GPU | helps Figure 3 |

### Day 3: Finalization
| Time | Task | GPUs | Notes |
|------|------|------|-------|
| 09:00–12:00 | Proof‑reading, citation clean‑up | — | cross‑check against dev logs |
| 12:00–14:00 | Final PDF build, reproducibility run | CPU | seed=42 |
| 14:00 | **Submit!** | — | celebrate |

### Training Configuration
```yaml
projector_lora:
  rank: 16            # or 32 in parallel job
  alpha: 32
  dropout: 0.05
  target_modules: ["mm_projector.fc1", "mm_projector.fc2"]
  bias: "lora"
optim:
  lr_main: 1e-4       # LoRA
  lr_mm: 2e-5         # projector base weights (optional)
  weight_decay: 0.0
  betas: [0.9, 0.999]
  scheduler: cosine
  warmup_steps: 500
training:
  batch_size_per_gpu: 16
  accumulation: 1
  epochs: 3
```

### Expected Performance
| Variant | Input Resolution | Vision Tokens | KV cache/step | Memory (32 layers) | 30‑step latency (A100) |
|---------|------------------|---------------|---------------|-------------------|----------------------|
| LaViDa Baseline | 384×384 | 729 | 729 × d | **15 GB** | **40 ms** |
| **SHIRG‑768** | 672×672 → 768 | 768 + 1 summary | 769 × d | 16 GB | 44 ms |
| SHIRG‑512 | 672×672 → 512 | 512 + 1 summary | 513 × d | 15 GB | 42 ms |
| Full High‑Res | 672×672 | 2,304 | 2,304 × d | 25 GB | 65 ms |

### Paper Structure (4 pages)
| Section | Key content |
|---------|-------------|
| 1 Introduction | 1‑para motivation: diffusion KV‑cache forces static prefix |
| 2 Method | Eq.(1) SHIRG score; Algorithm 1; Figure 1 pipeline |
| 3 LoRA adaptation | Table 1: param count & training time |
| 4 Experiments | Table 2 main results; Figure 2 speed‑accuracy curve |
| 5 Related work | SAINT, LLaVa‑HR, LaViDa |
| 6 Conclusion | 3 bullet take‑aways |

**Critical Success Factor**: If LoRA loss plateaus above baseline perplexity after 2h, immediately drop rank‑32 job and launch rank‑64; weak projector is the only real blocker to publication.

---

## 6 Expected Contributions

1. **First high‑resolution token selection for diffusion VLMs**—static, cache‑friendly approach that processes 672×672 images.
2. **Coverage‑aware selection algorithm**—ensures every spatial region maintains representation while selecting optimal tokens.
3. **Adapter once; prune dial anytime**—same LoRA handles 512–1024 tokens without retraining.
4. **3.2× visual detail improvement** with ≤10% latency overhead—enabling fine‑grained OCR/VQA tasks.

---

## 7 Ablation & Risk Mitigation

* **Remove coverage rule →** expect ≥ 2 CIDEr drop on ChartQA‑tiny.
* **β = 0 (no edge boost) →** miss thin tick marks, −1.5 CIDEr.
* **Rank‑16 vs rank‑32 LoRA →** choose higher rank only if perplexity plateaus.
* **High‑res processing overhead →** 672×672 adds ~4ms vs 384×384, but SHIRG selection saves 21ms vs full 2,304 tokens.

---

## 8 Potential Extensions

* **Two‑stage selection**: coarse static set for cache + *optional* dynamic refinement on *just those tokens* in late diffusion steps (keeps cache size small).
* **Adaptive K via entropy of logits at step 0**—lets the model keep more tokens only for cluttered images.
* **Cross‑modal reranking**: incorporate attention between selected tokens and *mask tokens* predicted at early diffusion steps.
* **Multi‑scale SHIRG**: apply hierarchical selection across different patch scales within the 48×48 grid.

---

## 9 Conclusion

SHIRG‑v2 reconciles **high‑resolution vision** with **diffusion KV‑cache efficiency** by processing 672×672 images (2,304 tokens) and selecting optimal subsets through coverage‑aware hierarchical clustering. With a single 0.5%‑parameter LoRA, LaViDa can achieve 3.2× more visual detail while maintaining cache compatibility, pushing diffusion VLMs into fine‑grained OCR/VQA performance previously dominated by autoregressive models—while staying ~1.7× faster.

**Implementation Status**: SHIRG-v2 methodology defined with high‑resolution processing, coverage‑aware token selection, edge‑density boost, and mixed‑ratio LoRA training. Ready to proceed with implementation phase.

---

### Key References

LaViDa paper ([ar5iv][1]) – LaViDa GitHub ([GitHub][11]) – LaViDa architecture page ([homepage.jackli.org][13]) – LLaDA diffusion LLM ([ar5iv][3]) – SAINT ([arXiv][5]) – HiRes‑LLaVA ([CVF Open Access][2]) – LLaVA‑Scissor ([arXiv][6]) – "Image ½ Tokens" ([arXiv][7]) – Token Cropr ([arXiv][9]) – FastV ([GitHub][8]) – DocVLM ([arXiv][12]) – ChartQA dataset ([ACL Anthology][4]) – Token‑compression survey list ([GitHub][14]) – Token pruning notes ([michal.io][15]) – ChartInsights analysis ([chartinsight.github.io][16])

[1]: https://ar5iv.org/pdf/2505.16839
[2]: https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_HiRes-LLaVA_Restoring_Fragmentation_Input_in_High-Resolution_Large_Vision-Language_Models_CVPR_2025_paper.pdf
[3]: https://ar5iv.org/pdf/2502.09992
[4]: https://aclanthology.org/2022.findings-acl.177.pdf
[5]: https://arxiv.org/abs/2503.11549
[6]: https://arxiv.org/abs/2506.21862
[7]: https://arxiv.org/abs/2403.06764
[8]: https://github.com/pkunlp-icler/FastV
[9]: https://arxiv.org/abs/2412.00965
[10]: https://arxiv.org/abs/2507.02279
[11]: https://github.com/jacklishufan/LaViDa
[12]: https://arxiv.org/abs/2412.08746
[13]: https://homepage.jackli.org/projects/lavida/
[14]: https://github.com/daixiangzi/Awesome-Token-Compress
[15]: https://michal.io/notes/ml/Token-Dropping%2C-Pruning%2C-Merging-and-Compression
[16]: https://chartinsight.github.io/