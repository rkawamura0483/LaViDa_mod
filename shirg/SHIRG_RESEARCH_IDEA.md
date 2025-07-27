How valid is below research idea? Is this really novel, and most importantly, is this really feasible? Thoroughly understand lavida and llada, as well as existing literature that bakcs or negates this idea

LaViDa’s bidirectional diffusion‑language backbone gives it fast parallel decoding but forces the whole multimodal prompt (≈1 k visual + text tokens) to be kept in the **prefix KV‑cache** across \~12‑30 diffusion steps. Because the vision encoder first produces 3 645 high‑resolution patch tokens and then *average‑pools* them down to 980, fine‑grained cues needed for OCR/VQA are already lost before the language model sees them ([ar5iv][1]).  Existing high‑resolution add‑ons—Slice/Restore windows in HiRes‑LLaVA ([CVF Open Access][2]) or multi‑view pooling in LaViDa itself ([ar5iv][1])—either fragment objects or still blur details.  At the same time, dynamic token dropping per diffusion step would invalidate the cache and explode latency.

Below I (1) survey *training‑free* token–selection techniques, (2) analyse why they cannot be used unchanged in diffusion VLMs, and (3) propose a **one‑week research plan** around a new, cache‑friendly algorithm—**STATIC‑HIerarchical Relevance Gate (SHIRG)**—that keeps only the right high‑resolution tokens once, before diffusion starts.

---

## 1 Background

### 1.1 LaViDa and its base LLM (LLaDA)

* **LaViDa** pairs a SigLIP vision encoder with an 8 B‑param diffusion LLM; five image views (4×336² + 1×672²) yield 3 645 patch embeddings, which are average‑pooled to 980 tokens to fit the context window ([ar5iv][1]).
* **Complementary masking** and **prefix‑DLM caching** let LaViDa train efficiently and reuse the image/text prefix at every reverse step, cutting inference time nearly ×2 vs. AR VLMs ([ar5iv][1]).
* The language core is **LLaDA**, a masked discrete diffusion model with bidirectional attention and no inherent KV‑cache; LaViDa’s prefix mask adds that cache but only if the visual prefix stays unchanged ([ar5iv][3]).

### 1.2 Why pooled tokens hurt OCR/VQA

Benchmarks such as ChartQA and DocVQA require locating 4‑6 pt text or thin tick marks; pooled 14×14 CLIP patches smear those pixels, so LaViDa under‑performs LLaVA on ChartQA by \~9 CIDEr despite stronger language modelling ([ACL Anthology][4], [ar5iv][1]).

---

## 2 Related *Training‑Free* Token‑Selection Methods

| Category                          | Key idea                                              | Diffusion‑ready?                        | Sources                   |
| --------------------------------- | ----------------------------------------------------- | --------------------------------------- | ------------------------- |
| **Similarity‑aware pruning**      | Graph‑based clustering to keep diverse tokens (SAINT) | Needs single pass; compatible           | ([arXiv][5])              |
| **Semantic connected components** | LLaVA‑Scissor keeps one token per region              | Designed for video; single pass         | ([arXiv][6])              |
| **Layer‑2 halving**               | “Image is Worth ½ Tokens” drops half after layer 2    | Requires mid‑encoder hook (allowed)     | ([arXiv][7])              |
| **FastV / TokenPacker**           | Rule‑based deep‑layer pruning                         | Mid‑encoder hooks                       | ([GitHub][8])             |
| **HiRes‑LLaVA SMS**               | Self‑Mining Sampler compresses tokens by affinity     | *Learns* an adapter → not training‑free | ([CVF Open Access][2])    |
| **Token Cropr / LaCo**            | Supervised token routers                              | Not training‑free                       | ([arXiv][9], [arXiv][10]) |
| **SAINT‑Hybrid**                  | Joint ViT/LLM pruning                                 | One‑shot, but assumes causal attention  | ([arXiv][5])              |

**Research Update**: None of these handle *bidirectional diffusion with a frozen KV‑prefix*: dynamic pruning across steps breaks the cache, while methods requiring training were initially out of scope. However, our implementation reveals that accessing genuine high‑resolution tokens (3,645 vs. 729) requires lightweight LoRA adaptation of LaViDa's mm_projector, following proven HiRes‑LLaVA methodology. This minimal training (3.5h on 8×A100) enables testing the genuine research hypothesis versus interpolated features.

---

## 3 Challenges Unique to Diffusion VLMs

1. **Prefix immutability.** Vision tokens are cached once; changing them after step 0 voids all later reuse.
2. **Bidirectional attention.** Tokens influence *all* others, so naively dropping “unimportant” ones can erase global context.
3. **Multi‑view embeddings.** Five scales are concatenated; importance must be compared across vastly different receptive fields.
4. **Latency budget.** Any pre‑selection must finish in < 30 ms to keep LaViDa’s \~1.9 × speed‑up over AR baselines ([ar5iv][1]).

---

## 4 Proposed Method: **STATIC‑Hierarchical Relevance Gate (SHIRG)**

### 4.1 Intuition

Keep the **smallest token set that preserves question‑relevant detail** *once*, before diffusion starts, so KV‑cache stays valid.  Combine *local information gain* with *text‑image relevance*—but compute both from already‑available embeddings, not from gradients or extra models.

### 4.2 Algorithm (training‑free)

1. **Patch embedding extraction**
   Use the unpooled 3 645 tokens that LaViDa already computes (no extra vision pass).
2. **Per‑token saliency score**
   *Information term* = local variance of the vision feature (entropy proxy).
   *Relevance term* = max cosine similarity with any text token in the *question* (available before diffusion).
   $s_i = \alpha \,\mathrm{Var}(v_i) + (1-\alpha)\max_j \cos(v_i, t_j)$
   where $\alpha≈0.3$ balances detail and semantics (tuned offline on a held‑out set).
3. **Hierarchical clustering**
   Run a one‑shot agglomerative merge on the 7×7 spatial grid per view to form connected components; keep top‑$K$ tokens per component proportional to its total saliency.
4. **Context‑token budget**
   Target 1 024 total visual tokens (‑≈20 % vs. default 980 pooling but at *higher resolution*).  If budget exceeded, iteratively merge the lowest‑score neighbours (adapting SAINT’s graph pruning ([arXiv][5])).
5. **Global back‑off token**
   Add one pooled “summary” token for **all dropped tokens** so the language model can still reference coarse context (as in MeanPool‑Adaptor).
6. **Static prefix export**
   Concatenate the selected tokens + summary token + question tokens → cached once; all diffusion steps reuse it, so latency hit is just the \~3 ms scoring pass.

**Training scope**: Only mm_projector LoRA weights are learned (≪1% of total parameters); SHIRG selection logic uses existing embeddings and remains training‑free.

### 4.3 Why it meets diffusion constraints

* **Static after step 0** → KV‑cache intact.
* **Bidirectional friendly** → summary token gives a global fallback; hierarchical selection keeps neighbourhood continuity (unlike window slicing).
* **Multi‑view aware** → scoring normalises by view‑level variance so high‑res crops are not unfairly down‑ranked.

---

## 5 One‑Week Research Plan

| Day | Milestone                                                                            | Notes                                                        |
| --- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------ |
| 1   | Clone LaViDa repo & add hook to intercept 3 645 tokens before pooling ([GitHub][11]) |                                                              |
| 2   | Implement saliency + relevance scorer in CUDA (≈150 loc)                             | Uses in‑batch torch.cosine\_similarity                       |
| 3   | Hierarchical component grouping & budgeted selection                                 | Re‑use SAINT’s open‑source graph code ([arXiv][5])           |
| 4   | Fast summary‑token pooling and prefix export                                         |                                                              |
| 5   | Evaluation on ChartQA, DocVQA, MMMU‑OCR splits                                       | Datasets readily available ([ACL Anthology][4], [arXiv][12]) |
| 6   | Latency benchmarking vs. Baseline (pooled 980) and Full 3 645                        | Expect +7 % accuracy on ChartQA with < 10 % extra latency    |
| 7   | Write 4‑page workshop paper; ablate α, K, and summary token                          |                                                              |

**Research Plan Update**: Extended to 2 weeks to incorporate LoRA training phase (Days 3-5) for mm_projector adaptation, enabling genuine high-resolution token access. This follows proven HiRes-LLaVA methodology and ensures valid research hypothesis testing.

---

## 6 Expected Contributions

1. **First token‑selection strategy tailored to diffusion VLMs**—static, cache‑compatible, training‑free.
2. **Improved fine‑grained VQA**: projected +5‑10 % on OCR‑heavy sets at similar compute.
3. **Open‑source reference code** (< 300 loc) that others can plug into any diffusion‑based LVLM.

---

## 7 Potential Extensions

* **Two‑stage selection**: coarse static set for cache + *optional* dynamic refinement on *just those tokens* in late diffusion steps (keeps cache size small).
* **Adaptive K via entropy of logits at step 0**—lets the model keep more tokens only for cluttered images.
* **Cross‑modal reranking**: incorporate attention between selected tokens and *mask tokens* predicted at early diffusion steps.

---

## 8 Conclusion

SHIRG offers a pragmatic path to high‑resolution, low‑latency LaViDa with minimal LoRA adaptation. **Updated approach**: While initially conceived as training‑free, our implementation reveals that accessing genuine high‑resolution tokens (3,645 vs. 729) requires lightweight LoRA adaptation of LaViDa's mm_projector, following proven HiRes‑LLaVA methodology. This minimal training (3.5h on 8×A100, ≪1% of model parameters) enables testing the genuine research hypothesis with real high‑resolution features rather than interpolation artifacts. By choosing *once‑for‑all* the most relevant visual tokens through lightweight variance‑and‑similarity heuristics from a genuine high‑resolution candidate set, we respect the prefix KV‑cache and bidirectional attention requirements unique to diffusion VLMs, while reclaiming the fine detail needed for real‑world OCR and dense VQA tasks.

**Implementation Status**: Steps 1-3 of the LaViDa fork modification plan have been completed, providing genuine high-resolution token extraction (3,645 tokens) with comprehensive validation. Ready to proceed with LoRA training phase.

---

### Key References

LaViDa paper ([ar5iv][1]) – LaViDa GitHub ([GitHub][11]) – LaViDa architecture page ([homepage.jackli.org][13]) – LLaDA diffusion LLM ([ar5iv][3]) – SAINT ([arXiv][5]) – HiRes‑LLaVA ([CVF Open Access][2]) – LLaVA‑Scissor ([arXiv][6]) – “Image ½ Tokens” ([arXiv][7]) – Token Cropr ([arXiv][9]) – FastV ([GitHub][8]) – DocVLM ([arXiv][12]) – ChartQA dataset ([ACL Anthology][4]) – Token‑compression survey list ([GitHub][14]) – Token pruning notes ([michal.io][15]) – ChartInsights analysis ([chartinsight.github.io][16])

[1]: https://ar5iv.org/pdf/2505.16839 "[2505.16839] LaViDa: A Large Diffusion Language Model for Multimodal Understanding"
[2]: https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_HiRes-LLaVA_Restoring_Fragmentation_Input_in_High-Resolution_Large_Vision-Language_Models_CVPR_2025_paper.pdf "HiRes-LLaVA: Restoring Fragmentation Input in High-Resolution Large Vision-Language Models"
[3]: https://ar5iv.org/pdf/2502.09992 "[2502.09992] Large Language Diffusion Models"
[4]: https://aclanthology.org/2022.findings-acl.177.pdf?utm_source=chatgpt.com "ChartQA: A Benchmark for Question Answering about Charts with Visual ..."
[5]: https://arxiv.org/abs/2503.11549 "[2503.11549] Similarity-Aware Token Pruning: Your VLM but Faster"
[6]: https://arxiv.org/abs/2506.21862 "[2506.21862] LLaVA-Scissor: Token Compression with Semantic Connected Components for Video LLMs"
[7]: https://arxiv.org/abs/2403.06764?utm_source=chatgpt.com "An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference ..."
[8]: https://github.com/pkunlp-icler/FastV?utm_source=chatgpt.com "GitHub - pkunlp-icler/FastV: [ECCV 2024 Oral] Code for paper: An Image ..."
[9]: https://arxiv.org/abs/2412.00965?utm_source=chatgpt.com "Token Cropr: Faster ViTs for Quite a Few Tasks"
[10]: https://arxiv.org/abs/2507.02279?utm_source=chatgpt.com "[2507.02279] LaCo: Efficient Layer-wise Compression of Visual Tokens ..."
[11]: https://github.com/jacklishufan/LaViDa?utm_source=chatgpt.com "LaViDa:A Large Diffusion Language Model for Multimodal ... - GitHub"
[12]: https://arxiv.org/abs/2412.08746?utm_source=chatgpt.com "[2412.08746] DocVLM: Make Your VLM an Efficient Reader"
[13]: https://homepage.jackli.org/projects/lavida/?utm_source=chatgpt.com "LaViDa - homepage.jackli.org"
[14]: https://github.com/daixiangzi/Awesome-Token-Compress?utm_source=chatgpt.com "daixiangzi/Awesome-Token-Compress - GitHub"
[15]: https://michal.io/notes/ml/Token-Dropping%2C-Pruning%2C-Merging-and-Compression?utm_source=chatgpt.com "Token Dropping, Pruning, Merging and Compression"
[16]: https://chartinsight.github.io/?utm_source=chatgpt.com "ChartInsights: Evaluating Multimodal Large Language Models for Low ..."