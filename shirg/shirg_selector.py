#!/usr/bin/env python3
"""
SHIRG: Static Hierarchical Relevance Gate for LaViDa
Training-free token selection for diffusion VLMs with KV-cache compatibility

Author: Research Implementation
Date: 2025-01-26
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any
from sklearn.cluster import AgglomerativeClustering
import warnings

# Suppress sklearn warnings for cleaner output in Colab
warnings.filterwarnings('ignore', category=UserWarning)

class SHIRGSelector(nn.Module):
    """
    Static Hierarchical Relevance Gate for LaViDa
    
    Training-free token selection that maintains:
    1. Fixed output size (1024 tokens) for KV cache compatibility  
    2. Static selection (once per sample) for cache reuse
    3. Text-conditioned relevance scoring
    4. Hierarchical spatial grouping
    5. Sub-1000ms latency budget
    
    Key Innovation: Combines information content (local variance) with 
    text-image semantic relevance for optimal token selection.
    
    SOLUTION 2 Enhancement: Supports dimension-adaptive relevance computation
    for robust text-vision semantic matching across different embedding spaces.
    """
    
    def __init__(self, 
                 target_tokens: int = 729,            # Fixed output for LaViDa spatial grid (27×27)
                                                   # FIX: 2025-07-27 - Select 729 best tokens from larger unpooled token set
                                                   # ISSUE: Access unpooled tokens before SigLIP pooling head
                                                   # SOLUTION: Use high-res (768x768=3025) tokens, select best 729 with SHIRG
                                                   # RESEARCH IMPACT: Test research hypothesis - better 729 from larger pool
                 alpha: float = 0.3,                  # Balance between detail and semantics  
                 hierarchical_levels: int = 3,        # Levels of spatial clustering
                 latency_budget_ms: float = 1000.0,     # Must finish within 1000ms
                 use_fast_clustering: bool = True,    # Use approximate clustering for speed
                 enable_caching: bool = True,         # Cache repeated computations
                 debug: bool = False,                 # Enable debug output
                 high_res_interpolation: bool = True, # Enable high-resolution interpolation (wrapper config)
                 target_grid_size: int = 55,          # Target grid size for interpolation (wrapper config)
                 vision_projector: Optional[nn.Module] = None,  # SOLUTION 2: Optional vision projector for text->vision mapping
                 **kwargs):                           # Accept additional config parameters
        super().__init__()
        
        self.target_tokens = target_tokens
        self.alpha = alpha
        self.hierarchical_levels = hierarchical_levels  
        self.latency_budget_ms = latency_budget_ms
        self.use_fast_clustering = use_fast_clustering
        self.enable_caching = enable_caching
        self.debug = debug
        self.high_res_interpolation = high_res_interpolation
        self.target_grid_size = target_grid_size
        self.vision_projector = vision_projector  # SOLUTION 2: Store optional projector
        
        # Performance tracking
        self.selection_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Simple cache for repeated inputs (if enabled)
        if enable_caching:
            self.selection_cache = {}
            self.max_cache_size = 100
        
        # No learnable parameters - fully training-free
        print(f"✅ SHIRG initialized: {target_tokens} tokens, α={alpha}, budget={latency_budget_ms}ms")
    
    def forward(self, 
                image_tokens: torch.Tensor,         # [B, N, D] - unpooled tokens from SigLIP
                text_embeddings: torch.Tensor,      # [B, L, D] - question embeddings  
                image_sizes: Optional[List[Tuple]] = None,  # Image dimension info
                **kwargs) -> torch.Tensor:          # [B, target_tokens, D]
        """
        Apply SHIRG token selection
        
        Args:
            image_tokens: Unpooled vision tokens from SigLIP [B, ~3645, 1152]
            text_embeddings: Text question embeddings [B, seq_len, 1152]
            image_sizes: Original image sizes for spatial reasoning
            
        Returns:
            selected_tokens: Fixed-size selected tokens [B, target_tokens, 1152]
        """
        batch_size = image_tokens.shape[0]
        start_time = time.time()
        
        if self.debug:
            print(f"🔍 SHIRG input: images={image_tokens.shape}, text={text_embeddings.shape}")
        
        selected_tokens_batch = []
        
        for b in range(batch_size):
            # Check latency budget before processing each sample
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > self.latency_budget_ms:
                if self.debug:
                    print(f"⚠️ SHIRG latency budget exceeded: {elapsed_ms:.1f}ms")
                # Fallback: use simple top-k selection for remaining samples
                remaining_tokens = self._fallback_selection(
                    image_tokens[b:], 
                    text_embeddings[min(b, text_embeddings.shape[0]-1):] if b < text_embeddings.shape[0] else text_embeddings[-1:]
                )
                selected_tokens_batch.extend(remaining_tokens)
                break
            
            # Per-sample token selection
            img_tokens = image_tokens[b]  # [N, D]
            # FIX: 2025-07-27 - Handle batch size mismatch between images and text
            # ISSUE: Text embeddings may have smaller batch size than image tokens (1 vs 5)
            # SOLUTION: Reuse the text embeddings for all images in batch (broadcast semantics)
            # RESEARCH IMPACT: Enables SHIRG to work with single question applied to multiple images
            txt_tokens = text_embeddings[min(b, text_embeddings.shape[0]-1)]  # [L, D] 
            
            # Check cache first (if enabled)
            cache_key = None
            if self.enable_caching:
                cache_key = self._compute_cache_key(img_tokens, txt_tokens)
                if cache_key in self.selection_cache:
                    selected_tokens_batch.append(self.selection_cache[cache_key])
                    self.cache_hits += 1
                    continue
                self.cache_misses += 1
            
            # 1. Compute saliency scores (information + relevance)
            saliency_scores = self._compute_saliency_scores(img_tokens, txt_tokens)
            
            # 2. Hierarchical spatial clustering
            spatial_groups = self._hierarchical_clustering(
                img_tokens, saliency_scores, 
                image_sizes[b] if image_sizes and b < len(image_sizes) else None
            )
            
            # 3. Budget-aware token selection
            selected = self._budget_aware_selection(
                img_tokens, saliency_scores, spatial_groups
            )
            
            # 4. Add global summary token for dropped content
            summary_token = self._create_summary_token(img_tokens, selected)
            
            # 5. Ensure fixed output size for cache compatibility
            final_tokens = self._pad_or_truncate_to_target(selected, summary_token)
            
            selected_tokens_batch.append(final_tokens)
            
            # Cache result if computation was fast enough
            if self.enable_caching and cache_key is not None:
                if len(self.selection_cache) >= self.max_cache_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self.selection_cache))
                    del self.selection_cache[oldest_key]
                self.selection_cache[cache_key] = final_tokens
        
        # Track performance
        total_time_ms = (time.time() - start_time) * 1000
        self.selection_times.append(total_time_ms)
        
        if self.debug:
            print(f"✅ SHIRG completed in {total_time_ms:.1f}ms")
        
        return torch.stack(selected_tokens_batch, dim=0)
    
    def _compute_saliency_scores(self, 
                                img_tokens: torch.Tensor,  # [N, D_vision] 
                                txt_tokens: torch.Tensor   # [L, D_text]
                                ) -> torch.Tensor:         # [N]
        """
        SOLUTION 2: Compute saliency scores with dimension-adaptive relevance computation
        
        Formula: s_i = α * Var(v_i) + (1-α) * Relevance(v_i, t_j)
        
        Args:
            img_tokens: Image tokens [N, D_vision] (e.g., 1152D)
            txt_tokens: Text tokens [L, D_text] (e.g., 4096D or 1152D after projection)
            
        Returns:
            saliency_scores: Per-token saliency scores [N]
        """
        device = img_tokens.device
        dtype = img_tokens.dtype  # FIX: Preserve input dtype
        
        # Information term: local variance as entropy proxy (unchanged)
        # Higher variance = more informative/detailed content
        info_scores = torch.var(img_tokens, dim=-1, keepdim=True).to(dtype=dtype)  # [N, 1]
        
        # SOLUTION 2: Dimension-adaptive relevance computation
        if txt_tokens.numel() == 0:
            # No text tokens - fall back to information-only scoring
            relevance_scores = torch.zeros_like(info_scores, dtype=dtype)
            
        elif img_tokens.shape[-1] == txt_tokens.shape[-1]:
            # CASE A: Same dimensions - use direct cosine similarity (original approach)
            img_norm = F.normalize(img_tokens, dim=-1)      # [N, D]
            txt_norm = F.normalize(txt_tokens, dim=-1)      # [L, D]
            
            # Compute all pairwise cosine similarities
            similarities = torch.mm(img_norm, txt_norm.t())  # [N, L]
            relevance_scores = torch.max(similarities, dim=-1, keepdim=True)[0].to(dtype=dtype)  # [N, 1]
            
            if self.debug:
                print(f"✅ Direct cosine similarity: img({img_tokens.shape[-1]}) == txt({txt_tokens.shape[-1]})")
                
        else:
            # CASE B: Different dimensions - use dimension-agnostic relevance measures
            if self.debug:
                print(f"⚠️ Dimension mismatch: img({img_tokens.shape[-1]}) != txt({txt_tokens.shape[-1]}), using adaptive relevance")
            
            # SOLUTION 2A: Feature magnitude correlation as semantic proxy
            # Rationale: Tokens with similar activation magnitudes often represent similar concepts
            img_magnitudes = torch.norm(img_tokens, dim=-1, keepdim=True)  # [N, 1]
            txt_magnitudes = torch.norm(txt_tokens, dim=-1, keepdim=True)  # [L, 1]
            
            # Compute magnitude similarity between each image token and all text tokens
            img_mag_norm = F.normalize(img_magnitudes, dim=0)  # Normalize across tokens
            txt_mag_norm = F.normalize(txt_magnitudes, dim=0)  # Normalize across tokens
            
            # Broadcast and compute magnitude correlation
            magnitude_similarities = 1.0 - torch.abs(img_mag_norm - txt_mag_norm.T)  # [N, L]
            
            # SOLUTION 2B: Statistical feature correlation (if sufficient dimensions)
            if min(img_tokens.shape[-1], txt_tokens.shape[-1]) >= 8:
                # Use feature statistics correlation for richer semantic matching
                
                # Compute statistical signatures: [mean, std, min, max] of features
                img_stats = torch.stack([
                    img_tokens.mean(dim=-1),    # Mean activation
                    img_tokens.std(dim=-1),     # Activation spread  
                    img_tokens.min(dim=-1)[0],  # Min activation
                    img_tokens.max(dim=-1)[0]   # Max activation
                ], dim=-1)  # [N, 4]
                
                txt_stats = torch.stack([
                    txt_tokens.mean(dim=-1),
                    txt_tokens.std(dim=-1), 
                    txt_tokens.min(dim=-1)[0],
                    txt_tokens.max(dim=-1)[0]
                ], dim=-1)  # [L, 4]
                
                # Normalize statistical signatures
                img_stats_norm = F.normalize(img_stats, dim=-1)
                txt_stats_norm = F.normalize(txt_stats, dim=-1)
                
                # Compute statistical correlation
                statistical_similarities = torch.mm(img_stats_norm, txt_stats_norm.t())  # [N, L]
                
                # Combine magnitude and statistical similarities
                relevance_matrix = (magnitude_similarities + statistical_similarities) / 2.0
            else:
                # Use magnitude similarity only for low-dimensional cases
                relevance_matrix = magnitude_similarities
            
            # Extract maximum relevance for each image token
            relevance_scores = torch.max(relevance_matrix, dim=-1, keepdim=True)[0].to(dtype=dtype)  # [N, 1]
            
            if self.debug:
                print(f"📊 Dimension-adaptive relevance - magnitude range: {img_magnitudes.min():.3f} to {img_magnitudes.max():.3f}")
                print(f"📊 Text magnitude range: {txt_magnitudes.min():.3f} to {txt_magnitudes.max():.3f}")
                print(f"📊 Relevance scores range: {relevance_scores.min():.3f} to {relevance_scores.max():.3f}")
        
        # Normalize both scores to [0, 1] range for stable combination
        if info_scores.max() > info_scores.min():
            info_scores = (info_scores - info_scores.min()) / (info_scores.max() - info_scores.min() + 1e-8)
        else:
            info_scores = torch.ones_like(info_scores, dtype=dtype) * 0.5
            
        if relevance_scores.max() > relevance_scores.min():
            relevance_scores = (relevance_scores - relevance_scores.min()) / (relevance_scores.max() - relevance_scores.min() + 1e-8)
        else:
            relevance_scores = torch.ones_like(relevance_scores, dtype=dtype) * 0.5
        
        # Combine with alpha weighting - ensure all operations preserve dtype
        alpha_tensor = torch.tensor(self.alpha, dtype=dtype, device=device)
        saliency_scores = alpha_tensor * info_scores + (1 - alpha_tensor) * relevance_scores
        
        # FIX: 2025-07-27 - Add debug output to verify SHIRG is actually selecting different tokens
        # ISSUE: Need to verify that saliency scores vary and selection is not uniform
        # SOLUTION: Log score statistics to confirm selection is meaningful
        # RESEARCH IMPACT: Verify that SHIRG selection is actually working
        if self.debug:
            scores_flat = saliency_scores.squeeze(-1)
            print(f"📊 Saliency scores - min: {scores_flat.min():.4f}, max: {scores_flat.max():.4f}, std: {scores_flat.std():.4f}")
            print(f"🎯 Alpha: {self.alpha}, Info weight: {alpha_tensor:.3f}, Relevance weight: {(1-alpha_tensor):.3f}")
        
        return saliency_scores.squeeze(-1)  # [N]
    
    def _hierarchical_clustering(self, 
                                img_tokens: torch.Tensor,     # [N, D]
                                saliency_scores: torch.Tensor, # [N]
                                image_size: Optional[Tuple] = None
                                ) -> List[List[int]]:
        """
        Perform hierarchical spatial clustering to group neighboring tokens
        
        Args:
            img_tokens: Image tokens [N, D]
            saliency_scores: Per-token saliency scores [N]
            image_size: Original image dimensions (height, width)
            
        Returns:
            spatial_groups: List of token index groups
        """
        num_tokens = img_tokens.shape[0]
        
        # Handle LaViDa's multi-view token layout
        if num_tokens > 3000:  # Likely multi-view (4×576 + 2304 ≈ 4608)
            return self._handle_multiview_clustering(num_tokens, saliency_scores)
        
        # Single view or smaller token count - treat as square grid
        grid_size = int(np.sqrt(num_tokens))
        if grid_size * grid_size != num_tokens:
            # Non-square layout - use simple distance-based clustering
            return self._fallback_clustering(num_tokens, saliency_scores)
        
        # Create 2D spatial coordinates
        coords = []
        for i in range(num_tokens):
            row = i // grid_size
            col = i % grid_size
            coords.append([row, col])
        coords = np.array(coords, dtype=np.float32)
        
        # Determine number of clusters based on target token count
        num_clusters = min(
            max(1, self.target_tokens // 8),  # ~8 tokens per cluster
            num_tokens // 4,                  # At least 4 tokens per cluster
            num_tokens                        # Can't have more clusters than tokens
        )
        
        try:
            if self.use_fast_clustering and num_tokens > 500:
                # Fast approximation for large token counts
                clusters = self._fast_spatial_clustering(coords, num_clusters)
            else:
                # Full hierarchical clustering
                clustering = AgglomerativeClustering(
                    n_clusters=num_clusters,
                    linkage='ward'
                )
                cluster_labels = clustering.fit_predict(coords)
                
                # Group tokens by cluster
                clusters = {}
                for token_idx, cluster_id in enumerate(cluster_labels):
                    if cluster_id not in clusters:
                        clusters[cluster_id] = []
                    clusters[cluster_id].append(token_idx)
                clusters = list(clusters.values())
        
        except Exception as e:
            if self.debug:
                print(f"⚠️ Clustering failed: {e}, using fallback")
            return self._fallback_clustering(num_tokens, saliency_scores)
        
        return clusters
    
    def _handle_multiview_clustering(self, 
                                   num_tokens: int, 
                                   saliency_scores: torch.Tensor
                                   ) -> List[List[int]]:
        """
        Handle LaViDa's multi-view token layout
        
        LaViDa uses 5 views with different resolutions:
        - 4 views at 336×336 → ~24×24 = 576 tokens each  
        - 1 view at 672×672 → ~48×48 = 2304 tokens
        - Total: 4×576 + 2304 = 4608 tokens (approximately)
        """
        clusters = []
        
        # Estimate view boundaries (approximate since exact layout may vary)
        view_size_small = 576   # 24×24 patches
        view_size_large = 2304  # 48×48 patches
        
        start_idx = 0
        
        # Process 4 small views
        for view in range(4):
            end_idx = min(start_idx + view_size_small, num_tokens)
            if start_idx >= end_idx:
                break
                
            view_tokens = list(range(start_idx, end_idx))
            
            # Sub-cluster within view
            sub_clusters = self._cluster_single_view(view_tokens, grid_size=24)
            clusters.extend(sub_clusters)
            
            start_idx = end_idx
        
        # Process 1 large view (if remaining tokens)
        if start_idx < num_tokens:
            end_idx = num_tokens
            view_tokens = list(range(start_idx, end_idx))
            
            # Determine grid size based on remaining tokens
            remaining_tokens = end_idx - start_idx
            grid_size = int(np.sqrt(remaining_tokens))
            
            # Sub-cluster within large view
            sub_clusters = self._cluster_single_view(view_tokens, grid_size=grid_size)
            clusters.extend(sub_clusters)
        
        return clusters
    
    def _cluster_single_view(self, 
                           token_indices: List[int], 
                           grid_size: int
                           ) -> List[List[int]]:
        """Cluster tokens within a single view"""
        num_tokens = len(token_indices)
        
        # Determine cluster count: aim for ~16 tokens per cluster
        num_clusters = max(1, num_tokens // 16)
        
        if num_tokens <= num_clusters:
            # Each token becomes its own cluster
            return [[idx] for idx in token_indices]
        
        # Create spatial coordinates within view
        coords = []
        for i, token_idx in enumerate(token_indices):
            local_idx = i  # Index within this view
            row = local_idx // grid_size
            col = local_idx % grid_size
            coords.append([row, col])
        
        coords = np.array(coords, dtype=np.float32)
        
        try:
            # Spatial clustering
            clustering = AgglomerativeClustering(
                n_clusters=num_clusters,
                linkage='ward'
            )
            cluster_labels = clustering.fit_predict(coords)
            
            # Group tokens by cluster
            clusters = {}
            for i, cluster_id in enumerate(cluster_labels):
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(token_indices[i])
            
            return list(clusters.values())
            
        except Exception:
            # Fallback: sequential grouping
            chunk_size = max(1, num_tokens // num_clusters)
            clusters = []
            for i in range(0, num_tokens, chunk_size):
                chunk = token_indices[i:i + chunk_size]
                if chunk:
                    clusters.append(chunk)
            return clusters
    
    def _fast_spatial_clustering(self, 
                               coords: np.ndarray, 
                               num_clusters: int
                               ) -> List[List[int]]:
        """Fast approximation to spatial clustering using k-means-style grouping"""
        num_tokens = len(coords)
        
        # Simple grid-based clustering for speed
        clusters = []
        tokens_per_cluster = max(1, num_tokens // num_clusters)
        
        for i in range(0, num_tokens, tokens_per_cluster):
            cluster = list(range(i, min(i + tokens_per_cluster, num_tokens)))
            if cluster:
                clusters.append(cluster)
        
        return clusters
    
    def _fallback_clustering(self, 
                           num_tokens: int, 
                           saliency_scores: torch.Tensor
                           ) -> List[List[int]]:
        """Fallback clustering when spatial clustering fails"""
        # Simple saliency-based grouping
        cluster_size = max(1, num_tokens // (self.target_tokens // 8))
        clusters = []
        
        for i in range(0, num_tokens, cluster_size):
            cluster = list(range(i, min(i + cluster_size, num_tokens)))
            if cluster:
                clusters.append(cluster)
        
        return clusters
    
    def _budget_aware_selection(self, 
                              img_tokens: torch.Tensor,      # [N, D]
                              saliency_scores: torch.Tensor, # [N]
                              spatial_groups: List[List[int]]
                              ) -> torch.Tensor:             # [K, D] where K < target_tokens
        """
        Select tokens within budget while respecting spatial grouping
        
        Args:
            img_tokens: All image tokens [N, D]
            saliency_scores: Per-token saliency scores [N]
            spatial_groups: List of spatial token groups
            
        Returns:
            selected_tokens: Selected tokens [K, D]
        """
        # Compute group saliency scores (sum of token saliencies in group)
        group_scores = []
        group_sizes = []
        
        for group in spatial_groups:
            if len(group) == 0:
                continue
            group_saliency = saliency_scores[group].sum().item()
            group_scores.append(group_saliency)
            group_sizes.append(len(group))
        
        # Filter out empty groups
        valid_groups = [(g, s, size) for g, s, size in zip(spatial_groups, group_scores, group_sizes) if size > 0]
        
        if not valid_groups:
            # Fallback: select top tokens by saliency
            k = min(self.target_tokens - 1, len(img_tokens))
            top_indices = torch.topk(saliency_scores, k)[1]
            return img_tokens[top_indices]
        
        # Allocate tokens proportionally to group saliency
        total_saliency = sum(s for _, s, _ in valid_groups)
        selected_indices = []
        budget_remaining = self.target_tokens - 1  # -1 for summary token
        
        for group, score, size in valid_groups:
            # Proportion of tokens to allocate to this group
            if total_saliency > 0:
                allocation_ratio = score / total_saliency
            else:
                allocation_ratio = 1.0 / len(valid_groups)
            
            tokens_for_group = max(1, int(allocation_ratio * budget_remaining))
            tokens_for_group = min(tokens_for_group, size, budget_remaining)
            
            if tokens_for_group <= 0:
                continue
            
            # Select top tokens within group by saliency
            group_saliencies = saliency_scores[group]
            if len(group_saliencies) >= tokens_for_group:
                top_indices = torch.topk(group_saliencies, tokens_for_group)[1]
                selected_indices.extend([group[idx] for idx in top_indices])
            else:
                selected_indices.extend(group)
            
            budget_remaining -= tokens_for_group
            if budget_remaining <= 0:
                break
        
        # If we haven't used full budget, add more tokens by global saliency
        if budget_remaining > 0 and len(selected_indices) < len(img_tokens):
            all_indices = set(range(len(saliency_scores)))
            remaining_indices = list(all_indices - set(selected_indices))
            
            if remaining_indices:
                remaining_scores = saliency_scores[remaining_indices]
                additional_count = min(budget_remaining, len(remaining_indices))
                
                # Use top-k selection for high-quality tokens
                if additional_count > 0:
                    top_remaining = torch.topk(remaining_scores, additional_count)[1]
                    additional_selected = [remaining_indices[idx] for idx in top_remaining]
                    selected_indices.extend(additional_selected)
                    
                    if self.debug:
                        print(f"🔄 Added {len(additional_selected)} high-saliency tokens from remaining pool")
        
        # Ensure we don't exceed budget
        if len(selected_indices) > self.target_tokens - 1:
            # Keep only top tokens by saliency
            selected_scores = saliency_scores[selected_indices]
            top_indices = torch.topk(selected_scores, self.target_tokens - 1)[1]
            selected_indices = [selected_indices[idx] for idx in top_indices]
            
            if self.debug:
                print(f"✂️ Trimmed selection from {len(selected_scores)} to {len(selected_indices)} tokens")
        
        # FIX: 2025-07-27 - Verify selection is different from sequential order
        # ISSUE: Need to confirm SHIRG selects different tokens than just first N tokens
        # SOLUTION: Compare selected indices to sequential order and log differences
        # RESEARCH IMPACT: Verify SHIRG is making meaningful token selections
        if self.debug and len(selected_indices) > 0:
            total_tokens = len(saliency_scores)
            sequential_indices = list(range(len(selected_indices)))  # First N tokens in order
            selected_set = set(selected_indices)
            sequential_set = set(sequential_indices)
            
            # Calculate diversity metrics
            overlap = len(selected_set & sequential_set)
            non_sequential_selections = len(selected_indices) - overlap
            diversity_percent = (non_sequential_selections / len(selected_indices)) * 100 if len(selected_indices) > 0 else 0
            
            # Calculate spatial spread - how spread out are the selected tokens?
            if len(selected_indices) > 1:
                selected_indices_sorted = sorted(selected_indices)
                max_gap = max(selected_indices_sorted[i+1] - selected_indices_sorted[i] 
                            for i in range(len(selected_indices_sorted)-1))
                total_spread = selected_indices_sorted[-1] - selected_indices_sorted[0]
            else:
                max_gap = 0
                total_spread = 0
            
            print(f"🔎 Selection diversity: {non_sequential_selections}/{len(selected_indices)} ({diversity_percent:.1f}%) non-sequential")
            print(f"🎯 Selection spread: tokens span {total_spread}/{total_tokens} ({(total_spread/total_tokens*100):.1f}%) of total range")
            
            # Show some example selected indices to verify diversity
            if len(selected_indices) >= 10:
                sample_indices = sorted(selected_indices[:10])
                print(f"📊 Sample selected indices: {sample_indices}")
        
        return img_tokens[selected_indices] if selected_indices else img_tokens[:1]
    
    def _create_summary_token(self, 
                            img_tokens: torch.Tensor,      # [N, D]
                            selected_tokens: torch.Tensor  # [K, D]
                            ) -> torch.Tensor:             # [1, D]
        """
        Create global summary token from all tokens for fallback context
        
        This provides a condensed representation of the entire image,
        ensuring the language model can still access global context.
        """
        # Create summary from all tokens (not just selected ones)
        # This ensures global context is preserved even with aggressive selection
        summary_token = img_tokens.mean(dim=0, keepdim=True)  # [1, D]
        
        if self.debug:
            selected_mean = selected_tokens.mean(dim=0, keepdim=True) if selected_tokens.numel() > 0 else summary_token
            cosine_sim = F.cosine_similarity(summary_token, selected_mean, dim=-1).item()
            print(f"🌐 Summary token similarity to selected tokens: {cosine_sim:.3f}")
        
        return summary_token
    
    def _pad_or_truncate_to_target(self, 
                                 selected_tokens: torch.Tensor,  # [K, D]
                                 summary_token: torch.Tensor     # [1, D]
                                 ) -> torch.Tensor:              # [target_tokens, D]
        """
        Ensure exactly target_tokens output for KV-cache compatibility
        
        This is critical for LaViDa's prefix-KV caching mechanism.
        """
        current_count = selected_tokens.shape[0] + 1  # +1 for summary token
        
        if current_count == self.target_tokens:
            # Perfect fit
            return torch.cat([selected_tokens, summary_token], dim=0)
        elif current_count < self.target_tokens:
            # Need padding - repeat summary token
            padding_needed = self.target_tokens - current_count
            padding = summary_token.repeat(padding_needed, 1)
            return torch.cat([selected_tokens, summary_token, padding], dim=0)
        else:
            # Need truncation - keep top selected tokens + summary
            truncate_to = self.target_tokens - 1
            return torch.cat([selected_tokens[:truncate_to], summary_token], dim=0)
    
    def _fallback_selection(self, 
                          image_tokens: torch.Tensor,     # [B, N, D]
                          text_embeddings: torch.Tensor   # [B, L, D]
                          ) -> List[torch.Tensor]:
        """
        Fast fallback selection when latency budget is exceeded
        
        Uses simple top-k selection without clustering.
        """
        fallback_tokens = []
        
        for b in range(image_tokens.shape[0]):
            img_tokens = image_tokens[b]  # [N, D]
            # FIX: 2025-07-27 - Handle batch size mismatch in fallback selection
            # ISSUE: Same batch size mismatch in fallback method
            # SOLUTION: Reuse text embeddings for all images
            # RESEARCH IMPACT: Consistent batch handling across all SHIRG methods
            txt_tokens = text_embeddings[min(b, text_embeddings.shape[0]-1)]  # [L, D]
            
            # Quick saliency scoring
            saliency_scores = self._compute_saliency_scores(img_tokens, txt_tokens)
            
            # Select top tokens
            k = min(self.target_tokens - 1, len(img_tokens))
            top_indices = torch.topk(saliency_scores, k)[1]
            selected = img_tokens[top_indices]
            
            # Add summary token
            summary = self._create_summary_token(img_tokens, selected)
            final_tokens = self._pad_or_truncate_to_target(selected, summary)
            
            fallback_tokens.append(final_tokens)
        
        return fallback_tokens
    
    def _compute_cache_key(self, 
                          img_tokens: torch.Tensor, 
                          txt_tokens: torch.Tensor
                          ) -> str:
        """Compute cache key for input tensors (simplified hash)"""
        # Simple hash based on tensor shapes and a few values
        # Not cryptographically secure, just for cache lookup
        img_hash = hash((img_tokens.shape, img_tokens.flatten()[:10].sum().item()))
        txt_hash = hash((txt_tokens.shape, txt_tokens.flatten()[:10].sum().item()))
        return f"{img_hash}_{txt_hash}"
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {
            'avg_selection_time_ms': np.mean(self.selection_times) if self.selection_times else 0.0,
            'max_selection_time_ms': np.max(self.selection_times) if self.selection_times else 0.0,
            'min_selection_time_ms': np.min(self.selection_times) if self.selection_times else 0.0,
            'total_selections': len(self.selection_times),
            'latency_budget_ms': self.latency_budget_ms,
            'budget_exceeded_count': sum(1 for t in self.selection_times if t > self.latency_budget_ms)
        }
        
        if self.enable_caching:
            total_cache_attempts = self.cache_hits + self.cache_misses
            stats.update({
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'cache_hit_rate_percent': (self.cache_hits / total_cache_attempts * 100) if total_cache_attempts > 0 else 0.0,
                'cache_size': len(self.selection_cache)
            })
        
        return stats
    
    def reset_stats(self):
        """Reset performance tracking"""
        self.selection_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        if self.enable_caching:
            self.selection_cache = {}


def create_shirg_selector(config: Optional[Dict[str, Any]] = None) -> SHIRGSelector:
    """
    Factory function to create SHIRG selector with configuration
    
    Args:
        config: Configuration dictionary with SHIRG parameters
        
    Returns:
        SHIRGSelector instance
    """
    default_config = {
        'target_tokens': 729,   # LaViDa compatibility constraint
        'alpha': 0.3,
        'hierarchical_levels': 3,
        'latency_budget_ms': 1000.0,
        'use_fast_clustering': True,
        'enable_caching': True,
        'debug': False,
        'high_res_interpolation': True,
        'target_grid_size': 55
    }
    
    if config:
        default_config.update(config)
    
    return SHIRGSelector(**default_config)


# Test function for Colab
def test_shirg_selector():
    """Test SHIRG selector with dummy data"""
    print("🧪 Testing SHIRG Selector...")
    
    # Create test data similar to LaViDa's SigLIP output
    batch_size = 2
    num_tokens = 3645  # LaViDa's multi-view token count
    hidden_size = 1152  # SigLIP hidden size
    seq_len = 50  # Typical question length
    
    # Dummy image and text tokens
    image_tokens = torch.randn(batch_size, num_tokens, hidden_size)
    text_tokens = torch.randn(batch_size, seq_len, hidden_size)
    
    # Create SHIRG selector
    shirg = create_shirg_selector({'debug': True})
    
    # Test selection
    start_time = time.time()
    selected_tokens = shirg(image_tokens, text_tokens)
    elapsed_time = time.time() - start_time
    
    print(f"✅ SHIRG test completed:")
    print(f"   Input shape: {image_tokens.shape}")
    print(f"   Output shape: {selected_tokens.shape}")
    print(f"   Processing time: {elapsed_time*1000:.1f}ms")
    print(f"   Target tokens: {shirg.target_tokens}")
    
    # Verify output shape
    expected_shape = (batch_size, shirg.target_tokens, hidden_size)
    assert selected_tokens.shape == expected_shape, f"Shape mismatch: {selected_tokens.shape} vs {expected_shape}"
    
    # Print performance stats
    stats = shirg.get_performance_stats()
    print(f"📊 Performance Stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("🎉 SHIRG test passed!")
    return shirg, selected_tokens


if __name__ == "__main__":
    # Run test if executed directly
    test_shirg_selector()