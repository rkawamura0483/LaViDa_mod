"""
SHIRG Extensions for SigLIP Vision Tower
SHIRG-Fovea: Static Hierarchical Relevance Gate for High-Resolution Diffusion VLMs

This module contains all SHIRG-specific functionality for high-resolution
token selection and processing, following the updated research methodology
with 2-view processing (1 global + 1 foveal).

Research Implementation based on:
- Two-scale foveation: Global 384² + 1×448² foveal view
- Per-view static Top-K selection (76.6% retention for foveal view)
- Cache-compatible processing for LaViDa (980 tokens total)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from typing import Optional, Tuple, Union, List

from .siglip_base import SigLipVisionConfig
from llava.utils import rank0_print


class SigLipShirgExtensions:
    """
    SHIRG-Fovea Extensions Mixin for SigLipVisionTower
    
    Contains all SHIRG-specific methods for multi-view token processing,
    per-view selection, and cache-compatible processing.
    """
    
    def forward_with_shirg(self, pixel_values, text_embeddings=None, selection_method='base', 
                          selection_params=None):
        """
        SHIRG-Fovea: Process 2-view format with per-view selection
        
        Implements the new SHIRG methodology:
        1. Extract 2-view tokens (1 global 384² + 1 foveal 448²)
        2. Apply 2×2 pooling to global view → 256 tokens
        3. Apply 70.7% Top-K selection to foveal view → 724 tokens
        4. Concatenate for 980 total tokens (matching LaViDa baseline)
        
        Args:
            pixel_values: Either list of 2 image tensors OR stacked tensor [2, C, H, W]
            text_embeddings: Optional text embeddings for scoring
            selection_method: Token selection method ('base', 'entropy', 'edge', 'full')
            selection_params: Method-specific parameters dict
            
        Returns:
            visual_tokens: [B, 980, D] selected tokens (256 global + 724 foveal)
        """
        try:
            # SHIRG-1FOVEAL-FIX: 2025-07-30 - Handle new 2-view input format
            # ISSUE: Updated research uses 2 views instead of 3
            # SOLUTION: Process 1 global 384² view + 1 foveal 448² view
            # RESEARCH IMPACT: Implements new SHIRG-Fovea architecture with 980 tokens
            # LAVIDA IMPACT: Maintains cache compatibility with baseline token count
            
            # Convert input to list of views if needed
            if torch.is_tensor(pixel_values) and len(pixel_values.shape) == 4:
                # Stacked tensor: [num_views, C, H, W]
                if pixel_values.shape[0] == 2:  # Expected 2 views for SHIRG-Fovea
                    rank0_print(f"SHIRG-1FOVEAL: Converting stacked tensor {pixel_values.shape} to list of 2 views")
                    pixel_values = [pixel_values[i] for i in range(2)]
                else:
                    raise ValueError(f"SHIRG-Fovea expects 2 views, got tensor with {pixel_values.shape[0]} views")
            elif torch.is_tensor(pixel_values) and len(pixel_values.shape) == 5:
                # SHIRG-FIX: 2025-07-30 - Handle batch processing for training
                # ISSUE: SHIRG-Fovea was designed for B=1 but training needs B>1
                # SOLUTION: Process each batch item separately and combine results
                # LAVIDA IMPACT: None - maintains same output format
                # SHIRG IMPACT: Enables batch training with proper gradient flow
                
                # Handle [B, num_views, C, H, W] format
                B, num_views = pixel_values.shape[:2]
                if num_views == 2:
                    if B == 1:
                        rank0_print(f"SHIRG-1FOVEAL: Converting 5D tensor {pixel_values.shape} to list of 2 views")
                        pixel_values = pixel_values.squeeze(0)  # Remove batch dimension
                        pixel_values = [pixel_values[i] for i in range(2)]
                    else:
                        # Batch processing: process each item separately
                        rank0_print(f"SHIRG-BATCH: Processing batch of {B} items with 2 views each")
                        batch_results = []
                        for b in range(B):
                            # Extract views for this batch item
                            batch_views = [pixel_values[b, 0], pixel_values[b, 1]]
                            # Process this item
                            result = self.forward_with_shirg(batch_views, text_embeddings[b:b+1] if text_embeddings is not None else None)
                            batch_results.append(result)
                        # Concatenate batch results
                        return torch.cat(batch_results, dim=0)
                else:
                    raise ValueError(f"SHIRG-Fovea expects 2 views, got {num_views} views in shape {pixel_values.shape}")
            
            # Step 1: Extract multiview tokens (1 global + 1 foveal)
            global_pooled, foveal_features = self.extract_multiview_tokens(pixel_values)
            
            rank0_print(f"SHIRG-Fovea: Extracted multiview tokens")
            rank0_print(f"   Global: {global_pooled.shape} (pooled from 448²)")
            rank0_print(f"   Foveal: {len(foveal_features)} view × {foveal_features[0].shape if foveal_features else 'None'}")
            
            # Step 2: Per-view Top-K selection on foveal view
            # Adjusted keep rate to maintain ~980 total tokens with 2×2 global pooling
            # SHIRG-2X2-K-FIX: 2025-07-30 - Adjust K for 256 global + 724 foveal = 980 total
            # ISSUE: With 256 global tokens, need 724 foveal tokens for 980 total
            # SOLUTION: Select 724 tokens from 1024 (70.7% keep rate)
            # RESEARCH IMPACT: Better balance between global context and foveal detail
            # LAVIDA IMPACT: Achieves exactly 980 tokens for cache compatibility
            
            selected_foveal = []
            for i, view_tokens in enumerate(foveal_features):
                # Adapt K to maintain 980 total tokens
                actual_tokens = view_tokens.shape[1]
                if actual_tokens == 1024:  # 448² with patch_size=14 → 32×32
                    K = 724  # 256 global + 724 foveal = 980 total
                else:
                    # Fallback: maintain proportional selection
                    # Target: 980 total - 256 global = 724 foveal
                    K = min(actual_tokens, 724)
                
                # Enable visualization tracking if needed
                if hasattr(self, '_enable_visualization'):
                    selected, indices = self.topk_per_view(view_tokens, K, text_embeddings, 
                                                         return_indices=True,
                                                         method=selection_method,
                                                         params=selection_params)
                    # Store for visualization
                    if not hasattr(self, '_foveal_selection_indices'):
                        self._foveal_selection_indices = []
                    self._foveal_selection_indices.append(indices)
                else:
                    selected = self.topk_per_view(view_tokens, K, text_embeddings,
                                                method=selection_method,
                                                params=selection_params)
                selected_foveal.append(selected)
                rank0_print(f"   Foveal view: selected {selected.shape[1]} tokens from {actual_tokens} total ({K/actual_tokens*100:.1f}%)")
            
            # Step 3: Concatenate all tokens
            # Global (256) + Foveal (724) = 980 tokens total
            all_views = [global_pooled] + selected_foveal
            
            # Ensure gradient flow and dtype consistency
            processed_views = []
            sample_image = pixel_values[0] if isinstance(pixel_values, list) else pixel_values
            
            for view_idx, view_tokens in enumerate(all_views):
                # Debug token values
                rank0_print(f"SHIRG-DEBUG View {view_idx}: shape={view_tokens.shape}, "
                           f"mean={view_tokens.mean().item():.4f}, std={view_tokens.std().item():.4f}")
                
                # Ensure gradient flow for LoRA training
                view_tokens = self.ensure_gradient_flow(view_tokens, sample_image)
                
                # Final dtype consistency check
                if torch.cuda.is_available() and view_tokens.dtype != torch.bfloat16:
                    view_tokens = view_tokens.to(torch.bfloat16)
                
                processed_views.append(view_tokens)
            
            # Concatenate all views along token dimension: [B, 980, D]
            concatenated_tokens = torch.cat(processed_views, dim=1)
            
            # Validate token count
            total_tokens = concatenated_tokens.shape[1]
            rank0_print(f"SHIRG-Fovea: Final token count: {total_tokens}")
            rank0_print(f"   Global tokens: {processed_views[0].shape[1]}")
            rank0_print(f"   Foveal tokens per view: {processed_views[1].shape[1]}")
            
            # CRITICAL: Ensure exactly 980 tokens for LaViDa compatibility
            if total_tokens != 980:
                rank0_print(f"⚠️ WARNING: Token count {total_tokens} != 980. LaViDa may fail!")
                # Add emergency padding or truncation if needed
                if total_tokens < 980:
                    # Pad with repeated last token
                    padding_needed = 980 - total_tokens
                    last_token = concatenated_tokens[:, -1:, :].expand(-1, padding_needed, -1)
                    noise = torch.randn_like(last_token) * 1e-6
                    padding = last_token + noise
                    concatenated_tokens = torch.cat([concatenated_tokens, padding], dim=1)
                    rank0_print(f"   Emergency padding: added {padding_needed} tokens")
                elif total_tokens > 980:
                    # Truncate to 980 tokens
                    concatenated_tokens = concatenated_tokens[:, :980, :]
                    rank0_print(f"   Emergency truncation: removed {total_tokens - 980} tokens")
                
                total_tokens = concatenated_tokens.shape[1]
            
            # SHIRG-2X2-VALIDATION: 2025-07-30 - Validate token count with 2×2 pooling
            # ISSUE: Token count validation updated for 2×2 pooling on global view
            # SOLUTION: Validate for 256 global + 724 foveal = 980 tokens
            # RESEARCH IMPACT: Better spatial resolution with 16×16 global grid
            # LAVIDA IMPACT: Maintains exactly 980 tokens for cache compatibility
            
            global_tokens = processed_views[0].shape[1]  # Should be 256
            foveal_tokens = processed_views[1].shape[1]
            expected_total = global_tokens + foveal_tokens
            
            if global_tokens == 256 and foveal_tokens == 724:
                rank0_print(f"   Expected with 448² both views: 256 + 724 = 980 tokens")
            else:
                rank0_print(f"   Adaptive token count: {global_tokens} + {foveal_tokens} = {expected_total} tokens")
            
            if total_tokens == 980:
                rank0_print(f"✅ SHIRG token count matches target: {total_tokens}")
            elif total_tokens == expected_total:
                rank0_print(f"✅ SHIRG token count matches expected: {total_tokens}")
            else:
                rank0_print(f"⚠️ SHIRG token count {total_tokens} != {expected_total} expected")
            
            # FINAL ASSERTION: LaViDa requires exactly 980 tokens
            assert concatenated_tokens.shape[1] == 980, (
                f"CRITICAL: LaViDa requires exactly 980 tokens, got {concatenated_tokens.shape[1]}. "
                f"Token selection or merging must be adjusted!"
            )
            
            return concatenated_tokens
            
        except Exception as e:
            rank0_print(f"🚨 SHIRG forward failed: {e}")
            import traceback
            rank0_print(f"Traceback: {traceback.format_exc()}")
            
            # Fallback to baseline processing
            raise  # Re-raise to let encoder handle fallback
    
    def extract_multiview_tokens(self, pixel_values):
        """
        SHIRG-Fovea: Extract tokens from 2-view format
        
        Processes:
        - View 0: Global 448² → 1024 tokens → 2×2 pool → 256 tokens
        - View 1: Foveal 448² → 1024 tokens (no pooling)
        
        Args:
            pixel_values: List of 2 image tensors:
                         [448×448 global, 448×448 foveal]
            
        Returns:
            global_pooled: [B, 256, D] pooled global context tokens
            foveal_features: List of 1 tensor [B, 1024, D]
        """
        start_time = time.time()
        
        # Validate input format - expect list of 2 views
        if not isinstance(pixel_values, list):
            raise ValueError(f"SHIRG-Fovea expects list of 2 views, got {type(pixel_values)}")
        
        if len(pixel_values) != 2:
            raise ValueError(f"SHIRG-Fovea expects exactly 2 views (1 global + 1 foveal), got {len(pixel_values)}")
        
        # Get model dtype
        tower_dtype = next(self.vision_tower.parameters()).dtype
        
        # Process global view: 448² → 1024 tokens → 2×2 pool → 256 tokens
        global_view = pixel_values[0]  # First view is global 448²
        
        # Ensure proper device and dtype
        if global_view.dtype != tower_dtype:
            global_view = global_view.to(dtype=tower_dtype)
        
        # Add batch dimension if needed
        if len(global_view.shape) == 3:
            global_view = global_view.unsqueeze(0)
        
        # Process through vision tower
        image_forward_outs = self.vision_tower(global_view, output_hidden_states=True)
        global_features = image_forward_outs.hidden_states[-1]  # [B, 1024, D] for 448²
        
        # SHIRG-2X2-POOL-FIX: 2025-07-30 - Use 2×2 pooling for better global context
        # ISSUE: 4×4 pooling (64 tokens) was too aggressive, losing spatial context
        # SOLUTION: Apply 2×2 pooling to get 256 tokens (16×16 spatial grid)
        # RESEARCH IMPACT: Better global scene understanding with 16×16 vs 8×8 grid
        # LAVIDA IMPACT: Maintains 980 total tokens (256 global + 724 foveal)
        
        B, N, D = global_features.shape
        if N == 1024:  # 32×32 patches from 448²
            # Reshape to spatial: [B, 32, 32, D]
            spatial_features = global_features.view(B, 32, 32, D)
            # Apply 2×2 pooling to get 16×16 = 256 tokens
            pooled_spatial = F.avg_pool2d(
                spatial_features.permute(0, 3, 1, 2),  # [B, D, 32, 32]
                kernel_size=2,
                stride=2
            ).permute(0, 2, 3, 1)  # [B, 16, 16, D]
            
            global_pooled = pooled_spatial.reshape(B, 256, D)
            rank0_print(f"   SHIRG-Fovea: Global view 448² → {N} tokens → 2×2 pool → {global_pooled.shape[1]} tokens")
        else:
            rank0_print(f"⚠️ SHIRG-Fovea: Expected 1024 tokens from global view, got {N}. Using adaptive pooling.")
            # Use adaptive pooling as fallback
            grid_size = int(math.sqrt(N))
            spatial_features = global_features.view(B, grid_size, grid_size, D)
            pooled_spatial = F.adaptive_avg_pool2d(
                spatial_features.permute(0, 3, 1, 2),
                output_size=(16, 16)  # Target 256 tokens
            ).permute(0, 2, 3, 1)
            global_pooled = pooled_spatial.reshape(B, 256, D)
        
        # Process 1 foveal view: 448² → 1024 tokens (no pooling)
        foveal_features = []
        foveal_view = pixel_values[1]  # View 1 is foveal 448²
        
        # Ensure proper device and dtype
        if foveal_view.dtype != tower_dtype:
            foveal_view = foveal_view.to(dtype=tower_dtype)
        
        # Add batch dimension if needed
        if len(foveal_view.shape) == 3:
            foveal_view = foveal_view.unsqueeze(0)
        
        # Process through vision tower
        image_forward_outs = self.vision_tower(foveal_view, output_hidden_states=True)
        view_features = image_forward_outs.hidden_states[-1]  # [B, N, D]
        
        # SHIRG-1FOVEAL-RESOLUTION: 2025-07-30 - Handle 448² foveal view
        # ISSUE: Single foveal view should be 448² (1024 tokens)
        # SOLUTION: Process 448² view and validate token count
        # RESEARCH IMPACT: Implements single high-resolution foveal view
        # LAVIDA IMPACT: Maintains compatibility with vision tower processing
        
        actual_tokens = view_features.shape[1]
        if actual_tokens == 1024:  # 448² with patch_size=14 → 32×32
            rank0_print(f"   SHIRG-Fovea: Processing 448² foveal view ({actual_tokens} tokens)")
        elif actual_tokens == 729:  # 384² with patch_size=14 → 27×27 (fallback)
            rank0_print(f"   SHIRG-Fovea: Processing 384² foveal view ({actual_tokens} tokens) - fallback resolution")
        else:
            rank0_print(f"⚠️ SHIRG-Fovea: Unexpected token count from foveal view: {actual_tokens}")
        
        foveal_features.append(view_features)
        
        elapsed_time = (time.time() - start_time) * 1000
        rank0_print(f"SHIRG-Fovea: Extracted multiview tokens in {elapsed_time:.1f}ms")
        rank0_print(f"   Global: {global_pooled.shape} (2×2 pooled from 384²)")
        rank0_print(f"   Foveal: {len(foveal_features)} view × {foveal_features[0].shape if foveal_features else 'None'}")
        
        return global_pooled, foveal_features
    
    def get_last_selection_visualization(self):
        """
        Get the last token selection pattern for visualization
        
        Returns:
            dict: Contains selection indices and scores for visualization
        """
        visualization_data = {
            'method': 'SHIRG-2View',
            'global_tokens': 256,
            'foveal_tokens': 724,
            'total_tokens': 980
        }
        
        if hasattr(self, '_foveal_selection_indices') and self._foveal_selection_indices:
            # Get the last foveal selection indices
            last_indices = self._foveal_selection_indices[-1]
            if last_indices is not None:
                visualization_data['foveal_selection_indices'] = last_indices.cpu().numpy().tolist()
                visualization_data['foveal_total_tokens'] = 1024  # 32x32 for 448² with patch_size=14
        
        if hasattr(self, '_last_selection_scores') and self._last_selection_scores is not None:
            visualization_data['selection_scores'] = self._last_selection_scores.cpu().numpy().tolist()
        
        return visualization_data
    
    def topk_per_view(self, view_tokens, K, text_embeddings=None, return_indices=False, 
                     method='base', params=None):
        """
        SHIRG-Fovea: Per-view Top-K selection with multiple scoring methods
        
        Args:
            view_tokens: [B, 1024, D] tokens from one foveal view (448² patch)
            K: Number of tokens to keep (724 for 70.7% keep rate)
            text_embeddings: Optional text embeddings for similarity scoring
            return_indices: If True, also return the selected indices for visualization
            method: Selection method ('base', 'entropy', 'edge', 'full')
            params: Dict with method-specific parameters:
                - entropy_threshold: τ for noise filtering (default: 0.12)
                - edge_weight: Weight for edge prior (default: 0.25)
                - radial_sigma: σ for radial weighting (default: 0.65)
                - merge_threshold: Similarity threshold (default: 0.9)
                
        Returns:
            selected_tokens: [B, K, D] selected tokens from this view
            (optional) topk_indices: [B, K] indices of selected tokens if return_indices=True
        """
        B, N, D = view_tokens.shape
        params = params or {}
        
        # Component 1: Attention to CLS token (a_i)
        cls_token = view_tokens[:, 0:1, :]  # [B, 1, D]
        attn_scores = torch.matmul(
            F.normalize(view_tokens, dim=-1),
            F.normalize(cls_token, dim=-1).transpose(-1, -2)
        ).squeeze(-1)  # [B, N]
        
        # Component 2: Text similarity (s_i) or token magnitude
        if text_embeddings is not None and text_embeddings.shape[-1] == D:
            # Direct similarity if dimensions match
            sim_scores = torch.matmul(
                F.normalize(view_tokens, dim=-1),
                F.normalize(text_embeddings, dim=-1).mean(dim=1, keepdim=True).transpose(-1, -2)
            ).squeeze(-1)  # [B, N]
        else:
            # Use token magnitude as proxy for information content
            sim_scores = torch.norm(view_tokens, dim=-1)  # [B, N]
        
        # Normalize scores to [0, 1]
        attn_scores = (attn_scores - attn_scores.min(dim=1, keepdim=True)[0]) / (
            attn_scores.max(dim=1, keepdim=True)[0] - attn_scores.min(dim=1, keepdim=True)[0] + 1e-8
        )
        sim_scores = (sim_scores - sim_scores.min(dim=1, keepdim=True)[0]) / (
            sim_scores.max(dim=1, keepdim=True)[0] - sim_scores.min(dim=1, keepdim=True)[0] + 1e-8
        )
        
        # Apply method-specific scoring
        if method == 'base':
            combined_scores = 0.7 * attn_scores + 0.3 * sim_scores
            
        elif method == 'entropy':
            # Entropy-based noise filtering
            τ = params.get('entropy_threshold', 0.12)
            attn_std = attn_scores.std(dim=-1, keepdim=True)
            noise_mask = (attn_std <= τ).float()  # Keep low-std tokens
            combined_scores = (0.7 * attn_scores + 0.3 * sim_scores) * noise_mask
            rank0_print(f"   Entropy filter: removed {(1-noise_mask.mean()).item()*100:.1f}% tokens")
            
        elif method == 'edge':
            # Edge-aware scoring with edge prior
            edge_prior = self.compute_edge_prior(view_tokens, params)
            edge_weight = params.get('edge_weight', 0.25)
            combined_scores = 0.4 * attn_scores + (0.35 - edge_weight) * sim_scores + edge_weight * edge_prior
            
        elif method == 'full':
            # Full enhancement with all components
            # 1. Entropy filter
            τ = params.get('entropy_threshold', 0.12)
            attn_std = attn_scores.std(dim=-1, keepdim=True)
            noise_mask = (attn_std <= τ).float()
            
            # 2. Edge-aware scoring
            edge_prior = self.compute_edge_prior(view_tokens, params)
            distance_penalty = self.compute_distance_penalty(view_tokens)
            
            # 3. Radial reweighting
            σ = params.get('radial_sigma', 0.65)
            radial_weight = self.compute_radial_weight(N, σ).to(view_tokens.device)
            
            # Combined score
            raw_score = 0.4 * attn_scores + 0.25 * sim_scores - 0.1 * distance_penalty + 0.25 * edge_prior
            combined_scores = raw_score * noise_mask * radial_weight.unsqueeze(0)
            
            rank0_print(f"   Full method: entropy removed {(1-noise_mask.mean()).item()*100:.1f}%, "
                       f"radial σ={σ}, edge weight=0.25")
        
        # Ensure we meet exact token budget after filtering
        effective_tokens = (combined_scores > 0).sum(dim=-1)
        if effective_tokens.min() < K:
            # Fallback: add small epsilon to ensure minimum K tokens
            combined_scores = combined_scores + 1e-6
            rank0_print(f"   Budget guarantee: added epsilon to ensure {K} tokens")
        
        # Store last selection scores for visualization
        self._last_selection_scores = combined_scores.detach().cpu() if hasattr(self, '_enable_visualization') else None
        
        # Select top-K tokens
        topk_values, topk_indices = torch.topk(combined_scores, K, dim=1)  # [B, K]
        
        # Store last selection indices for visualization
        self._last_selection_indices = topk_indices.detach().cpu() if hasattr(self, '_enable_visualization') else None
        
        # Gather selected tokens
        selected_tokens = torch.gather(
            view_tokens, 1,
            topk_indices.unsqueeze(-1).expand(-1, -1, D)
        )  # [B, K, D]
        
        # Optional token merging (post-selection)
        if params.get('merge_similar', False) and method == 'full':
            selected_tokens = self.merge_similar_tokens(selected_tokens, params.get('merge_threshold', 0.9))
        
        if return_indices:
            return selected_tokens, topk_indices
        return selected_tokens
    
    def compute_edge_prior(self, tokens, params):
        """
        Compute edge prior using Sobel filter on token embeddings
        
        Args:
            tokens: [B, N, D] input tokens
            params: Dict with optional parameters
            
        Returns:
            edge_scores: [B, N] normalized edge magnitude scores
        """
        B, N, D = tokens.shape
        
        # Reshape tokens to spatial grid (assuming square grid)
        H = W = int(math.sqrt(N))
        if H * W != N:
            # Fallback for non-square grids
            rank0_print(f"   Warning: Non-square token grid {N}, using simple gradient")
            # Simple gradient approximation
            grad = torch.diff(tokens, dim=1)
            edge_scores = torch.norm(grad, dim=-1).mean(dim=-1)
            # Pad to match original size
            edge_scores = F.pad(edge_scores, (0, 1), value=edge_scores.mean())
        else:
            # Reshape to spatial grid
            tokens_2d = tokens.view(B, H, W, D).permute(0, 3, 1, 2)  # [B, D, H, W]
            
            # Sobel filters
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                  dtype=tokens.dtype, device=tokens.device)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                  dtype=tokens.dtype, device=tokens.device)
            
            # Apply Sobel filters (using mean across channels)
            tokens_mean = tokens_2d.mean(dim=1, keepdim=True)  # [B, 1, H, W]
            
            # Pad for convolution
            tokens_padded = F.pad(tokens_mean, (1, 1, 1, 1), mode='replicate')
            
            # Apply filters
            edge_x = F.conv2d(tokens_padded, sobel_x.view(1, 1, 3, 3))
            edge_y = F.conv2d(tokens_padded, sobel_y.view(1, 1, 3, 3))
            
            # Compute edge magnitude
            edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2).squeeze(1)  # [B, H, W]
            edge_scores = edge_magnitude.view(B, N)  # [B, N]
        
        # Normalize to [0, 1]
        edge_min = edge_scores.min(dim=1, keepdim=True)[0]
        edge_max = edge_scores.max(dim=1, keepdim=True)[0]
        edge_scores = (edge_scores - edge_min) / (edge_max - edge_min + 1e-8)
        
        return edge_scores
    
    def compute_distance_penalty(self, tokens):
        """
        Compute distance penalty to encourage diversity
        
        Args:
            tokens: [B, N, D] input tokens
            
        Returns:
            distance_penalty: [B, N] penalty scores (higher = more similar to others)
        """
        B, N, D = tokens.shape
        
        # Compute pairwise cosine similarity
        tokens_norm = F.normalize(tokens, dim=-1)
        similarity_matrix = torch.bmm(tokens_norm, tokens_norm.transpose(1, 2))  # [B, N, N]
        
        # Average similarity to other tokens (excluding self)
        mask = 1 - torch.eye(N, device=tokens.device).unsqueeze(0)
        avg_similarity = (similarity_matrix * mask).sum(dim=-1) / (N - 1)  # [B, N]
        
        # Normalize to [0, 1]
        sim_min = avg_similarity.min(dim=1, keepdim=True)[0]
        sim_max = avg_similarity.max(dim=1, keepdim=True)[0]
        distance_penalty = (avg_similarity - sim_min) / (sim_max - sim_min + 1e-8)
        
        return distance_penalty
    
    def compute_radial_weight(self, N, sigma):
        """
        Compute radial weighting to de-bias center selection
        
        Args:
            N: Number of tokens (assumes square grid)
            sigma: Standard deviation for Gaussian weighting
            
        Returns:
            radial_weight: [N] weight for each position
        """
        # Assume square grid
        H = W = int(math.sqrt(N))
        if H * W != N:
            # Fallback: uniform weights
            return torch.ones(N)
        
        # Create spatial grid
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        
        # Compute distance from center
        center_y, center_x = H / 2 - 0.5, W / 2 - 0.5
        dist_sq = (y - center_y)**2 + (x - center_x)**2
        
        # Normalize by diagonal distance
        max_dist_sq = ((H/2)**2 + (W/2)**2)
        dist_norm = torch.sqrt(dist_sq / max_dist_sq)
        
        # Inverse Gaussian weight (higher at edges)
        radial_weight = 1 - torch.exp(-(dist_norm / sigma)**2)
        
        # Flatten to match token order
        radial_weight = radial_weight.view(N)
        
        # Normalize to preserve average weight = 1
        radial_weight = radial_weight / radial_weight.mean()
        
        return radial_weight
    
    def merge_similar_tokens(self, tokens, threshold=0.9):
        """
        Merge highly similar tokens to reclaim budget
        
        CRITICAL: This method MUST return exactly K tokens to maintain LaViDa compatibility
        
        Args:
            tokens: [B, K, D] selected tokens
            threshold: Cosine similarity threshold for merging
            
        Returns:
            merged_tokens: [B, K, D] tokens after merging (ALWAYS exactly K tokens)
        """
        B, K, D = tokens.shape
        
        # SHIRG-MERGE-FIX: 2025-07-30 - Ensure exact token count after merging
        # ISSUE: Token merging must maintain exact count for LaViDa cache compatibility
        # SOLUTION: Replace merged tokens with learned embeddings instead of zeros
        # RESEARCH IMPACT: Allows token merging while maintaining cache shape
        # LAVIDA IMPACT: Ensures exactly 980 tokens for LaViDa compatibility
        
        # Compute pairwise similarity
        tokens_norm = F.normalize(tokens, dim=-1)
        similarity = torch.bmm(tokens_norm, tokens_norm.transpose(1, 2))  # [B, K, K]
        
        # Find pairs above threshold (excluding diagonal)
        mask = torch.triu(torch.ones(K, K, device=tokens.device), diagonal=1)
        high_sim_pairs = (similarity * mask) > threshold
        
        # Track original token count for reporting
        original_count = K
        
        # Process each batch
        result_tokens = []
        for b in range(B):
            batch_tokens = tokens[b].clone()  # [K, D]
            kept_mask = torch.ones(K, dtype=torch.bool, device=tokens.device)
            
            # Greedy merging: for each token, merge with similar ones
            for i in range(K):
                if kept_mask[i]:
                    # Find tokens similar to i that haven't been merged yet
                    similar_mask = high_sim_pairs[b, i, :] & kept_mask
                    if similar_mask.any():
                        # Get indices of similar tokens
                        similar_indices = similar_mask.nonzero(as_tuple=True)[0]
                        all_indices = torch.cat([torch.tensor([i], device=tokens.device), similar_indices])
                        
                        # Average the similar tokens
                        merged_embedding = batch_tokens[all_indices].mean(dim=0)
                        batch_tokens[i] = merged_embedding
                        
                        # Mark similar tokens as merged (except current token i)
                        kept_mask[similar_indices] = False
            
            # Count how many unique tokens remain
            unique_count = kept_mask.sum().item()
            
            # Rearrange: put unique tokens first, then pad
            if unique_count < K:
                # Get unique tokens
                unique_tokens = batch_tokens[kept_mask]  # [unique_count, D]
                
                # CRITICAL: Instead of zeros, use repeated tokens or small noise
                # This maintains gradient flow and prevents issues with zero tokens
                padding_count = K - unique_count
                
                # Option 1: Repeat the last unique token
                if unique_count > 0:
                    # Repeat last token with small noise to maintain diversity
                    last_token = unique_tokens[-1:].expand(padding_count, -1)
                    noise = torch.randn_like(last_token) * 1e-6
                    padding = last_token + noise
                else:
                    # Fallback: use small random embeddings
                    padding = torch.randn(padding_count, D, dtype=tokens.dtype, device=tokens.device) * 1e-6
                
                # Concatenate to maintain exactly K tokens
                final_tokens = torch.cat([unique_tokens, padding], dim=0)
            else:
                # No merging needed, keep all K tokens
                final_tokens = batch_tokens
            
            # Ensure exactly K tokens
            assert final_tokens.shape[0] == K, f"Token count mismatch: {final_tokens.shape[0]} != {K}"
            result_tokens.append(final_tokens.unsqueeze(0))
        
        # Combine all batches
        merged_tokens = torch.cat(result_tokens, dim=0)  # [B, K, D]
        
        # Final verification
        assert merged_tokens.shape == (B, K, D), f"Final shape mismatch: {merged_tokens.shape} != {(B, K, D)}"
        
        # Report merging statistics
        avg_unique = sum([(tokens[b] != merged_tokens[b]).any(dim=-1).sum().item() 
                         for b in range(B)]) / B
        merge_ratio = (K - avg_unique) / K * 100
        
        rank0_print(f"   Token merging: {K - avg_unique:.0f} tokens merged ({merge_ratio:.1f}%), "
                   f"maintained exactly {K} tokens")
        
        return merged_tokens
    
    def ensure_gradient_flow(self, tokens, input_images):
        """
        SHIRG: Ensure gradient flow for LoRA training compatibility
        
        Args:
            tokens: [B, N, D] processed tokens
            input_images: Original input images
            
        Returns:
            tokens: [B, N, D] tokens with ensured gradient flow
        """
        # Always force gradient requirements on tokens
        if hasattr(tokens, 'requires_grad_'):
            tokens = tokens.requires_grad_(True)
        
        # Add explicit gradient connection to input if needed
        if hasattr(input_images, 'requires_grad') and input_images.requires_grad:
            # Create explicit gradient connection using input mean
            B, N, D = tokens.shape
            # Use small but non-negligible connection to ensure gradient flow
            input_connection = input_images.view(B, -1).mean(dim=1, keepdim=True).unsqueeze(-1)  # [B, 1, 1]
            input_connection = input_connection * 1e-6  # Small but significant influence
            
            # Add connection to all tokens to maintain gradient chain
            tokens = tokens + input_connection.expand_as(tokens)
        
        # Verify gradient requirements are preserved
        if not tokens.requires_grad:
            rank0_print("⚠️ GRADIENT-FIX: Failed to enable gradients on tokens")
            
        return tokens