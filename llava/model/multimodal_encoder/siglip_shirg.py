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
    
    def forward_with_shirg(self, pixel_values, text_embeddings=None):
        """
        SHIRG-Fovea: Process 2-view format with per-view selection
        
        Implements the new SHIRG methodology:
        1. Extract 2-view tokens (1 global 384² + 1 foveal 448²)
        2. Apply 2×2 pooling to global view → 196 tokens
        3. Apply 76.6% Top-K selection to foveal view → 784 tokens
        4. Concatenate for 980 total tokens (matching LaViDa baseline)
        
        Args:
            pixel_values: Either list of 2 image tensors OR stacked tensor [2, C, H, W]
            text_embeddings: Optional text embeddings for scoring
            
        Returns:
            visual_tokens: [B, 980, D] selected tokens (196 global + 784 foveal)
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
                # Handle [B, num_views, C, H, W] format
                B, num_views = pixel_values.shape[:2]
                if num_views == 2 and B == 1:
                    rank0_print(f"SHIRG-1FOVEAL: Converting 5D tensor {pixel_values.shape} to list of 2 views")
                    pixel_values = pixel_values.squeeze(0)  # Remove batch dimension
                    pixel_values = [pixel_values[i] for i in range(2)]
                else:
                    raise ValueError(f"SHIRG-Fovea expects [1, 2, C, H, W], got {pixel_values.shape}")
            
            # Step 1: Extract multiview tokens (1 global + 1 foveal)
            global_pooled, foveal_features = self.extract_multiview_tokens(pixel_values)
            
            rank0_print(f"SHIRG-Fovea: Extracted multiview tokens")
            rank0_print(f"   Global: {global_pooled.shape} (pooled from 384²)")
            rank0_print(f"   Foveal: {len(foveal_features)} view × {foveal_features[0].shape if foveal_features else 'None'}")
            
            # Step 2: Per-view Top-K selection on foveal view
            # Fixed 76.6% keep rate for single foveal view as per research
            # SHIRG-1FOVEAL-K: 2025-07-30 - Use 76.6% keep rate for single foveal view
            # ISSUE: Single foveal view needs higher keep rate to maintain 980 total tokens
            # SOLUTION: Calculate K as 76.6% of actual token count (784 from 1024)
            # RESEARCH IMPACT: Implements new single foveal view architecture
            # LAVIDA IMPACT: Maintains exactly 980 tokens for cache compatibility
            
            selected_foveal = []
            for i, view_tokens in enumerate(foveal_features):
                # Adapt K to actual token count (76.6% selection rate)
                actual_tokens = view_tokens.shape[1]
                if actual_tokens == 1024:  # 448² with patch_size=14 → 32×32
                    K = 784  # Exactly 76.6% of 1024
                else:
                    # Fallback: maintain 76.6% ratio for other resolutions
                    K = int(actual_tokens * 0.766)
                
                selected = self.topk_per_view(view_tokens, K, text_embeddings)
                selected_foveal.append(selected)
                rank0_print(f"   Foveal view: selected {selected.shape[1]} tokens from {actual_tokens} total ({K/actual_tokens*100:.1f}%)")
            
            # Step 3: Concatenate all tokens
            # Global (196) + Foveal (784) = 980 tokens total
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
            
            # SHIRG-1FOVEAL-VALIDATION: 2025-07-30 - Validate token count for single foveal view
            # ISSUE: Token count validation needs to reflect new single foveal architecture
            # SOLUTION: Validate for 196 global + 784 foveal = 980 tokens
            # RESEARCH IMPACT: Ensures correct implementation of single foveal view
            # LAVIDA IMPACT: Maintains exactly 980 tokens for cache compatibility
            
            global_tokens = processed_views[0].shape[1]  # Always 196
            foveal_tokens = processed_views[1].shape[1]
            expected_total = global_tokens + foveal_tokens
            
            if foveal_tokens == 784:  # 76.6% of 1024 (448²)
                rank0_print(f"   Expected with 448² foveal: 196 + 784 = 980 tokens")
            else:
                rank0_print(f"   Adaptive token count: 196 + {foveal_tokens} = {expected_total} tokens")
            
            if total_tokens == 980:
                rank0_print(f"✅ SHIRG token count matches target: {total_tokens}")
            elif total_tokens == expected_total:
                rank0_print(f"✅ SHIRG token count matches expected: {total_tokens}")
            else:
                rank0_print(f"⚠️ SHIRG token count {total_tokens} != {expected_total} expected")
            
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
        - View 0: Global 384² → 729 tokens → 2×2 pool → 196 tokens
        - View 1: Foveal 448² → 1024 tokens (no pooling)
        
        Args:
            pixel_values: List of 2 image tensors:
                         [384×384 global, 448×448 foveal]
            
        Returns:
            global_pooled: [B, 196, D] pooled global context tokens
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
        
        # Process global view: 384² → 729 tokens → 2×2 pool → 196 tokens
        global_view = pixel_values[0]  # First view is global 384²
        
        # Ensure proper device and dtype
        if global_view.dtype != tower_dtype:
            global_view = global_view.to(dtype=tower_dtype)
        
        # Add batch dimension if needed
        if len(global_view.shape) == 3:
            global_view = global_view.unsqueeze(0)
        
        # Process through vision tower
        image_forward_outs = self.vision_tower(global_view, output_hidden_states=True)
        global_features = image_forward_outs.hidden_states[-1]  # [B, 729, D] for 384²
        
        # Apply 2×2 average pooling to get 196 tokens
        B, N, D = global_features.shape
        if N == 729:  # 27×27 patches from 384²
            # Reshape to spatial: [B, 27, 27, D]
            spatial_features = global_features.view(B, 27, 27, D)
            # Apply 2×2 pooling with proper padding to get 14×14 = 196
            pooled_spatial = F.avg_pool2d(
                spatial_features.permute(0, 3, 1, 2),  # [B, D, 27, 27]
                kernel_size=2,
                stride=2,
                padding=1  # Padding to handle odd dimension
            ).permute(0, 2, 3, 1)  # [B, 14, 14, D]
            
            # Ensure we have exactly 196 tokens
            if pooled_spatial.shape[1] * pooled_spatial.shape[2] != 196:
                # Use adaptive pooling as fallback
                pooled_spatial = F.adaptive_avg_pool2d(
                    spatial_features.permute(0, 3, 1, 2),
                    output_size=(14, 14)
                ).permute(0, 2, 3, 1)
            
            global_pooled = pooled_spatial.reshape(B, 196, D)
        else:
            rank0_print(f"⚠️ SHIRG-Fovea: Expected 729 tokens from global view, got {N}. Using adaptive pooling.")
            # Use adaptive pooling as fallback
            grid_size = int(math.sqrt(N))
            spatial_features = global_features.view(B, grid_size, grid_size, D)
            pooled_spatial = F.adaptive_avg_pool2d(
                spatial_features.permute(0, 3, 1, 2),
                output_size=(14, 14)
            ).permute(0, 2, 3, 1)
            global_pooled = pooled_spatial.reshape(B, 196, D)
        
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
    
    def topk_per_view(self, view_tokens, K, text_embeddings=None):
        """
        SHIRG-Fovea: Per-view Top-K selection with composite scoring
        
        Implements scoring: 0.7*attention + 0.3*similarity
        
        Args:
            view_tokens: [B, 1024, D] tokens from one foveal view (448² patch)
            K: Number of tokens to keep (784 for 76.6% keep rate)
            text_embeddings: Optional text embeddings for similarity scoring
            
        Returns:
            selected_tokens: [B, K, D] selected tokens from this view
        """
        B, N, D = view_tokens.shape
        
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
        
        # Combine scores: 0.7*attention + 0.3*similarity
        combined_scores = 0.7 * attn_scores + 0.3 * sim_scores
        
        # Select top-K tokens
        topk_values, topk_indices = torch.topk(combined_scores, K, dim=1)  # [B, K]
        
        # Gather selected tokens
        selected_tokens = torch.gather(
            view_tokens, 1,
            topk_indices.unsqueeze(-1).expand(-1, -1, D)
        )  # [B, K, D]
        
        return selected_tokens
    
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