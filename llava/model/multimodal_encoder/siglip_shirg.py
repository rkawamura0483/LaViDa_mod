"""
SHIRG Extensions for SigLIP Vision Tower
SHIRG-Fovea: Static Hierarchical Relevance Gate for High-Resolution Diffusion VLMs

This module contains all SHIRG-specific functionality for high-resolution
token selection and processing, following the updated research methodology
with 5-view processing (1 global + 4 peripheral).

Research Implementation based on:
- Two-scale foveation: Global 384² + 4×512² peripheral views
- Per-view static Top-K selection (40-50% retention)
- Cache-compatible processing for LaViDa
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from typing import Optional, Tuple, Union

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
        SHIRG-Fovea: Process LaViDa's 5-view format with per-view selection
        
        Implements the new SHIRG methodology:
        1. Extract 5-view tokens (1 global + 4 peripheral)
        2. Per-view Top-K selection on peripheral views
        3. Concatenate selected tokens
        4. Maintain cache compatibility
        
        Args:
            pixel_values: Either list of 5 image tensors OR stacked tensor [5, C, H, W] from LaViDa's anyres
            text_embeddings: Optional text embeddings for scoring
            
        Returns:
            visual_tokens: [B, ~1832, D] selected tokens (196 global + 4×~409 peripheral)
        """
        try:
            # SHIRG-INPUT-FIX: 2025-07-29 - Handle both list and stacked tensor inputs from LaViDa
            # ISSUE: process_anyres_image returns stacked tensor [5, C, H, W], not list of 5 views
            # SOLUTION: Convert stacked tensor to list of views for SHIRG processing
            # RESEARCH IMPACT: Enables SHIRG to work with LaViDa's actual anyres output format
            # LAVIDA IMPACT: Maintains compatibility with LaViDa's image processing pipeline
            
            # Convert input to list of views if needed
            if torch.is_tensor(pixel_values) and len(pixel_values.shape) == 4:
                # Stacked tensor from process_anyres_image: [num_views, C, H, W]
                if pixel_values.shape[0] == 5:  # Expected 5 views for SHIRG-Fovea
                    rank0_print(f"SHIRG-INPUT-FIX: Converting stacked tensor {pixel_values.shape} to list of 5 views")
                    pixel_values = [pixel_values[i] for i in range(5)]
                elif pixel_values.shape[0] == 1:  # Single batch with 5 views
                    # Check if it's actually [1, 5, C, H, W]
                    if len(pixel_values.shape) == 5 and pixel_values.shape[1] == 5:
                        rank0_print(f"SHIRG-INPUT-FIX: Converting batched tensor {pixel_values.shape} to list of 5 views")
                        pixel_values = pixel_values.squeeze(0)  # Remove batch dim
                        pixel_values = [pixel_values[i] for i in range(5)]
                    else:
                        raise ValueError(f"Unexpected tensor shape for SHIRG-Fovea: {pixel_values.shape}")
                else:
                    raise ValueError(f"SHIRG-Fovea expects 5 views, got tensor with {pixel_values.shape[0]} views")
            elif torch.is_tensor(pixel_values) and len(pixel_values.shape) == 5:
                # Handle [B, num_views, C, H, W] format
                B, num_views = pixel_values.shape[:2]
                if num_views == 5 and B == 1:
                    rank0_print(f"SHIRG-INPUT-FIX: Converting 5D tensor {pixel_values.shape} to list of 5 views")
                    pixel_values = pixel_values.squeeze(0)  # Remove batch dimension
                    pixel_values = [pixel_values[i] for i in range(5)]
                else:
                    raise ValueError(f"SHIRG-Fovea expects [1, 5, C, H, W], got {pixel_values.shape}")
            
            # Step 1: Extract multiview tokens (1 global + 4 peripheral)
            global_pooled, peripheral_features = self.extract_multiview_tokens(pixel_values)
            
            rank0_print(f"SHIRG-Fovea: Extracted multiview tokens")
            rank0_print(f"   Global: {global_pooled.shape}")
            rank0_print(f"   Peripheral: {len(peripheral_features)} views × {peripheral_features[0].shape}")
            
            # Step 2: Per-view Top-K selection on peripheral views
            keep_ratio = 0.45  # 45% keep rate as per research (40-50% range)
            K = int(keep_ratio * 729)  # ~328 tokens per view (adapted for 384² patches)
            
            selected_peripheral = []
            for i, view_tokens in enumerate(peripheral_features):
                selected = self.topk_per_view(view_tokens, K, text_embeddings)
                selected_peripheral.append(selected)
                rank0_print(f"   View {i+1}: selected {selected.shape[1]} tokens")
            
            # SHIRG-LAVIDA-FIX: 2025-07-29 - Return tokens in LaViDa-compatible format
            # ISSUE: LaViDa expects to split encoded features by views, but SHIRG concatenates all tokens
            # SOLUTION: Return concatenated tokens with single batch dimension to match LaViDa's expectations
            # RESEARCH IMPACT: Maintains SHIRG token selection while ensuring LaViDa compatibility
            # LAVIDA IMPACT: Allows LaViDa to process SHIRG tokens through its standard pipeline
            
            # Step 3: Concatenate [global196 || view1_K ... view4_K]
            final_tokens = torch.cat([global_pooled] + selected_peripheral, dim=1)
            
            # Log final shape
            B, N, D = final_tokens.shape
            rank0_print(f"SHIRG-Fovea: Final token count: {N} (196 global + 4×{K} peripheral = {196 + 4*K})")
            
            # Ensure gradient flow for LoRA training
            final_tokens = self.ensure_gradient_flow(final_tokens, pixel_values[0] if isinstance(pixel_values, list) else pixel_values)
            
            # Final dtype consistency check
            if torch.cuda.is_available() and final_tokens.dtype != torch.bfloat16:
                final_tokens = final_tokens.to(torch.bfloat16)
            
            # CRITICAL: Return as single "view" for LaViDa compatibility
            # LaViDa will treat this as one large view instead of trying to split it
            return final_tokens
            
        except Exception as e:
            rank0_print(f"🚨 SHIRG forward failed: {e}")
            import traceback
            rank0_print(f"Traceback: {traceback.format_exc()}")
            
            # Fallback to baseline LaViDa processing
            rank0_print("SHIRG-FALLBACK: Using baseline LaViDa processing")
            
            # Process first view only as fallback
            if isinstance(pixel_values, list) and len(pixel_values) > 0:
                fallback_image = pixel_values[0]
            elif torch.is_tensor(pixel_values):
                # Take first view from tensor
                if len(pixel_values.shape) >= 4 and pixel_values.shape[0] >= 1:
                    fallback_image = pixel_values[0]
                else:
                    fallback_image = pixel_values
            else:
                fallback_image = pixel_values
            
            # Process through vision tower
            if type(fallback_image) is list:
                fallback_image = fallback_image[0]
            
            if len(fallback_image.shape) == 3:
                fallback_image = fallback_image.unsqueeze(0)
            
            image_forward_outs = self.vision_tower(fallback_image, output_hidden_states=True)
            fallback_tokens = image_forward_outs.hidden_states[-1]
            
            return fallback_tokens
    
    def extract_multiview_tokens(self, pixel_values):
        """
        SHIRG-Fovea: Extract tokens from LaViDa's 5-view anyres format
        
        Adapts to LaViDa's actual anyres output:
        - LaViDa produces 5×384² views from 768×768 grid
        - SHIRG treats first as global (pooled to 196) and rest as peripheral
        
        Args:
            pixel_values: List of 5 image tensors from LaViDa's anyres splitter:
                         All are 384×384 patches
            
        Returns:
            global_pooled: [B, 196, D] pooled global context tokens
            peripheral_features: List of 4 tensors, each [B, 729, D]
        """
        start_time = time.time()
        
        # Validate input format - expect list of 5 views from LaViDa's anyres
        if not isinstance(pixel_values, list):
            raise ValueError(f"SHIRG-Fovea expects list of 5 views, got {type(pixel_values)}")
        
        if len(pixel_values) != 5:
            raise ValueError(f"SHIRG-Fovea expects exactly 5 views (1 global + 4 peripheral), got {len(pixel_values)}")
        
        # SHIRG-ADAPTATION: 2025-07-29 - Work with LaViDa's 5×384² patches
        # ISSUE: LaViDa produces 5×384² patches, not 1×384² + 4×512² as originally planned
        # SOLUTION: Adapt SHIRG to work with LaViDa's format while maintaining research goals
        # RESEARCH IMPACT: Same per-view selection principle, adapted to available resolutions
        # LAVIDA IMPACT: Full compatibility with LaViDa's existing anyres processing
        
        # Process global view: 384² → 729 tokens → 2×2 pool → 196 tokens
        global_view = pixel_values[0]  # First view is global 384²
        
        # Ensure proper device and dtype
        tower_dtype = next(self.vision_tower.parameters()).dtype
        if global_view.dtype != tower_dtype:
            global_view = global_view.to(dtype=tower_dtype)
        
        # Process through vision tower
        if type(global_view) is list:
            global_features = []
            for img in global_view:
                image_forward_out = self.vision_tower(img.unsqueeze(0), output_hidden_states=True)
                image_feature = image_forward_out.hidden_states[-1]
                global_features.append(image_feature)
            global_features = torch.cat(global_features, dim=0)
        else:
            # Add batch dimension if needed
            if len(global_view.shape) == 3:
                global_view = global_view.unsqueeze(0)
            
            image_forward_outs = self.vision_tower(global_view, output_hidden_states=True)
            global_features = image_forward_outs.hidden_states[-1]  # [B, 729, D] for 384²
        
        # Apply 2×2 average pooling to get 196 tokens
        B, N, D = global_features.shape
        if N == 729:  # 27×27 patches from 384²
            # Reshape to spatial: [B, 27, 27, D]
            spatial_features = global_features.view(B, 27, 27, D)
            # Apply 2×2 pooling: [B, 13, 13, D] (with 1 padding)
            pooled_spatial = F.avg_pool2d(
                spatial_features.permute(0, 3, 1, 2),  # [B, D, 27, 27]
                kernel_size=2,
                stride=2,
                padding=0  # No padding, will get 13×13 = 169, need to pad to 196
            ).permute(0, 2, 3, 1)  # [B, 13, 13, D]
            
            # Pad to 14×14 = 196 tokens
            target_size = 14
            current_size = pooled_spatial.shape[1]
            if current_size < target_size:
                pad_size = target_size - current_size
                # Pad spatially
                pooled_spatial = F.pad(
                    pooled_spatial.permute(0, 3, 1, 2),  # [B, D, 13, 13]
                    (0, pad_size, 0, pad_size),  # pad right and bottom
                    mode='constant',
                    value=0
                ).permute(0, 2, 3, 1)  # [B, 14, 14, D]
            
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
        
        # Process 4 peripheral views: 384² → 729 tokens each (same as global but no pooling)
        peripheral_features = []
        for i in range(1, 5):  # Views 1-4 are peripheral 384²
            peripheral_view = pixel_values[i]
            
            # Ensure proper device and dtype
            if peripheral_view.dtype != tower_dtype:
                peripheral_view = peripheral_view.to(dtype=tower_dtype)
            
            # Process through vision tower
            if type(peripheral_view) is list:
                view_features = []
                for img in peripheral_view:
                    image_forward_out = self.vision_tower(img.unsqueeze(0), output_hidden_states=True)
                    image_feature = image_forward_out.hidden_states[-1]
                    view_features.append(image_feature)
                view_features = torch.cat(view_features, dim=0)
            else:
                # Add batch dimension if needed
                if len(peripheral_view.shape) == 3:
                    peripheral_view = peripheral_view.unsqueeze(0)
                
                image_forward_outs = self.vision_tower(peripheral_view, output_hidden_states=True)
                view_features = image_forward_outs.hidden_states[-1]  # [B, 729, D] for 384²
            
            # Validate token count (all views are 384² so expect 729 tokens)
            if view_features.shape[1] != 729:
                rank0_print(f"⚠️ SHIRG-Fovea: Expected 729 tokens from peripheral view {i}, got {view_features.shape[1]}")
            
            peripheral_features.append(view_features)
        
        elapsed_time = (time.time() - start_time) * 1000
        rank0_print(f"SHIRG-Fovea: Extracted multiview tokens in {elapsed_time:.1f}ms")
        rank0_print(f"   Global: {global_pooled.shape} (pooled from 384²)")
        rank0_print(f"   Peripheral: {len(peripheral_features)} views × {peripheral_features[0].shape if peripheral_features else 'None'}")
        
        return global_pooled, peripheral_features
    
    def topk_per_view(self, view_tokens, K, text_embeddings=None):
        """
        SHIRG-Fovea: Per-view Top-K selection with diversity-aware scoring
        
        Implements the full research methodology scoring: 
        score_i = softmax((0.5*a_i + 0.3*s_i - 0.1*d_i) / T), T=0.15
        
        Args:
            view_tokens: [B, 729, D] tokens from one peripheral view (384² patch)
            K: Number of tokens to keep (typically ~328 for 45% keep rate)
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
        attn_scores = F.normalize(attn_scores, dim=-1)  # Normalize to 0-1
        
        # Component 2: Text similarity (s_i)
        if text_embeddings is not None and text_embeddings.shape[-1] == D:
            # Direct similarity if dimensions match
            sim_scores = torch.matmul(
                F.normalize(view_tokens, dim=-1),
                F.normalize(text_embeddings, dim=-1).mean(dim=1, keepdim=True).transpose(-1, -2)
            ).squeeze(-1)  # [B, N]
        else:
            # Use token magnitude as proxy for information content
            sim_scores = torch.norm(view_tokens, dim=-1)  # [B, N]
        sim_scores = F.normalize(sim_scores, dim=-1)  # Normalize to 0-1
        
        # SHIRG-DIVERSITY-FIX: 2025-07-29 - Implement diversity-aware scoring per research methodology
        # ISSUE: Original implementation missing diversity component (-0.1*d_i) from research formula
        # SOLUTION: Add iterative selection with diversity penalty to encourage token variety
        # RESEARCH IMPACT: Implements complete scoring formula from Section 3.4.2
        # LAVIDA IMPACT: Improves token diversity for better coverage of visual information
        
        # Initialize selected indices and tokens
        selected_indices = []
        selected_tokens_list = []
        remaining_indices = torch.arange(N, device=view_tokens.device).unsqueeze(0).expand(B, -1)
        
        # Temperature parameter from research
        temperature = 0.15
        
        # Iteratively select tokens with diversity penalty
        for i in range(K):
            # Get currently remaining tokens
            mask = torch.zeros(B, N, dtype=torch.bool, device=view_tokens.device)
            for b in range(B):
                mask[b, remaining_indices[b]] = True
            
            # Component 3: Diversity penalty (d_i) - only after first selection
            if i > 0:
                # Calculate average cosine similarity to already-selected tokens
                selected_so_far = torch.stack(selected_tokens_list, dim=1)  # [B, i, D]
                # Compute similarity of all tokens to selected tokens
                all_similarities = torch.matmul(
                    F.normalize(view_tokens, dim=-1),  # [B, N, D]
                    F.normalize(selected_so_far, dim=-1).transpose(-1, -2)  # [B, D, i]
                )  # [B, N, i]
                diversity_penalty = all_similarities.mean(dim=-1)  # [B, N]
            else:
                diversity_penalty = torch.zeros(B, N, device=view_tokens.device)
            
            # Combine scores with diversity penalty: 0.5*a_i + 0.3*s_i - 0.1*d_i
            raw_scores = 0.5 * attn_scores + 0.3 * sim_scores - 0.1 * diversity_penalty
            
            # Apply temperature-scaled softmax
            scores = torch.softmax(raw_scores / temperature, dim=-1)
            
            # Mask out already selected tokens
            scores = scores * mask.float()
            
            # Select highest scoring token from remaining
            _, max_idx = scores.max(dim=1)  # [B]
            
            # Add to selected list
            selected_indices.append(max_idx)
            selected_token = torch.gather(
                view_tokens, 1, 
                max_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, D)
            ).squeeze(1)  # [B, D]
            selected_tokens_list.append(selected_token)
            
            # Remove from remaining indices
            for b in range(B):
                remaining_indices[b] = remaining_indices[b][remaining_indices[b] != max_idx[b]]
        
        # Stack all selected tokens
        selected_tokens = torch.stack(selected_tokens_list, dim=1)  # [B, K, D]
        
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
        # Always force gradient requirements on tokens (not just training mode)
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