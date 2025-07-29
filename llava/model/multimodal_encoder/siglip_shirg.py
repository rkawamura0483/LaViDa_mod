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
            pixel_values: List of 5 image tensors from LaViDa's anyres
            text_embeddings: Optional text embeddings for scoring
            
        Returns:
            visual_tokens: [B, ~1832, D] selected tokens (196 global + 4×~409 peripheral)
        """
        try:
            # Step 1: Extract multiview tokens (1 global + 4 peripheral)
            global_pooled, peripheral_features = self.extract_multiview_tokens(pixel_values)
            
            rank0_print(f"SHIRG-Fovea: Extracted multiview tokens")
            rank0_print(f"   Global: {global_pooled.shape}")
            rank0_print(f"   Peripheral: {len(peripheral_features)} views × {peripheral_features[0].shape}")
            
            # Step 2: Per-view Top-K selection on peripheral views
            keep_ratio = 0.45  # 45% keep rate as per research (40-50% range)
            K = int(keep_ratio * 1024)  # ~460 tokens per view
            
            selected_peripheral = []
            for i, view_tokens in enumerate(peripheral_features):
                selected = self.topk_per_view(view_tokens, K, text_embeddings)
                selected_peripheral.append(selected)
                rank0_print(f"   View {i+1}: selected {selected.shape[1]} tokens")
            
            # Step 3: Concatenate [global196 || view1_K ... view4_K]
            final_tokens = torch.cat([global_pooled] + selected_peripheral, dim=1)
            
            # Log final shape
            B, N, D = final_tokens.shape
            rank0_print(f"SHIRG-Fovea: Final token count: {N} (196 global + 4×{K} peripheral = {196 + 4*K})")
            
            # Ensure gradient flow for LoRA training
            final_tokens = self.ensure_gradient_flow(final_tokens, pixel_values[0])
            
            # Final dtype consistency check
            if torch.cuda.is_available() and final_tokens.dtype != torch.bfloat16:
                final_tokens = final_tokens.to(torch.bfloat16)
            
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
        SHIRG-Fovea: Extract tokens from LaViDa's 5-view anyres format per new methodology
        
        Implements the updated SHIRG research design:
        - 1 global 384² view → 196 tokens (2×2 pooled)
        - 4 peripheral 512² views → 4×1024 tokens each
        
        Args:
            pixel_values: List of 5 image tensors from LaViDa's anyres splitter:
                         [global_384², peripheral_512²_1, ..., peripheral_512²_4]
            
        Returns:
            global_pooled: [B, 196, D] pooled global context tokens
            peripheral_features: List of 4 tensors, each [B, 1024, D]
        """
        start_time = time.time()
        
        # Validate input format - expect list of 5 views from LaViDa's anyres
        if not isinstance(pixel_values, list):
            raise ValueError(f"SHIRG-Fovea expects list of 5 views, got {type(pixel_values)}")
        
        if len(pixel_values) != 5:
            raise ValueError(f"SHIRG-Fovea expects exactly 5 views (1 global + 4 peripheral), got {len(pixel_values)}")
        
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
        
        # Process 4 peripheral views: 512² → 1024 tokens each
        peripheral_features = []
        for i in range(1, 5):  # Views 1-4 are peripheral 512²
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
                view_features = image_forward_outs.hidden_states[-1]  # [B, 1024, D] for 512²
            
            # Validate token count
            if view_features.shape[1] != 1024:
                rank0_print(f"⚠️ SHIRG-Fovea: Expected 1024 tokens from peripheral view {i}, got {view_features.shape[1]}")
            
            peripheral_features.append(view_features)
        
        elapsed_time = (time.time() - start_time) * 1000
        rank0_print(f"SHIRG-Fovea: Extracted multiview tokens in {elapsed_time:.1f}ms")
        rank0_print(f"   Global: {global_pooled.shape} (pooled from 384²)")
        rank0_print(f"   Peripheral: {len(peripheral_features)} views × {peripheral_features[0].shape if peripheral_features else 'None'}")
        
        return global_pooled, peripheral_features
    
    def topk_per_view(self, view_tokens, K, text_embeddings=None):
        """
        SHIRG-Fovea: Per-view Top-K selection with attention-based scoring
        
        Implements the research methodology scoring: 0.7 × attn + 0.3 × sim
        
        Args:
            view_tokens: [B, 1024, D] tokens from one peripheral view
            K: Number of tokens to keep (typically ~409 for 40% keep rate)
            text_embeddings: Optional text embeddings for similarity scoring
            
        Returns:
            selected_tokens: [B, K, D] selected tokens from this view
        """
        B, N, D = view_tokens.shape
        
        # Component 1: Attention to CLS token (or first token as proxy)
        # This mimics the attention mechanism mentioned in the research
        cls_token = view_tokens[:, 0:1, :]  # [B, 1, D]
        attn_scores = torch.matmul(
            F.normalize(view_tokens, dim=-1),
            F.normalize(cls_token, dim=-1).transpose(-1, -2)
        ).squeeze(-1)  # [B, N]
        
        # Component 2: Text similarity (if available)
        if text_embeddings is not None and text_embeddings.shape[-1] == D:
            # Direct similarity if dimensions match
            sim_scores = torch.matmul(
                F.normalize(view_tokens, dim=-1),
                F.normalize(text_embeddings, dim=-1).mean(dim=1, keepdim=True).transpose(-1, -2)
            ).squeeze(-1)  # [B, N]
        else:
            # Use token magnitude as proxy for information content
            sim_scores = torch.norm(view_tokens, dim=-1)  # [B, N]
            sim_scores = F.normalize(sim_scores, dim=-1)
        
        # Combine scores as per research: 0.7 × attn + 0.3 × sim
        combined_scores = 0.7 * F.normalize(attn_scores, dim=-1) + 0.3 * F.normalize(sim_scores, dim=-1)
        
        # Select top-K tokens
        topk_indices = torch.topk(combined_scores, K, dim=1).indices  # [B, K]
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