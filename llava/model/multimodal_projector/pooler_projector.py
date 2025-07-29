import torch
import torch.nn as nn

import math

from transformers.models.clip.modeling_clip import CLIPVisionModel


class PoolerProjector(nn.Module):
    def __init__(self, config, vision_cfg,pooler_ratio=2):
        super().__init__()
        self._config = config
        self.hw = vision_cfg.image_size // vision_cfg.patch_size
        self.ratio = pooler_ratio

        self.conv_pool = nn.Conv2d(config.mm_hidden_size, config.hidden_size, kernel_size=self.ratio, stride=self.ratio)

        self.proj = nn.Sequential(
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )

    def forward(self, x, *args, **kwargs):
        # POOLER-FIX: 2025-07-29 - Handle multi-view LaViDa token format properly
        # ISSUE: LaViDa multi-view produces [5, 729, 1152] but pooler expects [batch, tokens, features]
        # SOLUTION: Process each view separately, then stack results
        # LAVIDA IMPACT: Enables proper baseline pooling per LaViDa paper specification
        # RESEARCH IMPACT: Provides correct baseline token count (5 × 196 = 980 tokens)
        
        original_shape = x.shape
        
        # Handle multi-view input: [5, 729, 1152] or single-view: [1, 729, 1152]
        if len(x.shape) == 3 and x.shape[1] == 729:  # LaViDa format: [views, tokens_per_view, features]
            num_views = x.shape[0]
            tokens_per_view = x.shape[1]  # Should be 729 (27×27)
            feature_dim = x.shape[2]
            
            height = width = self.hw  # Should be 27 for SigLIP 384×384
            
            # Verify token count matches expected grid
            if height * width != tokens_per_view:
                print(f"⚠️ POOLER-WARNING: Expected {height}×{width}={height*width} tokens, got {tokens_per_view}")
                # Fallback: assume square grid
                height = width = int(tokens_per_view ** 0.5)
                print(f"   Using fallback: {height}×{width} grid")
            
            pooled_views = []
            for view_idx in range(num_views):
                # Extract single view: [tokens_per_view, features]
                view_tokens = x[view_idx]  # [729, 1152]
                
                # Reshape to spatial grid: [1, height, width, features] → [1, features, height, width]
                view_spatial = view_tokens.view(1, height, width, feature_dim).permute(0, 3, 1, 2)
                
                # Apply 2×2 conv pooling: [1, features, height, width] → [1, features, height/2, width/2]
                pooled_spatial = self.conv_pool(view_spatial)
                
                # Flatten back to token sequence: [1, features, h/2, w/2] → [1, h/2*w/2, features]
                pooled_tokens = pooled_spatial.flatten(2).transpose(1, 2)  # [1, 196, features]
                
                pooled_views.append(pooled_tokens.squeeze(0))  # [196, features]
            
            # Stack pooled views: [num_views, 196, features]
            x = torch.stack(pooled_views, dim=0)
            
            print(f"POOLER-DEBUG: Multi-view pooling {original_shape} → {x.shape}")
            print(f"   Per view: {tokens_per_view} → {x.shape[1]} tokens ({height}×{width} → {int(height/self.ratio)}×{int(width/self.ratio)})")
            
        else:
            # Standard single-batch processing: [batch, tokens, features]
            height = width = self.hw
            if height * width != x.shape[1]:
                print(f"⚠️ POOLER-WARNING: Expected {height*width} tokens, got {x.shape[1]}")
                height = width = int(x.shape[1] ** 0.5)
            
            x = x.view(x.shape[0], height, width, -1).permute(0, 3, 1, 2)
            x = self.conv_pool(x)
            x = x.flatten(2).transpose(1, 2)
        
        # Apply final projection
        x = self.proj(x)
        return x

    @property
    def config(self):
        return {"mm_projector_type": "pooler"}
