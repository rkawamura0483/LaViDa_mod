#!/usr/bin/env python3
"""
SHIRG Token Dropout Implementation
Token-level dropout for stabilizing LoRA training on high-resolution tokens

Based on PatchDropout technique, randomly zeros out tokens during training
to improve robustness and prevent overfitting to specific token positions.

Author: Research Implementation
Date: 2025-07-30
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
import numpy as np


class ShirgTokenDropout(nn.Module):
    """
    Token dropout for SHIRG training stabilization
    
    Randomly drops tokens during training to:
    1. Prevent overfitting to specific high-res token positions
    2. Improve robustness to missing tokens
    3. Stabilize LoRA adaptation
    """
    
    def __init__(
        self,
        dropout_rate: float = 0.1,
        min_tokens_to_keep: int = 256,
        structured_dropout: bool = True,
        spatial_consistency: bool = True,
    ):
        """
        Initialize token dropout module
        
        Args:
            dropout_rate: Fraction of tokens to drop (0.1 = 10%)
            min_tokens_to_keep: Minimum number of tokens to keep (safety)
            structured_dropout: Use structured dropout patterns
            spatial_consistency: Maintain spatial coherence when dropping
        """
        super().__init__()
        self.dropout_rate = dropout_rate
        self.min_tokens_to_keep = min_tokens_to_keep
        self.structured_dropout = structured_dropout
        self.spatial_consistency = spatial_consistency
        
    def forward(
        self,
        tokens: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        force_keep_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply token dropout
        
        Args:
            tokens: Input tokens [batch, num_tokens, hidden_dim]
            token_type_ids: Optional token types (0=global, 1=foveal)
            force_keep_indices: Indices of tokens to always keep
            
        Returns:
            dropped_tokens: Tokens with dropout applied
            dropout_mask: Binary mask indicating kept tokens (1=kept, 0=dropped)
        """
        if not self.training or self.dropout_rate <= 0:
            # No dropout during inference or if rate is 0
            return tokens, torch.ones(tokens.shape[:2], device=tokens.device, dtype=torch.bool)
        
        batch_size, num_tokens, hidden_dim = tokens.shape
        device = tokens.device
        
        # Calculate number of tokens to keep
        num_to_keep = max(
            int(num_tokens * (1 - self.dropout_rate)),
            self.min_tokens_to_keep
        )
        num_to_keep = min(num_to_keep, num_tokens)  # Can't keep more than we have
        
        if self.structured_dropout and token_type_ids is not None:
            # Structured dropout: different rates for global vs foveal tokens
            dropout_mask = self._structured_dropout(
                batch_size, num_tokens, num_to_keep,
                token_type_ids, force_keep_indices, device
            )
        elif self.spatial_consistency:
            # Spatial dropout: drop spatially coherent regions
            dropout_mask = self._spatial_dropout(
                batch_size, num_tokens, num_to_keep,
                force_keep_indices, device
            )
        else:
            # Random dropout
            dropout_mask = self._random_dropout(
                batch_size, num_tokens, num_to_keep,
                force_keep_indices, device
            )
        
        # Apply dropout with proper scaling
        scale = 1.0 / (1.0 - self.dropout_rate)
        dropped_tokens = tokens * dropout_mask.unsqueeze(-1).float() * scale
        
        return dropped_tokens, dropout_mask
    
    def _random_dropout(
        self,
        batch_size: int,
        num_tokens: int,
        num_to_keep: int,
        force_keep_indices: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Random token dropout"""
        dropout_mask = torch.zeros(batch_size, num_tokens, device=device, dtype=torch.bool)
        
        for b in range(batch_size):
            # Random permutation
            perm = torch.randperm(num_tokens, device=device)
            
            # Keep top num_to_keep tokens
            keep_indices = perm[:num_to_keep]
            
            # Force keep certain indices if specified
            if force_keep_indices is not None and force_keep_indices[b] is not None:
                keep_indices = torch.unique(
                    torch.cat([keep_indices, force_keep_indices[b]])
                )
            
            dropout_mask[b, keep_indices] = True
        
        return dropout_mask
    
    def _structured_dropout(
        self,
        batch_size: int,
        num_tokens: int,
        num_to_keep: int,
        token_type_ids: torch.Tensor,
        force_keep_indices: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Structured dropout based on token types
        Preserves more global tokens (low-res context) than foveal tokens
        """
        dropout_mask = torch.zeros(batch_size, num_tokens, device=device, dtype=torch.bool)
        
        for b in range(batch_size):
            # Separate global and foveal tokens
            global_mask = token_type_ids[b] == 0
            foveal_mask = token_type_ids[b] == 1
            
            num_global = global_mask.sum().item()
            num_foveal = foveal_mask.sum().item()
            
            # Different dropout rates for different token types
            # Keep 95% of global tokens, adjust foveal to meet total target
            global_keep_rate = 0.95
            num_global_keep = min(int(num_global * global_keep_rate), num_global)
            num_foveal_keep = max(num_to_keep - num_global_keep, 0)
            num_foveal_keep = min(num_foveal_keep, num_foveal)
            
            # Select which tokens to keep
            if num_global > 0:
                global_indices = torch.where(global_mask)[0]
                global_perm = torch.randperm(num_global, device=device)
                keep_global = global_indices[global_perm[:num_global_keep]]
                dropout_mask[b, keep_global] = True
            
            if num_foveal > 0 and num_foveal_keep > 0:
                foveal_indices = torch.where(foveal_mask)[0]
                foveal_perm = torch.randperm(num_foveal, device=device)
                keep_foveal = foveal_indices[foveal_perm[:num_foveal_keep]]
                dropout_mask[b, keep_foveal] = True
            
            # Force keep certain indices
            if force_keep_indices is not None and force_keep_indices[b] is not None:
                dropout_mask[b, force_keep_indices[b]] = True
        
        return dropout_mask
    
    def _spatial_dropout(
        self,
        batch_size: int,
        num_tokens: int,
        num_to_keep: int,
        force_keep_indices: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Spatial dropout that maintains spatial coherence
        Drops contiguous regions rather than scattered tokens
        """
        dropout_mask = torch.zeros(batch_size, num_tokens, device=device, dtype=torch.bool)
        
        # Assume tokens are arranged spatially (need to know grid size)
        # For SHIRG with 980 tokens: 256 global + 724 foveal
        # Foveal tokens from 448² image = 32×32 = 1024 patches, keep 724
        
        for b in range(batch_size):
            if num_tokens == 980:  # SHIRG-specific layout
                # First 256 are global tokens - keep most of them
                num_global = 256
                num_global_keep = min(int(num_global * 0.95), num_global)
                global_perm = torch.randperm(num_global, device=device)
                dropout_mask[b, global_perm[:num_global_keep]] = True
                
                # Remaining are foveal tokens - spatial dropout
                foveal_start = 256
                foveal_tokens = num_tokens - foveal_start
                
                # Create spatial grid for foveal tokens
                grid_size = int(np.sqrt(foveal_tokens * 1.414))  # Approximate grid
                
                # Drop square regions
                region_size = 4  # Drop 4x4 regions
                num_regions = (grid_size // region_size) ** 2
                num_regions_to_keep = int(num_regions * (1 - self.dropout_rate))
                
                # Random region selection
                region_perm = torch.randperm(num_regions, device=device)
                keep_regions = region_perm[:num_regions_to_keep]
                
                # Convert regions back to token indices
                for region_idx in keep_regions:
                    row = (region_idx // (grid_size // region_size)) * region_size
                    col = (region_idx % (grid_size // region_size)) * region_size
                    
                    for r in range(region_size):
                        for c in range(region_size):
                            if row + r < grid_size and col + c < grid_size:
                                token_idx = (row + r) * grid_size + (col + c)
                                if token_idx < foveal_tokens:
                                    dropout_mask[b, foveal_start + token_idx] = True
            else:
                # Fallback to random dropout for unknown layouts
                dropout_mask[b] = self._random_dropout(
                    1, num_tokens, num_to_keep, 
                    force_keep_indices[b:b+1] if force_keep_indices is not None else None,
                    device
                )[0]
            
            # Force keep certain indices
            if force_keep_indices is not None and force_keep_indices[b] is not None:
                dropout_mask[b, force_keep_indices[b]] = True
        
        return dropout_mask


class ShirgDropoutScheduler:
    """
    Dropout rate scheduler for progressive training
    Gradually reduces dropout as training progresses
    """
    
    def __init__(
        self,
        initial_dropout: float = 0.1,
        final_dropout: float = 0.0,
        warmup_steps: int = 500,
        total_steps: int = 10000,
        schedule_type: str = "linear",
    ):
        self.initial_dropout = initial_dropout
        self.final_dropout = final_dropout
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.schedule_type = schedule_type
        
    def get_dropout_rate(self, current_step: int) -> float:
        """Get dropout rate for current training step"""
        if current_step < self.warmup_steps:
            # Linear warmup
            return self.initial_dropout * (current_step / self.warmup_steps)
        
        # Calculate progress after warmup
        progress = (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        
        if self.schedule_type == "linear":
            # Linear decay
            rate = self.initial_dropout - (self.initial_dropout - self.final_dropout) * progress
        elif self.schedule_type == "cosine":
            # Cosine decay
            rate = self.final_dropout + (self.initial_dropout - self.final_dropout) * (
                0.5 * (1 + np.cos(np.pi * progress))
            )
        elif self.schedule_type == "exponential":
            # Exponential decay
            decay_rate = -np.log((self.final_dropout + 1e-8) / self.initial_dropout)
            rate = self.initial_dropout * np.exp(-decay_rate * progress)
        else:
            rate = self.initial_dropout
        
        return max(self.final_dropout, rate)


def create_dropout_wrapper(model: nn.Module, config: dict) -> nn.Module:
    """
    Wrap model with token dropout for training
    
    Args:
        model: Base model
        config: Dropout configuration
        
    Returns:
        Model with dropout wrapper
    """
    
    class ModelWithDropout(nn.Module):
        def __init__(self, base_model, dropout_module):
            super().__init__()
            self.base_model = base_model
            self.dropout = dropout_module
            
        def forward(self, *args, **kwargs):
            # Extract image features if present
            if hasattr(self.base_model, 'extract_image_features'):
                image_features = kwargs.get('image_features')
                if image_features is not None:
                    # Apply dropout to image features
                    image_features, mask = self.dropout(image_features)
                    kwargs['image_features'] = image_features
                    kwargs['feature_mask'] = mask
            
            return self.base_model(*args, **kwargs)
    
    # Create dropout module
    dropout = ShirgTokenDropout(
        dropout_rate=config.get('dropout_rate', 0.1),
        min_tokens_to_keep=config.get('min_tokens_to_keep', 256),
        structured_dropout=config.get('structured_dropout', True),
        spatial_consistency=config.get('spatial_consistency', True),
    )
    
    return ModelWithDropout(model, dropout)


# Example usage and testing
if __name__ == "__main__":
    # Test token dropout
    print("🧪 Testing SHIRG Token Dropout")
    
    # Create dummy tokens
    batch_size = 2
    num_tokens = 980  # SHIRG token count
    hidden_dim = 1152
    
    tokens = torch.randn(batch_size, num_tokens, hidden_dim)
    
    # Create token type IDs (first 256 are global, rest are foveal)
    token_type_ids = torch.zeros(batch_size, num_tokens, dtype=torch.long)
    token_type_ids[:, 256:] = 1  # Mark foveal tokens
    
    # Test different dropout modes
    print("\n1. Random Dropout (10%)")
    dropout_random = ShirgTokenDropout(dropout_rate=0.1, structured_dropout=False)
    dropout_random.train()
    dropped_tokens, mask = dropout_random(tokens)
    kept_ratio = mask.float().mean()
    print(f"   Kept ratio: {kept_ratio:.3f}")
    print(f"   Output shape: {dropped_tokens.shape}")
    
    print("\n2. Structured Dropout (10%)")
    dropout_structured = ShirgTokenDropout(dropout_rate=0.1, structured_dropout=True)
    dropout_structured.train()
    dropped_tokens, mask = dropout_structured(tokens, token_type_ids)
    global_kept = mask[:, :256].float().mean()
    foveal_kept = mask[:, 256:].float().mean()
    print(f"   Global kept ratio: {global_kept:.3f}")
    print(f"   Foveal kept ratio: {foveal_kept:.3f}")
    
    print("\n3. Spatial Dropout (10%)")
    dropout_spatial = ShirgTokenDropout(dropout_rate=0.1, spatial_consistency=True)
    dropout_spatial.train()
    dropped_tokens, mask = dropout_spatial(tokens)
    print(f"   Kept ratio: {mask.float().mean():.3f}")
    
    print("\n4. Dropout Scheduler")
    scheduler = ShirgDropoutScheduler(
        initial_dropout=0.1,
        final_dropout=0.0,
        warmup_steps=100,
        total_steps=1000,
        schedule_type="cosine"
    )
    
    test_steps = [0, 50, 100, 250, 500, 750, 1000]
    for step in test_steps:
        rate = scheduler.get_dropout_rate(step)
        print(f"   Step {step}: dropout rate = {rate:.3f}")
    
    print("\n✅ Token dropout tests completed")