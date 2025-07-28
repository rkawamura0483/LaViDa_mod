"""
SHIRG LoRA Setup Module

This module provides utilities for setting up LoRA training on SHIRG-specified layers.
Implements the LoRA configuration from Section 3.3.3 of the SHIRG research proposal.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer implementation
    
    Implements LoRA as specified in SHIRG research proposal:
    - Rank-64 for projector and SigLIP attention layers
    - Rank-8 for coordinate embedding layer
    """
    def __init__(self, original_layer: nn.Linear, rank: int = 64, alpha: int = 128, dropout: float = 0.1):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False
            
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.randn(rank, original_layer.in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(original_layer.out_features, rank))
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor
        self.scaling = alpha / rank
        
    def forward(self, x):
        # Original layer output
        original_output = self.original_layer(x)
        
        # LoRA adaptation
        lora_output = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        
        return original_output + lora_output


def add_lora_to_linear_layer(layer: nn.Linear, rank: int = 64, alpha: int = 128) -> LoRALayer:
    """
    Replace a linear layer with LoRA-enabled version
    
    Args:
        layer: Original nn.Linear layer
        rank: LoRA rank
        alpha: LoRA alpha parameter
        
    Returns:
        LoRA-enabled layer
    """
    return LoRALayer(layer, rank=rank, alpha=alpha)


def setup_shirg_lora_modules(vision_tower, shirg_config):
    """
    Setup LoRA modules on SHIRG-specified layers
    
    Implements Section 3.3.3 LoRA Target Modules from research proposal:
    - mm_projector.fc1, mm_projector.fc2 (rank-64)
    - blocks.0-3.attn.qkv (rank-64) 
    - coord_linear (rank-8)
    
    Args:
        vision_tower: SigLipVisionTower instance
        shirg_config: SHIRG configuration object
        
    Returns:
        Dict of LoRA-enabled modules
    """
    lora_modules = {}
    
    logger.info("Setting up SHIRG LoRA modules...")
    
    # 1. SigLIP attention layers (blocks 0-3)
    if hasattr(vision_tower, 'vision_tower') and hasattr(vision_tower.vision_tower.vision_model, 'encoder'):
        encoder_layers = vision_tower.vision_tower.vision_model.encoder.layers
        
        for i in range(min(4, len(encoder_layers))):  # blocks 0-3
            layer = encoder_layers[i]
            if hasattr(layer, 'self_attn'):
                # Add LoRA to Q, K, V projections
                if hasattr(layer.self_attn, 'q_proj'):
                    original_q = layer.self_attn.q_proj
                    lora_q = add_lora_to_linear_layer(original_q, rank=shirg_config.lora_rank_siglip)
                    layer.self_attn.q_proj = lora_q
                    lora_modules[f'blocks.{i}.attn.q_proj'] = lora_q
                    
                if hasattr(layer.self_attn, 'k_proj'):
                    original_k = layer.self_attn.k_proj
                    lora_k = add_lora_to_linear_layer(original_k, rank=shirg_config.lora_rank_siglip)
                    layer.self_attn.k_proj = lora_k
                    lora_modules[f'blocks.{i}.attn.k_proj'] = lora_k
                    
                if hasattr(layer.self_attn, 'v_proj'):
                    original_v = layer.self_attn.v_proj
                    lora_v = add_lora_to_linear_layer(original_v, rank=shirg_config.lora_rank_siglip)
                    layer.self_attn.v_proj = lora_v
                    lora_modules[f'blocks.{i}.attn.v_proj'] = lora_v
                    
                logger.info(f"✅ Added LoRA to SigLIP block {i} attention layers")
    
    # 2. Coordinate embedding layer (rank-8)
    if hasattr(vision_tower, 'coord_linear'):
        original_coord = vision_tower.coord_linear
        lora_coord = add_lora_to_linear_layer(original_coord, rank=shirg_config.lora_rank_coordinate)
        vision_tower.coord_linear = lora_coord
        lora_modules['coord_linear'] = lora_coord
        logger.info("✅ Added LoRA to coordinate embedding layer")
    
    # 3. MM Projector layers (if accessible from vision tower)
    # Note: These are typically in the main LaViDa model, not the vision tower
    # This would be handled during main model LoRA setup
    
    logger.info(f"SHIRG LoRA setup complete: {len(lora_modules)} modules enabled")
    return lora_modules


def get_shirg_lora_parameters(lora_modules: Dict) -> List[nn.Parameter]:
    """
    Get all trainable LoRA parameters for optimizer
    
    Args:
        lora_modules: Dictionary of LoRA modules
        
    Returns:
        List of trainable parameters
    """
    lora_params = []
    for module_name, lora_module in lora_modules.items():
        if isinstance(lora_module, LoRALayer):
            lora_params.extend([lora_module.lora_A, lora_module.lora_B])
            logger.info(f"Added {module_name} LoRA parameters: A{lora_module.lora_A.shape}, B{lora_module.lora_B.shape}")
    
    return lora_params


def validate_lora_setup(vision_tower, lora_modules: Dict) -> bool:
    """
    Validate that LoRA setup is correct
    
    Args:
        vision_tower: Vision tower with LoRA modules
        lora_modules: Dictionary of LoRA modules
        
    Returns:
        bool: True if validation passes
    """
    validation_errors = []
    
    # Check that LoRA modules are present
    expected_modules = [
        'blocks.0.attn.q_proj', 'blocks.0.attn.k_proj', 'blocks.0.attn.v_proj',
        'blocks.1.attn.q_proj', 'blocks.1.attn.k_proj', 'blocks.1.attn.v_proj',
        'blocks.2.attn.q_proj', 'blocks.2.attn.k_proj', 'blocks.2.attn.v_proj',
        'blocks.3.attn.q_proj', 'blocks.3.attn.k_proj', 'blocks.3.attn.v_proj',
        'coord_linear'
    ]
    
    for module_name in expected_modules:
        if module_name not in lora_modules:
            validation_errors.append(f"Missing LoRA module: {module_name}")
    
    # Check gradient flow
    for module_name, lora_module in lora_modules.items():
        if isinstance(lora_module, LoRALayer):
            if not lora_module.lora_A.requires_grad:
                validation_errors.append(f"{module_name} LoRA_A gradients disabled")
            if not lora_module.lora_B.requires_grad:
                validation_errors.append(f"{module_name} LoRA_B gradients disabled")
    
    # Check parameter counts
    total_lora_params = sum(p.numel() for p in get_shirg_lora_parameters(lora_modules))
    expected_params = (
        4 * 3 * (64 * 1152 + 1152 * 64) +  # SigLIP attention LoRA (4 blocks × 3 projections)
        (8 * 4 + 128 * 8)  # Coordinate embedding LoRA
    )
    
    if abs(total_lora_params - expected_params) > expected_params * 0.1:  # 10% tolerance
        validation_errors.append(f"Parameter count mismatch: {total_lora_params} vs expected ~{expected_params}")
    
    if validation_errors:
        logger.error("LoRA validation failed:")
        for error in validation_errors:
            logger.error(f"  ❌ {error}")
        return False
    else:
        logger.info("✅ LoRA validation passed")
        logger.info(f"Total LoRA parameters: {total_lora_params:,} (~{total_lora_params/1e6:.1f}M)")
        return True


def compute_lora_parameter_efficiency(lora_modules: Dict, total_model_params: int) -> float:
    """
    Compute LoRA parameter efficiency
    
    Args:
        lora_modules: Dictionary of LoRA modules
        total_model_params: Total model parameters
        
    Returns:
        Percentage of total parameters that are LoRA-trainable
    """
    lora_params = sum(p.numel() for p in get_shirg_lora_parameters(lora_modules))
    efficiency = (lora_params / total_model_params) * 100
    
    logger.info(f"LoRA Parameter Efficiency: {lora_params:,} / {total_model_params:,} = {efficiency:.2f}%")
    return efficiency


if __name__ == "__main__":
    # Example usage for testing
    print("SHIRG LoRA Setup Module")
    print("This module provides LoRA setup utilities for SHIRG training.")
    print("Import and use setup_shirg_lora_modules() in your training script.")