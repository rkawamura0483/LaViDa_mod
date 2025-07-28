"""
SHIRG Integration Configuration

This module provides configuration utilities for integrating SHIRG with LaViDa.
It handles enabling/disabling SHIRG, LoRA setup, and integration parameters.
"""

import torch
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class SHIRGConfig:
    """
    SHIRG Configuration for LaViDa Integration
    
    Implements Section 4.1 configuration from SHIRG research proposal.
    """
    # Core SHIRG parameters
    enable_shirg: bool = True
    high_res_size: int = 672  # Input image resolution for SHIRG
    hi_detail_tokens: int = 2304  # 48×48 patches from 672×672
    lo_res_scaffold_tokens: int = 144  # 12×12 scaffold from 4×4 pooling
    selected_tokens: int = 768  # Selected hi-detail tokens
    total_output_tokens: int = 912  # 768 + 144 = 912 total
    
    # Distance-aware scoring weights (from research proposal)
    similarity_weight: float = 0.7
    neighbor_distance_weight: float = 0.2
    center_distance_weight: float = 0.1
    
    # Token merging parameters
    merging_epsilon: float = 0.05
    enable_neighbor_merging: bool = True
    
    # LoRA training configuration
    enable_lora: bool = True
    coordinate_embedding_dim: int = 128
    lora_rank_projector: int = 64
    lora_rank_siglip: int = 64
    lora_rank_coordinate: int = 8
    lora_alpha: int = 128
    
    # LoRA target modules (as specified in research proposal)
    lora_targets_projector: List[str] = None
    lora_targets_siglip: List[str] = None
    lora_targets_coordinate: List[str] = None
    
    # Performance optimization
    max_selection_time_ms: float = 30.0  # Target <30ms for token selection
    max_total_time_ms: float = 50.0  # Target <50ms for total latency
    
    # Cache compatibility
    ensure_static_tokens: bool = True  # Maintain prefix KV-cache compatibility
    
    def __post_init__(self):
        """Initialize default LoRA targets if not specified"""
        if self.lora_targets_projector is None:
            self.lora_targets_projector = ["mm_projector.fc1", "mm_projector.fc2"]
        
        if self.lora_targets_siglip is None:
            self.lora_targets_siglip = [
                "blocks.0.attn.qkv", "blocks.1.attn.qkv", 
                "blocks.2.attn.qkv", "blocks.3.attn.qkv"
            ]
        
        if self.lora_targets_coordinate is None:
            self.lora_targets_coordinate = ["coord_linear"]
        
        # Validate configuration
        assert self.selected_tokens + self.lo_res_scaffold_tokens == self.total_output_tokens, \
            f"Token counts don't match: {self.selected_tokens} + {self.lo_res_scaffold_tokens} != {self.total_output_tokens}"
        
        assert self.similarity_weight + self.neighbor_distance_weight + self.center_distance_weight <= 1.0, \
            "Distance-aware scoring weights should sum to ≤ 1.0"


def create_shirg_vision_config(**kwargs):
    """
    Create vision tower configuration with SHIRG integration
    
    Args:
        **kwargs: Additional configuration parameters
        
    Returns:
        vision_config: Configuration object with SHIRG settings
    """
    class VisionConfig:
        def __init__(self, **config_kwargs):
            # Default LaViDa settings
            self.unfreeze_mm_vision_tower = False
            self.mm_tunable_parts = []
            
            # SHIRG integration settings
            self.enable_shirg = config_kwargs.get('enable_shirg', True)
            self.shirg_config = SHIRGConfig(**{k: v for k, v in config_kwargs.items() 
                                            if k.startswith('shirg_') or k in SHIRGConfig.__dataclass_fields__})
            
            # Apply additional kwargs
            for key, value in config_kwargs.items():
                if not key.startswith('shirg_') and key not in SHIRGConfig.__dataclass_fields__:
                    setattr(self, key, value)
    
    return VisionConfig(**kwargs)


def setup_shirg_lora_config(shirg_config: SHIRGConfig):
    """
    Setup LoRA configuration for SHIRG training
    
    Args:
        shirg_config: SHIRG configuration object
        
    Returns:
        lora_config: LoRA configuration dictionary
    """
    lora_config = {
        "task_type": "FEATURE_EXTRACTION",
        "inference_mode": False,
        "r": shirg_config.lora_rank_projector,
        "lora_alpha": shirg_config.lora_alpha,
        "lora_dropout": 0.1,
        "target_modules": shirg_config.lora_targets_projector + shirg_config.lora_targets_siglip,
        "modules_to_save": shirg_config.lora_targets_coordinate,
    }
    
    return lora_config


def validate_shirg_requirements():
    """
    Validate that SHIRG requirements are met for training
    
    Returns:
        bool: True if all requirements are satisfied
        List[str]: List of validation errors (empty if valid)
    """
    errors = []
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        errors.append("CUDA not available - SHIRG requires GPU")
    
    # Check memory requirements (approximate)
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if total_memory < 35.0:  # Conservative estimate for 40GB target
            errors.append(f"Insufficient GPU memory: {total_memory:.1f}GB < 35GB minimum")
    
    # Check PyTorch version compatibility
    torch_version = torch.__version__
    if not torch_version.startswith(('1.13', '2.0', '2.1', '2.2', '2.3')):
        errors.append(f"PyTorch version {torch_version} may not be compatible")
    
    return len(errors) == 0, errors


def print_shirg_config(shirg_config: SHIRGConfig):
    """
    Print SHIRG configuration summary
    
    Args:
        shirg_config: SHIRG configuration to display
    """
    print("=" * 60)
    print("SHIRG: Static Hierarchical Relevance Gate Configuration")
    print("=" * 60)
    print(f"High-resolution processing: {shirg_config.high_res_size}×{shirg_config.high_res_size}")
    print(f"Token architecture:")
    print(f"  • Hi-detail tokens: {shirg_config.hi_detail_tokens} (from {shirg_config.high_res_size}×{shirg_config.high_res_size})")
    print(f"  • Selected tokens: {shirg_config.selected_tokens}")
    print(f"  • Lo-res scaffold: {shirg_config.lo_res_scaffold_tokens}")
    print(f"  • Total output: {shirg_config.total_output_tokens}")
    print(f"Distance-aware scoring:")
    print(f"  • Similarity weight: {shirg_config.similarity_weight}")
    print(f"  • Neighbor distance weight: {shirg_config.neighbor_distance_weight}")
    print(f"  • Center distance weight: {shirg_config.center_distance_weight}")
    print(f"LoRA configuration:")
    print(f"  • Projector rank: {shirg_config.lora_rank_projector}")
    print(f"  • SigLIP rank: {shirg_config.lora_rank_siglip}")
    print(f"  • Coordinate rank: {shirg_config.lora_rank_coordinate}")
    print(f"Performance targets:")
    print(f"  • Selection time: <{shirg_config.max_selection_time_ms}ms")
    print(f"  • Total latency: <{shirg_config.max_total_time_ms}ms")
    print("=" * 60)


if __name__ == "__main__":
    # Example usage
    config = SHIRGConfig()
    print_shirg_config(config)
    
    is_valid, errors = validate_shirg_requirements()
    if is_valid:
        print("✅ SHIRG requirements validated successfully")
    else:
        print("❌ SHIRG validation errors:")
        for error in errors:
            print(f"  • {error}")