#!/usr/bin/env python3
"""
LaViDa-SHIRG Integration
Integrates SHIRG token selection into LaViDa's vision processing pipeline

This module provides a drop-in replacement for LaViDa's standard pooling
mechanism, enabling high-resolution OCR/VQA capabilities while maintaining
the prefix-KV cache efficiency that makes LaViDa fast.

Author: Research Implementation  
Date: 2025-01-26
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import time
import warnings
import numpy as np

# Colab environment detection - Fixed method
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Add current directory to path for imports (LaViDa_mod is the repo root)
sys.path.append('./')

try:
    # Try importing without deepspeed first
    import llava
    from llava.model.builder import load_pretrained_model
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
    LAVIDA_AVAILABLE = True
except ImportError as e:
    # SHIRG-FIX: [2025-07-30] - Handle missing deepspeed gracefully
    # ISSUE: LaViDa imports fail when deepspeed is not installed
    # SOLUTION: Make deepspeed optional for SHIRG integration testing
    # LAVIDA IMPACT: LaViDa functionality limited without deepspeed
    # SHIRG IMPACT: SHIRG can still be tested without full LaViDa training
    print(f"⚠️ LaViDa imports not available: {e}")
    print("   This is expected if deepspeed is not installed.")
    print("   For full LaViDa functionality, install deepspeed.")
    LAVIDA_AVAILABLE = False

# PrefixKV cache compression integration
try:
    from prefixkv import PrefixKVWrapper
    PREFIXKV_AVAILABLE = True
    print("✅ PrefixKV available for cache compression")
except ImportError:
    PREFIXKV_AVAILABLE = False
    print("⚠️ PrefixKV not available - install with: pip install prefixkv")


class SHIRGCacheManager:
    """
    Memory-efficient KV cache management for SHIRG tokens
    
    Memory-efficient KV cache management for SHIRG tokens
    
    SHIRG-FOVEA: PrefixKV integration for cache compression
    Reduces memory footprint with 16-bit compression
    """
    
    def __init__(self, enable_compression=True):
        self.enable_compression = enable_compression and PREFIXKV_AVAILABLE
        self.compression_ratio = 0.5  # 16-bit compression
        
    def wrap_model_with_cache_compression(self, model):
        """
        Wrap diffusion model with PrefixKV compression for SHIRG
        
        Args:
            model: LaViDa model instance
            
        Returns:
            wrapped_model: Model with cache compression if available
        """
        if self.enable_compression:
            try:
                wrapped_model = PrefixKVWrapper(model, compression_ratio=self.compression_ratio)
                print(f"✅ PrefixKV cache compression enabled (16-bit, ratio={self.compression_ratio})")
                return wrapped_model
            except Exception as e:
                print(f"⚠️ PrefixKV wrapping failed: {e}, using standard caching")
                return model
        else:
            if PREFIXKV_AVAILABLE:
                print("ℹ️ PrefixKV available but compression disabled")
            else:
                print("ℹ️ PrefixKV not available, using standard caching")
            return model
    
    def get_cache_memory_info(self, model):
        """Get cache memory information"""
        if hasattr(model, 'get_cache_memory_usage'):
            return model.get_cache_memory_usage()
        else:
            return {"status": "standard_cache", "compression": "none"}




class TextVisionAligner(nn.Module):
    """
    Text-vision dimension alignment for SHIRG
    Projects text embeddings to vision feature space for similarity computation
    """
    
    def __init__(self, text_dim=4096, vision_dim=1152):
        super().__init__()
        self.align_layer = nn.Linear(text_dim, vision_dim, bias=False)
        
        # Initialize with Xavier uniform for stable training
        nn.init.xavier_uniform_(self.align_layer.weight)
        
    def forward(self, text_embeddings):
        """
        Project text embeddings to vision space
        
        Args:
            text_embeddings: [B, seq_len, text_dim] text token embeddings
            
        Returns:
            aligned_embeddings: [B, seq_len, vision_dim] aligned embeddings
        """
        return self.align_layer(text_embeddings)

# IN_COLAB already defined above

class LaViDaSHIRGWrapper:
    """
    Wrapper around LaViDa model that integrates SHIRG token selection
    
    This class provides the same interface as the original LaViDa model
    but uses SHIRG for vision token selection instead of standard pooling.
    """
    
    def __init__(self, 
                 model_path: str = "KonstantinosKK/lavida-llada-v1.0-instruct-hf-transformers",
                 shirg_config: Optional[Dict[str, Any]] = None,
                 device_map: str = "auto",
                 torch_dtype = torch.bfloat16,
                 vision_kwargs: Optional[Dict[str, Any]] = None,
                 selection_method: str = "base",
                 selection_params: Optional[Dict[str, Any]] = None):
        """
        Initialize LaViDa with SHIRG integration
        
        Args:
            model_path: HuggingFace model path for LaViDa
            shirg_config: SHIRG configuration parameters
            device_map: Device mapping for model
            torch_dtype: Data type for model
            vision_kwargs: Additional vision model kwargs
            selection_method: Token selection method ('base', 'entropy', 'edge', 'full')
            selection_params: Method-specific parameters
            device_map: Device mapping for model loading
            torch_dtype: Torch data type for model
            vision_kwargs: Vision tower configuration
        """
        
        if not LAVIDA_AVAILABLE:
            raise ImportError("LaViDa not available. Please install dependencies first.")
        
        self.model_path = model_path
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        
        # Default SHIRG configuration optimized for LaViDa
        default_shirg_config = {
            'target_tokens': 980,       # Fixed for LaViDa compatibility (196 global + 784 foveal)
            'alpha': 0.3,               # Balance detail vs semantics (>0 enables SHIRG)
            'hierarchical_levels': 3,   # Spatial clustering depth
            'latency_budget_ms': 1000.0,
            'use_fast_clustering': True,
            'enable_caching': True,
            'debug': True,              # Enable for research
            'high_res_interpolation': True,  # Enable high-resolution token interpolation
            'target_grid_size': 55      # Target high-resolution grid (55x55 = 3025 tokens)
        }
        
        if shirg_config:
            default_shirg_config.update(shirg_config)
        self.shirg_config = default_shirg_config
        
        # Default vision configuration for LaViDa with 5-view anyres support
        default_vision_kwargs = {
            "mm_vision_tower": "google/siglip-so400m-patch14-384",  # SigLIP-SO400M
            "mm_resampler_type": None,
            "mm_projector_type": 'mlp2x_gelu', 
            "mm_hidden_size": 1152,           # SigLIP hidden size
            "use_mm_proj": True,
            "mm_pooler_ratio": 2,             # For baseline pooling when SHIRG disabled
            "mm_patch_merge_type": 'spatial_unpad',  # Enable high-res processing
            "image_aspect_ratio": "anyres",           # Enable any resolution processing
            "image_grid_pinpoints": "[(384, 384), (512, 512)]",  # Support both global and peripheral views
        }
        
        if vision_kwargs:
            default_vision_kwargs.update(vision_kwargs)
        self.vision_kwargs = default_vision_kwargs
        
        # Store selection method and parameters
        self.selection_method = selection_method
        self.selection_params = selection_params or {}
        
        # Initialize model components
        self.tokenizer = None
        self.model = None  
        self.image_processor = None
        self.context_len = None
        self.conv_template = None
        self.shirg_selector = None
        
        # SHIRG cache management
        self.cache_manager = SHIRGCacheManager(enable_compression=True)
        
        # Performance tracking
        self.generation_times = []
        self.shirg_times = []
        
        print(f"🚀 LaViDa-SHIRG wrapper initialized")
        print(f"   Model: {model_path}")
        print(f"   SHIRG config: {self.shirg_config}")
    
    def load_model(self):
        """Load LaViDa model with SHIRG integration"""
        print("🔄 Loading LaViDa model with SHIRG integration...")
        
        try:
            # FIX: 2025-07-26 - Use string format for torch_dtype to match LaViDa examples
            # ISSUE: torch_dtype object may not be handled correctly by LaViDa's loader
            # SOLUTION: Use string format 'bfloat16' as in official predict.py
            # RESEARCH IMPACT: Ensures proper dtype handling throughout LaViDa loading
            
            # SHIRG-FIX: 2025-07-30 - Handle device_map for testing vs production
            # ISSUE: device_map="auto" causes device mismatches in multi-GPU setup
            # SOLUTION: Allow disabling device_map for testing while keeping for production
            # LAVIDA IMPACT: Testing can use single GPU, production uses multi-GPU
            # SHIRG IMPACT: Fixes device mismatch errors during testing
            
            # SHIRG-FIX: 2025-07-30 - Disable device_map for LoRA training
            # ISSUE: device_map="auto" (model parallelism) breaks LoRA gradient flow
            # SOLUTION: Check environment variable to disable device_map for DDP training
            # LAVIDA IMPACT: Requires loading full model per GPU but enables gradient flow
            # SHIRG IMPACT: Fixes zero gradient issue in 8 GPU LoRA training
            
            # Check if we should disable device_map for LoRA training
            disable_device_map = os.environ.get('SHIRG_NO_DEVICE_MAP', '0') == '1'
            if disable_device_map:
                print("⚠️ SHIRG_NO_DEVICE_MAP=1: Disabling device_map for LoRA training")
                actual_device_map = None
            else:
                # Use device_map only if not explicitly disabled
                actual_device_map = self.device_map if self.device_map is not None else "auto"
            
            # Load base model with LaViDa-compatible parameters
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path=self.model_path,
                model_base=None,
                model_name="llava_llada",  # Critical: Use LaViDa model name
                device_map=actual_device_map,
                vision_kwargs=self.vision_kwargs,
                torch_dtype='bfloat16',  # Use string format like official examples
                attn_implementation="flash_attention_2"
            )
            
            # Apply critical configuration fixes (from implementation plan)
            self._apply_config_fixes()
            
            # SHIRG-FIX: 2025-07-30 - Enable SHIRG 2-view mode immediately after model loading
            # ISSUE: LaViDa creates 5 views by default, but SHIRG-Fovea expects 2 views
            # SOLUTION: Set shirg_3view_mode=True in model config right after loading
            # LAVIDA IMPACT: Changes image preprocessing from 5 views to 2 views when SHIRG is enabled
            # SHIRG IMPACT: Fixes "expects 2 views, got 5 views" error during training
            if self.shirg_config.get('alpha', 0) > 0:
                print("🔧 Configuring SHIRG 2-view mode...")
                self.model.config.enable_shirg = True
                self.model.config.shirg_3view_mode = True  # Enable 2-view mode
                print("   ✅ SHIRG 2-view mode enabled (shirg_3view_mode=True)")
            
            # FIX: 2025-07-26 - Apply LaViDa-specific model setup from official examples
            # ISSUE: Missing tie_weights() and proper dtype setup causes generation errors
            # SOLUTION: Follow exact setup sequence from LaViDa's predict.py
            # RESEARCH IMPACT: Enables proper LaViDa model execution without dtype errors
            
            # Critical LaViDa setup steps (from official predict.py + lmms_eval)
            self.model.eval()
            self.model.tie_weights()  # Essential for LaViDa masked diffusion
            self.model.to(self.torch_dtype)  # Ensure all model components are bfloat16
            
            # Additional setup from lmms_eval (lines 164-166)
            self.model.model.set_activation_checkpointing(None)
            self.model.requires_grad_(False)
            
            # SHIRG selector is already integrated into the vision tower
            # No separate selector needed - SHIRG is implemented in SigLipShirgExtensions
            
            
            # Text-vision aligner for SHIRG (if needed for future extensions)
            # Currently SHIRG-Fovea doesn't require text alignment as scoring happens in vision space
            
            # Setup conversation template
            self._setup_conversation_template()
            
            # Apply cache compression if enabled
            if self.shirg_config.get('enable_cache_compression', False):
                self.model = self.cache_manager.wrap_model_with_cache_compression(self.model)
            
            # Integrate SHIRG into model
            self._integrate_shirg()
            
            print("✅ LaViDa-SHIRG model loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to load LaViDa-SHIRG model: {e}")
            raise
    
    def _apply_config_fixes(self):
        """Apply critical configuration fixes from U-HiRID experience"""
        
        # Ensure model is in eval mode
        self.model.eval()
        
        # Fix attention implementation
        if getattr(self.model.config, 'attn_implementation', 'unknown') != 'flash_attention_2':
            self.model.config.attn_implementation = "flash_attention_2"
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'config'):
                self.model.model.config.attn_implementation = "flash_attention_2"
        
        # Fix mm_patch_merge_type for SigLIP compatibility  
        if getattr(self.model.config, 'mm_patch_merge_type', 'unknown') != 'flat':
            self.model.config.mm_patch_merge_type = 'flat'
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'config'):
                self.model.model.config.mm_patch_merge_type = 'flat'
        
        # Add SHIRG configuration to model config
        self.model.config.use_shirg = True
        self.model.config.enable_shirg = True  # Both flags needed for SHIRG
        self.model.config.shirg_target_tokens = self.shirg_config['target_tokens']
        self.model.config.shirg_alpha = self.shirg_config['alpha']
        self.model.config.shirg_selection_method = self.selection_method
        self.model.config.shirg_selection_params = self.selection_params
        
        print("✅ Applied LaViDa configuration fixes")
    
    def _ensure_device_consistency(self):
        """
        Ensure all SHIRG components have consistent device/dtype
        """
        # Get target device and dtype from the main model
        target_device = self.model.device if hasattr(self.model, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu'
        target_dtype = self.torch_dtype
        
        # Currently no additional components needed for SHIRG-Fovea
        # This method is kept for potential future extensions
    
    def _setup_conversation_template(self):
        """Setup LaViDa conversation template with fallback"""
        from llava.conversation import conv_templates
        
        # FIX: 2025-07-26 - Use LaViDa's official conversation template
        # ISSUE: Wrong conversation template causes generation format issues
        # SOLUTION: Use 'llada' template from LaViDa's predict.py
        # RESEARCH IMPACT: Ensures proper conversation format for LaViDa generation
        
        # FIX: 2025-07-26 - Work around llada template's LLAMA_3 tokenizer requirement
        # ISSUE: llada template uses LLAMA_3 style which requires special tokenizer permissions
        # SOLUTION: Use llada template but bypass the tokenizer requirement
        # RESEARCH IMPACT: Enables LaViDa generation without Llama 3 tokenizer permissions
        
        # FIX: 2025-07-26 - Use exact same approach as LaViDa evaluation code
        # ISSUE: llada template tokenizer loading causes permission errors
        # SOLUTION: Use copy.deepcopy() and avoid get_prompt() call as in lmms_eval
        # RESEARCH IMPACT: Enables LaViDa generation using official evaluation approach
        
        # Store the template name but don't create the conversation object yet
        # We'll create it fresh for each generation call (like lmms_eval does)
        if "llada" in conv_templates:
            self.conv_template_name = "llada"
            print(f"✅ LaViDa template registered: {self.conv_template_name}")
        elif "plain" in conv_templates:
            self.conv_template_name = "plain"
            print(f"⚠️ Using fallback template: {self.conv_template_name}")
        else:
            self.conv_template_name = list(conv_templates.keys())[0]
            print(f"⚠️ Using fallback template: {self.conv_template_name}")
        
        # Don't create the conversation object here - do it in generate() method
        self.conv_template = None
            
        # Ensure tokenizer is properly set
        if self.tokenizer is None:
            print("⚠️ Tokenizer is None, conversation template may not work properly")
    
    def _integrate_shirg(self):
        """Integrate SHIRG into LaViDa's vision processing pipeline"""
        
        # FIX: 2025-07-27 - Fix SHIRG integration by patching correct method with correct signature
        # ISSUE: Previous patch had wrong method signature and integration point
        # SOLUTION: Patch encode_images with correct signature and integrate after vision tower
        # RESEARCH IMPACT: Enables actual SHIRG vs baseline comparison
        
        # Store SHIRG selector and wrapper reference
        self.model.shirg_selector = self.shirg_selector
        self.model.shirg_wrapper = self  # Allow patched method to access wrapper
        
        # Store original method for reference
        if hasattr(self.model, 'encode_images'):
            self.model._original_encode_images = self.model.encode_images
            
            # Only patch if we're actually using SHIRG (alpha > 0 enables SHIRG selection)
            shirg_enabled = (self.shirg_config.get('alpha', 0) > 0)
            # FIX: 2025-07-30 - Remove target_tokens check to enable SHIRG with 980 tokens
            # ISSUE: Condition required target_tokens != 980, but we need 980 for LaViDa compatibility
            # SOLUTION: Only check alpha > 0 to distinguish baseline (alpha=0) from SHIRG (alpha>0)
            # RESEARCH IMPACT: Enables SHIRG integration while maintaining LaViDa compatibility
            
            if shirg_enabled:
                # Patch encode_images to use SHIRG with correct signature
                def patched_encode_images(self, images):
                    """Patched encode_images that applies SHIRG-Fovea token selection"""
                    
                    print(f"🔄 SHIRG-Fovea patched encode_images called with {type(images)} images")
                    
                    # SHIRG-FOVEA-FIX: 2025-07-29 - Handle LaViDa's 5-view list format
                    # ISSUE: New methodology requires processing list of 5 views
                    # SOLUTION: Pass list directly to SHIRG forward_with_shirg
                    # RESEARCH IMPACT: Implements correct 5-view processing per methodology
                    
                    wrapper = getattr(self, 'shirg_wrapper', None)
                    vision_tower = self.get_model().get_vision_tower()
                    
                    if (wrapper is not None and 
                        hasattr(wrapper, '_current_question_tokens') and 
                        wrapper._current_question_tokens is not None and
                        wrapper.shirg_config.get('alpha', 0) > 0):
                        
                        try:
                            # SHIRG-Fovea: Process 5-view format (1 global + 4 peripheral)
                            if hasattr(vision_tower, 'forward_with_shirg'):
                                if wrapper.shirg_config.get('debug', False):
                                    if isinstance(images, list):
                                        print(f"🔍 Using SHIRG-Fovea processing with {len(images)} views")
                                    else:
                                        print(f"🔍 Using SHIRG-Fovea processing with tensor input")
                                
                                try:
                                    # SHIRG-Fovea method with optional text embeddings
                                    text_embeddings = wrapper._current_question_tokens
                                    if text_embeddings is not None:
                                        # Validate text embeddings shape and dtype
                                        if text_embeddings.dim() != 3:
                                            text_embeddings = text_embeddings.unsqueeze(0) if text_embeddings.dim() == 2 else text_embeddings
                                        # Ensure text embeddings are on correct device
                                        if isinstance(images, list) and len(images) > 0:
                                            ref_device = images[0].device if hasattr(images[0], 'device') else None
                                            ref_dtype = images[0].dtype if hasattr(images[0], 'dtype') else None
                                        else:
                                            ref_device = images.device if hasattr(images, 'device') else None
                                            ref_dtype = images.dtype if hasattr(images, 'dtype') else None
                                            
                                        if ref_device and text_embeddings.device != ref_device:
                                            text_embeddings = text_embeddings.to(device=ref_device, dtype=ref_dtype)
                                    
                                    # Call SHIRG-Fovea method with selection parameters
                                    selected_features = vision_tower.forward_with_shirg(
                                        images, 
                                        text_embeddings=text_embeddings,
                                        selection_method=wrapper.selection_method,
                                        selection_params=wrapper.selection_params
                                    )
                                    
                                    if wrapper.shirg_config.get('debug', False):
                                        print(f"✅ SHIRG-Fovea processing: {selected_features.shape}")
                                    
                                    # Apply projector to selected features
                                    image_features = self.get_model().mm_projector(selected_features)
                                    return image_features
                                    
                                except Exception as shirg_error:
                                    if wrapper.shirg_config.get('debug', False):
                                        print(f"❌ SHIRG forward_with_shirg failed: {shirg_error}")
                                        import traceback
                                        traceback.print_exc()
                                    
                                    # Fallback to baseline - process through standard vision tower
                                    if wrapper.shirg_config.get('debug', False):
                                        print("📉 Falling back to baseline LaViDa processing")
                                    # Use standard encode_images
                                    return self._original_encode_images(images)
                            else:
                                # No SHIRG available - use baseline
                                if wrapper.shirg_config.get('debug', False):
                                    print("📉 No SHIRG available, using baseline")
                                return self._original_encode_images(images)
                        except Exception as e:
                            if wrapper.shirg_config.get('debug', False):
                                print(f"⚠️ SHIRG selection failed: {e}")
                                import traceback
                                traceback.print_exc()
                            # Fallback to standard processing
                            return self._original_encode_images(images)
                    else:
                        # Baseline: use standard encode_images
                        return self._original_encode_images(images)
                
                # Bind method to model instance
                import types
                self.model.encode_images = types.MethodType(patched_encode_images, self.model)
                print(f"✅ SHIRG integration enabled - encode_images method patched (alpha={self.shirg_config.get('alpha', 0)})")
            else:
                print(f"✅ SHIRG integration disabled (baseline mode, alpha={self.shirg_config.get('alpha', 0)})")
                print(f"   - Using standard LaViDa encode_images (no token selection)")
        else:
            print("⚠️ Could not find encode_images method to patch")
    
    
    def _prepare_inputs_labels_for_multimodal_with_shirg(self, 
                                                        original_method,
                                                        input_ids, position_ids, attention_mask, past_key_values, 
                                                        labels, images, modalities=["image"], **kwargs):
        """
        DEPRECATED: This method is no longer used to avoid integration complexity.
        Always falls back to original LaViDa method.
        """
        # Simple fallback - no integration to avoid dtype issues
        return original_method(input_ids, position_ids, attention_mask,
                             past_key_values, labels, images, modalities, **kwargs)
    
    def _apply_shirg_to_vision_features(self, vision_features, text_context):
        """Apply SHIRG selection to vision features"""
        
        if self.shirg_selector is None:
            print("⚠️ SHIRG selector not initialized, using fallback")
            return vision_features
        
        try:
            start_time = time.time()
            
            # FIX: 2025-07-26 - Ensure dtype consistency before SHIRG processing
            # ISSUE: Vision features (bfloat16) and text context (float32) have mismatched dtypes
            # SOLUTION: Convert text context to match vision features dtype
            # RESEARCH IMPACT: Prevents dtype mismatch errors in SHIRG computation
            
            # Ensure both inputs have same dtype as vision features
            target_dtype = vision_features.dtype
            target_device = vision_features.device
            
            if text_context is not None:
                text_context = text_context.to(dtype=target_dtype, device=target_device)
            else:
                # Create dummy text context if None
                batch_size = vision_features.shape[0]
                embed_dim = vision_features.shape[-1]
                text_context = torch.zeros((batch_size, 1, embed_dim), 
                                         dtype=target_dtype, device=target_device)
            
            # Apply SHIRG selection
            selected_features = self.shirg_selector(
                image_tokens=vision_features,
                text_embeddings=text_context,
                image_sizes=getattr(self.model, '_current_image_sizes', None)
            )
            
            shirg_time = time.time() - start_time
            self.shirg_times.append(shirg_time * 1000)  # Convert to ms
            
            # Validate output shape
            expected_shape = (vision_features.shape[0], 
                            self.shirg_config['target_tokens'], 
                            vision_features.shape[-1])
            
            if selected_features.shape != expected_shape:
                print(f"⚠️ SHIRG output shape mismatch: {selected_features.shape} vs {expected_shape}")
                return vision_features  # Fallback
            
            if self.shirg_config.get('debug', False):
                print(f"✅ SHIRG applied: {vision_features.shape} → {selected_features.shape} ({shirg_time*1000:.1f}ms)")
            
            return selected_features
            
        except Exception as e:
            print(f"⚠️ SHIRG selection failed: {e}")
            return vision_features  # Fallback to original
    
    def generate(self, 
                image_path: str, 
                question: str, 
                max_new_tokens: int = 32,     # LaViDa constraint
                temperature: float = 0.0,
                top_p: float = 1.0,
                num_beams: int = 1,
                do_sample: bool = False,
                use_cache: bool = True,
                prefix_lm: bool = True,       # Enable prefix-DLM for cache reuse
                **kwargs) -> str:
        """
        Generate response using LaViDa with SHIRG token selection
        
        Args:
            image_path: Path to input image
            question: Question about the image
            max_new_tokens: Maximum new tokens (LaViDa constraint: 32)
            temperature: Generation temperature
            top_p: Top-p sampling
            num_beams: Number of beams
            do_sample: Whether to use sampling
            use_cache: Enable KV caching
            prefix_lm: Enable prefix-DLM for cache reuse
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        
        if self.model is None:
            self.load_model()
        
        try:
            start_time = time.time()
            
            # Load and process image  
            from PIL import Image
            image = Image.open(image_path).convert('RGB')
            image_tensor = process_images([image], self.image_processor, self.model.config)
            
            if type(image_tensor) is list:
                image_tensor = [img.to(device=self.model.device, dtype=self.torch_dtype) for img in image_tensor]
            else:
                image_tensor = image_tensor.to(device=self.model.device, dtype=self.torch_dtype)
            
            # FIX: 2025-07-26 - Use LaViDa evaluation approach for conversation template
            # ISSUE: Creating conversation template at init causes tokenizer permission errors
            # SOLUTION: Create fresh template for each generation using copy.deepcopy() as in lmms_eval
            # RESEARCH IMPACT: Enables LaViDa generation without tokenizer permission issues
            
            # Prepare conversation exactly like lmms_eval does (line 563-584)
            import copy
            from llava.conversation import conv_templates
            
            # Use deepcopy for llada templates as in lmms_eval
            if "llama_3" in self.conv_template_name or 'llada' in self.conv_template_name:
                conv = copy.deepcopy(conv_templates[self.conv_template_name])
            else:
                conv = conv_templates[self.conv_template_name].copy()
            
            # Prepare question with image token
            question_with_image = f"{DEFAULT_IMAGE_TOKEN}\n{question}"
            
            # Add messages to conversation
            conv.append_message(conv.roles[0], question_with_image)
            conv.append_message(conv.roles[1], None)
            
            # Get prompt
            prompt = conv.get_prompt()
            
            # Tokenize
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
            
            # FIX: 2025-07-27 - Set question context for SHIRG token selection
            # ISSUE: SHIRG needs question embeddings to compute text-image relevance
            # SOLUTION: Embed question tokens and store for SHIRG access during encode_images
            # RESEARCH IMPACT: Enables SHIRG's text-conditioned token selection mechanism
            
            # Store question context for SHIRG (only if SHIRG is actually enabled)
            if (hasattr(self.model, 'shirg_selector') and 
                self.model.shirg_selector is not None and 
                self.shirg_config.get('alpha', 0) > 0):
                try:
                    # Extract question tokens (exclude image tokens and system tokens)
                    question_only = question.strip()
                    question_tokens = self.tokenizer(question_only, return_tensors='pt')['input_ids']
                    question_tokens = question_tokens.to(device=self.model.device)
                    
                    # Get question embeddings using model's word embeddings
                    with torch.no_grad():
                        question_embeddings = self.model.get_model().embed_tokens(question_tokens)  # [1, seq_len, 4096]
                        
                        # SHIRG-X-FIX: 2025-07-28 - Consolidated text-vision alignment using learned aligner only
                        # ISSUE: Multiple fallback mechanisms cause inconsistent text-vision alignment
                        # SOLUTION: Use only the learned text_vision_aligner for consistent behavior
                        # RESEARCH IMPACT: Ensures stable cross-modal similarity computation for SHIRG-X
                        
                        text_dim = question_embeddings.shape[-1]  # 4096 (LLaDA text embedding size)
                        vision_dim = 1152  # SigLIP hidden size
                        
                        if text_dim != vision_dim:
                            # Always use the learned text-vision aligner (initialized in load_model)
                            if not hasattr(self, 'text_vision_aligner'):
                                raise RuntimeError("text_vision_aligner not initialized. Call load_model() first.")
                            
                            # Ensure aligner is on correct device/dtype
                            if (self.text_vision_aligner.align_layer.weight.device != question_embeddings.device or
                                self.text_vision_aligner.align_layer.weight.dtype != question_embeddings.dtype):
                                self.text_vision_aligner = self.text_vision_aligner.to(
                                    device=question_embeddings.device, 
                                    dtype=question_embeddings.dtype
                                )
                            
                            # Apply learned alignment
                            question_embeddings = self.text_vision_aligner(question_embeddings)
                            
                            if self.shirg_config.get('debug', False):
                                print(f"✅ Text-vision alignment: {text_dim} -> {vision_dim} (learned aligner)")
                                print(f"   Input shape: {question_embeddings.shape}")
                        else:
                            if self.shirg_config.get('debug', False):
                                print(f"✅ No alignment needed: text_dim == vision_dim ({text_dim})")
                    
                    # Store for SHIRG access during encode_images
                    self._current_question_tokens = question_embeddings
                    self.model._current_image_sizes = [image.size] if hasattr(image, 'size') else None
                    
                except Exception as e:
                    if self.shirg_config.get('debug', False):
                        print(f"⚠️ Could not set question context for SHIRG: {e}")
                    self._current_question_tokens = None
            
            # FIX: 2025-07-26 - Use exact generation parameters from LaViDa lmms_eval
            # ISSUE: Wrong generation parameters cause errors in LaViDa diffusion
            # SOLUTION: Use exact parameters from llava_llada.py evaluation code (lines 587-610)
            # RESEARCH IMPACT: Enables successful LaViDa generation using official evaluation method
            
            # Build generation config exactly like lmms_eval (lines 587-610)
            gen_kwargs = {}
            
            # Set defaults exactly like lmms_eval
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = max_new_tokens
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0  # Force to 0 like lmms_eval
            if "do_sample" not in gen_kwargs:
                gen_kwargs["do_sample"] = do_sample
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = num_beams
                
            # FIX: 2025-07-27 - Add critical LaViDa prefix-DLM parameters for fast inference
            # ISSUE: Missing prefix_lm=True and proper diffusion scheduling causes slow inference
            # SOLUTION: Use exact parameters from LaViDa's predict.py for optimal performance
            # RESEARCH IMPACT: Enables LaViDa's 2x speed advantage through prefix-DLM caching
            
            # LaViDa-specific scheduling parameters (from predict.py and lmms_eval)
            if 'block_length' not in gen_kwargs:
                gen_kwargs['block_length'] = min(64, gen_kwargs["max_new_tokens"])  # Use 64 like predict.py
            if 'step_per_block' not in gen_kwargs and 'step_ratio' not in gen_kwargs:
                gen_kwargs['step_ratio'] = 0.5  # Use step_ratio for faster diffusion (predict.py line 78)
            
            # Critical LaViDa parameters for fast inference
            gen_kwargs["prefix_lm"] = True  # Enable prefix-DLM caching (key for 2x speedup)
            gen_kwargs["tokenizer"] = self.tokenizer  # Required for diffusion generation
            gen_kwargs["verbose"] = False  # Disable verbose output for cleaner evaluation
            
            # Force temperature to 0 for deterministic evaluation
            gen_kwargs["temperature"] = 0
            
            # Add image_sizes for image processing
            gen_kwargs["image_sizes"] = [image.size]
                
            # Add any additional kwargs
            gen_kwargs.update(kwargs)
            
            # Generate using LaViDa diffusion with prefix-DLM caching
            with torch.inference_mode():
                # Use exact same parameters as lmms_eval llava_llada.py (lines 613-635)
                pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                attention_masks = input_ids.ne(pad_token_ids).to(self.model.device)
                
                # Call LaViDa's diffusion generation with prefix-DLM caching
                output_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_masks,
                    pad_token_id=pad_token_ids,
                    images=image_tensor,
                    use_cache=use_cache,
                    **gen_kwargs
                )
            
            # FIX: 2025-07-26 - Use exact output processing from lmms_eval
            # ISSUE: Output processing must match evaluation code exactly
            # SOLUTION: Follow exact steps from llava_llada.py lines 638-639
            # RESEARCH IMPACT: Proper text extraction consistent with LaViDa evaluation
            
            # Process output exactly like lmms_eval (lines 638-639)
            text_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            text_outputs = [text_output.lstrip('!') for text_output in text_outputs]
            
            # Extract first response and strip whitespace
            result = text_outputs[0].strip() if text_outputs else ""
            
            # Track performance
            total_time = time.time() - start_time
            self.generation_times.append(total_time * 1000)
            
            # Clean up question context
            if hasattr(self, '_current_question_tokens'):
                delattr(self, '_current_question_tokens')
            if hasattr(self.model, '_current_image_sizes'):
                delattr(self.model, '_current_image_sizes')
            
            return result
            
        except Exception as e:
            print(f"⚠️ Generation failed: {e}")
            return f"Generation failed: {str(e)}"
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        
        stats = {
            'total_generations': len(self.generation_times),
            'avg_generation_time_ms': np.mean(self.generation_times) if self.generation_times else 0.0,
            'avg_shirg_time_ms': np.mean(self.shirg_times) if self.shirg_times else 0.0,
        }
        
        # Add SHIRG-specific stats
        if self.shirg_selector:
            shirg_stats = self.shirg_selector.get_performance_stats()
            stats.update({f'shirg_{k}': v for k, v in shirg_stats.items()})
        
        return stats
    
    def benchmark_latency(self, image_path: str, question: str, num_runs: int = 10) -> Dict[str, float]:
        """Benchmark generation latency"""
        print(f"🔄 Benchmarking latency over {num_runs} runs...")
        
        times = []
        for i in range(num_runs):
            start_time = time.time()
            result = self.generate(image_path, question)
            elapsed_time = time.time() - start_time
            times.append(elapsed_time * 1000)  # Convert to ms
            
            if i == 0:  # Print first result
                print(f"   Sample result: {result[:100]}...")
        
        import numpy as np
        return {
            'avg_latency_ms': np.mean(times),
            'min_latency_ms': np.min(times),
            'max_latency_ms': np.max(times),
            'std_latency_ms': np.std(times),
            'num_runs': num_runs
        }

    def cleanup(self):
        """
        Clean up model resources and free GPU memory
        
        FIX: 2025-07-26 - Add proper cleanup method for memory management
        ISSUE: LaViDa wrapper doesn't release GPU memory when deleted
        SOLUTION: Explicitly delete model components and clear CUDA cache
        RESEARCH IMPACT: Enables sequential model loading within GPU constraints
        """
        print("🧹 Cleaning up LaViDa-SHIRG model...")
        
        # Delete model components
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None
            
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            
        if hasattr(self, 'image_processor') and self.image_processor is not None:
            del self.image_processor
            self.image_processor = None
            
        if hasattr(self, 'shirg_selector') and self.shirg_selector is not None:
            del self.shirg_selector
            self.shirg_selector = None
            
            
        if hasattr(self, 'text_vision_aligner') and self.text_vision_aligner is not None:
            del self.text_vision_aligner
            self.text_vision_aligner = None
            
        if hasattr(self, 'adaptive_k_head') and self.adaptive_k_head is not None:
            del self.adaptive_k_head
            self.adaptive_k_head = None
            
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        print("✅ LaViDa-SHIRG model cleanup complete")

    def check_gpu_memory_shirg_x(self, context="general"):
        """
        SHIRG-X-FIX: 2025-07-28 - GPU memory monitoring for 40GB constraint
        ISSUE: SHIRG-X processes 2,448 tokens vs LaViDa's 729 = 3.4x memory increase
        SOLUTION: Proactive memory monitoring with warnings and cleanup suggestions
        RESEARCH IMPACT: Prevents OOM failures during SHIRG-X token processing
        
        Args:
            context: Description of current operation for logging
            
        Returns:
            memory_info: Dictionary with memory statistics
        """
        if not torch.cuda.is_available():
            return {"available": False, "warning": "CUDA not available"}
        
        # Get memory statistics
        allocated_gb = torch.cuda.memory_allocated() / 1e9
        reserved_gb = torch.cuda.memory_reserved() / 1e9
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        free_gb = total_gb - allocated_gb
        
        memory_info = {
            "allocated_gb": allocated_gb,
            "reserved_gb": reserved_gb,
            "total_gb": total_gb,
            "free_gb": free_gb,
            "usage_percent": (allocated_gb / total_gb) * 100,
            "context": context
        }
        
        # SHIRG-X specific thresholds (conservatively allow for 3.4x token increase)
        warning_threshold = 30.0  # Warn at 30GB (75% of 40GB)
        critical_threshold = 35.0  # Critical at 35GB (87.5% of 40GB)
        
        if allocated_gb > critical_threshold:
            print(f"🚨 CRITICAL GPU Memory ({context}): {allocated_gb:.1f}GB / {total_gb:.1f}GB ({memory_info['usage_percent']:.1f}%)")
            print(f"   ⚠️ SHIRG-X may cause OOM with 3.4x token increase!")
            print(f"   💡 Consider: torch.cuda.empty_cache() or reducing batch size")
            memory_info["level"] = "critical"
            
        elif allocated_gb > warning_threshold:
            print(f"⚠️ High GPU Memory ({context}): {allocated_gb:.1f}GB / {total_gb:.1f}GB ({memory_info['usage_percent']:.1f}%)")
            print(f"   📊 SHIRG-X processes {2448} tokens vs LaViDa's {729} (3.4x increase)")
            memory_info["level"] = "warning"
            
        elif self.shirg_config.get('debug', False):
            print(f"✅ GPU Memory ({context}): {allocated_gb:.1f}GB / {total_gb:.1f}GB ({memory_info['usage_percent']:.1f}%)")
            memory_info["level"] = "normal"
        
        return memory_info


def create_lavida_shirg_model(shirg_config: Optional[Dict[str, Any]] = None) -> LaViDaSHIRGWrapper:
    """
    Factory function to create LaViDa-SHIRG model
    
    Args:
        shirg_config: SHIRG configuration parameters
        
    Returns:
        LaViDaSHIRGWrapper instance
    """
    return LaViDaSHIRGWrapper(shirg_config=shirg_config)


def compare_baseline_vs_shirg(image_path: str, question: str) -> Dict[str, Any]:
    """
    Compare baseline LaViDa vs SHIRG-enhanced LaViDa
    
    Args:
        image_path: Path to test image
        question: Question about the image
        
    Returns:
        Comparison results
    """
    print("🔄 Comparing Baseline vs SHIRG...")
    
    # Test baseline (no SHIRG)
    print("Testing baseline LaViDa...")
    baseline_model = LaViDaSHIRGWrapper(shirg_config={'target_tokens': 980, 'alpha': 0.0})  # Disable SHIRG
    baseline_model.load_model()
    
    baseline_start = time.time()
    baseline_result = baseline_model.generate(image_path, question)
    baseline_time = time.time() - baseline_start
    
    # Test SHIRG
    print("Testing SHIRG-enhanced LaViDa...")
    shirg_model = create_lavida_shirg_model()
    shirg_model.load_model()
    
    shirg_start = time.time()
    shirg_result = shirg_model.generate(image_path, question)
    shirg_time = time.time() - shirg_start
    
    comparison = {
        'baseline': {
            'result': baseline_result,
            'time_ms': baseline_time * 1000,
            'tokens': 980
        },
        'shirg': {
            'result': shirg_result,
            'time_ms': shirg_time * 1000,
            'tokens': shirg_model.shirg_config['target_tokens']
        },
        'improvement': {
            'latency_overhead_percent': ((shirg_time - baseline_time) / baseline_time) * 100,
            'token_change': shirg_model.shirg_config['target_tokens'] - 980
        }
    }
    
    print(f"📊 Comparison Results:")
    print(f"   Baseline: {baseline_time*1000:.1f}ms → '{baseline_result[:50]}...'")
    print(f"   SHIRG: {shirg_time*1000:.1f}ms → '{shirg_result[:50]}...'")  
    print(f"   Latency overhead: {comparison['improvement']['latency_overhead_percent']:+.1f}%")
    
    return comparison


def setup_shirg_lora(model, lora_config=None):
    """
    Setup LoRA for SHIRG-Fovea: projector + early SigLIP layers
    
    Following the research methodology:
    - Rank-64 LoRA on mm_projector.fc1 
    - Rank-64 LoRA on SigLIP blocks 0-3 Q/K matrices
    - Target ~0.9% parameters (70M out of 8B)
    
    Args:
        model: LaViDa model with SHIRG integration
        lora_config: Optional LoRA configuration overrides
    
    Returns:
        model: Model with LoRA adapters configured
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        print("⚠️ PEFT not available. Please install PEFT for LoRA training.")
        return model
    
    # SHIRG-Fovea LoRA configuration (rank-64 as per research plan)
    shirg_lora_config = {
        'r': 64,                    # Rank: 64 for sufficient capacity
        'lora_alpha': 128,          # Alpha: 2x rank for stable training
        'target_modules': [
            # MM Projector first FC layer (primary target)
            "mm_projector.fc1",
            
            # Early SigLIP layers (vision adaptation) - Q/K only as per plan
            "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj",
            "vision_tower.vision_model.encoder.layers.0.self_attn.k_proj",
            "vision_tower.vision_model.encoder.layers.1.self_attn.q_proj", 
            "vision_tower.vision_model.encoder.layers.1.self_attn.k_proj",
            "vision_tower.vision_model.encoder.layers.2.self_attn.q_proj",
            "vision_tower.vision_model.encoder.layers.2.self_attn.k_proj",
            "vision_tower.vision_model.encoder.layers.3.self_attn.q_proj",
            "vision_tower.vision_model.encoder.layers.3.self_attn.k_proj",
        ],
        'lora_dropout': 0.05,
        'bias': 'none',
        'task_type': TaskType.FEATURE_EXTRACTION,
        'inference_mode': False
    }
    
    # Apply user overrides if provided
    if lora_config:
        shirg_lora_config.update(lora_config)
    
    # Create LoRA config object
    lora_config_obj = LoraConfig(**shirg_lora_config)
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config_obj)
    
    # Freeze everything except LoRA parameters  
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            print(f"✓ SHIRG-Fovea parameter enabled: {name}")
    
    return model


def verify_shirg_setup(model):
    """
    Verify SHIRG-Fovea LoRA setup is correct
    
    Validates that trainable parameters match research plan:
    - Target ~0.9% parameters (70M out of 8B)
    - Rank-64 LoRA configuration
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"SHIRG-Fovea LoRA Configuration:")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable ratio: {trainable_params/total_params:.4f}")
    
    # Should be ~0.9% trainable for SHIRG-Fovea (rank-64)
    expected_ratio = 0.009  # 0.9%
    actual_ratio = trainable_params/total_params
    
    if actual_ratio < expected_ratio * 0.8:
        print(f"⚠️ Warning: Trainable ratio {actual_ratio:.4f} is lower than target {expected_ratio}")
        print("   Consider increasing LoRA rank")
    elif actual_ratio > expected_ratio * 1.5:
        print(f"⚠️ Warning: Trainable ratio {actual_ratio:.4f} is higher than target {expected_ratio}")
        print("   Consider reducing LoRA rank")
    else:
        print(f"✅ SHIRG-Fovea LoRA ratio within target range: {actual_ratio:.4f}")
    
    return True


def prepare_shirg_training_data():
    """
    Prepare training data for SHIRG-Fovea adaptation
    
    Returns:
        training_config: Configuration for SHIRG training
    """
    
    # Training configuration for SHIRG-Fovea
    training_config = {
        "datasets": {
            "chartqa": {"weight": 0.3, "focus": "High-res spatial features"},
            "docvqa": {"weight": 0.3, "focus": "Document understanding"},  
            "vqa_v2": {"weight": 0.4, "focus": "General VQA"},
        },
        "batch_size": 16,        # Per GPU
        "gradient_accumulation": 1,
        "lr": 2e-5,             # LoRA learning rate
        "warmup_ratio": 0.03,
        "epochs": 1,
        "mixed_precision": "bf16",
        "lora_config": {
            "rank": 64,
            "alpha": 128,
            "dropout": 0.05
        }
    }
    
    return training_config


# Test function for Colab
def test_lavida_shirg_integration():
    """Test LaViDa-SHIRG integration with dummy setup"""
    print("🧪 Testing LaViDa-SHIRG Integration...")
    
    try:
        # Create model wrapper
        model = create_lavida_shirg_model({
            'target_tokens': 980,
            'alpha': 0.3,
            'debug': True
        })
        
        print("✅ Model wrapper created successfully")
        
        # Test would require actual model loading which needs GPU
        print("ℹ️ Full test requires GPU and model download in Colab")
        
        return model
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None


if __name__ == "__main__":
    # Run test if executed directly
    test_lavida_shirg_integration()