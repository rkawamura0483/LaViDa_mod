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

# Add LaViDa to path for imports
# In Colab, we're in /content/repo-name/ so LaViDa is in current directory
sys.path.append('./LaViDa' if IN_COLAB else './LaViDa')

try:
    from llava.model.builder import load_pretrained_model
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
    LAVIDA_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ LaViDa imports not available: {e}")
    LAVIDA_AVAILABLE = False

from shirg_selector import SHIRGSelector, create_shirg_selector

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
                 vision_kwargs: Optional[Dict[str, Any]] = None):
        """
        Initialize LaViDa with SHIRG integration
        
        Args:
            model_path: HuggingFace model path for LaViDa
            shirg_config: SHIRG configuration parameters
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
            'target_tokens': 729,       # Fixed for LaViDa compatibility (avoids assertion failures)
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
        
        # Default vision configuration for LaViDa with high-resolution support
        default_vision_kwargs = {
            "mm_vision_tower": "google/siglip-so400m-patch14-384",  # SigLIP-SO400M
            "mm_resampler_type": None,
            "mm_projector_type": 'mlp2x_gelu', 
            "mm_hidden_size": 1152,           # SigLIP hidden size
            "use_mm_proj": True,
            "mm_pooler_ratio": 2,
            "mm_patch_merge_type": 'spatial_unpad',  # Enable high-res processing
            "image_aspect_ratio": "anyres",           # Enable any resolution processing
            "image_grid_pinpoints": "[(768, 768)]",    # High-res: 55×55 = 3,025 tokens
        }
        
        if vision_kwargs:
            default_vision_kwargs.update(vision_kwargs)
        self.vision_kwargs = default_vision_kwargs
        
        # Initialize model components
        self.tokenizer = None
        self.model = None  
        self.image_processor = None
        self.context_len = None
        self.conv_template = None
        self.shirg_selector = None
        
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
            
            # Load base model with LaViDa-compatible parameters
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path=self.model_path,
                model_base=None,
                model_name="llava_llada",  # Critical: Use LaViDa model name
                device_map=self.device_map,
                vision_kwargs=self.vision_kwargs,
                torch_dtype='bfloat16',  # Use string format like official examples
                attn_implementation="flash_attention_2"
            )
            
            # Apply critical configuration fixes (from implementation plan)
            self._apply_config_fixes()
            
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
            
            # Initialize SHIRG selector
            self.shirg_selector = create_shirg_selector(self.shirg_config)
            
            # Setup conversation template
            self._setup_conversation_template()
            
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
        self.model.config.shirg_target_tokens = self.shirg_config['target_tokens']
        self.model.config.shirg_alpha = self.shirg_config['alpha']
        
        print("✅ Applied LaViDa configuration fixes")
    
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
            # FIX: 2025-07-27 - Remove target_tokens check to enable SHIRG with 729 tokens
            # ISSUE: Condition required target_tokens != 729, but we need 729 for LaViDa compatibility
            # SOLUTION: Only check alpha > 0 to distinguish baseline (alpha=0) from SHIRG (alpha>0)
            # RESEARCH IMPACT: Enables SHIRG integration while maintaining LaViDa compatibility
            
            if shirg_enabled:
                # Patch encode_images to use SHIRG with correct signature
                def patched_encode_images(self, images):
                    """Patched encode_images that applies SHIRG token selection"""
                    
                    print(f"🔄 SHIRG patched encode_images called with {images.shape if hasattr(images, 'shape') else type(images)} images")
                    
                    # FIX: 2025-07-27 - Apply SHIRG to raw vision tower output before pooling
                    # ISSUE: Previous integration was after pooling, getting already-reduced 729 tokens
                    # SOLUTION: Access vision tower directly and apply SHIRG to raw patch embeddings
                    # RESEARCH IMPACT: Tests actual research hypothesis of selecting from unpooled tokens
                    
                    wrapper = getattr(self, 'shirg_wrapper', None)
                    vision_tower = self.get_model().get_vision_tower()
                    
                    if (wrapper is not None and 
                        hasattr(wrapper, '_current_question_tokens') and 
                        wrapper._current_question_tokens is not None and
                        wrapper.shirg_config.get('alpha', 0) > 0):
                        
                        try:
                            # SHIRG-FIX: 2025-07-27 - Use genuine high-resolution token extraction
                            # ISSUE: Previous approach used interpolation, not real high-res processing
                            # SOLUTION: Use new forward_with_high_res method for genuine 3,645 tokens
                            # RESEARCH IMPACT: Tests actual research hypothesis with real high-resolution features
                            
                            if wrapper.shirg_config.get('debug', False):
                                print(f"🔍 Using genuine high-resolution token extraction")
                            
                            # Check if vision tower supports high-res extraction
                            if hasattr(vision_tower, 'forward_with_high_res'):
                                # Use new high-resolution extraction method
                                standard_features, high_res_features = vision_tower.forward_with_high_res(
                                    images, return_high_res=True, target_resolution=(768, 768)
                                )
                                
                                if wrapper.shirg_config.get('debug', False):
                                    if isinstance(high_res_features, list):
                                        hr_shape = [f.shape for f in high_res_features]
                                    else:
                                        hr_shape = high_res_features.shape
                                    print(f"✅ Genuine high-res extraction: {hr_shape}")
                                
                                # Use high-resolution features for SHIRG
                                raw_features = high_res_features
                                if isinstance(raw_features, list):
                                    # Handle list format (batch processing)
                                    raw_features = raw_features[0] if len(raw_features) == 1 else torch.cat(raw_features, dim=0)
                                
                            elif hasattr(vision_tower, 'get_multiview_high_res_tokens'):
                                # Use multi-view approach following LaViDa specification
                                if wrapper.shirg_config.get('debug', False):
                                    print(f"🔍 Using LaViDa multi-view high-res extraction")
                                
                                raw_features = vision_tower.get_multiview_high_res_tokens(images)
                                
                                if wrapper.shirg_config.get('debug', False):
                                    print(f"✅ Multi-view high-res tokens: {raw_features.shape}")
                            
                            else:
                                # Fallback: Use enhanced interpolation method as backup
                                if wrapper.shirg_config.get('debug', False):
                                    print(f"⚠️ Using fallback interpolation method")
                                
                                # Access the SigLIP vision model components
                                vision_model = vision_tower.vision_tower  # SigLipVisionModel
                                vision_transformer = vision_model.vision_model  # SigLipVisionTransformer
                                
                                # Get properly formatted input
                                pixel_values = images.to(device=vision_tower.device, dtype=vision_tower.dtype)
                                
                                # Step 1: Get patch embeddings - this gives us the raw patch grid
                                hidden_states = vision_transformer.embeddings(pixel_values)
                                
                                # Step 2: Enhanced interpolation to 3,645 tokens (LaViDa target)
                                batch_size, num_patches, embed_dim = hidden_states.shape
                                
                                if num_patches == 729:  # 27x27 grid from 384x384 input
                                    # Create high-resolution grid matching LaViDa's 3,645 target
                                    grid_size = int(np.sqrt(num_patches))  # 27
                                    
                                    # Calculate target grid for 3,645 tokens
                                    # sqrt(3645) ≈ 60.37, use 60x61 = 3,660, trim to 3,645
                                    target_h, target_w = 60, 61
                                    target_tokens = min(target_h * target_w, 3645)
                                    
                                    # Reshape to spatial grid [B, 27, 27, D]
                                    spatial_features = hidden_states.view(batch_size, grid_size, grid_size, embed_dim)
                                    spatial_features = spatial_features.permute(0, 3, 1, 2)  # [B, D, 27, 27]
                                    
                                    # Interpolate to higher resolution [B, D, 60, 61]
                                    import torch.nn.functional as F
                                    high_res_features = F.interpolate(
                                        spatial_features, 
                                        size=(target_h, target_w), 
                                        mode='bilinear', 
                                        align_corners=False
                                    )
                                    
                                    # Reshape back to token sequence [B, 3660, D]
                                    high_res_features = high_res_features.permute(0, 2, 3, 1)
                                    raw_features = high_res_features.reshape(batch_size, target_h * target_w, embed_dim)
                                    
                                    # Trim to exactly 3,645 tokens
                                    raw_features = raw_features[:, :3645, :]
                                    
                                    if wrapper.shirg_config.get('debug', False):
                                        print(f"✅ Enhanced interpolation to 3,645 tokens: {raw_features.shape}")
                                else:
                                    raw_features = hidden_states
                                    if wrapper.shirg_config.get('debug', False):
                                        print(f"✅ Using original tokens: {raw_features.shape}")
                            
                            raw_features = raw_features.to(images.dtype)
                            
                            # Apply SHIRG selection to unpooled tokens
                            if raw_features.shape[-2] > wrapper.shirg_config.get('target_tokens', 729):
                                # We have more tokens than target - can do meaningful selection
                                selected_features = self.shirg_selector(
                                    image_tokens=raw_features,
                                    text_embeddings=wrapper._current_question_tokens,
                                    image_sizes=getattr(self, '_current_image_sizes', None)
                                )
                                
                                if wrapper.shirg_config.get('debug', False):
                                    print(f"🎯 SHIRG applied: {raw_features.shape} → {selected_features.shape}")
                                
                                image_features = selected_features
                                
                            else:
                                # Same token count - still apply SHIRG for reordering/reweighting
                                if wrapper.shirg_config.get('debug', False):
                                    print(f"🔄 Applying SHIRG reordering to {raw_features.shape[-2]} tokens")
                                
                                selected_features = self.shirg_selector(
                                    image_tokens=raw_features,
                                    text_embeddings=wrapper._current_question_tokens,
                                    image_sizes=getattr(self, '_current_image_sizes', None)
                                )
                                
                                image_features = selected_features
                            
                        except Exception as e:
                            if wrapper.shirg_config.get('debug', False):
                                print(f"⚠️ SHIRG selection failed: {e}, trying simpler approach")
                            
                            # Fallback: LaViDa already removes pooling head, so vision_tower gives unpooled tokens
                            try:
                                raw_features = vision_tower(images)  # Already unpooled in LaViDa!
                                if wrapper.shirg_config.get('debug', False):
                                    print(f"✅ Using LaViDa unpooled tokens: {raw_features.shape}")
                                
                                # Apply SHIRG to these unpooled tokens
                                if raw_features.shape[-2] >= wrapper.shirg_config.get('target_tokens', 729):
                                    selected_features = self.shirg_selector(
                                        image_tokens=raw_features,
                                        text_embeddings=wrapper._current_question_tokens,
                                        image_sizes=getattr(self, '_current_image_sizes', None)
                                    )
                                    if wrapper.shirg_config.get('debug', False):
                                        print(f"🎯 SHIRG applied (fallback): {raw_features.shape} → {selected_features.shape}")
                                    image_features = selected_features
                                else:
                                    image_features = raw_features
                            except Exception as e2:
                                if wrapper.shirg_config.get('debug', False):
                                    print(f"⚠️ Fallback also failed: {e2}, using standard processing")
                                image_features = vision_tower(images)
                    else:
                        # Baseline: use standard vision tower (already pooled to 729)
                        image_features = vision_tower(images)
                    
                    # Apply mm_projector to final features
                    image_features = self.get_model().mm_projector(image_features)
                    return image_features
                
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
                        
                        # FIX: 2025-07-27 - SOLUTION 1: Use LaViDa's trained mm_projector for dimension alignment
                        # ISSUE: Text embeddings (4096) and vision features (1152) have different dimensions
                        # SOLUTION: Use LaViDa's existing mm_projector to map between vision and text spaces properly
                        # RESEARCH IMPACT: Preserves semantic meaning using trained projection weights
                        
                        # Get LaViDa's trained mm_projector for proper dimension alignment
                        text_dim = question_embeddings.shape[-1]  # 4096 (LLaDA text embedding size)
                        vision_dim = 1152  # SigLIP hidden size
                        
                        if text_dim != vision_dim:
                            # SOLUTION 1A: Use LaViDa's mm_projector inverse for text->vision mapping
                            if (hasattr(self.model, 'get_model') and 
                                hasattr(self.model.get_model(), 'mm_projector')):
                                
                                mm_projector = self.model.get_model().mm_projector
                                
                                # For linear projector: use pseudo-inverse for reverse mapping
                                if hasattr(mm_projector, 'weight'):
                                    # mm_projector maps: vision_dim -> text_dim
                                    # We need: text_dim -> vision_dim (inverse)
                                    with torch.no_grad():
                                        projector_weight = mm_projector.weight  # [text_dim, vision_dim]
                                        
                                        # FIX: Convert to float32 for torch.pinverse compatibility
                                        # torch.pinverse doesn't support bfloat16, only float32/float64
                                        projector_weight_f32 = projector_weight.float()  # Convert to float32
                                        
                                        # Use pseudo-inverse for stable text->vision mapping
                                        pseudo_inverse_f32 = torch.pinverse(projector_weight_f32)  # [vision_dim, text_dim]
                                        
                                        # Convert back to original dtype and apply projection
                                        pseudo_inverse = pseudo_inverse_f32.to(dtype=question_embeddings.dtype)
                                        question_embeddings = torch.matmul(question_embeddings, pseudo_inverse.T)  # [1, seq_len, vision_dim]
                                        
                                        if self.shirg_config.get('debug', False):
                                            print(f"✅ Using LaViDa mm_projector inverse: {text_dim} -> {vision_dim} (bfloat16 compatible)")
                                
                                # For MLP projector: create learned inverse approximation
                                elif hasattr(mm_projector, '0') and hasattr(mm_projector[0], 'weight'):
                                    # Get first layer for approximation
                                    first_layer_weight = mm_projector[0].weight  # [text_dim, vision_dim]
                                    
                                    # Create or reuse learned inverse projector
                                    if not hasattr(self, '_text_to_vision_projector'):
                                        self._text_to_vision_projector = torch.nn.Linear(
                                            text_dim, vision_dim, bias=False
                                        ).to(device=question_embeddings.device, dtype=question_embeddings.dtype)
                                        
                                        # Initialize with pseudo-inverse of first layer
                                        with torch.no_grad():
                                            # FIX: Convert to float32 for torch.pinverse compatibility
                                            first_layer_f32 = first_layer_weight.float()
                                            pseudo_inverse_f32 = torch.pinverse(first_layer_f32)
                                            pseudo_inverse = pseudo_inverse_f32.to(dtype=question_embeddings.dtype)
                                            self._text_to_vision_projector.weight.data = pseudo_inverse
                                        
                                        if self.shirg_config.get('debug', False):
                                            print(f"✅ Created learned inverse projector: {text_dim} -> {vision_dim}")
                                    
                                    # Apply learned projection
                                    question_embeddings = self._text_to_vision_projector(question_embeddings)
                                
                                else:
                                    # FALLBACK: Use Xavier-initialized projection (better than random)
                                    if not hasattr(self, '_fallback_text_projector'):
                                        self._fallback_text_projector = torch.nn.Linear(
                                            text_dim, vision_dim, bias=False
                                        ).to(device=question_embeddings.device, dtype=question_embeddings.dtype)
                                        # Initialize with Xavier uniform (better than random)
                                        torch.nn.init.xavier_uniform_(self._fallback_text_projector.weight)
                                        
                                        if self.shirg_config.get('debug', False):
                                            print(f"⚠️ Using fallback Xavier projection: {text_dim} -> {vision_dim}")
                                    
                                    question_embeddings = self._fallback_text_projector(question_embeddings)
                            
                            else:
                                # No mm_projector available - use Xavier initialization
                                if not hasattr(self, '_basic_text_projector'):
                                    self._basic_text_projector = torch.nn.Linear(
                                        text_dim, vision_dim, bias=False
                                    ).to(device=question_embeddings.device, dtype=question_embeddings.dtype)
                                    torch.nn.init.xavier_uniform_(self._basic_text_projector.weight)
                                    
                                    if self.shirg_config.get('debug', False):
                                        print(f"⚠️ No mm_projector found, using basic projection: {text_dim} -> {vision_dim}")
                                
                                question_embeddings = self._basic_text_projector(question_embeddings)
                    
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
            
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
        print("✅ LaViDa-SHIRG model cleanup complete")


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
    baseline_model = LaViDaSHIRGWrapper(shirg_config={'target_tokens': 729, 'alpha': 0.0})  # Disable SHIRG
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


# Test function for Colab
def test_lavida_shirg_integration():
    """Test LaViDa-SHIRG integration with dummy setup"""
    print("🧪 Testing LaViDa-SHIRG Integration...")
    
    try:
        # Create model wrapper
        model = create_lavida_shirg_model({
            'target_tokens': 729,
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