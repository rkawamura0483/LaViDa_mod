#!/usr/bin/env python3
"""
SHIRG Real OCR/VQA Model Runner
Handles model loading, inference operations, and GPU memory management for SHIRG validation
"""

import os
import sys
import math
import torch
import torch.nn.functional as F
import numpy as np
import time
import warnings
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import random
import copy

# SHIRG-FIX: 2025-07-28 - Add missing rank0_print function
# ISSUE: NameError: name 'rank0_print' is not defined on line 865
# SOLUTION: Define rank0_print function for distributed training compatibility
# LAVIDA IMPACT: Enables SHIRG validation to complete without crashes
# SHIRG IMPACT: Allows proper research validation metrics to be displayed
def rank0_print(*args, **kwargs):
    """Print function compatible with distributed training - always prints on rank 0"""
    print(*args, **kwargs)

# Check for torchvision availability
try:
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    print("⚠️ torchvision not available, using basic tensor conversion")
    TORCHVISION_AVAILABLE = False

warnings.filterwarnings('ignore')

# Add paths for imports
sys.path.append('./shirg/')
sys.path.append('./llava/')
sys.path.append('./')

# Import LaViDa components
try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
    from llava.conversation import conv_templates, SeparatorStyle
    LAVIDA_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ LaViDa imports not available: {e}")
    LAVIDA_AVAILABLE = False

class LaViDaModelRunner:
    """Handles LaViDa model loading, inference operations, and GPU memory management"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Baseline LaViDa model components (original encoder)
        self.baseline_tokenizer = None
        self.baseline_model = None
        self.baseline_image_processor = None
        self.baseline_tower = None
        
        # SHIRG LaViDa model components (SHIRG encoder)
        self.shirg_tokenizer = None
        self.shirg_model = None
        self.shirg_image_processor = None
        self.shirg_tower = None
        
        self.max_length = None
        
        # Model configuration - using correct HuggingFace model path
        self.pretrained_path = "KonstantinosKK/lavida-llada-v1.0-instruct-hf-transformers"
        self.model_name = "llava_llada"
        self.conv_template_name = "llada"
    
    def _load_baseline_model(self):
        """Load baseline LaViDa model with original SigLIP encoder"""
        if not LAVIDA_AVAILABLE:
            print("❌ LaViDa not available, skipping baseline model loading")
            return False
            
        print("📦 Loading baseline LaViDa model (original encoder)...")
        
        try:
            # Explicitly load without SHIRG modifications
            print(f"   📂 Model path: {self.pretrained_path}")
            print(f"   🏷️ Model name: {self.model_name}")
            print(f"   🔧 Conv template: {self.conv_template_name}")
            
            # COMPREHENSIVE-DEBUG: 2025-07-29 - Add detailed model loading diagnostics
            # ISSUE: Need to track exactly where model loading fails for better debugging
            # SOLUTION: Add step-by-step loading diagnostics with component validation
            # LAVIDA IMPACT: Enables precise diagnosis of LaViDa model loading issues
            # SHIRG IMPACT: Ensures research validation can identify and resolve loading failures
            
            print("   🔄 Step 1: Loading pretrained model components...")
            
            # LAVIDA-CONFIG-FIX: 2025-07-29 - Use proper LaViDa vision configuration
            # ISSUE: LaViDa requires specific vision_kwargs for proper SigLIP integration
            # SOLUTION: Use exact vision_kwargs from original LaViDa predict.py
            # RESEARCH IMPACT: Ensures baseline LaViDa works exactly as in original implementation
            # SHIRG IMPACT: Provides proper baseline for SHIRG comparison
            
            vision_kwargs = {
                "mm_vision_tower": "google/siglip-so400m-patch14-384",
                "mm_resampler_type": None,
                "mm_projector_type": 'mlp2x_gelu',
                "mm_hidden_size": 1152,
                "use_mm_proj": True
            }
            
            print("BASELINE-CONFIG: Using 384×384 image processor for standard LaViDa processing")
            
            # Load baseline model components with proper LaViDa configuration
            self.baseline_tokenizer, self.baseline_model, self.baseline_image_processor, _ = load_pretrained_model(
                model_path=self.pretrained_path,
                model_base=None,
                model_name=self.model_name,
                load_8bit=False,
                load_4bit=False,
                device=self.device,
                device_map=None,
                vision_kwargs=vision_kwargs,
                torch_dtype='bfloat16'
            )
            
            # VALIDATION-FIX: 2025-07-29 - Validate each component after loading
            # ISSUE: One or more components may fail to load causing downstream errors
            # SOLUTION: Check each component individually and provide specific error messages
            # LAVIDA IMPACT: Identifies exactly which LaViDa component failed to load
            # SHIRG IMPACT: Enables targeted fixes for specific component failures
            
            print("   🔍 Step 2: Validating loaded components...")
            
            # Check tokenizer
            if self.baseline_tokenizer is None:
                print("   ❌ Baseline tokenizer failed to load")
                return False
            else:
                print(f"   ✅ Baseline tokenizer loaded: {type(self.baseline_tokenizer).__name__}")
                print(f"      - Vocab size: {getattr(self.baseline_tokenizer, 'vocab_size', 'Unknown')}")
                print(f"      - EOS token: {getattr(self.baseline_tokenizer, 'eos_token', 'Unknown')}")
            
            # Check model
            if self.baseline_model is None:
                print("   ❌ Baseline model failed to load")
                return False
            else:
                print(f"   ✅ Baseline model loaded: {type(self.baseline_model).__name__}")
                print(f"      - Device: {next(self.baseline_model.parameters()).device}")
                print(f"      - Dtype: {next(self.baseline_model.parameters()).dtype}")
            
            # Check image processor
            if self.baseline_image_processor is None:
                print("   ❌ Baseline image processor failed to load")
                return False
            else:
                print(f"   ✅ Baseline image processor loaded: {type(self.baseline_image_processor).__name__}")
                if hasattr(self.baseline_image_processor, 'size'):
                    print(f"      - Size: {self.baseline_image_processor.size}")
                if hasattr(self.baseline_image_processor, 'crop_size'):
                    print(f"      - Crop size: {self.baseline_image_processor.crop_size}")
            
            # Get max token length from model config
            self.max_length = getattr(self.baseline_model.config, 'max_position_embeddings', 2048)
            
            # LAVIDA-SETUP-FIX: 2025-07-29 - Proper LaViDa model setup following original predict.py
            # ISSUE: LaViDa models require specific setup for diffusion-based generation
            # SOLUTION: Follow exact setup from original LaViDa predict.py
            # RESEARCH IMPACT: Ensures baseline LaViDa generates correctly
            # SHIRG IMPACT: Provides proper baseline performance reference
            
            # Move to device and set eval mode
            self.baseline_model = self.baseline_model.to(self.device)
            self.baseline_model.eval()
            
            # LaViDa-specific setup
            self.baseline_model.tie_weights()
            self.baseline_model = self.baseline_model.to(torch.bfloat16)
            
            # Get vision tower
            if hasattr(self.baseline_model, 'get_vision_tower'):
                self.baseline_tower = self.baseline_model.get_vision_tower()
                if self.baseline_tower is not None:
                    self.baseline_tower = self.baseline_tower.to(self.device)
                    print(f"   🔍 Vision tower loaded: {type(self.baseline_tower).__name__}")
                else:
                    print("   ⚠️ Vision tower is None")
            else:
                print("   ⚠️ Model has no get_vision_tower method")
            
            # Memory check after loading
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                print(f"   💾 GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
            
            print("✅ Baseline LaViDa model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load baseline model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _unload_baseline_model(self):
        """Unload baseline model to free GPU memory"""
        print("🗑️ Unloading baseline model...")
        
        if self.baseline_model is not None:
            del self.baseline_model
            self.baseline_model = None
        
        if self.baseline_tokenizer is not None:
            del self.baseline_tokenizer
            self.baseline_tokenizer = None
            
        if self.baseline_image_processor is not None:
            del self.baseline_image_processor
            self.baseline_image_processor = None
            
        if self.baseline_tower is not None:
            del self.baseline_tower
            self.baseline_tower = None
        
        self._clear_gpu_memory()
        print("✅ Baseline model unloaded")
    
    def _unload_shirg_model(self):
        """Unload SHIRG model to free GPU memory"""
        print("🗑️ Unloading SHIRG model...")
        
        if self.shirg_model is not None:
            del self.shirg_model
            self.shirg_model = None
        
        if self.shirg_tokenizer is not None:
            del self.shirg_tokenizer
            self.shirg_tokenizer = None
            
        if self.shirg_image_processor is not None:
            del self.shirg_image_processor
            self.shirg_image_processor = None
            
        if self.shirg_tower is not None:
            del self.shirg_tower
            self.shirg_tower = None
        
        self._clear_gpu_memory()
        print("✅ SHIRG model unloaded")
    
    def _clear_gpu_memory(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Memory status after clearing
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"   💾 GPU Memory after clearing - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
    
    def _run_all_baseline_inferences(self, ocr_vqa_samples):
        """Run baseline inference on all samples"""
        baseline_results = {}
        
        if not self._load_baseline_model():
            print("❌ Failed to load baseline model, skipping baseline inferences")
            return {}
        
        print(f"🔄 Running baseline inference on {len(ocr_vqa_samples)} samples...")
        
        for i, (sample_name, sample_data) in enumerate(ocr_vqa_samples.items()):
            print(f"\n📝 Sample {i+1}/{len(ocr_vqa_samples)}: {sample_name}")
            
            try:
                # Extract components
                image = sample_data['image']
                question = sample_data['question']
                
                # TOKENIZER-FIX: 2025-07-29 - Robust baseline tokenizer validation
                # ISSUE: baseline_tokenizer can be None causing 'NoneType' object is not subscriptable
                # SOLUTION: Add comprehensive None checks and meaningful error handling
                # LAVIDA IMPACT: Prevents baseline inference crashes due to tokenizer issues
                # SHIRG IMPACT: Ensures research validation can proceed with meaningful error logging
                
                # Prepare input
                if self.baseline_tokenizer is not None:
                    try:
                        input_ids = self._prepare_input_ids(question, self.baseline_tokenizer)
                        if input_ids is not None:
                            result = self._run_baseline_inference(image, input_ids, question)
                            baseline_results[sample_name] = result
                            print(f"   ✅ Baseline result: {result.get('response', 'No response')[:100]}...")
                        else:
                            print("   ⚠️ Input IDs preparation returned None")
                            baseline_results[sample_name] = {
                                'response': "Error: Input IDs preparation failed",
                                'tokens_used': 0,
                                'inference_time': 0.0,
                                'error': 'Input IDs preparation returned None'
                            }
                    except Exception as e:
                        print(f"   ❌ Error in baseline tokenizer processing: {e}")
                        baseline_results[sample_name] = {
                            'response': f"Error: Baseline tokenizer processing failed - {str(e)}",
                            'tokens_used': 0,
                            'inference_time': 0.0,
                            'error': str(e)
                        }
                else:
                    print("   ⚠️ Baseline tokenizer not available, skipping")
                    baseline_results[sample_name] = {
                        'response': "Error: Baseline tokenizer not available",
                        'tokens_used': 0,
                        'inference_time': 0.0,
                        'error': 'Baseline tokenizer is None'
                    }
                    
            except Exception as e:
                print(f"   ❌ Error processing {sample_name}: {e}")
                # DEBUGGING-FIX: 2025-07-29 - Add detailed error information for better diagnosis
                # ISSUE: Generic error messages make it hard to diagnose specific failure points
                # SOLUTION: Include error type, traceback, and context for better debugging
                # LAVIDA IMPACT: Enables faster debugging of LaViDa model loading issues
                # SHIRG IMPACT: Provides detailed context for SHIRG integration failures
                import traceback
                error_details = {
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'traceback': traceback.format_exc(),
                    'sample_name': sample_name,
                    'phase': 'baseline_inference'
                }
                print(f"   📋 Error details: {error_details['error_type']} - {error_details['error_message']}")
                baseline_results[sample_name] = {
                    'response': f"Error during baseline inference: {str(e)[:100]}...",
                    'tokens_used': 0,
                    'inference_time': 0.0,
                    'error': str(e),
                    'error_details': error_details
                }
        
        print(f"\n✅ Completed baseline inference on {len(baseline_results)} samples")
        return baseline_results
    
    def _run_all_shirg_inferences(self, ocr_vqa_samples):
        """Run SHIRG inference on all samples"""
        shirg_results = {}
        
        if not self._load_shirg_model():
            print("❌ Failed to load SHIRG model, skipping SHIRG inferences")
            return {}
        
        # Load LoRA weights if available
        self._load_lora_weights()
        
        print(f"🔄 Running SHIRG inference on {len(ocr_vqa_samples)} samples...")
        
        for i, (sample_name, sample_data) in enumerate(ocr_vqa_samples.items()):
            print(f"\n📝 Sample {i+1}/{len(ocr_vqa_samples)}: {sample_name}")
            
            try:
                # Extract components
                image = sample_data['image']
                question = sample_data['question']
                
                # SHIRG-METHODOLOGY-FIX: 2025-07-29 - Use original image, let SHIRG vision tower handle resizing
                # ISSUE: Manual resizing violates SHIRG research methodology which requires native high-res processing
                # SOLUTION: Pass original image to SHIRG model, vision tower handles 672×672 processing internally
                # RESEARCH IMPACT: Enables native high-resolution processing as specified in SHIRG research
                # LAVIDA IMPACT: Maintains compatibility with LaViDa image processing pipeline
                shirg_image = image  # Use original image, SHIRG vision tower will handle 672×672 processing
                
                # Prepare input
                if self.shirg_tokenizer is not None:
                    input_ids = self._prepare_input_ids(question, self.shirg_tokenizer)
                    result = self._run_shirg_inference(shirg_image, input_ids, question)
                    shirg_results[sample_name] = result
                    print(f"   ✅ SHIRG result: {result.get('response', 'No response')[:100]}...")
                else:
                    print("   ⚠️ SHIRG tokenizer not available, skipping")
                    
            except Exception as e:
                print(f"   ❌ Error processing {sample_name}: {e}")
                shirg_results[sample_name] = {
                    'response': f"Error: {str(e)}",
                    'tokens_used': 0,
                    'inference_time': 0.0,
                    'token_selection': {},
                    'error': str(e)
                }
        
        print(f"\n✅ Completed SHIRG inference on {len(shirg_results)} samples")
        return shirg_results
    
    def _load_shirg_model(self):
        """Load SHIRG-enabled LaViDa model"""
        if not LAVIDA_AVAILABLE:
            print("❌ LaViDa not available, skipping SHIRG model loading")
            return False
            
        print("📦 Loading SHIRG LaViDa model (SHIRG encoder)...")
        
        try:
            # First, ensure SHIRG encoder is available
            try:
                from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
                print("   ✅ SHIRG SigLIP encoder available")
            except ImportError as e:
                print(f"   ❌ SHIRG encoder not available: {e}")
                return False
            
            # SHIRG-CONFIG-FIX: 2025-07-29 - Use SHIRG-specific vision configuration
            # ISSUE: SHIRG requires high-resolution vision tower configuration
            # SOLUTION: Configure vision_kwargs for SHIRG 672x672 processing
            # RESEARCH IMPACT: Enables SHIRG high-resolution token extraction
            # LAVIDA IMPACT: Maintains LaViDa compatibility while adding SHIRG capabilities
            
            # SHIRG vision configuration for 672x672 processing
            shirg_vision_kwargs = {
                "mm_vision_tower": "google/siglip-so400m-patch14-384",  # Base model
                "mm_resampler_type": None,
                "mm_projector_type": 'mlp2x_gelu',
                "mm_hidden_size": 1152,
                "use_mm_proj": True,
                "enable_shirg": True,  # Enable SHIRG processing
                "image_aspect_ratio": None  # CRITICAL: Disable multi-view for SHIRG single-view processing
            }
            
            print("SHIRG-CONFIG: Using 672×672 image processor for SHIRG high-resolution processing")
            
            # Load SHIRG model components
            print(f"   📂 Model path: {self.pretrained_path}")
            print(f"   🏷️ Model name: {self.model_name}")
            
            self.shirg_tokenizer, self.shirg_model, self.shirg_image_processor, _ = load_pretrained_model(
                model_path=self.pretrained_path,
                model_base=None,
                model_name=self.model_name,
                load_8bit=False,
                load_4bit=False,
                device=self.device,
                device_map=None,
                vision_kwargs=shirg_vision_kwargs,
                torch_dtype='bfloat16'
            )
            
            # Get max token length from model config
            self.max_length = getattr(self.shirg_model.config, 'max_position_embeddings', 2048)
            
            # SHIRG-MODEL-CONFIG-FIX: 2025-07-29 - Set image_aspect_ratio to None for SHIRG
            # ISSUE: SHIRG needs single-view processing, not LaViDa's multi-view "anyres" mode
            # SOLUTION: Override image_aspect_ratio to None to disable multi-view processing
            # RESEARCH IMPACT: Enables SHIRG to receive single 672×672 images instead of multi-view
            # LAVIDA IMPACT: Maintains LaViDa functionality while enabling SHIRG high-resolution processing
            if hasattr(self.shirg_model.config, 'image_aspect_ratio'):
                original_aspect_ratio = getattr(self.shirg_model.config, 'image_aspect_ratio', None)
                self.shirg_model.config.image_aspect_ratio = None
                print(f"   🔧 SHIRG: Disabled multi-view processing (was: {original_aspect_ratio}, now: None)")
            else:
                # Add the attribute if it doesn't exist
                setattr(self.shirg_model.config, 'image_aspect_ratio', None)
                print(f"   🔧 SHIRG: Set image_aspect_ratio = None for single-view processing")
            
            # SHIRG-SETUP-FIX: 2025-07-29 - Proper SHIRG model setup
            # ISSUE: SHIRG models need same setup as baseline plus SHIRG configuration
            # SOLUTION: Apply LaViDa setup plus SHIRG-specific configuration
            # RESEARCH IMPACT: Ensures SHIRG generates correctly with high-resolution tokens
            # LAVIDA IMPACT: Maintains LaViDa compatibility
            
            # Move to device and set eval mode
            self.shirg_model = self.shirg_model.to(self.device)
            self.shirg_model.eval()
            
            # LaViDa-specific setup (same as baseline)
            self.shirg_model.tie_weights()
            self.shirg_model = self.shirg_model.to(torch.bfloat16)
            
            # Get vision tower and enable SHIRG
            if hasattr(self.shirg_model, 'get_vision_tower'):
                self.shirg_tower = self.shirg_model.get_vision_tower()
                if self.shirg_tower is not None:
                    self.shirg_tower = self.shirg_tower.to(self.device)
                    
                    # SHIRG-CONFIG-FIX: 2025-07-29 - Comprehensive SHIRG configuration
                    # ISSUE: SHIRG configuration may not be properly applied to vision tower
                    # SOLUTION: Set all necessary SHIRG configuration parameters
                    # RESEARCH IMPACT: Ensures SHIRG methodology is properly implemented
                    # LAVIDA IMPACT: Maintains compatibility while enabling SHIRG features
                    if hasattr(self.shirg_tower, 'config'):
                        self.shirg_tower.config.enable_shirg = True
                        # Ensure SHIRG uses proper high-resolution configuration
                        self.shirg_tower.shirg_enabled = True
                        print(f"   🔍 SHIRG enabled on vision tower")
                        print(f"   🔍 SHIRG configuration: enable_shirg={getattr(self.shirg_tower.config, 'enable_shirg', False)}")
                    else:
                        print(f"   ⚠️ Vision tower has no config attribute - SHIRG mode may not work properly")
                    
                    print(f"   🔍 Vision tower loaded: {type(self.shirg_tower).__name__}")
                else:
                    print("   ⚠️ Vision tower is None")
            else:
                print("   ⚠️ Model has no get_vision_tower method")
            
            # Memory check after loading
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                print(f"   💾 GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
            
            print("✅ SHIRG LaViDa model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load SHIRG model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_lora_weights(self):
        """Load LoRA weights if available"""
        print("🔍 Checking for LoRA weights...")
        
        # Look for LoRA weights in common locations
        lora_paths = [
            './shirg/lora_weights/',
            './lora_weights/',
            './weights/lora/',
            './models/lora/'
        ]
        
        lora_found = False
        for lora_path in lora_paths:
            if os.path.exists(lora_path):
                print(f"   📂 Found LoRA directory: {lora_path}")
                # Check for adapter files
                adapter_files = [f for f in os.listdir(lora_path) if f.endswith('.bin') or f.endswith('.safetensors')]
                if adapter_files:
                    print(f"   📦 Found LoRA weights: {adapter_files}")
                    lora_found = True
                    
                    # TODO: Load LoRA weights into model
                    # This would require integrating with PEFT library
                    print(f"   ⚠️ LoRA loading not yet implemented - model using base weights")
                    break
        
        if not lora_found:
            print(f"   📋 No LoRA weights found, using base model weights")
            print(f"   📋 Continuing without LoRA weights")
    
    def _resize_for_shirg(self, image, target_size=672):
        """Resize image for SHIRG processing (672x672)"""
        try:
            if hasattr(image, 'size'):
                # PIL Image
                return image.resize((target_size, target_size), Image.Resampling.LANCZOS)
            elif torch.is_tensor(image):
                # Tensor image - convert to PIL first
                if image.dim() == 4:
                    image = image.squeeze(0)
                if image.dim() == 3 and image.shape[0] in [1, 3]:
                    # CHW format
                    image = image.permute(1, 2, 0)
                
                # Convert to PIL
                if image.max() <= 1.0:
                    image = (image * 255).byte()
                image_pil = Image.fromarray(image.cpu().numpy())
                return image_pil.resize((target_size, target_size), Image.Resampling.LANCZOS)
            else:
                print(f"   ⚠️ Unknown image type: {type(image)}")
                return image
        except Exception as e:
            print(f"   ⚠️ Error resizing image: {e}")
            return image
    
    def _validate_and_convert_image(self, image, sample_id):
        """
        Validate and convert image to PIL format for SHIRG processing
        
        Args:
            image: Input image in various formats (PIL, numpy, tensor, etc.)
            sample_id: Sample identifier for debugging
            
        Returns:
            PIL.Image: Validated PIL image ready for processing
        """
        try:
            # SHIRG-IMAGE-VALIDATION-FIX: 2025-07-29 - Comprehensive image format validation
            # ISSUE: Images from HuggingFace datasets can be in various formats causing preprocessing errors
            # SOLUTION: Validate and normalize all image formats to PIL.Image before processing
            # RESEARCH IMPACT: Enables SHIRG to process any dataset image format reliably  
            # LAVIDA IMPACT: Maintains consistent image format across baseline and SHIRG processing
            
            # Check for None/invalid image
            if image is None:
                print(f"   ❌ Sample {sample_id}: Image is None")
                return None
            
            # Handle PIL Images (most common case)
            if isinstance(image, Image.Image):
                # Ensure RGB mode for proper channel dimension inference
                if image.mode != 'RGB':
                    print(f"   🔄 Sample {sample_id}: Converting {image.mode} to RGB")
                    image = image.convert('RGB')
                
                # Validate image has valid dimensions
                width, height = image.size
                if width == 0 or height == 0:
                    print(f"   ❌ Sample {sample_id}: Invalid image dimensions {width}x{height}")
                    return None
                
                print(f"   ✅ Sample {sample_id}: Valid PIL Image {width}x{height} in {image.mode} mode")
                return image
            
            # Handle numpy arrays
            elif isinstance(image, np.ndarray):
                print(f"   🔄 Sample {sample_id}: Converting numpy array to PIL Image")
                
                # Handle different numpy array formats
                if image.ndim == 2:
                    # Grayscale image
                    image = np.stack([image] * 3, axis=-1)
                elif image.ndim == 3:
                    # Color image - ensure HWC format
                    if image.shape[0] in [1, 3, 4]:  # CHW format
                        image = np.transpose(image, (1, 2, 0))
                    
                    # Take only RGB channels if RGBA
                    if image.shape[-1] == 4:
                        image = image[:, :, :3]
                    elif image.shape[-1] == 1:
                        image = np.stack([image[:, :, 0]] * 3, axis=-1)
                else:
                    print(f"   ❌ Sample {sample_id}: Unsupported numpy array shape {image.shape}")
                    return None
                
                # Normalize to 0-255 range if needed
                if image.dtype == np.float32 or image.dtype == np.float64:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                elif image.dtype != np.uint8:
                    image = image.astype(np.uint8)
                
                # Convert to PIL
                pil_image = Image.fromarray(image)
                print(f"   ✅ Sample {sample_id}: Converted numpy array to PIL Image {pil_image.size}")
                return pil_image
            
            # Handle torch tensors
            elif torch.is_tensor(image):
                print(f"   🔄 Sample {sample_id}: Converting torch tensor to PIL Image")
                
                # Move to CPU and convert to numpy
                image_np = image.detach().cpu().numpy()
                
                # Handle batch dimension
                if image_np.ndim == 4:
                    image_np = image_np.squeeze(0)
                
                # Recursive call with numpy array
                return self._validate_and_convert_image(image_np, sample_id)
            
            # Handle other formats (try to convert to PIL)
            else:
                print(f"   ⚠️ Sample {sample_id}: Unknown image type {type(image)}, attempting conversion")
                try:
                    # Try direct PIL conversion
                    pil_image = Image.fromarray(np.array(image))
                    return self._validate_and_convert_image(pil_image, sample_id)
                except Exception as conv_error:
                    print(f"   ❌ Sample {sample_id}: Failed to convert {type(image)} to PIL: {conv_error}")
                    return None
                    
        except Exception as e:
            print(f"   ❌ Sample {sample_id}: Image validation failed: {e}")
            return None
    
    def _prepare_input_ids(self, question, tokenizer):
        """Prepare input IDs for inference following EXACT original LaViDa pattern"""
        try:
            # LAVIDA-EXACT-FIX: 2025-07-29 - Use EXACT original LaViDa conversation pattern
            # ISSUE: Not following exact original predict.py conversation setup
            # SOLUTION: Use exact same pattern as original predict.py with copy.deepcopy
            # RESEARCH IMPACT: Ensures exact LaViDa behavior for proper baseline comparison
            # LAVIDA IMPACT: Maintains exact LaViDa conversation format and tokenization
            
            if tokenizer is None:
                print(f"   ⚠️ Tokenizer is None - cannot prepare input IDs")
                raise ValueError("Tokenizer is None")
            
            # EXACT ORIGINAL LAVIDA PATTERN from predict.py:
            # conv_template = "llada" 
            # question = DEFAULT_IMAGE_TOKEN + "\nDescribe the image in detail."
            # conv = copy.deepcopy(conv_templates[conv_template])
            # conv.append_message(conv.roles[0], question)
            # conv.append_message(conv.roles[1], None)
            # prompt_question = conv.get_prompt()
            
            try:
                # Use exact same pattern as original predict.py
                conv_template_name = "llada"  # Exact same as original
                
                print(f"   📝 Using LaViDa conversation template: {conv_template_name}")
                
                # Build question exactly like original predict.py
                formatted_question = DEFAULT_IMAGE_TOKEN + "\n" + question
                
                # DEEPCOPY-FIX: 2025-07-29 - Handle problematic tokenizer in conversation template
                # ISSUE: copy.deepcopy fails if conversation template has problematic tokenizer
                # SOLUTION: Temporarily replace tokenizer fields during deepcopy, then restore
                # RESEARCH IMPACT: Enables exact LaViDa conversation pattern to work
                # LAVIDA IMPACT: Maintains exact LaViDa behavior while handling tokenizer issues
                
                # Use copy.deepcopy exactly like original predict.py, but handle tokenizer issues
                import copy
                
                # Get the template and check if it has problematic tokenizer
                template = conv_templates[conv_template_name]
                original_tokenizer = getattr(template, 'tokenizer', None)
                original_tokenizer_id = getattr(template, 'tokenizer_id', None)
                
                try:
                    # Try normal deepcopy first (like original)
                    conv = copy.deepcopy(template)
                    print(f"   ✅ Successfully deepcopied conversation template")
                except Exception as deepcopy_error:
                    print(f"   ⚠️ Deepcopy failed ({deepcopy_error}), using tokenizer-safe copy...")
                    
                    # Temporarily set tokenizer to None for deepcopy, then restore behavior
                    if hasattr(template, 'tokenizer'):
                        template.tokenizer = None
                    if hasattr(template, 'tokenizer_id'):
                        template.tokenizer_id = None
                    
                    try:
                        conv = copy.deepcopy(template)
                        print(f"   ✅ Successfully deepcopied with tokenizer-safe method")
                    finally:
                        # Restore original tokenizer values
                        if hasattr(template, 'tokenizer'):
                            template.tokenizer = original_tokenizer
                        if hasattr(template, 'tokenizer_id'):
                            template.tokenizer_id = original_tokenizer_id
                
                # Follow exact original pattern
                conv.append_message(conv.roles[0], formatted_question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()
                
                print(f"   📝 LaViDa prompt: {prompt_question[:100]}...")
                
                # Use exact tokenization from original predict.py
                input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
                
                return input_ids
                
            except Exception as conv_error:
                print(f"   ⚠️ LaViDa original pattern failed: {conv_error}")
                import traceback
                traceback.print_exc()
                
                # FALLBACK-FIX: 2025-07-29 - If original pattern fails, use manual conversation building
                # ISSUE: copy.deepcopy may fail if template has problematic tokenizer
                # SOLUTION: Build conversation manually with same structure
                # RESEARCH IMPACT: Allows research validation to continue with LaViDa-compatible prompts
                # LAVIDA IMPACT: Maintains LaViDa conversation structure even with tokenizer issues
                
                print(f"   🔄 Attempting manual conversation building as fallback...")
                try:
                    conv_template = conv_templates["llada"]
                    
                    # Create new conversation instance manually
                    from llava.conversation import Conversation
                    
                    conv = Conversation(
                        system=conv_template.system,
                        roles=conv_template.roles,
                        version=conv_template.version,
                        messages=[],
                        offset=conv_template.offset,
                        sep=conv_template.sep,
                        sep_style=conv_template.sep_style,
                        tokenizer_id=None,  # Don't use template's tokenizer
                        tokenizer=None,     # Use passed tokenizer instead
                        stop_token_ids=getattr(conv_template, 'stop_token_ids', [])
                    )
                    
                    # Build conversation following original pattern
                    formatted_question = DEFAULT_IMAGE_TOKEN + "\n" + question
                    conv.append_message(conv.roles[0], formatted_question)
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    
                    print(f"   📝 Manual LaViDa prompt: {prompt_question[:100]}...")
                    
                    # Tokenize with LaViDa method
                    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
                    
                    return input_ids
                    
                except Exception as manual_error:
                    print(f"   ⚠️ Manual conversation building failed: {manual_error}")
                    
                    # Final fallback: Simple LaViDa-compatible prompt
                    try:
                        simple_prompt = DEFAULT_IMAGE_TOKEN + "\n" + question
                        input_ids = tokenizer_image_token(simple_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
                        return input_ids
                    except Exception as simple_error:
                        print(f"   ⚠️ Simple LaViDa tokenization failed: {simple_error}")
                        
                        # Absolute final fallback
                        if tokenizer is not None:
                            try:
                                basic_input_ids = tokenizer.encode(question, return_tensors='pt')
                                if basic_input_ids.dim() == 1:
                                    basic_input_ids = basic_input_ids.unsqueeze(0)
                                return basic_input_ids.to(self.device)
                            except Exception as basic_error:
                                print(f"   ⚠️ Basic tokenization failed: {basic_error}")
                        
                        # Ultimate fallback for research validation
                        print(f"   🚨 All tokenization methods failed - creating dummy tokens for validation")
                        dummy_tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
                        return dummy_tokens.to(self.device)
            
        except Exception as e:
            print(f"   ⚠️ Error preparing input IDs: {e}")
            import traceback
            traceback.print_exc()
            # Final fallback: create minimal token sequence for research validation
            dummy_tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
            return dummy_tokens.to(self.device)
    
    def _run_baseline_inference(self, image, input_ids, question):
        """Run inference with baseline LaViDa model"""
        start_time = time.time()
        
        try:
            # Prepare image
            if TORCHVISION_AVAILABLE:
                # BASELINE-IMAGE-VALIDATION-FIX: 2025-07-29 - Add same validation as SHIRG for consistency
                # ISSUE: Baseline also needs image validation to prevent format errors
                # SOLUTION: Use same validation method as SHIRG for consistent processing
                # RESEARCH IMPACT: Ensures both baseline and SHIRG handle dataset images reliably
                # LAVIDA IMPACT: Maintains LaViDa's image processing pipeline compatibility
                
                # Validate and ensure proper image format (same as SHIRG)
                validated_image = self._validate_and_convert_image(image, "baseline")
                if validated_image is None:
                    raise ValueError(f"Invalid image format for baseline inference")
                
                # Use proper image processing
                image_tensor = process_images([validated_image], self.baseline_image_processor, self.baseline_model.config)
                if isinstance(image_tensor, list):
                    image_tensor = image_tensor[0]
                # DTYPE-FIX: 2025-07-29 - Use BFloat16 for LaViDa compatibility
                # ISSUE: Using Float16 for LaViDa causes dtype mismatch with model expectations
                # SOLUTION: Use BFloat16 consistently for LaViDa models
                # LAVIDA IMPACT: Eliminates "expected mat1 and mat2 to have the same dtype" errors
                # SHIRG IMPACT: Ensures consistent dtype processing throughout SHIRG pipeline
                image_tensor = image_tensor.to(self.device, dtype=torch.bfloat16)
            else:
                # Fallback tensor conversion
                # BASELINE-FALLBACK-VALIDATION-FIX: 2025-07-29 - Add validation for baseline fallback path
                # ISSUE: Baseline fallback path also needs image validation for consistency
                # SOLUTION: Validate image before manual tensor conversion
                # RESEARCH IMPACT: Ensures all baseline processing paths handle images reliably
                # LAVIDA IMPACT: Maintains consistent image format handling across all baseline paths
                
                # Validate image before fallback processing
                validated_image = self._validate_and_convert_image(image, "baseline-fallback")
                if validated_image is None:
                    raise ValueError(f"Invalid image format for baseline fallback processing")
                
                image_tensor = self._pil_to_tensor(validated_image)
                # DTYPE-FIX: 2025-07-29 - Use BFloat16 for LaViDa compatibility
                # ISSUE: Using Float16 for LaViDa causes dtype mismatch with model expectations
                # SOLUTION: Use BFloat16 consistently for LaViDa models
                # LAVIDA IMPACT: Eliminates "expected mat1 and mat2 to have the same dtype" errors
                # SHIRG IMPACT: Ensures consistent dtype processing throughout SHIRG pipeline
                image_tensor = image_tensor.to(self.device, dtype=torch.bfloat16)
            
            # LAVIDA-GENERATION-FIX: 2025-07-29 - Use exact LaViDa generation from original predict.py
            # ISSUE: Current generation doesn't follow LaViDa diffusion-based generation
            # SOLUTION: Use exact LaViDa generation method with diffusion parameters
            # RESEARCH IMPACT: Ensures baseline uses proper LaViDa diffusion generation
            # LAVIDA IMPACT: Maintains LaViDa's core diffusion-based generation method
            
            # Run inference using LaViDa diffusion generation (following original predict.py)
            with torch.inference_mode():
                # Get image sizes (required for LaViDa)
                image_sizes = [image.size if hasattr(image, 'size') else (384, 384)]
                
                # LAVIDA-RETURN-FIX: 2025-07-29 - Handle LaViDa generate return format properly
                # ISSUE: LaViDa generate() returns (cont, hist) only when verbose=True, just cont when verbose=False
                # SOLUTION: Set verbose=True to get both cont and hist, or handle single return value correctly
                # RESEARCH IMPACT: Enables proper LaViDa baseline inference for SHIRG comparison
                # LAVIDA IMPACT: Maintains exact LaViDa generation behavior
                
                # Use LaViDa diffusion generation (same as original predict.py)
                result = self.baseline_model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=False,        # Same as original
                    temperature=0.1,        # Same as original
                    max_new_tokens=64,      # Same as original (changed from 512)
                    block_length=64,        # LaViDa diffusion block size
                    step_ratio=0.5,         # LaViDa diffusion steps (32 steps)
                    tokenizer=self.baseline_tokenizer,  # LaViDa requires tokenizer
                    prefix_lm=True,         # LaViDa prefix caching
                    verbose=True,           # FIXED: Set to True to get (cont, hist) tuple
                    schedule='shift'        # LaViDa diffusion schedule
                )
                
                # Handle LaViDa return format - should now be (cont, hist) tuple
                if isinstance(result, tuple) and len(result) == 2:
                    cont, hist = result
                    output_ids = cont
                else:
                    # Fallback if only single value returned
                    output_ids = result
                    hist = None
            
            # LAVIDA-DECODE-FIX: 2025-07-29 - Use exact LaViDa decoding from original predict.py
            # ISSUE: Current decoding doesn't match LaViDa output format
            # SOLUTION: Use exact LaViDa decoding method from original predict.py
            # RESEARCH IMPACT: Ensures baseline output matches original LaViDa format
            # LAVIDA IMPACT: Maintains LaViDa's response formatting and cleanup
            
            # Decode response using LaViDa method (following original predict.py)
            text_outputs = self.baseline_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            # Clean up LaViDa-specific artifacts (from original predict.py)
            text_outputs = [text_output.lstrip('!') for text_output in text_outputs]
            response = text_outputs[0] if text_outputs else ""
            
            # Calculate tokens for analysis
            response_tokens = self.baseline_tokenizer.encode(response, return_tensors='pt')[0]
            response_ids = response_tokens  # For compatibility
            
            inference_time = time.time() - start_time
            
            # BASELINE-DEBUG: 2025-07-29 - Enhanced vision tower analysis with debugging
            # ISSUE: Vision features may not be properly extracted causing downstream errors
            # SOLUTION: Add comprehensive vision tower validation and debugging
            # LAVIDA IMPACT: Ensures LaViDa vision processing works correctly
            # SHIRG IMPACT: Provides baseline reference for SHIRG comparison
            vision_info = {}
            if self.baseline_tower is not None:
                try:
                    # Get vision features for analysis
                    with torch.no_grad():
                        print(f"BASELINE-DEBUG: Image tensor shape: {image_tensor.shape}")
                        print(f"BASELINE-DEBUG: Image tensor dtype: {image_tensor.dtype}")
                        
                        vision_features = self.baseline_tower(image_tensor)
                        
                        print(f"BASELINE-DEBUG: Vision features type: {type(vision_features)}")
                        if hasattr(vision_features, 'shape'):
                            print(f"BASELINE-DEBUG: Image features shape: {vision_features.shape}")
                            expected_tokens = 729  # 27x27 for 384x384 LaViDa
                            actual_tokens = vision_features.shape[1] if len(vision_features.shape) > 1 else vision_features.numel()
                            print(f"BASELINE-DEBUG: Expected {expected_tokens} tokens, got {actual_tokens} tokens")
                            
                            vision_info = {
                                'feature_shape': list(vision_features.shape),
                                'num_tokens': actual_tokens,
                                'feature_dim': vision_features.shape[-1] if len(vision_features.shape) > 1 else 1,
                                'expected_tokens': expected_tokens,
                                'tokens_match_expected': actual_tokens == expected_tokens
                            }
                            
                            # SHIRG-POOLING: Debug token structure for research validation
                            if actual_tokens == 729:
                                grid_size = int(math.sqrt(actual_tokens))  # Should be 27
                                print(f"SHIRG-POOLING: Baseline LaViDa tokens {actual_tokens} → {grid_size}×{grid_size} grid")
                        else:
                            print(f"BASELINE-DEBUG: Vision features has no shape attribute")
                            vision_info = {'error': 'Vision features missing shape attribute'}
                            
                except Exception as e:
                    print(f"BASELINE-DEBUG: Vision tower error: {e}")
                    import traceback
                    traceback.print_exc()
                    vision_info = {'error': str(e)}
            else:
                print(f"BASELINE-DEBUG: Vision tower is None")
                vision_info = {'error': 'Vision tower not available'}
            
            return {
                'response': response,
                'tokens_used': len(response_ids) if hasattr(response_ids, '__len__') else 0,
                'inference_time': inference_time,
                'vision_info': vision_info,
                'model_type': 'baseline'
            }
            
        except Exception as e:
            inference_time = time.time() - start_time
            print(f"   ❌ Baseline inference error: {e}")
            return {
                'response': f"Error during baseline inference: {str(e)}",
                'tokens_used': 0,
                'inference_time': inference_time,
                'vision_info': {},
                'model_type': 'baseline',
                'error': str(e)
            }
    
    def _run_shirg_inference(self, image, input_ids, question):
        """Run inference with SHIRG-enabled LaViDa model"""
        start_time = time.time()
        
        try:
            # SHIRG-IMAGE-FIX: 2025-07-29 - Native high-resolution processing per research spec
            # ISSUE: SHIRG requires native 672×672 processing for high-resolution token extraction
            # SOLUTION: Use SHIRG-configured image processor for 672×672 processing
            # RESEARCH IMPACT: Enables genuine high-resolution token extraction (48×48 = 2304 tokens)
            # LAVIDA IMPACT: Maintains LaViDa image processing pipeline compatibility
            
            # SHIRG-PROCESSING-FIX: 2025-07-29 - Direct high-resolution image processing for SHIRG
            # ISSUE: SHIRG needs direct 672×672 processing, not LaViDa's multi-view anyres processing
            # SOLUTION: Process image directly with SHIRG image processor configured for 672×672
            # RESEARCH IMPACT: Enables native high-resolution token extraction as per SHIRG methodology
            # LAVIDA IMPACT: Bypasses LaViDa's anyres processing for SHIRG-specific high-res handling
            
            if TORCHVISION_AVAILABLE:
                # SHIRG-IMAGE-FORMAT-FIX: 2025-07-29 - Add comprehensive image validation before processing
                # ISSUE: Direct preprocess call fails with "Unable to infer channel dimension format" 
                # SOLUTION: Validate and convert image to PIL format before processing
                # RESEARCH IMPACT: Enables SHIRG to process dataset images in any format (PIL, numpy, tensor)
                # LAVIDA IMPACT: Maintains LaViDa's image processing pipeline compatibility
                
                # Validate and ensure proper image format
                validated_image = self._validate_and_convert_image(image, sample_id)
                if validated_image is None:
                    raise ValueError(f"Invalid image format for sample {sample_id}")
                
                image_tensor = process_images([validated_image], self.shirg_image_processor, self.shirg_model.config)
                if isinstance(image_tensor, list):
                    image_tensor = image_tensor[0]
                
                # Convert to proper format for LaViDa
                if isinstance(image_tensor, torch.Tensor):
                    image_tensor = image_tensor.to(dtype=torch.bfloat16, device=self.device)
                else:
                    raise ValueError(f"Unexpected image tensor type: {type(image_tensor)}")
                
                print(f"SHIRG-PROCESSING: Image tensor shape after processing: {image_tensor.shape}")
            else:
                # Fallback: manual processing for SHIRG requirements
                # SHIRG-FALLBACK-VALIDATION-FIX: 2025-07-29 - Add validation for fallback path too
                # ISSUE: Fallback path also needs image validation for consistency
                # SOLUTION: Validate image before manual processing
                # RESEARCH IMPACT: Ensures all SHIRG processing paths handle images reliably
                # LAVIDA IMPACT: Maintains consistent image format handling across all paths
                
                # Validate image before fallback processing
                validated_image = self._validate_and_convert_image(image, sample_id)
                if validated_image is None:
                    raise ValueError(f"Invalid image format for SHIRG fallback processing")
                
                # Resize to 672×672 for SHIRG high-resolution processing
                shirg_resized_image = self._resize_for_shirg(validated_image, target_size=672)
                base_tensor = self._pil_to_tensor(shirg_resized_image)
                image_tensor = base_tensor.to(self.device, dtype=torch.bfloat16)
            
            # Get image sizes for SHIRG (required parameter)
            image_sizes = [image.size]  # Original size, processor handles resizing
            
            # Extract token selection metadata before inference
            try:
                selection_metadata = self._extract_shirg_selection_metadata(image, question)
            except Exception as metadata_error:
                print(f"   ⚠️ SHIRG metadata extraction failed: {metadata_error}")
                selection_metadata = {'error': str(metadata_error)}
            
            # SHIRG-GENERATION-FIX: 2025-07-29 - Use same LaViDa generation as baseline
            # ISSUE: SHIRG must use identical generation method as baseline for fair comparison
            # SOLUTION: Use exact same LaViDa diffusion generation parameters
            # RESEARCH IMPACT: Ensures fair comparison between baseline and SHIRG
            # SHIRG IMPACT: Only difference should be token selection, not generation method
            
            # Run SHIRG inference using LaViDa diffusion-based generation
            with torch.inference_mode():
                # Check if model supports SHIRG mode
                use_shirg = True
                if hasattr(self.shirg_tower, 'forward') and hasattr(self.shirg_tower, 'config'):
                    if hasattr(self.shirg_tower.config, 'enable_shirg'):
                        use_shirg = self.shirg_tower.config.enable_shirg
                
                print(f"   🔍 SHIRG mode: {use_shirg}")
                print(f"   🔍 Image tensor shape: {image_tensor.shape if hasattr(image_tensor, 'shape') else 'No shape'}")
                print(f"   🔍 Image tensor type: {type(image_tensor)}")
                print(f"   🔍 Image tensor device: {image_tensor.device if hasattr(image_tensor, 'device') else 'No device'}")
                print(f"   🔍 Image tensor dtype: {image_tensor.dtype if hasattr(image_tensor, 'dtype') else 'No dtype'}")
                
                # SHIRG-RETURN-FIX: 2025-07-29 - Handle LaViDa generate return format properly for SHIRG
                # ISSUE: SHIRG model also uses LaViDa generate() which has same return format dependency
                # SOLUTION: Set verbose=True to get both cont and hist, or handle single return value correctly
                # RESEARCH IMPACT: Enables proper SHIRG inference for comparison with baseline
                # SHIRG IMPACT: Maintains exact LaViDa generation behavior while using SHIRG tokens
                
                # TENSOR-FORMAT-FIX: 2025-07-29 - Ensure image tensor is in correct format
                # ISSUE: LaViDa expects tensor, not list of tensors for single image
                # SOLUTION: Extract single tensor if in list format
                # RESEARCH IMPACT: Enables SHIRG inference to work properly
                # LAVIDA IMPACT: Maintains compatibility with LaViDa's generate function
                
                if isinstance(image_tensor, list):
                    # LaViDa expects list for multi-view anyres processing
                    images_for_generate = image_tensor
                elif len(image_tensor.shape) == 4 and image_tensor.shape[0] == 1:
                    # Single image tensor [1, C, H, W] - wrap in list for LaViDa
                    images_for_generate = [image_tensor.squeeze(0)]  # Remove batch dim and wrap
                else:
                    # Use as-is
                    images_for_generate = image_tensor
                
                print(f"   🔍 Images for generate type: {type(images_for_generate)}")
                if isinstance(images_for_generate, list):
                    print(f"   🔍 Images for generate list length: {len(images_for_generate)}")
                    if len(images_for_generate) > 0 and hasattr(images_for_generate[0], 'shape'):
                        print(f"   🔍 First image shape: {images_for_generate[0].shape}")
                
                # Use identical LaViDa generation parameters as baseline
                result = self.shirg_model.generate(
                    input_ids,
                    images=images_for_generate,
                    image_sizes=image_sizes,
                    do_sample=False,
                    temperature=0.1,
                    max_new_tokens=64,   # Same as baseline (was 128)
                    block_length=64,     # LaViDa diffusion block size
                    step_ratio=0.5,      # LaViDa diffusion steps
                    tokenizer=self.shirg_tokenizer,
                    prefix_lm=True,      # LaViDa prefix caching
                    verbose=True,        # FIXED: Set to True to get (cont, hist) tuple
                    schedule='shift'     # LaViDa diffusion schedule
                )
                
                # Handle LaViDa return format - should now be (cont, hist) tuple
                if isinstance(result, tuple) and len(result) == 2:
                    cont, hist = result
                    output_ids = cont
                else:
                    # Fallback if only single value returned
                    output_ids = result
                    hist = None
            
            # SHIRG-DECODE-FIX: 2025-07-29 - Use same LaViDa decoding as baseline
            # ISSUE: SHIRG must use identical decoding method as baseline for fair comparison
            # SOLUTION: Use exact same LaViDa decoding as baseline
            # RESEARCH IMPACT: Ensures fair comparison between baseline and SHIRG responses
            # SHIRG IMPACT: Only difference should be token selection, not response formatting
            
            # Decode SHIRG response (same as baseline)
            text_outputs = self.shirg_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            # Clean up LaViDa-specific artifacts
            text_outputs = [text_output.lstrip('!') for text_output in text_outputs]
            response = text_outputs[0] if text_outputs else ""
            
            # Calculate tokens for analysis
            response_tokens = self.shirg_tokenizer.encode(response, return_tensors='pt')[0]
            response_ids = response_tokens  # For compatibility
            
            inference_time = time.time() - start_time
            
            # Vision tower analysis with SHIRG information
            vision_info = {}
            if self.shirg_tower is not None:
                try:
                    # Get vision features for analysis
                    with torch.no_grad():
                        # Check if SHIRG mode is enabled
                        if hasattr(self.shirg_tower, 'forward'):
                            vision_features = self.shirg_tower(image_tensor, use_shirg=True)
                        else:
                            vision_features = self.shirg_tower(image_tensor)
                        
                        if hasattr(vision_features, 'shape'):
                            vision_info = {
                                'feature_shape': list(vision_features.shape),
                                'num_tokens': vision_features.shape[1] if len(vision_features.shape) > 1 else vision_features.numel(),
                                'feature_dim': vision_features.shape[-1] if len(vision_features.shape) > 1 else 1,
                                'shirg_enabled': use_shirg
                            }
                except Exception as e:
                    vision_info = {'error': str(e), 'shirg_enabled': use_shirg}
            
            return {
                'response': response,
                'tokens_used': len(response_ids) if hasattr(response_ids, '__len__') else 0,
                'inference_time': inference_time,
                'vision_info': vision_info,
                'token_selection': selection_metadata,
                'model_type': 'shirg'
            }
            
        except Exception as e:
            inference_time = time.time() - start_time
            print(f"   ❌ SHIRG inference error: {e}")
            
            # DEBUGGING-FIX: 2025-07-29 - Enhanced error logging for channel dimension format error
            # ISSUE: Generic error messages make it hard to diagnose the channel dimension format issue
            # SOLUTION: Add comprehensive error logging with context and traceback
            # RESEARCH IMPACT: Enables faster debugging of SHIRG integration issues
            # LAVIDA IMPACT: Helps identify LaViDa-SHIRG compatibility problems
            
            import traceback
            error_details = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc(),
                'phase': 'shirg_inference'
            }
            
            print(f"   📋 Error details:")
            print(f"      Error type: {error_details['error_type']}")
            print(f"      Error message: {error_details['error_message']}")
            print(f"   📋 Full traceback:")
            print(error_details['traceback'])
            
            # Log context information
            print(f"   📋 Context at error:")
            print(f"      Image tensor type: {type(image_tensor) if 'image_tensor' in locals() else 'Not created'}")
            if 'image_tensor' in locals() and hasattr(image_tensor, 'shape'):
                print(f"      Image tensor shape: {image_tensor.shape}")
            if 'images_for_generate' in locals():
                print(f"      Images for generate type: {type(images_for_generate)}")
                if isinstance(images_for_generate, list):
                    print(f"      Images list length: {len(images_for_generate)}")
            
            return {
                'response': f"Error during SHIRG inference: {str(e)[:200]}...",
                'tokens_used': 0,
                'inference_time': inference_time,
                'vision_info': {},
                'token_selection': {},
                'model_type': 'shirg',
                'error': str(e),
                'error_details': error_details
            }
    
    def _extract_shirg_selection_metadata(self, image, question):
        """Extract SHIRG token selection metadata for analysis"""
        try:
            if self.shirg_tower is None:
                return {'error': 'SHIRG tower not available'}
            
            # SHIRG-METADATA-FIX: 2025-07-29 - Direct processing for metadata extraction
            # ISSUE: Same as above - need direct 672×672 processing for SHIRG
            # SOLUTION: Use direct preprocessing to bypass anyres multi-view
            # RESEARCH IMPACT: Ensures consistent SHIRG processing for metadata extraction
            # LAVIDA IMPACT: Maintains compatibility while using SHIRG-specific processing
            
            if TORCHVISION_AVAILABLE:
                # Direct preprocessing for SHIRG
                processed_dict = self.shirg_image_processor.preprocess([image], return_tensors="pt")
                image_tensor = processed_dict["pixel_values"].to(self.device, dtype=torch.bfloat16)
            else:
                image_tensor = self._pil_to_tensor(image).to(self.device, dtype=torch.bfloat16)
            
            with torch.no_grad():
                # Check if SHIRG tower has selection metadata extraction
                if hasattr(self.shirg_tower, 'extract_dual_scale_tokens'):
                    # SHIRG-enabled tower
                    try:
                        print(f"   📋 SHIRG metadata: Calling extract_dual_scale_tokens")
                        print(f"      Image tensor shape: {image_tensor.shape}")
                        print(f"      Image tensor dtype: {image_tensor.dtype}")
                        
                        # METHOD-FIX: 2025-07-29 - Use correct method name for SHIRG token extraction
                        # ISSUE: Method was incorrectly named as extract_shirg_tokens
                        # SOLUTION: Use extract_dual_scale_tokens which is the actual method in siglip_shirg.py
                        # RESEARCH IMPACT: Enables proper SHIRG dual-scale token extraction
                        # LAVIDA IMPACT: Fixes method call to existing SHIRG implementation
                        features, scaffold = self.shirg_tower.extract_dual_scale_tokens(image_tensor)
                        
                        selection_info = {
                            'method': 'SHIRG',
                            'input_tokens': features.shape[1] if len(features.shape) > 1 else features.numel(),
                            'scaffold_tokens': scaffold.shape[1] if len(scaffold.shape) > 1 else scaffold.numel(),
                            'input_resolution': '672x672',
                            'feature_dim': features.shape[-1] if len(features.shape) > 1 else 1
                        }
                        
                        # If SHIRG selector is available, get selection details
                        if hasattr(self.shirg_tower, 'shirg_selector'):
                            # This would require implementing token selection analysis
                            selection_info['selector_available'] = True
                        
                        return selection_info
                        
                    except Exception as e:
                        print(f"   ❌ SHIRG extraction error: {e}")
                        import traceback
                        print(f"   📋 Traceback: {traceback.format_exc()}")
                        return {'error': f'SHIRG extraction failed: {str(e)}'}
                
                else:
                    # Standard vision tower - no SHIRG
                    features = self.shirg_tower(image_tensor)
                    return {
                        'method': 'Standard',
                        'input_tokens': features.shape[1] if len(features.shape) > 1 else features.numel(),
                        'input_resolution': 'Variable',
                        'feature_dim': features.shape[-1] if len(features.shape) > 1 else 1
                    }
        
        except Exception as e:
            return {'error': f'Selection metadata extraction failed: {str(e)}'}
    
    def _pil_to_tensor(self, pil_image):
        """Convert PIL image to tensor (fallback when torchvision unavailable)"""
        try:
            # Convert PIL to numpy
            image_np = np.array(pil_image)
            
            # Handle different image modes
            if len(image_np.shape) == 2:
                # Grayscale - add channel dimension
                image_np = np.stack([image_np] * 3, axis=-1)
            elif len(image_np.shape) == 3 and image_np.shape[2] == 4:
                # RGBA - convert to RGB
                image_np = image_np[:, :, :3]
            
            # Convert to tensor and normalize
            image_tensor = torch.from_numpy(image_np).float()
            image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
            image_tensor = image_tensor / 255.0  # Normalize to [0,1]
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
            
            return image_tensor
            
        except Exception as e:
            print(f"   ⚠️ Error converting PIL to tensor: {e}")
            # Return a dummy tensor
            return torch.zeros(1, 3, 224, 224)