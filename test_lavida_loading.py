#!/usr/bin/env python3
"""
Test LaViDa Model Loading
Quick test to validate LaViDa model loading fixes before running full SHIRG validation
"""

import os
import sys
import torch
import warnings

warnings.filterwarnings('ignore')

# Add paths for imports
sys.path.append('./shirg/')
sys.path.append('./llava/')
sys.path.append('./')

def test_lavida_loading():
    """Test basic LaViDa model loading with fixes"""
    print("🔍 TESTING LAVIDA MODEL LOADING")
    print("=" * 50)
    
    try:
        # Import LaViDa components
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        
        print("✅ LaViDa imports successful")
        
        # Model configuration
        pretrained_path = "KonstantinosKK/lavida-llada-v1.0-instruct-hf-transformers"
        model_name = "llava_llada"
        
        # Vision configuration
        vision_kwargs = {
            'mm_vision_tower': "google/siglip-so400m-patch14-384",
            'mm_resampler_type': None,
            'mm_projector_type': 'mlp2x_gelu',
            'mm_hidden_size': 1152,
            'use_mm_proj': True,
            'enable_shirg': False  # Start with SHIRG disabled for baseline test
        }
        
        # Load with proper settings
        device_map_setting = "auto" if torch.cuda.is_available() else None
        torch_dtype_setting = "bfloat16" if torch.cuda.is_available() else "float32"
        
        print(f"🔄 Loading model: {pretrained_path}")
        print(f"   Device map: {device_map_setting}")
        print(f"   Torch dtype: {torch_dtype_setting}")
        print(f"   Vision tower: {vision_kwargs['mm_vision_tower']}")
        
        # Load model
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            pretrained_path,
            None,
            model_name,
            device_map=device_map_setting,
            torch_dtype=torch_dtype_setting,
            **vision_kwargs
        )
        
        print("✅ Model loaded successfully!")
        
        # Test model components
        model.eval()
        if hasattr(model, 'tie_weights'):
            model.tie_weights()
        
        # Get vision tower
        vision_tower = model.get_vision_tower()
        
        print(f"📊 Model Information:")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Model device: {next(model.parameters()).device}")
        print(f"   Model dtype: {next(model.parameters()).dtype}")
        print(f"   Vision tower: {vision_tower.vision_tower_name if vision_tower else 'None'}")
        print(f"   Vision tower loaded: {vision_tower.is_loaded if vision_tower else False}")
        print(f"   Tokenizer vocab size: {len(tokenizer)}")
        print(f"   Image processor: {type(image_processor).__name__}")
        
        # Test basic GPU memory if available
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            print(f"   GPU memory allocated: {memory_allocated:.2f} GB")
        
        print("\n🎉 LaViDa model loading test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ LaViDa model loading test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vision_tower_loading():
    """Test vision tower loading specifically"""
    print("\n🔍 TESTING VISION TOWER LOADING")
    print("=" * 50)
    
    try:
        from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
        from llava.model.multimodal_encoder.siglip_base import SigLipVisionConfig
        
        print("✅ SigLIP imports successful")
        
        # Test vision tower loading
        vision_tower_name = "google/siglip-so400m-patch14-384"
        config = SigLipVisionConfig()
        config.enable_shirg = True
        
        print(f"🔄 Loading vision tower: {vision_tower_name}")
        
        vision_tower = SigLipVisionTower(vision_tower_name, config, delay_load=False)
        
        print("✅ Vision tower loaded successfully!")
        print(f"   Tower name: {vision_tower.vision_tower_name}")
        print(f"   Is loaded: {vision_tower.is_loaded}")
        print(f"   SHIRG enabled: {vision_tower.shirg_enabled}")
        
        print("\n🎉 Vision tower loading test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Vision tower loading test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 LAVIDA LOADING VALIDATION TESTS")
    print("=" * 60)
    
    # Test LaViDa model loading
    lavida_success = test_lavida_loading()
    
    # Test vision tower loading
    vision_success = test_vision_tower_loading()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    print(f"LaViDa Model Loading: {'✅ PASS' if lavida_success else '❌ FAIL'}")
    print(f"Vision Tower Loading: {'✅ PASS' if vision_success else '❌ FAIL'}")
    
    if lavida_success and vision_success:
        print("\n🎉 ALL TESTS PASSED! Ready for SHIRG validation.")
        print("   Run: python shirg/real_ocr_vqa_validation.py")
    else:
        print("\n❌ SOME TESTS FAILED! Fix issues before proceeding.")
        
    print("=" * 60)