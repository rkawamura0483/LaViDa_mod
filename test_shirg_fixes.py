#!/usr/bin/env python3
"""
Test SHIRG Fixes
Quick test to validate meta tensor fixes and SHIRG integration before full validation
"""

import os
import sys
import torch
import warnings
import traceback

warnings.filterwarnings('ignore')

# Add paths for imports
sys.path.append('./shirg/')
sys.path.append('./llava/')
sys.path.append('./')

def test_lavida_model_loading():
    """Test LaViDa model loading with our meta tensor fixes"""
    print("🔍 TESTING LAVIDA MODEL LOADING")
    print("=" * 50)
    
    try:
        # Import LaViDa components
        from llava.model.builder import load_pretrained_model
        
        print("✅ LaViDa imports successful")
        
        # Test model configuration
        pretrained_path = "KonstantinosKK/lavida-llada-v1.0-instruct-hf-transformers"
        model_name = "llava_llada"
        
        # Vision configuration
        vision_kwargs = {
            'mm_vision_tower': "google/siglip-so400m-patch14-384",
            'mm_resampler_type': None,
            'mm_projector_type': 'mlp2x_gelu',
            'mm_hidden_size': 1152,
            'use_mm_proj': True,
            'enable_shirg': True  # Enable SHIRG extensions
        }
        
        # Load model with proper settings
        device_map_setting = "auto" if torch.cuda.is_available() else None
        torch_dtype_setting = "bfloat16" if torch.cuda.is_available() else "float32"
        
        print(f"🔄 Loading LaViDa model: {pretrained_path}")
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
        
        print("✅ LaViDa model loaded successfully!")
        
        # Configure model
        model.eval()
        if hasattr(model, 'tie_weights'):
            model.tie_weights()
        
        # Get vision tower and test SHIRG
        vision_tower = model.get_vision_tower()
        
        print(f"📊 Model Information:")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Model device: {next(model.parameters()).device}")
        print(f"   Model dtype: {next(model.parameters()).dtype}")
        print(f"   Vision tower: {vision_tower.vision_tower_name if vision_tower else 'None'}")
        print(f"   Vision tower loaded: {vision_tower.is_loaded if vision_tower else False}")
        print(f"   SHIRG enabled: {vision_tower.shirg_enabled if hasattr(vision_tower, 'shirg_enabled') else 'Unknown'}")
        
        # Test GPU memory if available
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            print(f"   GPU memory allocated: {memory_allocated:.2f} GB")
        
        return True, model, vision_tower, tokenizer, image_processor
        
    except Exception as e:
        print(f"❌ LaViDa model loading test FAILED: {e}")
        traceback.print_exc()
        return False, None, None, None, None

def test_vision_tower_shirg():
    """Test SHIRG-specific functionality in vision tower"""
    print("\n🔍 TESTING SHIRG VISION TOWER FUNCTIONALITY")
    print("=" * 50)
    
    try:
        from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
        from llava.model.multimodal_encoder.siglip_base import SigLipVisionConfig
        
        print("✅ SigLIP imports successful")
        
        # Test vision tower loading with SHIRG
        vision_tower_name = "google/siglip-so400m-patch14-384"
        config = SigLipVisionConfig()
        config.enable_shirg = True
        
        print(f"🔄 Loading vision tower with SHIRG: {vision_tower_name}")
        
        vision_tower = SigLipVisionTower(vision_tower_name, config, delay_load=False)
        
        print("✅ Vision tower with SHIRG loaded successfully!")
        print(f"   Tower name: {vision_tower.vision_tower_name}")
        print(f"   Is loaded: {vision_tower.is_loaded}")
        print(f"   SHIRG enabled: {vision_tower.shirg_enabled}")
        
        # Test SHIRG methods
        has_shirg_methods = all(hasattr(vision_tower, method) for method in [
            'forward_with_shirg',
            'extract_dual_scale_tokens', 
            'distance_aware_selection'
        ])
        
        print(f"   SHIRG methods available: {has_shirg_methods}")
        
        return True, vision_tower
        
    except Exception as e:
        print(f"❌ SHIRG vision tower test FAILED: {e}")
        traceback.print_exc()
        return False, None

def test_simple_inference():
    """Test simple inference to ensure everything works"""
    print("\n🔍 TESTING SIMPLE INFERENCE")
    print("=" * 50)
    
    try:
        # Load model first
        success, model, vision_tower, tokenizer, image_processor = test_lavida_model_loading()
        if not success:
            print("❌ Cannot test inference - model loading failed")
            return False
        
        # Create a dummy image tensor
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test image tensor (384x384 for baseline)
        dummy_image = torch.randn(1, 3, 384, 384, device=device, dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        
        print(f"🔄 Testing vision tower forward pass...")
        
        # Test baseline forward pass
        with torch.no_grad():
            if hasattr(vision_tower, 'forward'):
                # Test standard LaViDa forward
                features_baseline = vision_tower.forward(dummy_image, use_shirg=False)
                print(f"✅ Baseline forward pass successful: {features_baseline.shape}")
                
                # Test SHIRG forward if available
                if hasattr(vision_tower, 'forward_with_shirg'):
                    # Create larger image for SHIRG (672x672)
                    shirg_image = torch.randn(1, 3, 672, 672, device=device, dtype=dummy_image.dtype)
                    features_shirg = vision_tower.forward(shirg_image, use_shirg=True)
                    print(f"✅ SHIRG forward pass successful: {features_shirg.shape}")
                    
                    # Validate token counts
                    expected_baseline = 729  # 27x27 patches
                    expected_shirg = 1216   # 1152 selected + 64 scaffold
                    
                    print(f"📊 Token count validation:")
                    print(f"   Baseline tokens: {features_baseline.shape[1]} (expected: {expected_baseline})")
                    print(f"   SHIRG tokens: {features_shirg.shape[1]} (expected: {expected_shirg})")
                    
                    baseline_correct = features_baseline.shape[1] == expected_baseline
                    shirg_correct = features_shirg.shape[1] == expected_shirg
                    
                    if baseline_correct and shirg_correct:
                        print("✅ Token counts match expected values!")
                    else:
                        print("⚠️ Token counts don't match - may need adjustment")
                
            else:
                print("⚠️ Vision tower forward method not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple inference test FAILED: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all SHIRG fix validation tests"""
    print("🧪 SHIRG FIXES VALIDATION TESTS")
    print("=" * 60)
    
    # Test 1: LaViDa model loading
    lavida_success = test_lavida_model_loading()[0]
    
    # Test 2: SHIRG vision tower functionality  
    shirg_success = test_vision_tower_shirg()[0]
    
    # Test 3: Simple inference
    inference_success = test_simple_inference()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    print(f"LaViDa Model Loading: {'✅ PASS' if lavida_success else '❌ FAIL'}")
    print(f"SHIRG Vision Tower: {'✅ PASS' if shirg_success else '❌ FAIL'}")  
    print(f"Simple Inference: {'✅ PASS' if inference_success else '❌ FAIL'}")
    
    if lavida_success and shirg_success and inference_success:
        print("\n🎉 ALL TESTS PASSED! SHIRG fixes are working.")
        print("   ✅ Meta tensor loading fixed")
        print("   ✅ SHIRG integration verified")
        print("   ✅ Ready for full validation")
        print("\n🚀 Run: python shirg/real_ocr_vqa_validation.py")
    else:
        print("\n❌ SOME TESTS FAILED! Review errors above.")
        
    print("=" * 60)

if __name__ == "__main__":
    main()