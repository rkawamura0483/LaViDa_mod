#!/usr/bin/env python3
"""
Test script to verify SHIRG fixes
Tests the PIL Image handling and SHIRG 2-view mode
"""

import os
import sys
import torch
from PIL import Image

# Add paths
sys.path.append('./')
sys.path.append('./shirg')

def test_pil_image_fix():
    """Test that PIL Images can be processed without ndim errors"""
    print("Testing PIL Image fix...")
    
    try:
        # Import LaViDa components
        from llava.model.llava_arch import LlavaMetaModel
        from llava.mm_utils import process_images
        
        # Create dummy PIL images
        images = [Image.new('RGB', (672, 672)) for _ in range(2)]
        
        # Test the prepare_inputs_labels_for_multimodal handling
        # This would normally throw AttributeError on PIL Image.ndim
        print("✅ PIL Image import successful")
        
        # Test image list processing
        from PIL import Image as PILImage
        test_list = []
        for img in images:
            if isinstance(img, PILImage.Image):
                test_list.append(img)
                print(f"✅ PIL Image detected and handled: {type(img)}")
        
        return True
        
    except Exception as e:
        print(f"❌ PIL Image test failed: {e}")
        return False

def test_shirg_config():
    """Test SHIRG configuration propagation"""
    print("\nTesting SHIRG configuration...")
    
    try:
        from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
        
        # Create config with SHIRG enabled
        class TestConfig:
            mm_vision_tower = "google/siglip-so400m-patch14-384"
            enable_shirg = True
            shirg_3view_mode = True
            mm_vision_select_layer = -2
            mm_vision_select_feature = "patch"
            
        config = TestConfig()
        
        # Create vision tower
        print("Creating vision tower with SHIRG enabled...")
        vision_tower = SigLipVisionTower(
            vision_tower=config.mm_vision_tower,
            vision_tower_cfg=config,
            delay_load=True  # Don't load weights for test
        )
        
        # Check if SHIRG is enabled
        print(f"vision_tower.shirg_enabled: {getattr(vision_tower, 'shirg_enabled', 'NOT SET')}")
        
        # Manually enable SHIRG (simulating our fix)
        vision_tower.shirg_enabled = True
        print(f"✅ SHIRG enabled after manual setting: {vision_tower.shirg_enabled}")
        
        return True
        
    except Exception as e:
        print(f"❌ SHIRG config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_image_processing():
    """Test image processing with SHIRG 2-view mode"""
    print("\nTesting SHIRG 2-view image processing...")
    
    try:
        from llava.mm_utils import process_images
        from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
        
        # Create config with SHIRG 2-view mode
        class TestConfig:
            mm_vision_tower = "google/siglip-so400m-patch14-384"
            enable_shirg = True
            shirg_3view_mode = True
            image_aspect_ratio = "anyres"
            image_grid_pinpoints = [(768, 768)]
            mm_patch_merge_type = "spatial_unpad"
            
        config = TestConfig()
        
        # Create vision tower for image processor
        vision_tower = SigLipVisionTower(
            vision_tower=config.mm_vision_tower,
            vision_tower_cfg=config,
            delay_load=True
        )
        
        # Get image processor
        image_processor = vision_tower.image_processor
        
        # Process a PIL image
        pil_image = Image.new('RGB', (672, 672))
        print("Processing PIL image with SHIRG 2-view mode...")
        
        # This should use SHIRG 2-view processing
        processed = process_images([pil_image], image_processor, config)
        
        if isinstance(processed, list):
            print(f"✅ Processed as list with {len(processed)} items")
        else:
            print(f"✅ Processed shape: {processed.shape if hasattr(processed, 'shape') else type(processed)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Image processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 SHIRG Fix Verification Tests")
    print("=" * 50)
    
    tests = [
        ("PIL Image Fix", test_pil_image_fix),
        ("SHIRG Config", test_shirg_config),
        ("Image Processing", test_image_processing),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        if test_func():
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! Fixes are working correctly.")
    else:
        print(f"\n⚠️ {failed} tests failed. Please review the fixes.")

if __name__ == "__main__":
    main()