#!/usr/bin/env python3
"""
Debug script to trace tensor dimensions in SHIRG high-resolution extraction
"""
import torch
import sys
sys.path.append('./')

try:
    from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
    print("✅ Import successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def debug_dimensions():
    """Debug tensor dimensions step by step"""
    print("🔍 Debugging Tensor Dimensions")
    print("=" * 50)
    
    # Initialize vision tower
    vision_tower = SigLipVisionTower(
        vision_tower="google/siglip-so400m-patch14-384",
        vision_tower_cfg=None,
        delay_load=False
    )
    
    # Create test image batch
    batch_size = 2
    test_image = torch.randn(batch_size, 3, 384, 384, dtype=torch.bfloat16, device='cuda')
    print(f"Input image shape: {test_image.shape}")
    
    # Test baseline forward
    print("\n1. Testing baseline forward...")
    try:
        baseline_features = vision_tower.forward(test_image)
        print(f"   Baseline features shape: {baseline_features.shape}")
    except Exception as e:
        print(f"   ❌ Baseline forward failed: {e}")
        return False
    
    # Test forward_with_high_res 
    print("\n2. Testing forward_with_high_res...")
    try:
        standard_features, high_res_features = vision_tower.forward_with_high_res(
            test_image, return_high_res=True, target_resolution=(768, 768)
        )
        
        print(f"   Standard features type: {type(standard_features)}")
        if isinstance(standard_features, list):
            print(f"   Standard features list length: {len(standard_features)}")
            if len(standard_features) > 0:
                print(f"   Standard features[0] shape: {standard_features[0].shape}")
        else:
            print(f"   Standard features shape: {standard_features.shape}")
            
        print(f"   High-res features type: {type(high_res_features)}")
        if isinstance(high_res_features, list):
            print(f"   High-res features list length: {len(high_res_features)}")
            if len(high_res_features) > 0:
                print(f"   High-res features[0] shape: {high_res_features[0].shape}")
        else:
            print(f"   High-res features shape: {high_res_features.shape}")
            
    except Exception as e:
        print(f"   ❌ forward_with_high_res failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test _extract_high_res_tokens directly
    print("\n3. Testing _extract_high_res_tokens directly...")
    try:
        high_res_tokens = vision_tower._extract_high_res_tokens(
            test_image, target_resolution=(768, 768)
        )
        print(f"   High-res tokens shape: {high_res_tokens.shape}")
    except Exception as e:
        print(f"   ❌ _extract_high_res_tokens failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✅ All dimension tests passed!")
    return True

if __name__ == "__main__":
    success = debug_dimensions()
    sys.exit(0 if success else 1)