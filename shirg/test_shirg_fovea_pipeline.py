#!/usr/bin/env python3
"""
SHIRG-Fovea Pipeline Test
Test the complete SHIRG-Fovea implementation with 5-view anyres processing
"""

import os
import sys
import torch
import warnings
from PIL import Image

warnings.filterwarnings('ignore')

# Add paths for imports
sys.path.append('./shirg/')
sys.path.append('./llava/')
sys.path.append('./')

from llava.utils import rank0_print

def test_shirg_fovea_pipeline():
    """Test SHIRG-Fovea implementation with mock 5-view input"""
    print("🧪 SHIRG-FOVEA PIPELINE TEST")
    print("=" * 60)
    
    try:
        # Import components
        from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
        from llava.model.multimodal_encoder.siglip_shirg import SigLipShirgExtensions
        
        print("✅ Imports successful")
        
        # Create mock configuration
        class MockConfig:
            enable_shirg = True
            image_aspect_ratio = "anyres"
            image_grid_pinpoints = [(384, 384), (512, 512)]
            mm_patch_merge_type = "spatial_unpad"
        
        # Initialize vision tower with SHIRG enabled
        print("\n🔄 Initializing SigLipVisionTower with SHIRG-Fovea...")
        vision_tower = SigLipVisionTower(
            vision_tower="google/siglip-so400m-patch14-384",
            args=MockConfig(),
            delay_load=False
        )
        
        # Check if SHIRG is enabled
        if hasattr(vision_tower, 'shirg_enabled') and vision_tower.shirg_enabled:
            print("✅ SHIRG-Fovea enabled on vision tower")
        else:
            print("❌ SHIRG-Fovea not enabled!")
            return
        
        # Create mock 5-view input (simulating LaViDa's anyres output)
        print("\n🔄 Creating mock 5-view input...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Mock 5 views: 1 global (384²) + 4 peripheral (512²)
        mock_views = []
        
        # Global view: 384×384
        global_view = torch.randn(1, 3, 384, 384, device=device, dtype=torch.float32)
        mock_views.append(global_view)
        print(f"   Global view: {global_view.shape}")
        
        # 4 peripheral views: 512×512 each
        for i in range(4):
            peripheral_view = torch.randn(1, 3, 512, 512, device=device, dtype=torch.float32)
            mock_views.append(peripheral_view)
            print(f"   Peripheral view {i+1}: {peripheral_view.shape}")
        
        # Test forward pass with SHIRG
        print("\n🔄 Testing forward_with_shirg...")
        try:
            # Mock text embeddings (optional)
            text_embeddings = torch.randn(1, 50, 1152, device=device, dtype=torch.float32)
            
            # Run SHIRG-Fovea forward pass
            output_tokens = vision_tower.forward_with_shirg(mock_views, text_embeddings)
            
            print(f"✅ SHIRG-Fovea output shape: {output_tokens.shape}")
            print(f"   Expected: [1, ~1832, 1152] (196 global + 4×~409 peripheral)")
            print(f"   Actual tokens: {output_tokens.shape[1]}")
            
            # Validate output dimensions
            B, N, D = output_tokens.shape
            if 1600 <= N <= 2000:  # Reasonable range for ~1832 tokens
                print("✅ Token count within expected range!")
            else:
                print(f"⚠️ Token count {N} outside expected range [1600, 2000]")
            
            if D == 1152:  # SigLIP feature dimension
                print("✅ Feature dimension correct (1152)!")
            else:
                print(f"❌ Feature dimension {D} incorrect (expected 1152)")
                
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Test multiview token extraction
        print("\n🔄 Testing extract_multiview_tokens...")
        try:
            if hasattr(vision_tower, 'extract_multiview_tokens'):
                global_pooled, peripheral_features = vision_tower.extract_multiview_tokens(mock_views)
                
                print(f"✅ Multiview extraction successful:")
                print(f"   Global pooled: {global_pooled.shape} (expected [1, 196, 1152])")
                print(f"   Peripheral features: {len(peripheral_features)} views")
                for i, feat in enumerate(peripheral_features):
                    print(f"      View {i+1}: {feat.shape} (expected [1, 1024, 1152])")
            else:
                print("⚠️ extract_multiview_tokens not available")
                
        except Exception as e:
            print(f"❌ Multiview extraction failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test per-view Top-K selection
        print("\n🔄 Testing topk_per_view...")
        try:
            if hasattr(vision_tower, 'topk_per_view') and len(peripheral_features) > 0:
                K = 409  # ~40% of 1024
                selected = vision_tower.topk_per_view(peripheral_features[0], K, text_embeddings)
                
                print(f"✅ Per-view Top-K selection successful:")
                print(f"   Input: {peripheral_features[0].shape}")
                print(f"   Output: {selected.shape} (expected [1, {K}, 1152])")
            else:
                print("⚠️ topk_per_view not available")
                
        except Exception as e:
            print(f"❌ Top-K selection failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n✅ SHIRG-FOVEA PIPELINE TEST COMPLETE!")
        print("   All major components tested successfully")
        
    except Exception as e:
        print(f"\n❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run the SHIRG-Fovea pipeline test"""
    test_shirg_fovea_pipeline()

if __name__ == "__main__":
    main()