#!/usr/bin/env python3
"""
Final SHIRG Fix Validation

Tests the final corrected implementation:
- Same encoder, different layer outputs (layer -2 vs layer -1)
- Same resolution (384x384)
- No separate model loading
- Fast performance expected

Usage: python shirg/validate_final_fix.py
"""

import os
import sys
import torch
import torch.nn.functional as F
import time

# Add paths
sys.path.append('./')
sys.path.append('./shirg')

try:
    from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
    from llava.utils import rank0_print
    print("✅ LaViDa imports successful")
except ImportError as e:
    print(f"❌ LaViDa imports failed: {e}")
    sys.exit(1)

def main():
    print("🔬 Final SHIRG Fix Validation")
    print("=" * 50)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16
    
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    
    # Load vision tower
    print("\n🔄 Loading vision tower...")
    vision_tower = SigLipVisionTower(
        vision_tower="google/siglip-so400m-patch14-384",
        vision_tower_cfg=None,
        delay_load=False
    )
    
    # Create test image
    test_image = torch.randn(2, 3, 384, 384, dtype=dtype, device=device)
    print(f"\n🖼️ Test image: {test_image.shape}")
    
    # Test 1: LaViDa path (layer -2, should be fast since no separate model)
    print("\n📍 Test 1: LaViDa path (layer -2)")
    start_time = time.time()
    lavida_features = vision_tower.forward(test_image)
    lavida_time = time.time() - start_time
    
    print(f"   LaViDa features: {lavida_features.shape}")
    print(f"   Time: {lavida_time*1000:.1f}ms")
    
    # Test 2: SHIRG path (layer -1, should be fast since same forward pass)
    print("\n📍 Test 2: SHIRG path (layer -1)")
    start_time = time.time()
    _, shirg_features = vision_tower.forward_with_high_quality(
        test_image, return_high_quality=True
    )
    shirg_time = time.time() - start_time
    
    print(f"   SHIRG features: {shirg_features.shape}")
    print(f"   Time: {shirg_time*1000:.1f}ms")
    
    # Test 3: Feature comparison (should be high similarity now)
    print("\n📍 Test 3: Feature quality comparison")
    
    # Ensure same shapes
    assert lavida_features.shape == shirg_features.shape, f"Shape mismatch: {lavida_features.shape} vs {shirg_features.shape}"
    
    # Cosine similarity
    lavida_norm = F.normalize(lavida_features, p=2, dim=-1)
    shirg_norm = F.normalize(shirg_features, p=2, dim=-1)
    similarity = torch.sum(lavida_norm * shirg_norm, dim=-1).mean().item()
    
    # Feature stats
    lavida_stats = {
        'mean': lavida_features.mean().item(),
        'std': lavida_features.std().item()
    }
    
    shirg_stats = {
        'mean': shirg_features.mean().item(),
        'std': shirg_features.std().item()
    }
    
    print(f"   Cosine similarity: {similarity:.4f}")
    print(f"   LaViDa stats: mean={lavida_stats['mean']:.4f}, std={lavida_stats['std']:.4f}")
    print(f"   SHIRG stats: mean={shirg_stats['mean']:.4f}, std={shirg_stats['std']:.4f}")
    
    # Success criteria
    high_similarity = similarity > 0.7
    fast_lavida = lavida_time < 2.0  # Should be very fast
    fast_shirg = shirg_time < 5.0   # Should be reasonably fast
    same_shapes = lavida_features.shape == shirg_features.shape
    correct_tokens = lavida_features.shape[1] == 729
    
    print(f"\n🎯 VALIDATION RESULTS:")
    print(f"   High similarity (>0.7): {'✅' if high_similarity else '❌'} ({similarity:.4f})")
    print(f"   Fast LaViDa (<2s): {'✅' if fast_lavida else '❌'} ({lavida_time*1000:.1f}ms)")
    print(f"   Fast SHIRG (<5s): {'✅' if fast_shirg else '❌'} ({shirg_time*1000:.1f}ms)")
    print(f"   Same shapes: {'✅' if same_shapes else '❌'}")
    print(f"   Correct tokens (729): {'✅' if correct_tokens else '❌'}")
    
    # Overall success
    success = all([high_similarity, fast_lavida, fast_shirg, same_shapes, correct_tokens])
    
    print(f"\n{'='*50}")
    print(f"🏁 FINAL VALIDATION: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    if success:
        print("✅ SHIRG implementation is now correct and ready for research!")
        print("✅ Proceed with LoRA training")
    else:
        print("❌ Issues still remain, check output above")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)