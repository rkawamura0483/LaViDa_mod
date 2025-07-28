#!/usr/bin/env python3
"""
SHIRG-FIX-TEST: 2025-07-28 - Quick test for CUDA indexing fixes
ISSUE: Need to verify SHIRG methods work without CUDA errors
SOLUTION: Simple test script to validate token extraction and method availability
VALIDATION IMPACT: Ensures comprehensive validation can proceed without device-side assertions
"""

import torch
import sys
import os

# Add LaViDa paths
sys.path.append('.')
sys.path.append('./llava')

def test_shirg_methods():
    """Test that all required SHIRG methods are available and functional"""
    print("🧪 SHIRG Method Availability Test")
    print("=" * 50)
    
    try:
        from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
        
        # Create test vision tower
        vision_tower = SigLipVisionTower(
            vision_tower="google/siglip-so400m-patch14-384",
            vision_tower_cfg=None,
            delay_load=True
        )
        
        # Check method availability
        required_methods = [
            'forward_with_shirg',
            'get_highres_tokens_for_shirg', 
            'shirg_token_selection',
            'compare_baseline_vs_shirg',
            '_compute_edge_density_boost',
            '_get_coverage_guaranteed_tokens',
            'forward_with_shirg_x',
            'extract_shirg_x_tokens'
        ]
        
        print("✅ Checking method availability:")
        for method in required_methods:
            if hasattr(vision_tower, method):
                print(f"   ✓ {method}")
            else:
                print(f"   ✗ {method} - MISSING!")
                return False
                
        print("\n🎯 All required SHIRG methods are available!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing SHIRG methods: {e}")
        return False

def test_token_extraction():
    """Test token extraction with sample data"""
    print("\n🔍 SHIRG Token Extraction Test")
    print("=" * 50)
    
    try:
        from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
        
        # Create test vision tower (delay_load=True to avoid model loading)
        vision_tower = SigLipVisionTower(
            vision_tower="google/siglip-so400m-patch14-384",
            vision_tower_cfg=None,
            delay_load=True
        )
        
        # Test coordinate computation (doesn't require loaded model)
        print("✅ Testing patch coordinate computation...")
        coords = vision_tower.compute_patch_centroids(H=48, W=48)
        print(f"   ✓ Patch coordinates shape: {coords.shape} (expected: [2304, 4])")
        
        if coords.shape != (2304, 4):
            print(f"   ✗ Wrong coordinate shape! Expected [2304, 4], got {coords.shape}")
            return False
            
        # Test coordinate bounds
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        if x_coords.min() < 0 or x_coords.max() > 1 or y_coords.min() < 0 or y_coords.max() > 1:
            print(f"   ✗ Coordinates out of [0,1] range!")
            return False
        
        print("   ✓ Coordinates are properly normalized to [0,1]")
        
        # Test entropy computation with dummy tokens
        print("✅ Testing patch entropy computation...")
        dummy_tokens = torch.randn(2, 2304, 1152)  # [B, N, D]
        entropy = vision_tower.compute_patch_entropy(dummy_tokens)
        print(f"   ✓ Patch entropy shape: {entropy.shape} (expected: [2])")
        
        if entropy.shape != (2,):
            print(f"   ✗ Wrong entropy shape! Expected [2], got {entropy.shape}")
            return False
            
        print("   ✓ Entropy computation works correctly")
        
        # Test neighbor distance computation
        print("✅ Testing neighbor distance computation...")
        neighbor_dist = vision_tower.compute_neighbor_distances(dummy_tokens, H=48, W=48)
        print(f"   ✓ Neighbor distances shape: {neighbor_dist.shape} (expected: [2, 2304])")
        
        if neighbor_dist.shape != (2, 2304):
            print(f"   ✗ Wrong neighbor distance shape! Expected [2, 2304], got {neighbor_dist.shape}")
            return False
            
        print("   ✓ Neighbor distance computation works correctly")
        
        print("\n🎉 All token extraction tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing token extraction: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cuda_safety():
    """Test CUDA safety measures"""
    print("\n🛡️ CUDA Safety Test")
    print("=" * 50)
    
    try:
        from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
        
        vision_tower = SigLipVisionTower(
            vision_tower="google/siglip-so400m-patch14-384", 
            vision_tower_cfg=None,
            delay_load=True
        )
        
        # Test token merging with edge cases
        print("✅ Testing token merging safety...")
        
        # Create test data with potential indexing issues
        B, N, D = 2, 2304, 1152
        tokens = torch.randn(B, N, D)
        coords = vision_tower.compute_patch_centroids(H=48, W=48)
        scores = torch.randn(B, N)
        
        # Test merge with small epsilon (should not crash)
        merged_tokens, merged_coords = vision_tower.merge_neighboring_tokens(
            tokens, coords, scores, epsilon=0.01
        )
        
        print(f"   ✓ Merged tokens shape: {merged_tokens.shape}")
        print(f"   ✓ Merged coords shape: {merged_coords.shape}")
        
        # Ensure no NaN or Inf values
        if torch.isnan(merged_tokens).any():
            print("   ✗ NaN values in merged tokens!")
            return False
            
        if torch.isinf(merged_tokens).any():
            print("   ✗ Inf values in merged tokens!")
            return False
            
        print("   ✓ No NaN/Inf values in merged tokens")
        
        # Test edge density computation
        print("✅ Testing edge density computation...")
        edge_boost = vision_tower._compute_edge_density_boost(tokens)
        print(f"   ✓ Edge boost shape: {edge_boost.shape} (expected: [2, 2304])")
        
        if edge_boost.shape != (B, N):
            print(f"   ✗ Wrong edge boost shape! Expected [{B}, {N}], got {edge_boost.shape}")
            return False
            
        print("   ✓ Edge density computation safe")
        
        # Test coverage guaranteed tokens
        print("✅ Testing coverage guaranteed selection...")
        guaranteed_tokens = vision_tower._get_coverage_guaranteed_tokens(tokens)
        print(f"   ✓ Guaranteed tokens shape: {guaranteed_tokens.shape}")
        
        # Ensure indices are valid
        if guaranteed_tokens.max() >= N:
            print(f"   ✗ Invalid token indices! Max: {guaranteed_tokens.max()}, should be < {N}")
            return False
            
        print("   ✓ Coverage guaranteed selection safe")
        
        print("\n🛡️ All CUDA safety tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in CUDA safety test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all SHIRG fix tests"""
    print("🚀 SHIRG FIXES VALIDATION")
    print("=" * 60)
    
    all_passed = True
    
    # Test 1: Method availability
    if not test_shirg_methods():
        all_passed = False
    
    # Test 2: Token extraction
    if not test_token_extraction():
        all_passed = False
        
    # Test 3: CUDA safety
    if not test_cuda_safety():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED - SHIRG fixes are working!")
        print("✅ Ready to run comprehensive validation")
    else:
        print("❌ SOME TESTS FAILED - Need to fix remaining issues")
        print("⚠️ Do not run comprehensive validation yet")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)