#!/usr/bin/env python3
"""
SHIRG Enhanced Token Selection Test
Test the new token selection methods (base, entropy, edge, full) with parameter validation
"""

import os
import sys
import torch
import time
import warnings
import numpy as np
from PIL import Image

warnings.filterwarnings('ignore')

# Add paths for imports
sys.path.append('./shirg/')
sys.path.append('./llava/')
sys.path.append('./')

from llava.utils import rank0_print

def test_enhanced_selection():
    """Test SHIRG enhanced token selection methods"""
    print("🧪 SHIRG ENHANCED TOKEN SELECTION TEST")
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
            image_grid_pinpoints = [(448, 448)]
            mm_patch_merge_type = "spatial_unpad"
        
        # Initialize vision tower with SHIRG enabled
        print("\n🔄 Initializing SigLipVisionTower with SHIRG...")
        vision_tower = SigLipVisionTower(
            vision_tower="google/siglip-so400m-patch14-384",
            args=MockConfig(),
            delay_load=False
        )
        
        # Check if SHIRG is enabled
        if hasattr(vision_tower, 'shirg_enabled') and vision_tower.shirg_enabled:
            print("✅ SHIRG enabled on vision tower")
        else:
            print("❌ SHIRG not enabled!")
            return
        
        # Create mock 2-view input for SHIRG-Fovea
        print("\n🔄 Creating mock 2-view input (global + foveal)...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Mock 2 views: 1 global (448²) + 1 foveal (448²)
        mock_views = []
        
        # Global view: 448×448
        global_view = torch.randn(1, 3, 448, 448, device=device, dtype=torch.float32)
        mock_views.append(global_view)
        print(f"   Global view: {global_view.shape}")
        
        # Foveal view: 448×448
        foveal_view = torch.randn(1, 3, 448, 448, device=device, dtype=torch.float32)
        mock_views.append(foveal_view)
        print(f"   Foveal view: {foveal_view.shape}")
        
        # Mock text embeddings
        text_embeddings = torch.randn(1, 50, 1152, device=device, dtype=torch.float32)
        
        # Test different selection methods
        methods = ['base', 'entropy', 'edge', 'full']
        param_sets = [
            {},  # Base - no params
            {'entropy_threshold': 0.12},  # Entropy
            {'edge_weight': 0.25},  # Edge
            {'entropy_threshold': 0.12, 'radial_sigma': 0.65, 'edge_weight': 0.25, 'merge_similar': True}  # Full
        ]
        
        results = {}
        
        print("\n🧪 Testing different selection methods...")
        print("-" * 60)
        
        for method, params in zip(methods, param_sets):
            print(f"\n📊 Testing method: {method.upper()}")
            print(f"   Parameters: {params}")
            
            try:
                # Measure time
                start_time = time.time()
                
                # Run SHIRG forward pass with specific method
                output_tokens = vision_tower.forward_with_shirg(
                    mock_views, 
                    text_embeddings,
                    selection_method=method,
                    selection_params=params
                )
                
                elapsed_time = (time.time() - start_time) * 1000  # ms
                
                # Validate output
                B, N, D = output_tokens.shape
                
                print(f"✅ Output shape: {output_tokens.shape}")
                print(f"   Expected: [1, 980, 1152] (256 global + 724 foveal)")
                print(f"   Time: {elapsed_time:.1f}ms")
                
                # Check exact token budget
                if N == 980:
                    print("✅ Token budget met exactly!")
                else:
                    print(f"❌ Token budget violation: {N} != 980")
                
                # Store results
                results[method] = {
                    'shape': output_tokens.shape,
                    'time_ms': elapsed_time,
                    'budget_met': N == 980,
                    'mean': output_tokens.mean().item(),
                    'std': output_tokens.std().item()
                }
                
            except Exception as e:
                print(f"❌ Method {method} failed: {e}")
                import traceback
                traceback.print_exc()
                results[method] = {'error': str(e)}
        
        # Test token selection directly
        print("\n🧪 Testing topk_per_view with different methods...")
        print("-" * 60)
        
        if hasattr(vision_tower, 'topk_per_view'):
            # Extract features for testing
            _, foveal_features = vision_tower.extract_multiview_tokens(mock_views)
            test_tokens = foveal_features[0]  # [1, 1024, 1152]
            K = 724  # 70.7% of 1024
            
            for method, params in zip(methods, param_sets):
                print(f"\n📊 Testing topk_per_view - {method.upper()}")
                
                try:
                    start_time = time.time()
                    
                    selected = vision_tower.topk_per_view(
                        test_tokens, K, text_embeddings,
                        method=method,
                        params=params
                    )
                    
                    elapsed_time = (time.time() - start_time) * 1000
                    
                    print(f"✅ Selected shape: {selected.shape}")
                    print(f"   Expected: [1, {K}, 1152]")
                    print(f"   Time: {elapsed_time:.1f}ms")
                    
                    if selected.shape[1] == K:
                        print(f"✅ Exact K={K} tokens selected!")
                    else:
                        print(f"❌ Wrong number of tokens: {selected.shape[1]} != {K}")
                        
                except Exception as e:
                    print(f"❌ topk_per_view {method} failed: {e}")
        
        # Summary report
        print("\n📈 PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"{'Method':<10} {'Shape':<20} {'Time (ms)':<10} {'Budget OK':<10} {'Mean':<10} {'Std':<10}")
        print("-" * 60)
        
        for method, result in results.items():
            if 'error' not in result:
                shape_str = str(result['shape'])
                time_str = f"{result['time_ms']:.1f}"
                budget_str = "✅" if result['budget_met'] else "❌"
                mean_str = f"{result['mean']:.4f}"
                std_str = f"{result['std']:.4f}"
                print(f"{method:<10} {shape_str:<20} {time_str:<10} {budget_str:<10} {mean_str:<10} {std_str:<10}")
            else:
                print(f"{method:<10} {'ERROR':<20} {'-':<10} {'❌':<10} {'-':<10} {'-':<10}")
        
        # Latency targets check
        print("\n📊 LATENCY TARGET CHECK (1.2× baseline)")
        print("-" * 60)
        baseline_time = results.get('base', {}).get('time_ms', 40)
        target_time = baseline_time * 1.2
        
        for method, result in results.items():
            if 'time_ms' in result:
                ratio = result['time_ms'] / baseline_time
                status = "✅" if ratio <= 1.2 else "❌"
                print(f"{method:<10}: {result['time_ms']:>6.1f}ms ({ratio:.2f}× baseline) {status}")
        
        print("\n✅ ENHANCED SELECTION TEST COMPLETE!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

def test_parameter_sensitivity():
    """Test parameter sensitivity for different methods"""
    print("\n\n🔬 PARAMETER SENSITIVITY TEST")
    print("=" * 60)
    
    try:
        from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
        
        # Mock setup
        class MockConfig:
            enable_shirg = True
        
        vision_tower = SigLipVisionTower(
            vision_tower="google/siglip-so400m-patch14-384",
            args=MockConfig(),
            delay_load=False
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mock_views = [
            torch.randn(1, 3, 448, 448, device=device, dtype=torch.float32),
            torch.randn(1, 3, 448, 448, device=device, dtype=torch.float32)
        ]
        text_embeddings = torch.randn(1, 50, 1152, device=device, dtype=torch.float32)
        
        # Test entropy threshold sensitivity
        print("\n📊 Entropy Threshold Sensitivity")
        print("-" * 40)
        thresholds = [0.08, 0.10, 0.12, 0.15]
        
        for tau in thresholds:
            params = {'entropy_threshold': tau}
            try:
                output = vision_tower.forward_with_shirg(
                    mock_views, text_embeddings,
                    selection_method='entropy',
                    selection_params=params
                )
                print(f"τ = {tau}: Output shape {output.shape} ✅")
            except Exception as e:
                print(f"τ = {tau}: Failed - {e} ❌")
        
        # Test edge weight sensitivity
        print("\n📊 Edge Weight Sensitivity")
        print("-" * 40)
        edge_weights = [0.15, 0.20, 0.25, 0.30]
        
        for weight in edge_weights:
            params = {'edge_weight': weight}
            try:
                output = vision_tower.forward_with_shirg(
                    mock_views, text_embeddings,
                    selection_method='edge',
                    selection_params=params
                )
                print(f"edge_weight = {weight}: Output shape {output.shape} ✅")
            except Exception as e:
                print(f"edge_weight = {weight}: Failed - {e} ❌")
        
        # Test radial sigma sensitivity
        print("\n📊 Radial Sigma Sensitivity (Full method)")
        print("-" * 40)
        sigmas = [0.55, 0.65, 0.75]
        
        for sigma in sigmas:
            params = {
                'entropy_threshold': 0.12,
                'radial_sigma': sigma,
                'edge_weight': 0.25
            }
            try:
                output = vision_tower.forward_with_shirg(
                    mock_views, text_embeddings,
                    selection_method='full',
                    selection_params=params
                )
                print(f"σ = {sigma}: Output shape {output.shape} ✅")
            except Exception as e:
                print(f"σ = {sigma}: Failed - {e} ❌")
        
        print("\n✅ PARAMETER SENSITIVITY TEST COMPLETE!")
        
    except Exception as e:
        print(f"\n❌ Parameter test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all enhanced selection tests"""
    test_enhanced_selection()
    test_parameter_sensitivity()

if __name__ == "__main__":
    main()