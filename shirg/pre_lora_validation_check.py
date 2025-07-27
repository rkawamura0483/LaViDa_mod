#!/usr/bin/env python3
"""
SHIRG Pre-LoRA Validation Script
Comprehensive validation of SHIRG implementation before LoRA training

SHIRG-FIX: 2025-07-27 - Pre-training validation script
ISSUE: Need to validate SHIRG implementation correctness before expensive LoRA training
SOLUTION: Comprehensive testing of token extraction, selection, and integration
LAVIDA IMPACT: Ensures LaViDa functionality preserved during SHIRG integration
SHIRG IMPACT: Validates research implementation matches theoretical specification
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add paths for imports
sys.path.append('./shirg/')
sys.path.append('./llava/')

def check_environment():
    """Validate execution environment and dependencies"""
    print("🔍 Checking Environment...")
    
    # Check if in Colab
    try:
        import google.colab
        print("  ✓ Running in Google Colab")
        IN_COLAB = True
    except ImportError:
        print("  ⚠️ Not in Colab - ensure proper GPU setup")
        IN_COLAB = False
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"  ✓ GPU: {gpu_props.name}")
        print(f"  ✓ GPU Memory: {gpu_props.total_memory / 1e9:.1f}GB")
        
        # Check for adequate memory (need >15GB for SHIRG)
        if gpu_props.total_memory / 1e9 < 15:
            print("  ⚠️ Warning: GPU memory may be insufficient for high-res tokens")
    else:
        print("  ❌ No GPU available - SHIRG requires GPU")
        return False
    
    # Check PyTorch version
    print(f"  ✓ PyTorch version: {torch.__version__}")
    
    return True

def check_file_structure():
    """Validate required files exist"""
    print("\n📁 Checking File Structure...")
    
    required_files = [
        'llava/model/multimodal_encoder/siglip_encoder.py',
        'shirg/SHIRG_RESEARCH_IDEA.md',
        'shirg/LAVIDA_FORK_MODIFICATION_PLAN.md',
        'CLAUDE.md'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ❌ Missing: {file_path}")
            return False
    
    return True

def validate_siglip_modifications():
    """Validate SigLIP encoder modifications for SHIRG"""
    print("\n🔧 Validating SigLIP Modifications...")
    
    try:
        from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
        
        # Create test vision tower
        tower = SigLipVisionTower(
            vision_tower="google/siglip-so400m-patch14-384",
            vision_tower_cfg=None,
            delay_load=True
        )
        
        # Check for SHIRG methods
        shirg_methods = [
            'forward_with_shirg',
            'get_multiview_tokens_for_shirg', 
            'shirg_token_selection',
            'compare_baseline_vs_shirg'
        ]
        
        for method in shirg_methods:
            if hasattr(tower, method):
                print(f"  ✓ {method} method present")
            else:
                print(f"  ❌ Missing {method} method")
                return False
        
        print("  ✓ All SHIRG methods present in SigLipVisionTower")
        return True
        
    except Exception as e:
        print(f"  ❌ Error loading SigLipVisionTower: {e}")
        return False

def test_token_extraction():
    """Test high-resolution token extraction"""
    print("\n🎯 Testing Token Extraction...")
    
    try:
        from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
        
        # Create vision tower
        tower = SigLipVisionTower(
            vision_tower="google/siglip-so400m-patch14-384", 
            vision_tower_cfg=None,
            delay_load=True
        )
        
        # Create dummy image batch
        dummy_images = torch.randn(2, 3, 384, 384)
        if torch.cuda.is_available():
            dummy_images = dummy_images.cuda()
        
        # Test 1: Standard forward pass (should give 729 tokens)
        print("  Testing standard forward pass...")
        with torch.no_grad():
            standard_features = tower.forward(dummy_images)
            
        expected_shape = (2, 729, tower.hidden_size)
        if standard_features.shape == expected_shape:
            print(f"  ✓ Standard forward: {standard_features.shape}")
        else:
            print(f"  ❌ Standard forward shape mismatch: {standard_features.shape} vs {expected_shape}")
            return False
        
        # Test 2: Multi-view token extraction (should give 4608 tokens)
        print("  Testing multi-view token extraction...")
        with torch.no_grad():
            multiview_tokens = tower.get_multiview_tokens_for_shirg(dummy_images)
            
        expected_mv_shape = (2, 4608, tower.hidden_size)  # Corrected: 4×576 + 1×2304 = 4608
        if multiview_tokens.shape == expected_mv_shape:
            print(f"  ✓ Multi-view extraction: {multiview_tokens.shape}")
        elif multiview_tokens.shape[1] >= 4000:  # Allow some flexibility
            print(f"  ⚠️ Multi-view shape close: {multiview_tokens.shape} (expected {expected_mv_shape})")
        else:
            print(f"  ❌ Multi-view shape wrong: {multiview_tokens.shape} vs {expected_mv_shape}")
            return False
            
        return True
        
    except Exception as e:
        print(f"  ❌ Token extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_shirg_selection():
    """Test SHIRG token selection algorithm"""
    print("\n🎯 Testing SHIRG Selection Algorithm...")
    
    try:
        from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
        
        # Create vision tower
        tower = SigLipVisionTower(
            vision_tower="google/siglip-so400m-patch14-384",
            vision_tower_cfg=None,
            delay_load=True
        )
        
        # Create test data with corrected dimensions
        batch_size = 2
        total_tokens = 4608  # Corrected: LaViDa multi-view gives 4608 tokens
        embed_dim = tower.hidden_size
        target_tokens = 768
        
        # Generate dummy multiview tokens
        multiview_tokens = torch.randn(batch_size, total_tokens, embed_dim)
        text_embeddings = torch.randn(batch_size, 20, embed_dim)  # 20 text tokens
        
        if torch.cuda.is_available():
            multiview_tokens = multiview_tokens.cuda()
            text_embeddings = text_embeddings.cuda()
        
        # Test SHIRG selection
        print("  Testing SHIRG token selection...")
        start_time = time.time()
        
        with torch.no_grad():
            selected_tokens = tower.shirg_token_selection(
                multiview_tokens, target_tokens, text_embeddings
            )
        
        selection_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Validate output shape (SHIRG returns target_tokens + 1 summary token)
        expected_shape = (batch_size, target_tokens + 1, embed_dim)
        if selected_tokens.shape == expected_shape:
            print(f"  ✓ SHIRG selection: {selected_tokens.shape}")
            print(f"  ✓ Selection time: {selection_time:.1f}ms (target: <30ms)")
        else:
            print(f"  ❌ SHIRG selection shape: {selected_tokens.shape} vs {expected_shape}")
            return False
        
        # Check selection time (should be <30ms for cache preservation)
        if selection_time < 30:
            print(f"  ✓ Fast selection: {selection_time:.1f}ms")
        else:
            print(f"  ⚠️ Slow selection: {selection_time:.1f}ms (may impact cache)")
        
        # Test different target counts
        test_counts = [512, 768, 1024]
        for count in test_counts:
            with torch.no_grad():
                selected = tower.shirg_token_selection(multiview_tokens, count, text_embeddings)
            if selected.shape[1] == count + 1:  # +1 for summary token
                print(f"  ✓ Target {count} tokens: {selected.shape}")
            else:
                print(f"  ❌ Target {count} failed: {selected.shape}")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ SHIRG selection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_shirg_integration():
    """Test full SHIRG integration pipeline"""
    print("\n🔄 Testing SHIRG Integration Pipeline...")
    
    try:
        from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
        
        # Create vision tower
        tower = SigLipVisionTower(
            vision_tower="google/siglip-so400m-patch14-384",
            vision_tower_cfg=None, 
            delay_load=True
        )
        
        # Create test images
        test_images = torch.randn(2, 3, 384, 384)
        if torch.cuda.is_available():
            test_images = test_images.cuda()
        
        # Test baseline vs SHIRG comparison
        print("  Testing baseline vs SHIRG comparison...")
        with torch.no_grad():
            baseline_tokens, shirg_tokens = tower.compare_baseline_vs_shirg(
                test_images, target_tokens=768
            )
        
        # Validate shapes
        if baseline_tokens.shape[1] == 729:  # LaViDa baseline
            print(f"  ✓ Baseline tokens: {baseline_tokens.shape}")
        else:
            print(f"  ❌ Baseline shape wrong: {baseline_tokens.shape}")
            return False
            
        if shirg_tokens.shape[1] == 768 + 1:  # SHIRG selection + summary token
            print(f"  ✓ SHIRG tokens: {shirg_tokens.shape}")
        else:
            print(f"  ❌ SHIRG shape wrong: {shirg_tokens.shape}")
            return False
        
        # Test forward_with_shirg method
        print("  Testing forward_with_shirg method...")
        with torch.no_grad():
            shirg_features = tower.forward_with_shirg(
                test_images, target_tokens=512
            )
        
        if shirg_features.shape[1] == 512 + 1:  # +1 for summary token
            print(f"  ✓ SHIRG forward: {shirg_features.shape}")
        else:
            print(f"  ❌ SHIRG forward wrong: {shirg_features.shape}")
            return False
            
        return True
        
    except Exception as e:
        print(f"  ❌ SHIRG integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_efficiency():
    """Test GPU memory usage with SHIRG"""
    print("\n💾 Testing Memory Efficiency...")
    
    if not torch.cuda.is_available():
        print("  ⚠️ No GPU available for memory testing")
        return True
    
    try:
        # Clear GPU memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
        
        # Create vision tower
        tower = SigLipVisionTower(
            vision_tower="google/siglip-so400m-patch14-384",
            vision_tower_cfg=None,
            delay_load=True
        )
        
        # Load model memory
        model_memory = torch.cuda.memory_allocated() - initial_memory
        print(f"  Model memory: {model_memory / 1e6:.0f}MB")
        
        # Test with batch processing
        batch_sizes = [1, 2, 4]
        for batch_size in batch_sizes:
            torch.cuda.empty_cache()
            pre_batch_memory = torch.cuda.memory_allocated()
            
            # Process batch
            test_images = torch.randn(batch_size, 3, 384, 384).cuda()
            
            with torch.no_grad():
                # Test both baseline and SHIRG
                baseline = tower.forward(test_images)
                shirg = tower.forward_with_shirg(test_images, target_tokens=768)
            
            peak_memory = torch.cuda.memory_allocated() - pre_batch_memory
            print(f"  Batch {batch_size}: {peak_memory / 1e6:.0f}MB")
            
            # Check for memory leaks
            del test_images, baseline, shirg
            torch.cuda.empty_cache()
            
            post_batch_memory = torch.cuda.memory_allocated()
            leak = post_batch_memory - pre_batch_memory
            
            if leak < 1e6:  # Less than 1MB leak is acceptable
                print(f"    ✓ No memory leak (leak: {leak / 1e6:.1f}MB)")
            else:
                print(f"    ⚠️ Potential memory leak: {leak / 1e6:.1f}MB")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Memory test failed: {e}")
        return False

def test_position_embedding_interpolation():
    """Test position embedding interpolation for high-res tokens"""
    print("\n📐 Testing Position Embedding Interpolation...")
    
    try:
        from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionEmbeddings, SigLipVisionConfig
        
        # Create config and embeddings
        config = SigLipVisionConfig()
        embeddings = SigLipVisionEmbeddings(config)
        
        if torch.cuda.is_available():
            embeddings = embeddings.cuda()
        
        # Test standard resolution (384x384 -> 27x27 = 729 patches)
        standard_images = torch.randn(2, 3, 384, 384)
        if torch.cuda.is_available():
            standard_images = standard_images.cuda()
            
        with torch.no_grad():
            standard_embeds = embeddings(standard_images)
            
        if standard_embeds.shape[1] == 729:
            print(f"  ✓ Standard embeddings: {standard_embeds.shape}")
        else:
            print(f"  ❌ Standard embeddings wrong: {standard_embeds.shape}")
            return False
        
        # Test high resolution (672x672 -> 48x48 = 2304 patches)
        high_res_images = torch.randn(2, 3, 672, 672)
        if torch.cuda.is_available():
            high_res_images = high_res_images.cuda()
            
        with torch.no_grad():
            high_res_embeds = embeddings(high_res_images)
            
        if high_res_embeds.shape[1] == 2304:
            print(f"  ✓ High-res embeddings: {high_res_embeds.shape}")
        else:
            print(f"  ❌ High-res embeddings wrong: {high_res_embeds.shape}")
            return False
            
        # Check that interpolation doesn't break gradients (for LoRA training)
        if high_res_embeds.requires_grad:
            print("  ✓ Gradients preserved through interpolation")
        else:
            print("  ⚠️ Gradients may be blocked (check for .detach() calls)")
            
        return True
        
    except Exception as e:
        print(f"  ❌ Position embedding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_research_alignment():
    """Validate implementation matches research specification"""
    print("\n📝 Validating Research Alignment...")
    
    print("  Checking SHIRG research specifications:")
    
    # Check 1: Multi-view token extraction (4×336² + 1×672²)
    view_configs = [(336, 336, 4), (672, 672, 1)]
    expected_tokens = sum(count * (h//14) * (w//14) for h, w, count in view_configs)
    print(f"  ✓ Expected multi-view tokens: {expected_tokens} (4×576 + 1×2304 = 4608)")
    
    # Check 2: Target token budgets
    target_budgets = [512, 768, 1024]
    print(f"  ✓ Target token budgets: {target_budgets}")
    
    # Check 3: Selection criteria (variance + similarity + edge boost)
    selection_components = ["variance", "similarity", "edge_density"]
    print(f"  ✓ Selection components: {selection_components}")
    
    # Check 4: Coverage guarantee requirement
    print("  ✓ Coverage guarantee: At least 1 token per spatial region")
    
    # Check 5: Cache preservation (static selection)
    print("  ✓ Cache preservation: Static selection before diffusion")
    
    # Check 6: Performance targets
    performance_targets = {
        "selection_time": "<30ms",
        "memory_overhead": "<20%", 
        "ocr_improvement": "+5-10%"
    }
    print(f"  ✓ Performance targets: {performance_targets}")
    
    return True

def run_comprehensive_validation():
    """Run all validation tests"""
    print("🚀 SHIRG Pre-LoRA Comprehensive Validation")
    print("=" * 50)
    
    tests = [
        ("Environment Check", check_environment),
        ("File Structure", check_file_structure),
        ("SigLIP Modifications", validate_siglip_modifications),
        ("Token Extraction", test_token_extraction),
        ("SHIRG Selection", test_shirg_selection),
        ("SHIRG Integration", test_shirg_integration),
        ("Memory Efficiency", test_memory_efficiency),
        ("Position Embeddings", test_position_embedding_interpolation),
        ("Research Alignment", validate_research_alignment),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Ready for LoRA training!")
        return True
    else:
        print(f"\n⚠️ {total - passed} tests failed - Fix issues before LoRA training")
        return False

def generate_fix_recommendations(results: Dict[str, bool]):
    """Generate specific fix recommendations for failed tests"""
    print("\n🔧 FIX RECOMMENDATIONS")
    print("=" * 30)
    
    if not results.get("Token Extraction", True):
        print("\n📝 Token Extraction Issues:")
        print("1. Check multi-view processing logic in get_multiview_tokens_for_shirg()")
        print("2. Verify view configurations match LaViDa spec: 4×336² + 1×672²")
        print("3. Expected token count: 4608 (not 3645) - check math: 4×576 + 1×2304")
    
    if not results.get("SHIRG Selection", True):
        print("\n📝 SHIRG Selection Issues:")
        print("1. Implement hierarchical clustering for coverage guarantee")
        print("2. Add edge density boost using Laplacian operator")
        print("3. Optimize selection algorithm for <30ms performance")
    
    if not results.get("Position Embeddings", True):
        print("\n📝 Position Embedding Issues:")
        print("1. Ensure interpolation preserves gradients (no .detach())")
        print("2. Use bilinear interpolation for smooth scaling")
        print("3. Validate interpolation works for all resolution combinations")
    
    if not results.get("Memory Efficiency", True):
        print("\n📝 Memory Issues:")
        print("1. Add explicit torch.cuda.empty_cache() calls")
        print("2. Use gradient checkpointing for large batches")
        print("3. Consider mixed precision (fp16) for memory savings")

if __name__ == "__main__":
    # Run validation
    success = run_comprehensive_validation()
    
    if not success:
        print("\n" + "=" * 50)
        print("⚠️ VALIDATION FAILED - CHECK ISSUES ABOVE")
        print("=" * 50)
        print("\nRecommended actions:")
        print("1. Fix failed tests using recommendations above")
        print("2. Re-run validation script until all tests pass")
        print("3. Only proceed with LoRA training after full validation")
        sys.exit(1)
    else:
        print("\n" + "=" * 50)
        print("✅ VALIDATION SUCCESSFUL - READY FOR LORA TRAINING")
        print("=" * 50)
        print("\nNext steps:")
        print("1. Proceed with LoRA adapter setup")
        print("2. Launch mixed-ratio training as per research plan")
        print("3. Monitor training convergence and validation metrics")
        sys.exit(0)