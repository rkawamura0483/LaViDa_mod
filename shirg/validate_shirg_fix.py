#!/usr/bin/env python3
"""
SHIRG Architecture Validation Script

This script validates the comprehensive fix for SHIRG high-resolution token extraction,
ensuring genuine high-quality features are extracted for valid research.

SHIRG-FIX: 2025-07-27 - Validation for dual-architecture approach
ISSUE: Previous implementation used degraded features, violated SHIRG methodology
SOLUTION: Validate dual-path processing (LaViDa + SHIRG) with proper feature quality
RESEARCH IMPACT: Ensures research validity and feature quality for SHIRG

Usage (in Colab):
    !cd /content/LaViDa_mod && python shirg/validate_shirg_fix.py

Author: Research Implementation
Date: 2025-07-27
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional
import warnings

# Colab environment detection
try:
    import google.colab
    IN_COLAB = True
    print("🌐 Running in Google Colab environment")
except ImportError:
    IN_COLAB = False
    print("💻 Running in local environment")

# Add paths for imports
BASE_PATH = './' if IN_COLAB else './'
sys.path.append(BASE_PATH)
sys.path.append(f'{BASE_PATH}/shirg')

try:
    from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
    from llava.utils import rank0_print
    LAVIDA_AVAILABLE = True
    print("✅ LaViDa imports successful")
except ImportError as e:
    print(f"❌ LaViDa imports failed: {e}")
    LAVIDA_AVAILABLE = False

class SHIRGArchitectureValidator:
    """
    Validates the comprehensive SHIRG architecture fix
    
    Tests the dual-architecture approach (LaViDa + SHIRG) to ensure:
    1. LaViDa compatibility maintained (729 tokens, reduced encoder)
    2. SHIRG quality achieved (high-res tokens, full encoder) 
    3. Feature quality meets research standards (>0.7 cosine similarity)
    4. Performance within acceptable bounds (<5s extraction time)
    """
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.bfloat16
        self.vision_tower = None
        
        print(f"🔧 SHIRG Validator initialized")
        print(f"   Device: {self.device}")
        print(f"   Dtype: {self.dtype}")
    
    def setup_vision_tower(self) -> bool:
        """Setup vision tower with SHIRG modifications"""
        print("\n🔄 Setting up SigLIP Vision Tower with SHIRG architecture...")
        
        try:
            self.vision_tower = SigLipVisionTower(
                vision_tower="google/siglip-so400m-patch14-384",
                vision_tower_cfg=None,
                delay_load=False
            )
            
            # Verify dual architecture is present
            has_full_encoder_method = hasattr(self.vision_tower, '_get_full_encoder_for_shirg')
            has_high_res_method = hasattr(self.vision_tower, 'forward_with_high_res')
            has_multiview_method = hasattr(self.vision_tower, 'get_multiview_high_res_tokens')
            
            print(f"✅ Vision tower loaded")
            print(f"🔍 Architecture validation:")
            print(f"   _get_full_encoder_for_shirg: {'✅' if has_full_encoder_method else '❌'}")
            print(f"   forward_with_high_res: {'✅' if has_high_res_method else '❌'}")
            print(f"   get_multiview_high_res_tokens: {'✅' if has_multiview_method else '❌'}")
            
            if not (has_full_encoder_method and has_high_res_method and has_multiview_method):
                print("❌ SHIRG architecture incomplete!")
                return False
                
            return True
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return False
    
    def test_dual_architecture(self) -> Dict[str, Any]:
        """Test that both LaViDa and SHIRG paths work correctly"""
        print("\n🧪 Testing Dual Architecture (LaViDa + SHIRG)...")
        
        results = {}
        
        try:
            # Create test image
            test_image = torch.randn(2, 3, 384, 384, dtype=self.dtype, device=self.device)
            
            # Test 1: LaViDa standard path (should give 729 tokens)
            print("📍 Test 1: LaViDa standard path")
            start_time = time.time()
            lavida_features = self.vision_tower.forward(test_image)
            lavida_time = time.time() - start_time
            
            results['lavida'] = {
                'success': True,
                'shape': lavida_features.shape,
                'tokens': lavida_features.shape[1],
                'time_ms': lavida_time * 1000,
                'expected_tokens': 729,
                'tokens_correct': lavida_features.shape[1] == 729
            }
            
            print(f"   LaViDa features: {lavida_features.shape}")
            print(f"   Tokens: {lavida_features.shape[1]} (expected: 729) {'✅' if results['lavida']['tokens_correct'] else '❌'}")
            print(f"   Time: {lavida_time*1000:.1f}ms")
            
            # Test 2: SHIRG high-resolution path 
            print("📍 Test 2: SHIRG high-resolution path")
            start_time = time.time()
            _, shirg_features = self.vision_tower.forward_with_high_res(
                test_image, return_high_res=True, target_resolution=(768, 768)
            )
            shirg_time = time.time() - start_time
            
            # Calculate expected tokens for 768x768 at 14x14 patches
            expected_shirg_tokens = (768 // 14) ** 2  # 54x54 = 2916
            
            results['shirg'] = {
                'success': True,
                'shape': shirg_features.shape,
                'tokens': shirg_features.shape[1],
                'time_ms': shirg_time * 1000,
                'expected_tokens': expected_shirg_tokens,
                'high_resolution': shirg_features.shape[1] > 729
            }
            
            print(f"   SHIRG features: {shirg_features.shape}")
            print(f"   Tokens: {shirg_features.shape[1]} (expected: ~{expected_shirg_tokens}) {'✅' if results['shirg']['high_resolution'] else '❌'}")
            print(f"   Time: {shirg_time*1000:.1f}ms")
            
            # Test 3: Feature quality comparison
            print("📍 Test 3: Feature quality analysis")
            
            # Compare first 729 tokens between paths
            min_tokens = min(lavida_features.shape[1], shirg_features.shape[1])
            lavida_subset = F.normalize(lavida_features[:, :min_tokens, :], p=2, dim=-1)
            shirg_subset = F.normalize(shirg_features[:, :min_tokens, :], p=2, dim=-1)
            
            # Cosine similarity
            similarity = torch.sum(lavida_subset * shirg_subset, dim=-1).mean().item()
            
            # Feature statistics
            lavida_stats = {
                'mean': lavida_features.mean().item(),
                'std': lavida_features.std().item(),
                'norm': torch.norm(lavida_features).item()
            }
            
            shirg_stats = {
                'mean': shirg_features.mean().item(),
                'std': shirg_features.std().item(),
                'norm': torch.norm(shirg_features).item()
            }
            
            results['quality'] = {
                'cosine_similarity': similarity,
                'quality_grade': 'High' if similarity > 0.7 else 'Medium' if similarity > 0.5 else 'Low',
                'lavida_stats': lavida_stats,
                'shirg_stats': shirg_stats,
                'research_valid': similarity > 0.7  # SHIRG research validity threshold
            }
            
            print(f"   Cosine similarity: {similarity:.4f} ({results['quality']['quality_grade']})")
            print(f"   Research valid: {'✅' if results['quality']['research_valid'] else '❌'}")
            print(f"   LaViDa stats: mean={lavida_stats['mean']:.4f}, std={lavida_stats['std']:.4f}")
            print(f"   SHIRG stats: mean={shirg_stats['mean']:.4f}, std={shirg_stats['std']:.4f}")
            
            results['overall_success'] = (
                results['lavida']['tokens_correct'] and
                results['shirg']['high_resolution'] and 
                results['quality']['research_valid']
            )
            
        except Exception as e:
            print(f"❌ Dual architecture test failed: {e}")
            results['overall_success'] = False
            results['error'] = str(e)
            
        return results
    
    def test_encoder_separation(self) -> Dict[str, Any]:
        """Test that LaViDa and SHIRG use different encoders"""
        print("\n🧪 Testing Encoder Separation...")
        
        results = {}
        
        try:
            # Check LaViDa encoder (should have 26 layers)
            lavida_layers = len(self.vision_tower.vision_tower.vision_model.encoder.layers)
            
            # Load SHIRG encoder and check (should have 27 layers)
            shirg_encoder = self.vision_tower._get_full_encoder_for_shirg()
            shirg_layers = len(shirg_encoder.vision_model.encoder.layers)
            
            results = {
                'success': True,
                'lavida_layers': lavida_layers,
                'shirg_layers': shirg_layers,
                'expected_lavida': 26,  # One layer deleted
                'expected_shirg': 27,   # Full encoder
                'separation_correct': (lavida_layers == 26 and shirg_layers == 27)
            }
            
            print(f"   LaViDa encoder layers: {lavida_layers} (expected: 26) {'✅' if lavida_layers == 26 else '❌'}")
            print(f"   SHIRG encoder layers: {shirg_layers} (expected: 27) {'✅' if shirg_layers == 27 else '❌'}")
            print(f"   Separation correct: {'✅' if results['separation_correct'] else '❌'}")
            
        except Exception as e:
            print(f"❌ Encoder separation test failed: {e}")
            results['success'] = False
            results['error'] = str(e)
            
        return results
    
    def test_multiview_processing(self) -> Dict[str, Any]:
        """Test LaViDa 5-view multiview processing"""
        print("\n🧪 Testing Multi-view Processing...")
        
        results = {}
        
        try:
            # Create test image
            test_image = torch.randn(2, 3, 384, 384, dtype=self.dtype, device=self.device)
            
            start_time = time.time()
            multiview_features = self.vision_tower.get_multiview_high_res_tokens(test_image)
            processing_time = time.time() - start_time
            
            target_tokens = 3645  # LaViDa specification
            actual_tokens = multiview_features.shape[1]
            
            results = {
                'success': True,
                'shape': multiview_features.shape,
                'tokens': actual_tokens,
                'target_tokens': target_tokens,
                'time_ms': processing_time * 1000,
                'tokens_correct': actual_tokens == target_tokens,
                'feature_quality': {
                    'mean': multiview_features.mean().item(),
                    'std': multiview_features.std().item(),
                    'has_variation': multiview_features.std().item() > 0.1
                }
            }
            
            print(f"   Multi-view features: {multiview_features.shape}")
            print(f"   Tokens: {actual_tokens} (target: {target_tokens}) {'✅' if results['tokens_correct'] else '❌'}")
            print(f"   Time: {processing_time*1000:.1f}ms")
            print(f"   Feature variation: {'✅' if results['feature_quality']['has_variation'] else '❌'}")
            
        except Exception as e:
            print(f"❌ Multi-view processing test failed: {e}")
            results['success'] = False
            results['error'] = str(e)
            
        return results
    
    def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test performance of the new architecture"""
        print("\n🧪 Testing Performance Benchmarks...")
        
        results = {}
        
        try:
            test_image = torch.randn(2, 3, 384, 384, dtype=self.dtype, device=self.device)
            num_runs = 3
            
            # Warm up
            _ = self.vision_tower.forward(test_image)
            _ = self.vision_tower.forward_with_high_res(test_image, return_high_res=True)
            
            # Benchmark LaViDa path
            lavida_times = []
            for _ in range(num_runs):
                start = time.time()
                _ = self.vision_tower.forward(test_image)
                lavida_times.append((time.time() - start) * 1000)
            
            # Benchmark SHIRG path  
            shirg_times = []
            for _ in range(num_runs):
                start = time.time()
                _ = self.vision_tower.forward_with_high_res(test_image, return_high_res=True)
                shirg_times.append((time.time() - start) * 1000)
            
            # Benchmark multi-view
            multiview_times = []
            for _ in range(num_runs):
                start = time.time()
                _ = self.vision_tower.get_multiview_high_res_tokens(test_image)
                multiview_times.append((time.time() - start) * 1000)
            
            results = {
                'success': True,
                'lavida_ms': {
                    'mean': np.mean(lavida_times),
                    'std': np.std(lavida_times)
                },
                'shirg_ms': {
                    'mean': np.mean(shirg_times),
                    'std': np.std(shirg_times)
                },
                'multiview_ms': {
                    'mean': np.mean(multiview_times),
                    'std': np.std(multiview_times)
                },
                'overhead': {
                    'shirg_vs_lavida': np.mean(shirg_times) / np.mean(lavida_times),
                    'acceptable_performance': np.mean(shirg_times) < 5000  # <5s threshold
                }
            }
            
            print(f"   LaViDa: {results['lavida_ms']['mean']:.1f}±{results['lavida_ms']['std']:.1f}ms")
            print(f"   SHIRG: {results['shirg_ms']['mean']:.1f}±{results['shirg_ms']['std']:.1f}ms")
            print(f"   Multi-view: {results['multiview_ms']['mean']:.1f}±{results['multiview_ms']['std']:.1f}ms")
            print(f"   SHIRG overhead: {results['overhead']['shirg_vs_lavida']:.2f}x")
            print(f"   Performance acceptable: {'✅' if results['overhead']['acceptable_performance'] else '❌'}")
            
        except Exception as e:
            print(f"❌ Performance benchmark failed: {e}")
            results['success'] = False
            results['error'] = str(e)
            
        return results
    
    def run_validation(self) -> bool:
        """Run complete validation suite"""
        print("🔬 SHIRG Architecture Validation")
        print("=" * 60)
        
        if not self.setup_vision_tower():
            return False
        
        # Run all tests
        test_results = {}
        
        test_suite = [
            ('dual_architecture', self.test_dual_architecture),
            ('encoder_separation', self.test_encoder_separation),
            ('multiview_processing', self.test_multiview_processing),
            ('performance_benchmarks', self.test_performance_benchmarks),
        ]
        
        for test_name, test_func in test_suite:
            print(f"\n{'='*20} {test_name.upper()} {'='*20}")
            test_results[test_name] = test_func()
        
        # Generate final report
        self.generate_final_report(test_results)
        
        # Determine overall success
        critical_tests = ['dual_architecture', 'encoder_separation']
        critical_success = all(
            test_results[test].get('overall_success' if test == 'dual_architecture' else 'success', False)
            for test in critical_tests
        )
        
        return critical_success
    
    def generate_final_report(self, test_results: Dict):
        """Generate final validation report"""
        print(f"\n{'='*60}")
        print("📊 SHIRG ARCHITECTURE VALIDATION REPORT")
        print(f"{'='*60}")
        
        # Count successes
        total_tests = len(test_results)
        successful_tests = sum(
            1 for result in test_results.values() 
            if result.get('success', False) or result.get('overall_success', False)
        )
        
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        print(f"Tests passed: {successful_tests}/{total_tests} ({success_rate*100:.1f}%)")
        
        # Critical validations
        dual_arch_success = test_results.get('dual_architecture', {}).get('overall_success', False)
        encoder_sep_success = test_results.get('encoder_separation', {}).get('success', False)
        
        quality_valid = False
        if 'dual_architecture' in test_results:
            quality_valid = test_results['dual_architecture'].get('quality', {}).get('research_valid', False)
        
        print(f"\n🎯 CRITICAL VALIDATIONS:")
        print(f"   Dual architecture working: {'✅' if dual_arch_success else '❌'}")
        print(f"   Encoder separation correct: {'✅' if encoder_sep_success else '❌'}")
        print(f"   Feature quality (>0.7 similarity): {'✅' if quality_valid else '❌'}")
        
        # Overall recommendation
        ready_for_research = dual_arch_success and encoder_sep_success and quality_valid
        
        print(f"\n🚀 RECOMMENDATION:")
        if ready_for_research:
            print("✅ READY FOR SHIRG RESEARCH")
            print("✅ Architecture correctly implements dual-path processing")
            print("✅ Feature quality meets research standards")
            print("✅ Proceed with LoRA training")
        else:
            print("❌ NOT READY FOR RESEARCH")
            print("❌ Critical issues detected that must be fixed")
            print("❌ Do not proceed with LoRA training until issues resolved")
        
        return ready_for_research

def main():
    """Main validation function"""
    
    # Check dependencies
    if not LAVIDA_AVAILABLE:
        print("❌ LaViDa not available")
        return False
    
    # Run validation
    validator = SHIRGArchitectureValidator()
    success = validator.run_validation()
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        print(f"\n🏁 Validation {'PASSED' if success else 'FAILED'} (exit code: {exit_code})")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️ Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Validation crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)