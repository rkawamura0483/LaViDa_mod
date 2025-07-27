#!/usr/bin/env python3
"""
Corrected SHIRG Architecture Validation Script

Tests the fixed SHIRG implementation that uses:
- Same resolution (384x384) for both LaViDa and SHIRG
- Different encoder depth (26 vs 27 layers) for quality comparison
- Proper methodology aligned with SHIRG research objectives

SHIRG-FIX: 2025-07-27 - Validation for corrected SHIRG methodology
ISSUE: Previous approach used different resolutions = incompatible features
SOLUTION: Same resolution, different encoder quality for valid comparison
RESEARCH IMPACT: Enables valid SHIRG research with comparable features

Usage (in Colab):
    !cd /content/LaViDa_mod && python shirg/validate_corrected_shirg.py

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

class CorrectedSHIRGValidator:
    """
    Validates the corrected SHIRG implementation that properly follows research methodology
    """
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.bfloat16
        self.vision_tower = None
        
        print(f"🔧 Corrected SHIRG Validator initialized")
        print(f"   Device: {self.device}")
        print(f"   Dtype: {self.dtype}")
    
    def setup_vision_tower(self) -> bool:
        """Setup vision tower with corrected SHIRG implementation"""
        print("\n🔄 Setting up SigLIP Vision Tower with corrected SHIRG...")
        
        try:
            self.vision_tower = SigLipVisionTower(
                vision_tower="google/siglip-so400m-patch14-384",
                vision_tower_cfg=None,
                delay_load=False
            )
            
            # Check for corrected methods
            has_high_quality_method = hasattr(self.vision_tower, 'forward_with_high_quality')
            has_extract_method = hasattr(self.vision_tower, '_extract_high_quality_tokens')
            has_shirg_method = hasattr(self.vision_tower, 'get_high_quality_tokens_for_shirg')
            has_full_encoder_method = hasattr(self.vision_tower, '_get_full_encoder_for_shirg')
            
            print(f"✅ Vision tower loaded")
            print(f"🔍 Corrected SHIRG methods:")
            print(f"   forward_with_high_quality: {'✅' if has_high_quality_method else '❌'}")
            print(f"   _extract_high_quality_tokens: {'✅' if has_extract_method else '❌'}")
            print(f"   get_high_quality_tokens_for_shirg: {'✅' if has_shirg_method else '❌'}")
            print(f"   _get_full_encoder_for_shirg: {'✅' if has_full_encoder_method else '❌'}")
            
            all_methods_present = all([
                has_high_quality_method, has_extract_method, 
                has_shirg_method, has_full_encoder_method
            ])
            
            if not all_methods_present:
                print("❌ Missing corrected SHIRG methods!")
                return False
                
            return True
            
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            return False
    
    def test_corrected_approach(self) -> Dict[str, Any]:
        """Test the corrected SHIRG approach (same resolution, different encoder quality)"""
        print("\n🧪 Testing Corrected SHIRG Approach...")
        
        results = {}
        
        try:
            # Create test image at standard LaViDa resolution (384x384)
            test_image = torch.randn(2, 3, 384, 384, dtype=self.dtype, device=self.device)
            
            # Test 1: LaViDa path (26 layers, 729 tokens)
            print("📍 Test 1: LaViDa standard path (26 layers, 729 tokens)")
            start_time = time.time()
            lavida_features = self.vision_tower.forward(test_image)
            lavida_time = time.time() - start_time
            
            print(f"   LaViDa: {lavida_features.shape}, {lavida_time*1000:.1f}ms")
            
            # Test 2: SHIRG path (27 layers, SAME resolution, 729 tokens)
            print("📍 Test 2: SHIRG high-quality path (27 layers, same resolution, 729 tokens)")
            start_time = time.time()
            _, shirg_features = self.vision_tower.forward_with_high_quality(
                test_image, return_high_quality=True
            )
            shirg_time = time.time() - start_time
            
            print(f"   SHIRG: {shirg_features.shape}, {shirg_time*1000:.1f}ms")
            
            # Critical validation: both should have EXACTLY the same shape
            shapes_match = lavida_features.shape == shirg_features.shape
            both_729_tokens = (lavida_features.shape[1] == 729 and shirg_features.shape[1] == 729)
            
            # Feature quality comparison
            print("📍 Test 3: Feature quality comparison (critical for SHIRG research)")
            
            lavida_norm = F.normalize(lavida_features, p=2, dim=-1)
            shirg_norm = F.normalize(shirg_features, p=2, dim=-1)
            
            # Cosine similarity (should be HIGH now - same resolution, similar features)
            similarity = torch.sum(lavida_norm * shirg_norm, dim=-1).mean().item()
            
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
            
            # Performance check
            performance_acceptable = shirg_time < 5.0  # Should be much faster now
            
            results = {
                'success': True,
                'lavida_shape': lavida_features.shape,
                'shirg_shape': shirg_features.shape,
                'shapes_match': shapes_match,
                'both_729_tokens': both_729_tokens,
                'lavida_time_ms': lavida_time * 1000,
                'shirg_time_ms': shirg_time * 1000,
                'performance_acceptable': performance_acceptable,
                'cosine_similarity': similarity,
                'quality_grade': 'High' if similarity > 0.7 else 'Medium' if similarity > 0.5 else 'Low',
                'research_valid': similarity > 0.7,
                'lavida_stats': lavida_stats,
                'shirg_stats': shirg_stats,
                'overall_success': shapes_match and both_729_tokens and (similarity > 0.7) and performance_acceptable
            }
            
            print(f"   Shapes match: {'✅' if shapes_match else '❌'}")
            print(f"   Both 729 tokens: {'✅' if both_729_tokens else '❌'}")
            print(f"   Cosine similarity: {similarity:.4f} ({results['quality_grade']}) {'✅' if similarity > 0.7 else '❌'}")
            print(f"   Performance: {shirg_time*1000:.1f}ms {'✅' if performance_acceptable else '❌'}")
            print(f"   Research valid: {'✅' if results['research_valid'] else '❌'}")
            
        except Exception as e:
            print(f"❌ Corrected approach test failed: {e}")
            results = {'success': False, 'error': str(e), 'overall_success': False}
            
        return results
    
    def test_encoder_layers(self) -> Dict[str, Any]:
        """Test that encoder layers are correctly separated"""
        print("\n🧪 Testing Encoder Layer Separation...")
        
        results = {}
        
        try:
            # Check LaViDa encoder layers (should be 26)
            lavida_layers = len(self.vision_tower.vision_tower.vision_model.encoder.layers)
            
            # Load and check SHIRG encoder layers (should be 27)
            shirg_encoder = self.vision_tower._get_full_encoder_for_shirg()
            shirg_layers = len(shirg_encoder.vision_model.encoder.layers)
            
            layer_separation_correct = (lavida_layers == 26 and shirg_layers == 27)
            
            results = {
                'success': True,
                'lavida_layers': lavida_layers,
                'shirg_layers': shirg_layers,
                'layer_separation_correct': layer_separation_correct
            }
            
            print(f"   LaViDa layers: {lavida_layers} (expected: 26) {'✅' if lavida_layers == 26 else '❌'}")
            print(f"   SHIRG layers: {shirg_layers} (expected: 27) {'✅' if shirg_layers == 27 else '❌'}")
            print(f"   Separation correct: {'✅' if layer_separation_correct else '❌'}")
            
        except Exception as e:
            print(f"❌ Encoder layer test failed: {e}")
            results = {'success': False, 'error': str(e)}
            
        return results
    
    def test_shirg_token_quality(self) -> Dict[str, Any]:
        """Test SHIRG token extraction for research use"""
        print("\n🧪 Testing SHIRG Token Quality...")
        
        results = {}
        
        try:
            test_image = torch.randn(2, 3, 384, 384, dtype=self.dtype, device=self.device)
            
            start_time = time.time()
            shirg_tokens = self.vision_tower.get_high_quality_tokens_for_shirg(test_image)
            extraction_time = time.time() - start_time
            
            # Validate shape and performance
            correct_shape = shirg_tokens.shape == (2, 729, 1152)
            fast_extraction = extraction_time < 3.0  # Should be very fast
            
            # Feature quality metrics
            token_diversity = torch.std(shirg_tokens.mean(dim=-1)).item()
            has_good_diversity = token_diversity > 0.1
            
            results = {
                'success': True,
                'shape': shirg_tokens.shape,
                'correct_shape': correct_shape,
                'extraction_time_ms': extraction_time * 1000,
                'fast_extraction': fast_extraction,
                'token_diversity': token_diversity,
                'has_good_diversity': has_good_diversity,
                'ready_for_shirg': correct_shape and fast_extraction and has_good_diversity
            }
            
            print(f"   Shape: {shirg_tokens.shape} {'✅' if correct_shape else '❌'}")
            print(f"   Extraction time: {extraction_time*1000:.1f}ms {'✅' if fast_extraction else '❌'}")
            print(f"   Token diversity: {token_diversity:.4f} {'✅' if has_good_diversity else '❌'}")
            print(f"   Ready for SHIRG: {'✅' if results['ready_for_shirg'] else '❌'}")
            
        except Exception as e:
            print(f"❌ SHIRG token quality test failed: {e}")
            results = {'success': False, 'error': str(e)}
            
        return results
    
    def run_validation(self) -> bool:
        """Run complete corrected validation"""
        print("🔬 Corrected SHIRG Architecture Validation")
        print("=" * 60)
        
        if not self.setup_vision_tower():
            return False
        
        # Run corrected tests
        test_results = {}
        
        test_suite = [
            ('corrected_approach', self.test_corrected_approach),
            ('encoder_layers', self.test_encoder_layers),
            ('shirg_token_quality', self.test_shirg_token_quality),
        ]
        
        for test_name, test_func in test_suite:
            print(f"\n{'='*20} {test_name.upper()} {'='*20}")
            test_results[test_name] = test_func()
        
        # Generate final assessment
        self.generate_final_assessment(test_results)
        
        # Overall success criteria
        critical_success = (
            test_results.get('corrected_approach', {}).get('overall_success', False) and
            test_results.get('encoder_layers', {}).get('layer_separation_correct', False) and
            test_results.get('shirg_token_quality', {}).get('ready_for_shirg', False)
        )
        
        return critical_success
    
    def generate_final_assessment(self, test_results: Dict):
        """Generate final assessment of corrected SHIRG implementation"""
        print(f"\n{'='*60}")
        print("📊 CORRECTED SHIRG VALIDATION ASSESSMENT")
        print(f"{'='*60}")
        
        # Extract key metrics
        approach_success = test_results.get('corrected_approach', {})
        similarity = approach_success.get('cosine_similarity', 0.0)
        performance = approach_success.get('shirg_time_ms', float('inf'))
        
        layer_success = test_results.get('encoder_layers', {}).get('layer_separation_correct', False)
        token_quality = test_results.get('shirg_token_quality', {}).get('ready_for_shirg', False)
        
        print(f"🎯 CRITICAL METRICS:")
        print(f"   Cosine similarity: {similarity:.4f} {'✅ RESEARCH VALID' if similarity > 0.7 else '❌ TOO LOW'}")
        print(f"   SHIRG performance: {performance:.1f}ms {'✅ FAST' if performance < 5000 else '❌ TOO SLOW'}")
        print(f"   Encoder separation: {'✅ CORRECT' if layer_success else '❌ INCORRECT'}")
        print(f"   Token quality: {'✅ READY' if token_quality else '❌ NOT READY'}")
        
        # Overall recommendation
        all_critical_passed = (similarity > 0.7) and (performance < 5000) and layer_success and token_quality
        
        print(f"\n🚀 FINAL RECOMMENDATION:")
        if all_critical_passed:
            print("✅ CORRECTED SHIRG IMPLEMENTATION IS VALID")
            print("✅ Research methodology properly implemented")
            print("✅ Feature quality meets standards (>0.7 similarity)")
            print("✅ Performance acceptable (<5s extraction)")
            print("✅ READY TO PROCEED WITH LORA TRAINING")
        else:
            print("❌ CORRECTED IMPLEMENTATION STILL HAS ISSUES")
            if similarity <= 0.7:
                print(f"❌ Cosine similarity too low: {similarity:.4f} (need >0.7)")
            if performance >= 5000:
                print(f"❌ Performance too slow: {performance:.1f}ms (need <5000ms)")
            if not layer_success:
                print("❌ Encoder layer separation incorrect")
            if not token_quality:
                print("❌ Token quality insufficient for SHIRG")
            print("❌ DO NOT PROCEED UNTIL ISSUES FIXED")
        
        return all_critical_passed

def main():
    """Main validation function"""
    
    if not LAVIDA_AVAILABLE:
        print("❌ LaViDa not available")
        return False
    
    # Run corrected validation
    validator = CorrectedSHIRGValidator()
    success = validator.run_validation()
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        print(f"\n🏁 Corrected validation {'PASSED' if success else 'FAILED'} (exit code: {exit_code})")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️ Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Validation crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)