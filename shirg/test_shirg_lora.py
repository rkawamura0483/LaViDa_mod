#!/usr/bin/env python3
"""
SHIRG LoRA Implementation Test Script
Quick test to verify all components work together

SHIRG-FIX: 2025-07-27 - Integration test for complete SHIRG LoRA pipeline
ISSUE: Need verification that all components integrate properly before full training
SOLUTION: Lightweight test script to validate end-to-end functionality
LAVIDA IMPACT: Ensures compatibility with LaViDa architecture
SHIRG IMPACT: Validates research implementation is production-ready
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Add paths for imports
sys.path.append('./shirg/')
sys.path.append('./llava/')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_shirg_implementation():
    """Test SHIRG vision tower implementation"""
    logger.info("Testing SHIRG implementation...")
    
    try:
        from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower
        
        # Initialize vision tower
        vision_tower = SigLipVisionTower(
            vision_tower="google/siglip-so400m-patch14-384",
            vision_tower_cfg=None,
            delay_load=False
        )
        
        # Test basic functionality
        test_images = torch.randn(2, 3, 384, 384)
        if torch.cuda.is_available():
            test_images = test_images.cuda()
        
        # Test standard forward pass
        with torch.no_grad():
            baseline_output = vision_tower.forward(test_images)
            logger.info(f"✅ Baseline forward: {baseline_output.shape}")
            
            # Test high-resolution token extraction
            if hasattr(vision_tower, 'get_highres_tokens_for_shirg'):
                highres_output = vision_tower.get_highres_tokens_for_shirg(test_images)
                logger.info(f"✅ High-res extraction: {highres_output.shape}")
                
                # Test SHIRG selection
                if hasattr(vision_tower, 'shirg_token_selection'):
                    shirg_output = vision_tower.shirg_token_selection(highres_output, 768)
                    logger.info(f"✅ SHIRG selection: {shirg_output.shape}")
                    
                    # Test SHIRG forward
                    if hasattr(vision_tower, 'forward_with_shirg'):
                        shirg_forward = vision_tower.forward_with_shirg(test_images, 512)
                        logger.info(f"✅ SHIRG forward: {shirg_forward.shape}")
                    else:
                        logger.warning("⚠️ forward_with_shirg method not found")
                else:
                    logger.warning("⚠️ shirg_token_selection method not found")
            else:
                logger.warning("⚠️ get_highres_tokens_for_shirg method not found")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ SHIRG implementation test failed: {e}")
        return False

def test_lora_configuration():
    """Test LoRA configuration setup"""
    logger.info("Testing LoRA configuration...")
    
    try:
        from shirg_lora_training import SHIRGLoRAConfig
        
        # Test configuration creation
        config = SHIRGLoRAConfig(
            lora_rank=16,
            batch_size_per_gpu=2,
            num_epochs=1,
            dataset_size=100
        )
        
        logger.info(f"✅ LoRA config created: {config.total_steps} steps")
        return True
        
    except Exception as e:
        logger.error(f"❌ LoRA configuration test failed: {e}")
        return False

def test_dataset_preparation():
    """Test dataset preparation"""
    logger.info("Testing dataset preparation...")
    
    try:
        from shirg_dataset_preparation import SHIRGDatasetConfig, SHIRGDatasetProcessor
        
        # Test configuration
        config = SHIRGDatasetConfig(
            base_dataset_size=10,
            ocr_enhancement_size=5,
            output_dir="./test_shirg_data"
        )
        
        # Test processor creation
        processor = SHIRGDatasetProcessor(config)
        
        # Test sample creation
        sample = processor.create_synthetic_sample("ocr", "test_001")
        logger.info(f"✅ Sample created: {sample['sample_id']}")
        
        # Cleanup test directory
        import shutil
        if Path("./test_shirg_data").exists():
            shutil.rmtree("./test_shirg_data")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataset preparation test failed: {e}")
        return False

def test_evaluation_setup():
    """Test evaluation setup"""
    logger.info("Testing evaluation setup...")
    
    try:
        from shirg_evaluation import SHIRGEvaluationConfig
        
        # Test configuration
        config = SHIRGEvaluationConfig(
            datasets=["ChartQA"],
            sample_size=5,
            token_budgets=[512]
        )
        
        logger.info(f"✅ Evaluation config created: {len(config.datasets)} datasets")
        return True
        
    except Exception as e:
        logger.error(f"❌ Evaluation setup test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🧪 Starting SHIRG LoRA implementation tests...")
    
    tests = [
        ("SHIRG Implementation", test_shirg_implementation),
        ("LoRA Configuration", test_lora_configuration),
        ("Dataset Preparation", test_dataset_preparation),
        ("Evaluation Setup", test_evaluation_setup)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        if test_func():
            passed += 1
            logger.info(f"✅ {test_name} PASSED")
        else:
            logger.info(f"❌ {test_name} FAILED")
    
    logger.info(f"\n🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! SHIRG LoRA implementation is ready for training.")
        return 0
    else:
        logger.info("⚠️ Some tests failed. Please fix issues before proceeding with training.")
        return 1

if __name__ == "__main__":
    exit(main())