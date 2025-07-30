#!/usr/bin/env python3
"""
Quick test to verify the optimizer fix for test_training_step
"""

import os
import sys

# Add paths
sys.path.append('./')
sys.path.append('./shirg')

from shirg.test_shirg_lora_pretrain import ShirgLoraPreTrainTest
from shirg.shirg_lora_config import create_lora_training_config

def main():
    print("🧪 Testing optimizer fix...")
    print("=" * 60)
    
    # Create test config
    config = create_lora_training_config()
    
    # Create tester
    tester = ShirgLoraPreTrainTest(config)
    
    # Run only the training step test
    try:
        result = tester.test_training_step()
        
        if result.get("passed", False):
            print("\n✅ Training step test PASSED!")
            print(f"   Loss: {result.get('details', {}).get('loss', 'N/A')}")
            return 0
        else:
            print("\n❌ Training step test FAILED!")
            print(f"   Error: {result.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        print(f"\n❌ Test exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    print(f"\nExit code: {exit_code}")
    sys.exit(exit_code)