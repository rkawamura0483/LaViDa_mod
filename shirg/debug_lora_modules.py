#!/usr/bin/env python3
"""
Debug script to understand LoRA module targeting in LaViDa
"""

import os
import sys
import torch

# Add paths
sys.path.append('./')
sys.path.append('./shirg')

# Set environment to disable device_map for proper loading
os.environ['SHIRG_NO_DEVICE_MAP'] = '1'

from shirg.lavida_shirg_integration import LaViDaSHIRGWrapper

def main():
    print("🔍 Debugging LoRA module targeting for LaViDa-SHIRG")
    print("=" * 60)
    
    # Create wrapper
    wrapper = LaViDaSHIRGWrapper(
        shirg_config={
            'target_tokens': 980,
            'alpha': 0.3,
            'debug': False
        }
    )
    
    # Load model
    print("\n📥 Loading model...")
    wrapper.load_model()
    model = wrapper.model
    
    print("\n📋 Model structure analysis:")
    
    # Find all linear modules that could be LoRA targets
    linear_modules = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_modules.append(name)
    
    print(f"\nFound {len(linear_modules)} linear modules")
    
    # Group by component
    vision_modules = []
    projector_modules = []
    llm_modules = []
    other_modules = []
    
    for name in linear_modules:
        if 'vision_tower' in name:
            vision_modules.append(name)
        elif 'mm_projector' in name or 'projector' in name:
            projector_modules.append(name)
        elif 'model.layers' in name or 'transformer' in name:
            llm_modules.append(name)
        else:
            other_modules.append(name)
    
    print(f"\n🔍 Vision tower modules ({len(vision_modules)}):")
    for i, name in enumerate(vision_modules[:10]):  # Show first 10
        print(f"   {i+1}. {name}")
    if len(vision_modules) > 10:
        print(f"   ... and {len(vision_modules) - 10} more")
    
    print(f"\n🔍 Projector modules ({len(projector_modules)}):")
    for name in projector_modules:
        print(f"   - {name}")
    
    print(f"\n🔍 LLM modules ({len(llm_modules)}):")
    print(f"   Total: {len(llm_modules)} modules")
    
    print(f"\n🔍 Other modules ({len(other_modules)}):")
    for name in other_modules[:5]:
        print(f"   - {name}")
    
    # Now test LoRA targeting
    print("\n🎯 Testing LoRA target patterns:")
    
    # Test patterns from research plan
    test_patterns = [
        "mm_projector.0",
        "mm_projector.2", 
        "mm_projector.fc1",
        "mm_projector.fc2",
        "vision_tower.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj",
        "vision_tower.vision_tower.vision_model.encoder.layers.0.self_attn.k_proj",
        "vision_tower.vision_tower.vision_model.encoder.layers.0.self_attn.v_proj",
        "model.vision_tower.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj",
        "model.mm_projector.0",
        "model.mm_projector.2",
    ]
    
    for pattern in test_patterns:
        found = any(pattern in module_name for module_name in linear_modules)
        print(f"   {pattern}: {'✅ FOUND' if found else '❌ NOT FOUND'}")
    
    # Show exact matches for key components
    print("\n📍 Exact module paths for SHIRG LoRA targets:")
    
    # Find projector modules
    print("\nProjector modules:")
    for name in linear_modules:
        if 'mm_projector' in name and any(x in name for x in ['0', '2', 'fc1', 'fc2']):
            print(f"   ✅ {name}")
    
    # Find early vision encoder modules
    print("\nEarly vision encoder modules (layers 0-5):")
    for i in range(6):
        for proj_type in ['q_proj', 'k_proj', 'v_proj']:
            pattern = f"layers.{i}.self_attn.{proj_type}"
            matches = [name for name in linear_modules if pattern in name and 'vision' in name]
            if matches:
                print(f"   ✅ Layer {i} {proj_type}: {matches[0]}")

if __name__ == "__main__":
    main()