#!/bin/bash
# Test SHIRG LoRA with 8 GPU setup
# This script runs the pre-training tests with multi-GPU configuration

echo "🧪 Testing SHIRG LoRA with 8 GPU configuration"
echo "=============================================="

# Set environment variable to enable multi-GPU testing
export SHIRG_TEST_MULTI_GPU=true

# SHIRG-FIX: 2025-07-30 - Disable device_map for LoRA gradient flow testing
# ISSUE: device_map="auto" (model parallelism) breaks LoRA gradient flow
# SOLUTION: Set SHIRG_NO_DEVICE_MAP=1 to use data parallelism (DDP) instead
# LAVIDA IMPACT: Each GPU loads full model (~16GB) for proper gradient flow
# SHIRG IMPACT: Fixes zero gradient issue in multi-GPU LoRA testing
export SHIRG_NO_DEVICE_MAP=1

# Show GPU configuration
echo "📊 GPU Configuration:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""

# Set CUDA devices (all 8 GPUs)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Run the test
echo "🚀 Running tests with multi-GPU setup..."
python shirg/test_shirg_lora_pretrain.py --selection-method full

# Check result
if [ $? -eq 0 ]; then
    echo "✅ Multi-GPU tests passed!"
else
    echo "❌ Multi-GPU tests failed!"
    exit 1
fi

echo ""
echo "📝 To run actual 8 GPU training after tests pass:"
echo "   bash shirg/run_8gpu_training.sh"