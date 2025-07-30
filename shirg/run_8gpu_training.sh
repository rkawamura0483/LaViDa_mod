#!/bin/bash
# SHIRG LoRA Multi-GPU Training Script for 8 x A100 GPUs
# This script handles distributed training setup for SHIRG on Lambda Cloud
# Author: Research Implementation
# Date: 2025-07-30

# Set environment variables for distributed training
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WORLD_SIZE=8
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Disable tokenizers parallelism to avoid warnings
export TOKENIZERS_PARALLELISM=false

# Set OMP threads to avoid CPU oversubscription
export OMP_NUM_THREADS=4

# SHIRG-FIX: 2025-07-30 - Disable device_map for LoRA gradient flow
# ISSUE: device_map="auto" (model parallelism) breaks LoRA gradient flow
# SOLUTION: Set SHIRG_NO_DEVICE_MAP=1 to use data parallelism (DDP) instead
# LAVIDA IMPACT: Each GPU loads full model (~16GB) for proper gradient flow
# SHIRG IMPACT: Fixes zero gradient issue in multi-GPU LoRA training
export SHIRG_NO_DEVICE_MAP=1

echo "🚀 Starting SHIRG LoRA training on 8 GPUs"
echo "================================================"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "World size: $WORLD_SIZE"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "================================================"

# Training configuration
MODEL_PATH="KonstantinosKK/lavida-llada-v1.0-instruct-hf-transformers"
OUTPUT_DIR="./shirg_lora_checkpoints_8gpu"
SELECTION_METHOD="full"  # Options: base, entropy, edge, full
TOTAL_BATCH_SIZE=256     # Total batch size across all GPUs
LEARNING_RATE=1.8e-5
NUM_EPOCHS=3
GRADIENT_ACCUMULATION=1
NUM_WORKERS=4

# Calculate per-device batch size
PER_DEVICE_BATCH_SIZE=$((TOTAL_BATCH_SIZE / WORLD_SIZE))
echo "Per-device batch size: $PER_DEVICE_BATCH_SIZE"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run training with torchrun (PyTorch distributed launcher)
torchrun \
    --nproc_per_node=$WORLD_SIZE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    shirg/train_shirg_lora_multi_gpu.py \
    --model-path $MODEL_PATH \
    --output-dir $OUTPUT_DIR \
    --selection-method $SELECTION_METHOD \
    --batch-size $TOTAL_BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --num-epochs $NUM_EPOCHS \
    --gradient-accumulation-steps $GRADIENT_ACCUMULATION \
    --num-workers $NUM_WORKERS

# Check exit code
if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "Checkpoints saved to: $OUTPUT_DIR"
else
    echo "❌ Training failed with exit code $?"
fi