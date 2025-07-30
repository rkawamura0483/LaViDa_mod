#!/bin/bash
# SHIRG LoRA Multi-GPU Training Script for 8 x A100 GPUs
# This script handles distributed training setup for SHIRG on Lambda Cloud
# Author: Research Implementation
# Date: 2025-07-30

# SHIRG-FIX: 2025-07-30 - Add automatic GPU memory cleanup
# ISSUE: Stopped training leaves GPU memory allocated
# SOLUTION: Clear GPU memory before and after training, handle interrupts
# LAVIDA IMPACT: Ensures clean GPU state for any LaViDa operations
# SHIRG IMPACT: Prevents OOM errors from previous training runs

# Function to clean up GPU memory
cleanup_gpu_memory() {
    echo "🧹 Cleaning up GPU memory..."
    
    # Kill any existing training processes
    pkill -f train_shirg_lora_multi_gpu 2>/dev/null
    pkill -f torchrun 2>/dev/null
    
    # Small delay to ensure processes are killed
    sleep 2
    
    # Python command to clear CUDA cache
    python3 -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print('✅ GPU memory cleared')
else:
    print('⚠️ No CUDA devices available')
" 2>/dev/null || echo "⚠️ Could not clear GPU cache (Python/PyTorch not available yet)"
    
    # Show GPU status
    echo "📊 Current GPU status:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader,nounits || echo "nvidia-smi not available"
    echo ""
}

# Trap function to handle script interruption (Ctrl+C)
cleanup_on_exit() {
    echo -e "\n\n⚠️ Training interrupted! Cleaning up..."
    cleanup_gpu_memory
    exit 1
}

# Set up trap for SIGINT (Ctrl+C) and SIGTERM
trap cleanup_on_exit SIGINT SIGTERM

# Clean GPU memory before starting
cleanup_gpu_memory

# Set environment variables for distributed training
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WORLD_SIZE=8
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Disable tokenizers parallelism to avoid warnings
export TOKENIZERS_PARALLELISM=false

# Set OMP threads to avoid CPU oversubscription
export OMP_NUM_THREADS=4

# SHIRG-FIX: 2025-07-30 - Disable excessive debug output
# ISSUE: Too much debug output makes training logs hard to read
# SOLUTION: Set SHIRG_DEBUG=0 to disable debug prints
export SHIRG_DEBUG=0

# SHIRG-FIX: 2025-07-30 - Ensure spawn method for CUDA multiprocessing
# ISSUE: Fork method causes "Cannot re-initialize CUDA" errors in DataLoader
# SOLUTION: Python will use spawn method (set in training scripts)
# LAVIDA IMPACT: Prevents worker process crashes during data loading
# SHIRG IMPACT: Enables stable multi-worker data loading on all GPUs
# NOTE: If crashes persist, reduce NUM_WORKERS or set to 0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Additional memory and debugging settings to prevent segfaults
export CUDA_LAUNCH_BLOCKING=0  # Disable for better performance
export TORCH_DISTRIBUTED_DEBUG=OFF  # Turn ON for debugging
export NCCL_DEBUG=WARN  # Set to INFO for more debugging

# SHIRG-FIX: 2025-07-30 - Disable device_map for LoRA gradient flow
# ISSUE: device_map="auto" (model parallelism) breaks LoRA gradient flow
# SOLUTION: Set SHIRG_NO_DEVICE_MAP=1 to use data parallelism (DDP) instead
# LAVIDA IMPACT: Each GPU loads full model (~16GB) for proper gradient flow
# SHIRG IMPACT: Fixes zero gradient issue in multi-GPU LoRA training
export SHIRG_NO_DEVICE_MAP=1

# SHIRG-FIX: 2025-07-30 - Disable checkpoint cleanup to prevent crashes
# ISSUE: Checkpoint cleanup with shutil.rmtree() can crash on distributed filesystems
# SOLUTION: Disable cleanup during training, manually clean up old checkpoints later
# LAVIDA IMPACT: None
# SHIRG IMPACT: Prevents crashes at checkpoint-500 on Lambda Cloud
export SHIRG_DISABLE_CHECKPOINT_CLEANUP=1

echo "🚀 Starting SHIRG LoRA training on 8 GPUs"
echo "================================================"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "World size: $WORLD_SIZE"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "📊 Validation: DISABLED (to prevent OOM)"
echo "================================================"
echo ""
echo "⚠️  Note: Validation is disabled by default to prevent OOM errors."
echo "   Remove --skip-validation flag if you have enough memory."
echo ""

# Training configuration
MODEL_PATH="KonstantinosKK/lavida-llada-v1.0-instruct-hf-transformers"
OUTPUT_DIR="./shirg_lora_checkpoints_8gpu"
SELECTION_METHOD="full"  # Options: base, entropy, edge, full
TOTAL_BATCH_SIZE=32    # Total batch size across all GPUs (increased for real data)
# IMPORTANT: Reduced batch size might help with memory issues
# Try TOTAL_BATCH_SIZE=4 if you get segfaults
LEARNING_RATE=1.8e-5
NUM_EPOCHS=3
GRADIENT_ACCUMULATION=1
NUM_WORKERS=0  # Reduce to 0 if multiprocessing errors persist
DATA_DIR="./data/vqa_datasets"  # Path to downloaded VQA datasets

# Calculate per-device batch size
PER_DEVICE_BATCH_SIZE=$((TOTAL_BATCH_SIZE / WORLD_SIZE))
echo "Per-device batch size: $PER_DEVICE_BATCH_SIZE"

# Create output directory
mkdir -p $OUTPUT_DIR

# Check if datasets exist
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Dataset directory not found: $DATA_DIR"
    echo "   Please run: python shirg/download_vqa_datasets.py --data-dir $DATA_DIR"
    exit 1
fi

# Run training with torchrun (PyTorch distributed launcher)
# Add --rdzv-backend=c10d for more stable process group initialization
torchrun \
    --nproc_per_node=$WORLD_SIZE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --rdzv-backend=c10d \
    --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
    shirg/train_shirg_lora_multi_gpu.py \
    --model-path $MODEL_PATH \
    --output-dir $OUTPUT_DIR \
    --selection-method $SELECTION_METHOD \
    --batch-size $TOTAL_BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --num-epochs $NUM_EPOCHS \
    --gradient-accumulation-steps $GRADIENT_ACCUMULATION \
    --num-workers $NUM_WORKERS \
    --data-dir $DATA_DIR \
    --skip-validation  # Skip validation to prevent OOM

# Check exit code
EXIT_CODE=$?

# Always clean up GPU memory after training
cleanup_gpu_memory

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo "Checkpoints saved to: $OUTPUT_DIR"
else
    echo "❌ Training failed with exit code $EXIT_CODE"
    echo ""
    echo "Troubleshooting tips:"
    echo "1. If you see 'Cannot re-initialize CUDA in forked subprocess':"
    echo "   - The multiprocessing fix should handle this automatically"
    echo "   - If it persists, set NUM_WORKERS=0 in this script"
    echo "2. Run 'python shirg/test_multiprocessing_fix.py' to verify setup"
    echo "3. Check that all GPUs are visible: nvidia-smi"
    echo "4. Ensure sufficient GPU memory is available"
fi

# Final GPU status
echo ""
echo "📊 Final GPU memory status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader,nounits || echo "nvidia-smi not available"