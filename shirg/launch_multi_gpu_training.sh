#!/bin/bash
# Launch script for 8x A100 distributed training on Lambda Cloud

# Training configuration
MODEL_PATH="KonstantinosKK/lavida-llada-v1.0-instruct-hf-transformers"
OUTPUT_DIR="./shirg_lora_checkpoints_8gpu"
SELECTION_METHOD="full"
BATCH_SIZE=128  # Total batch size across all GPUs
LEARNING_RATE=1.8e-5
NUM_EPOCHS=3
GRADIENT_ACCUMULATION=1
NUM_WORKERS=4

# Create output directory
mkdir -p $OUTPUT_DIR
mkdir -p logs

# Get current timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/training_${TIMESTAMP}.log"

echo "🚀 Starting SHIRG LoRA training on 8x A100 GPUs"
echo "📁 Output directory: $OUTPUT_DIR"
echo "📊 Total batch size: $BATCH_SIZE"
echo "📝 Log file: $LOG_FILE"

# Launch distributed training using torchrun
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_shirg_lora_multi_gpu.py \
    --model-path $MODEL_PATH \
    --output-dir $OUTPUT_DIR \
    --selection-method $SELECTION_METHOD \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --num-epochs $NUM_EPOCHS \
    --gradient-accumulation-steps $GRADIENT_ACCUMULATION \
    --num-workers $NUM_WORKERS \
    2>&1 | tee $LOG_FILE

echo "✅ Training completed! Check $LOG_FILE for details."