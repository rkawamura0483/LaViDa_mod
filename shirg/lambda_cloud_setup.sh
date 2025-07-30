#!/bin/bash
# SHIRG LoRA Training Setup for Lambda Cloud
# This script sets up everything needed for SHIRG LoRA training

echo "🚀 SHIRG LoRA Training Setup for Lambda Cloud"
echo "============================================"

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python 3.10 if needed
echo "🐍 Setting up Python 3.10..."
sudo apt install python3.10 python3.10-venv python3.10-dev -y

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3.10 -m venv shirg_env
source shirg_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Clone repository (if not already cloned)
if [ ! -d "LaViDa_mod" ]; then
    echo "📥 Cloning LaViDa_mod repository..."
    git clone https://github.com/yourusername/LaViDa_mod.git
fi

cd LaViDa_mod/shirg

# Install dependencies
echo "📦 Installing dependencies..."
python install_dependencies.py

# Install additional required packages
echo "📦 Installing additional packages..."
pip install wandb accelerate datasets peft bitsandbytes flash-attn

# Login to Hugging Face (for model access)
echo "🤗 Please login to Hugging Face for model access..."
huggingface-cli login

# Login to Weights & Biases (optional but recommended)
echo "📊 Please login to Weights & Biases for experiment tracking..."
wandb login

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p shirg_lora_checkpoints
mkdir -p data_cache
mkdir -p logs

# Run integration check
echo "🔍 Running integration check..."
python check_lora_integration.py

# Download datasets (optional - can be done during training)
echo ""
echo "📊 Dataset Download (Optional)"
echo "The datasets will be automatically downloaded during training."
echo "To pre-download them, you can run:"
echo "  python -c \"from dataset_loaders import create_data_loaders; create_data_loaders(max_samples_per_dataset=100)\""

# Final instructions
echo ""
echo "✅ Setup complete!"
echo ""
echo "🧪 To run pre-training tests:"
echo "  python test_shirg_lora_pretrain.py --selection-method full"
echo ""
echo "🚀 To start LoRA training:"
echo "  python train_shirg_lora.py --selection-method full --batch-size 16"
echo ""
echo "📊 To monitor with TensorBoard:"
echo "  tensorboard --logdir ./logs"
echo ""
echo "💡 Tips:"
echo "  - Use screen or tmux for long-running training"
echo "  - Monitor GPU memory with: watch nvidia-smi"
echo "  - Check W&B dashboard for real-time metrics"