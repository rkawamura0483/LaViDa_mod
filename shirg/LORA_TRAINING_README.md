# SHIRG LoRA Training Guide for Lambda Cloud

This guide provides comprehensive instructions for training SHIRG with Extra-LoRA adaptation on Lambda Cloud GPUs.

## 📋 Overview

SHIRG (Static Hierarchical Relevance Gate) enhances LaViDa's vision-language capabilities through intelligent token selection. This LoRA training adapts the model for better high-resolution understanding while maintaining LaViDa's speed advantages.

### Key Features
- **Extra-LoRA Footprint**: Enhanced adaptation with value matrices and fc2 layer
- **Token Dropout**: 10% dropout for training stabilization
- **Optimal Batch Size**: Automatic GPU memory optimization
- **Mixed Precision**: BF16 training for A100/H100 GPUs
- **Multi-Dataset Training**: ChartQA, DocVQA, VQA-v2 combined

## 🚀 Quick Start

### 1. Clone Repository and Setup Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/LaViDa_mod.git
cd LaViDa_mod

# Install dependencies
cd shirg
python install_dependencies.py
```

### 2. Run Pre-Training Tests

Before starting training, run comprehensive tests to ensure everything is working:

```bash
python test_shirg_lora_pretrain.py --selection-method full
```

This will test:
- Environment setup
- GPU availability and memory
- Model loading
- LoRA module targeting
- Forward/backward passes
- Mixed precision
- Token dropout
- Data loading

### 3. Optimize Batch Size

Find the optimal batch size for your GPU:

```bash
python optimize_batch_size.py --max-batch 32 --safety-margin 0.9
```

This will:
- Test different batch sizes
- Measure memory usage
- Recommend optimal settings
- Save results to `batch_size_optimization_results.json`

### 4. Start LoRA Training

Launch training with optimized settings:

```bash
python train_shirg_lora.py \
    --selection-method full \
    --batch-size 16 \
    --learning-rate 1.8e-5 \
    --num-epochs 3 \
    --output-dir ./shirg_lora_checkpoints
```

## 📊 Training Configuration

### Extra-LoRA Architecture

The Extra-LoRA footprint targets ~1.4% of parameters (136M out of 8B):

| Module | LoRA Components | Rank | Purpose |
|--------|----------------|------|---------|
| **mm_projector.fc1** | Full layer | 64 | Primary adaptation |
| **mm_projector.fc2** | Full layer | 64 | Enhanced projection |
| **SigLIP blocks 0-3** | Q, K, V | 64 | Early vision adaptation |
| **SigLIP blocks 4-5** | Q, K only | 64 | Mid-layer adaptation |

### Selection Methods

Choose from four SHIRG selection methods:

1. **base**: Original attention + text similarity
2. **entropy**: Adds noise filtering (removes ~10-15% noisy tokens)
3. **edge**: Adds edge/text priors for document tasks
4. **full**: All enhancements with radial reweighting

### Training Hyperparameters

Default settings from research:
- **Learning Rate**: 1.8e-5 (reduced from 2e-5)
- **Batch Size**: 16 (adjust based on GPU)
- **Epochs**: 3
- **Warmup Steps**: 500
- **Token Dropout**: 10% with cosine schedule
- **Mixed Precision**: BF16 for A100/H100

## 💻 Lambda Cloud Setup

### GPU Requirements

Recommended Lambda Cloud instances:
- **Minimum**: 1x A100 40GB
- **Optimal**: 2x A100 80GB
- **Best**: 4x A100 80GB or H100

### Instance Setup Script

```bash
#!/bin/bash
# Lambda Cloud setup script

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.10
sudo apt install python3.10 python3.10-venv python3.10-dev -y

# Create virtual environment
python3.10 -m venv shirg_env
source shirg_env/bin/activate

# Clone repository
git clone https://github.com/yourusername/LaViDa_mod.git
cd LaViDa_mod/shirg

# Install dependencies
pip install --upgrade pip
python install_dependencies.py

# Install additional tools
pip install wandb accelerate datasets

# Login to Hugging Face (for model access)
huggingface-cli login

# Login to Weights & Biases (optional)
wandb login
```

### Multi-GPU Training

For multi-GPU training, use accelerate:

```bash
# Configure accelerate
accelerate config

# Launch training
accelerate launch train_shirg_lora.py \
    --selection-method full \
    --batch-size 8 \
    --gradient-accumulation 2
```

## 📈 Monitoring Training

### Weights & Biases

Training progress is automatically logged to W&B:
- Loss curves
- Learning rate schedule
- Dropout rate schedule
- GPU memory usage
- Validation metrics

### Checkpointing

Checkpoints are saved:
- Every 500 steps
- End of each epoch
- Best validation loss
- Final model

Structure:
```
shirg_lora_checkpoints/
├── checkpoint-500/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── training_state.json
├── checkpoint-best/
├── checkpoint-epoch-1/
└── checkpoint-final/
```

## 🧪 Evaluation

### Running Evaluation

After training, evaluate on test sets:

```bash
python run_shirg_evaluation.py \
    --model-path ./shirg_lora_checkpoints/checkpoint-best \
    --dataset chartqa \
    --selection-method full
```

### Expected Performance

Based on research targets:

| Dataset | Baseline | SHIRG-Full | Improvement |
|---------|----------|------------|-------------|
| ChartQA | 0.0 | +4.0 CIDEr | +4.0 |
| DocVQA | 0.0 | +2.5 EM | +2.5 |
| TextVQA | 0.0 | +2.8 Acc | +2.8 |

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use gradient accumulation
   
2. **Slow Training**
   - Ensure Flash Attention 2 is installed
   - Check GPU utilization
   - Verify mixed precision is enabled

3. **Poor Convergence**
   - Check learning rate (try 1e-5 to 5e-5)
   - Ensure proper data loading
   - Verify LoRA modules are targeted correctly

### Debug Mode

Enable debug mode for detailed logging:

```bash
export SHIRG_DEBUG=1
python train_shirg_lora.py --selection-method full
```

## 📚 Advanced Configuration

### Custom LoRA Configuration

Modify `shirg_lora_config.py` for custom settings:

```python
config = ShirgLoraConfig(
    rank=128,  # Increase for more capacity
    alpha=256,  # Usually 2x rank
    dropout=0.1,  # LoRA dropout
    learning_rate=2e-5,
    token_dropout_rate=0.15,  # Increase for regularization
)
```

### Dataset Customization

Add custom datasets in `train_shirg_lora.py`:

```python
dataset_configs = {
    "chartqa": {"weight": 0.3, "max_samples": 10000},
    "docvqa": {"weight": 0.3, "max_samples": 10000},
    "vqa_v2": {"weight": 0.4, "max_samples": 10000},
    "custom_dataset": {"weight": 0.2, "max_samples": 5000},  # Add custom
}
```

## 📄 Citation

If you use this code for research, please cite:

```bibtex
@article{shirg2025,
  title={SHIRG: Static Hierarchical Relevance Gate for High-Resolution Diffusion VLMs},
  author={Your Name},
  year={2025}
}
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Run pre-training tests before submitting PRs
2. Follow the coding style
3. Add tests for new features
4. Update documentation

## 📞 Support

For issues or questions:
- Open an issue on GitHub
- Check existing issues first
- Include error logs and system info