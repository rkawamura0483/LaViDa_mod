# SHIRG LoRA Training - Colab Quick Start Guide

## 🚀 Quick Start (5 minutes)

### 1. Open in Colab and Clone Repository

```python
# Cell 1: Clone repository
!git clone https://github.com/yourusername/LaViDa_mod.git
%cd LaViDa_mod

# Cell 2: Install dependencies
!python shirg/install_dependencies.py
```

### 2. Quick Training Demo

```python
# Cell 3: Run quick demo with synthetic data
%cd /content/LaViDa_mod
!python shirg/train_shirg_lora_colab.py
```

## 📊 Interactive Training (Recommended)

### Option 1: Interactive UI

```python
# Cell 4: Interactive training interface
import sys
sys.path.append('/content/LaViDa_mod')
sys.path.append('/content/LaViDa_mod/shirg')

from train_shirg_lora_colab import interactive_train
interactive_train()
```

### Option 2: Custom Configuration

```python
# Cell 5: Custom training
from train_shirg_lora_colab import train_in_colab, get_colab_optimized_config

# Get optimized config
config = get_colab_optimized_config()

# Modify as needed
config.num_train_epochs = 2
config.per_device_train_batch_size = 4

# Start training
train_in_colab(
    config=config,
    num_samples=200,  # Number of demo samples
    checkpoint_dir="./my_checkpoints"
)
```

## 🎯 Colab-Specific Features

### Memory-Optimized Defaults

The Colab version automatically detects your GPU and sets optimal parameters:

| GPU Type | Memory | Batch Size | Gradient Accum | LoRA Rank |
|----------|--------|------------|----------------|-----------|
| T4 (Free) | 15GB | 2 | 8 | 32 |
| V100 | 16GB | 4 | 4 | 32 |
| A100 | 40GB | 8 | 2 | 64 |

### Automatic Adjustments

- **FP16** instead of BF16 for T4 GPUs
- **Gradient checkpointing** enabled by default
- **Reduced LoRA rank** (32 instead of 64)
- **Simplified target modules** for faster training

## 🔧 Advanced Usage

### Use Real Data

```python
# Cell 6: Train with real dataset
from datasets import load_dataset

# Load your dataset
dataset = load_dataset("your_dataset", split="train[:1000]")

# Convert to required format
train_data = []
for item in dataset:
    train_data.append({
        "image": item["image"],
        "question": item["question"],
        "answer": item["answer"],
        "id": item["id"]
    })

# Train with real data
from train_shirg_lora import ShirgLoraTrainer
trainer = ShirgLoraTrainer(config=config)
# ... setup with real data
```

### Monitor GPU Usage

```python
# Cell 7: GPU monitoring
import GPUtil
import time

def monitor_gpu():
    while True:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"GPU: {gpu.name}")
            print(f"Memory: {gpu.memoryUsed}/{gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)")
            print(f"Temp: {gpu.temperature}°C")
            print("-" * 40)
        time.sleep(5)

# Run in background
import threading
monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
monitor_thread.start()
```

### Save to Google Drive

```python
# Cell 8: Save checkpoints to Drive
from google.colab import drive
drive.mount('/content/drive')

# Train with Drive output
train_in_colab(
    config=config,
    checkpoint_dir="/content/drive/MyDrive/shirg_checkpoints"
)
```

## 📈 Expected Performance

### Training Time (Colab Free T4)
- 100 samples: ~5 minutes
- 500 samples: ~20 minutes
- 1000 samples: ~40 minutes

### Memory Usage
- Base model: ~8GB
- With LoRA (rank 32): +0.5GB
- Per batch (size 2): +1GB

## 🚨 Troubleshooting

### Out of Memory

```python
# Reduce batch size to 1
config.per_device_train_batch_size = 1
config.gradient_accumulation_steps = 16

# Or reduce LoRA rank
config.rank = 16
config.alpha = 32
```

### Session Timeout

```python
# Keep session alive
import IPython
from IPython.display import display, HTML
import time

def keep_alive():
    while True:
        time.sleep(60)
        display(HTML('<script>console.log("Keep alive")</script>'))

# Run in background
import threading
keep_thread = threading.Thread(target=keep_alive, daemon=True)
keep_thread.start()
```

### Slow Training

```python
# Enable mixed precision
config.fp16 = True

# Disable logging
config.logging_steps = 50
config.save_steps = 500
```

## 🎓 Learning Resources

### Understanding the Code

1. **Selection Methods**:
   - `base`: Simple attention-based selection
   - `entropy`: Removes noisy tokens
   - `edge`: Better for documents
   - `full`: All enhancements

2. **LoRA Rank**:
   - Higher rank = more capacity but more memory
   - Start with 32 for Colab
   - Use 64 for A100

3. **Batch Size**:
   - Limited by GPU memory
   - Use gradient accumulation for larger effective batch

### Next Steps

1. Try different selection methods
2. Experiment with LoRA rank
3. Test on real datasets
4. Fine-tune hyperparameters

## 💡 Tips for Colab

1. **Use GPU Runtime**: Runtime → Change runtime type → GPU
2. **Monitor Usage**: Check GPU memory in runtime logs
3. **Save Progress**: Checkpoints every 100 steps
4. **Quick Iterations**: Use small datasets first
5. **Colab Pro**: Consider upgrading for longer sessions

## 📝 Example Notebook Structure

```python
# Complete Colab notebook

# Cell 1: Setup
!git clone https://github.com/yourusername/LaViDa_mod.git
%cd LaViDa_mod
!python shirg/install_dependencies.py

# Cell 2: Imports
import sys
sys.path.extend(['/content/LaViDa_mod', '/content/LaViDa_mod/shirg'])
from train_shirg_lora_colab import *

# Cell 3: Configuration
config = get_colab_optimized_config()
print(f"Config: {config}")

# Cell 4: Training
train_in_colab(config=config, num_samples=100)

# Cell 5: Evaluation (optional)
# Load checkpoint and evaluate
```

Happy training! 🎉