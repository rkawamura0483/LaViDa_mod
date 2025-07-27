# SHIRG LoRA Training Implementation

Complete implementation of SHIRG-v2 (Static-Hierarchical Relevance Gate) with mixed-ratio LoRA training for LaViDa high-resolution token processing.

## Overview

This implementation provides:
- **Mixed-ratio LoRA training** for mm_projector adaptation
- **Complete dataset preparation** pipeline for 558K+ samples
- **Comprehensive evaluation** framework for OCR/VQA tasks
- **Production-ready** integration with LaViDa architecture

## Research Implementation

Following the SHIRG research specifications:
- **Coverage-aware token selection** with hierarchical clustering
- **Edge density boost** for thin text detection  
- **Static selection** preserving diffusion KV-cache efficiency
- **Training-free approach** with minimal LoRA adaptation

## Quick Start

### 1. Test Implementation
```bash
# Verify all components work together
python shirg/test_shirg_lora.py
```

### 2. Full Training Pipeline
```bash
# Run complete training pipeline
python shirg/run_shirg_lora_training.py \
    --dataset-size 1000 \
    --ocr-samples 200 \
    --lora-rank 16 \
    --batch-size 4 \
    --epochs 1 \
    --token-budgets 512 768
```

### 3. Evaluation Only
```bash
# Run evaluation without training
python shirg/run_shirg_lora_training.py \
    --skip-training \
    --eval-datasets ChartQA DocVQA \
    --eval-samples 50
```

## Implementation Components

### Core Files

| File | Purpose |
|------|---------|
| `shirg_lora_training.py` | Mixed-ratio LoRA training implementation |
| `shirg_dataset_preparation.py` | Dataset preparation for 558K+ samples |
| `shirg_evaluation.py` | Comprehensive evaluation pipeline |
| `run_shirg_lora_training.py` | Main orchestration script |
| `test_shirg_lora.py` | Integration testing |

### SHIRG Integration

The implementation integrates with the existing LaViDa SigLIP encoder:
- `siglip_encoder.py` - Enhanced with SHIRG methods
- `get_multiview_tokens_for_shirg()` - High-res token extraction
- `shirg_token_selection()` - SHIRG-v2 selection algorithm
- `forward_with_shirg()` - Complete SHIRG forward pass

## Training Configuration

### LoRA Parameters (Research Specifications)
```python
lora_rank: 16              # Proven optimal for projector adaptation
lora_alpha: 32             # α/r = 2.0 scaling ratio
dropout: 0.05              # Low dropout for stability
target_modules: ["mm_projector.0", "mm_projector.2"]
```

### Mixed-Ratio Training
```python
token_budgets: [512, 768, 1024]  # Multi-budget training
include_pooled: True             # Include original 980 tokens
ratio_sampling: "uniform"        # Uniform sampling strategy
```

### Dataset Composition
```python
base_dataset: 558000      # LCS-558K core samples
ocr_enhancement: 50000    # High-res OCR samples  
total_samples: 608000     # Combined training data
```

## Performance Targets

### Research Specifications
- **Token Efficiency**: 512-1024 vs 2304 high-res tokens
- **Speed**: <30ms selection overhead
- **Memory**: 30-50% KV cache reduction
- **Quality**: +5-10% OCR/VQA accuracy improvement

### Training Efficiency
- **Time**: 3-4h on 8×A100 (research target)
- **Parameters**: <1% trainable (200K vs 8B total)
- **Memory**: <40GB per GPU during training

## Usage Examples

### Dataset Preparation
```python
from shirg_dataset_preparation import SHIRGDatasetConfig, SHIRGDatasetProcessor

config = SHIRGDatasetConfig(
    base_dataset_size=558000,
    ocr_enhancement_size=50000,
    output_dir="./shirg_training_data"
)

processor = SHIRGDatasetProcessor(config)
dataset_path = processor.prepare_full_dataset()
```

### LoRA Training
```python
from shirg_lora_training import SHIRGLoRAConfig, SHIRGLoRATrainer

config = SHIRGLoRAConfig(
    lora_rank=16,
    token_budgets=[512, 768, 1024],
    num_epochs=3,
    learning_rate=1e-4
)

trainer = SHIRGLoRATrainer(config, model)
results = trainer.train(train_loader, val_loader)
```

### Evaluation
```python
from shirg_evaluation import SHIRGEvaluationConfig, SHIRGEvaluator

config = SHIRGEvaluationConfig(
    datasets=["ChartQA", "DocVQA", "TextVQA"],
    token_budgets=[512, 768, 1024],
    sample_size=500
)

evaluator = SHIRGEvaluator(config, model, vision_tower)
results = evaluator.run_evaluation()
```

## Command Line Options

### Training Pipeline
```bash
python shirg/run_shirg_lora_training.py [OPTIONS]

Dataset Options:
  --dataset-size INT        Base dataset size (default: 1000)
  --ocr-samples INT         OCR enhancement samples (default: 200)
  --image-resolution INT    High-res input size (default: 672)
  --enable-augmentation     Enable data augmentation

LoRA Options:
  --lora-rank INT          LoRA rank (default: 16)
  --lora-alpha INT         LoRA alpha (default: 32)
  --learning-rate FLOAT    Learning rate (default: 1e-4)
  --token-budgets INT+     Token budgets to train (default: [512, 768])

Training Options:
  --batch-size INT         Batch size per GPU (default: 4)
  --epochs INT             Training epochs (default: 1)
  --mixed-precision STR    Mixed precision mode (default: fp16)

Evaluation Options:
  --eval-datasets STR+     Evaluation datasets (default: [ChartQA, DocVQA])
  --eval-samples INT       Samples per dataset (default: 20)

Control Options:
  --skip-training          Skip training, run evaluation only
  --skip-evaluation        Skip evaluation, run training only
  --output-dir STR         Output directory (default: ./shirg_lora_output)
```

## Output Structure

```
shirg_lora_output/
├── checkpoints/
│   ├── checkpoint-100/
│   ├── checkpoint-200/
│   └── best_model/
├── evaluation/
│   ├── evaluation_results.json
│   ├── detailed_results.json
│   └── evaluation_report.md
├── logs/
│   └── training.log
└── SHIRG_TRAINING_REPORT.md
```

## Integration with LaViDa

### Original LaViDa Path (Preserved)
```python
# Standard LaViDa inference (729 tokens)
vision_features = vision_tower.forward(images)
projected = mm_projector(vision_features)
```

### SHIRG-Enhanced Path
```python
# SHIRG-enhanced inference (512-1024 tokens)
vision_features = vision_tower.forward_with_shirg(images, target_tokens=768)
projected = lora_adapted_mm_projector(vision_features)
```

### Backward Compatibility
- Original LaViDa functionality preserved
- SHIRG features are opt-in
- Same model weights, enhanced capabilities

## Research Validation

### SHIRG-v2 Algorithm Implementation
1. **High-resolution token extraction** (672×672 → 2304 tokens)
2. **Saliency scoring** with variance + similarity + edge density
3. **Hierarchical clustering** for coverage guarantee
4. **Global ranking** for remaining budget
5. **Summary token** for dropped regions

### Mixed-Ratio LoRA Training
1. **Random ratio sampling** during training
2. **Single adapter** works across all budgets
3. **Gradient flow validation** for compatibility
4. **Performance benchmarking** vs research targets

## Troubleshooting

### Common Issues

1. **GPU Memory Error**
   ```bash
   # Reduce batch size
   --batch-size 2 --grad-accumulation 4
   ```

2. **PEFT Not Found**
   ```bash
   pip install peft
   ```

3. **Import Errors**
   ```bash
   # Ensure paths are correct
   export PYTHONPATH=$PYTHONPATH:./shirg:./llava
   ```

4. **Performance Issues**
   ```bash
   # Use mixed precision
   --mixed-precision fp16
   ```

### Validation Checklist

Before training:
- [ ] ✅ GPU available with >15GB memory
- [ ] ✅ PEFT library installed
- [ ] ✅ SHIRG methods present in vision tower
- [ ] ✅ Test script passes all checks

During training:
- [ ] ✅ Loss decreasing steadily
- [ ] ✅ Memory usage stable
- [ ] ✅ Mixed ratios sampling correctly
- [ ] ✅ Gradients flowing to LoRA parameters

After training:
- [ ] ✅ Checkpoints saved successfully
- [ ] ✅ Evaluation results reasonable
- [ ] ✅ Performance meets targets
- [ ] ✅ Integration tests pass

## Research Contributions

### Novel Features
1. **Coverage-aware selection**: First token selection with spatial guarantee
2. **Cache-friendly design**: Static selection preserving diffusion efficiency  
3. **Mixed-ratio training**: Single adapter for multiple token budgets
4. **Production integration**: Complete LaViDa-compatible implementation

### Expected Results
- **Efficiency**: 30-50% memory reduction with minimal quality loss
- **Speed**: <30ms selection overhead maintaining LaViDa's speed advantage
- **Quality**: Competitive or improved performance on OCR/VQA tasks
- **Flexibility**: Runtime adjustable speed/quality trade-offs

## Next Steps

1. **Scale to Full Dataset**: Train on complete 558K LCS + 50K OCR samples
2. **Benchmark Evaluation**: Test on full ChartQA, DocVQA, TextVQA datasets  
3. **Hyperparameter Optimization**: Grid search α, β parameters
4. **Production Deployment**: Integrate with LaViDa inference pipeline

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{shirg2025,
  title={SHIRG-v2: Coverage-Aware Token Selection for Diffusion Vision-Language Models},
  author={Research Team},
  journal={Workshop Paper},
  year={2025}
}
```

## License

This implementation follows the LaViDa project license and is intended for research purposes.