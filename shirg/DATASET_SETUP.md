# SHIRG Dataset Setup Guide

## Quick Start

1. **Download VQA v2 Dataset** (required for training):
```bash
python shirg/download_vqa_datasets.py --datasets vqa_v2 --data-dir ./data/vqa_datasets
```

2. **Other datasets** will auto-download from HuggingFace when training starts:
   - ChartQA
   - TextVQA
   - OCRVQA
   - InfoVQA

## Available Datasets

### Training Datasets
- **ChartQA**: Chart understanding (auto-downloads from HuggingFace)
- **VQA v2**: General VQA (requires manual download)
- **TextVQA**: Scene text reading (auto-downloads)
- **OCRVQA**: Book cover OCR (auto-downloads)
- **InfoVQA**: Infographic understanding (auto-downloads)

### Validation-Only Datasets
- **DocVQA**: Document understanding (no train split available)
- **MathVista**: Mathematical reasoning (no train split available)

## Training Configuration

The default configuration uses these datasets with weights:
- ChartQA: 20% (18k samples)
- TextVQA: 25% (35k samples)
- OCRVQA: 35% (70k samples)
- InfoVQA: 20% (24k samples)

Total: ~147k samples optimized for 8-hour training on 8xA100 GPUs.

## Troubleshooting

### "No real training data found" Error
This means datasets aren't available. Solutions:
1. Ensure internet connection for HuggingFace auto-downloads
2. Download VQA v2: `python shirg/download_vqa_datasets.py --datasets vqa_v2`
3. Check `--data-dir` points to correct location

### Validation Split Errors
- ChartQA uses 'val' not 'validation' (handled automatically)
- DocVQA only has validation/test splits (no training data)

### Memory Issues
- Reduce batch size if OOM errors occur
- Default: 2 samples per GPU (16 total across 8 GPUs)