# How to Use Real VQA Datasets for SHIRG Training

This guide explains how to download and use real VQA datasets for SHIRG training instead of synthetic data.

## Step 1: Download VQA Datasets

First, download the real VQA datasets (ChartQA, DocVQA, VQA v2, TextVQA, OCR-VQA):

```bash
python shirg/download_vqa_datasets.py --data-dir ./data/vqa_datasets
```

This will download:
- **ChartQA**: ~18k training samples for chart understanding
- **DocVQA**: ~40k training samples for document VQA  
- **VQA v2**: ~440k training samples for general VQA
- **TextVQA**: ~35k training samples for text-based VQA
- **OCR-VQA**: ~200k training samples for OCR tasks

**Total**: ~750k+ training samples (vs 2400 synthetic samples)

## Step 2: Verify Downloaded Datasets

Check what datasets were successfully downloaded:

```bash
python shirg/create_real_dataset_config.py --data-dir ./data/vqa_datasets
```

This will show:
- Number of samples per dataset
- Total available training samples
- Suggested training configuration

## Step 3: Run Training with Real Data

The training script automatically detects and uses real datasets:

```bash
# For 8 GPU training
bash shirg/run_8gpu_training.sh

# For single GPU training  
python shirg/train_shirg_lora.py \
    --batch-size 8 \
    --data-dir ./data/vqa_datasets
```

## What Changed?

1. **Batch Size**: Increased from 8 to 64 for 8 GPUs (to match research paper)
2. **Dataset Size**: From 2,400 synthetic samples to 750k+ real samples
3. **Steps per Epoch**: From ~300 to ~45,000 (as expected in research)
4. **Training Time**: Now properly utilizes 8 hours on 8xA100 GPUs

## Expected Training Output

With real datasets, you should see:

```
📊 Preparing datasets...
🔍 Using real datasets from: ./data/vqa_datasets
✅ ChartQA found: 18,317 training samples
✅ DocVQA found: 39,463 training samples  
✅ VQA v2 found: 443,757 training samples
✅ TextVQA found: 34,602 training samples
✅ OCR-VQA found: 197,293 training samples

📊 Total samples loaded: 733,432
✅ Training samples: 733,432
✅ Validation samples: 5,000

Starting epoch 1/3
Steps per epoch: 45,839
```

## Troubleshooting

### If datasets don't download:
- Check internet connection
- Some datasets require accepting terms (visit HuggingFace pages)
- Try downloading individual datasets: `--datasets chartqa vqa_v2`

### If training still uses synthetic data:
- Verify data directory path is correct
- Check that JSON files exist in dataset subdirectories
- Look for error messages during dataset loading

### Memory issues with large datasets:
- Reduce max_samples_per_dataset in training config
- Use gradient accumulation to simulate larger batches
- Enable gradient checkpointing

## Performance Tips

1. **Multi-GPU Training**: Use all 8 GPUs for ~45k steps/epoch
2. **Mixed Precision**: Enable fp16/bf16 for memory efficiency
3. **Gradient Accumulation**: Set to 4-8 for effective batch size
4. **Data Loading**: Keep num_workers=0 to avoid CUDA issues

## Research Alignment

With real datasets, your training now matches the SHIRG research paper:
- 3 epochs × 45k steps = 135k total steps
- Batch size 64 across 8 GPUs
- ~750k diverse VQA samples
- Proper convergence in 7-8 hours

This ensures your SHIRG LoRA training produces meaningful results comparable to the research baseline.