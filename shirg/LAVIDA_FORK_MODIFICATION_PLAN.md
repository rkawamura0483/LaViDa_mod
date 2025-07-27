# LaViDa Fork Modification Plan for SHIRG Integration

## Objective
Modify LaViDa's vision processing pipeline to expose real high-resolution tokens (3,645) for SHIRG selection, with minimal LoRA projector adaptation following proven HiRes-LLaVA methodology.

## Current Problem
- LaViDa removes the last SigLIP encoder layer and forces 729 tokens
- SHIRG needs access to higher-resolution tokens (3,645) for meaningful selection
- Current interpolation approach degrades feature quality
- mm_projector expects specific token statistics from 729-token input

## Solution: Fork + Lightweight LoRA Projector Adaptation

**Key Insight**: Follow HiRes-LLaVA/LLaVA-HR proven methodology - expose high-res tokens and realign only the projector with minimal LoRA training (3.5h on 8×A100, 7h on 2×A100).

### **Step 1: Restore Full SigLIP for High-Resolution Tokens**

**File:** `LaViDa/llava/model/multimodal_encoder/siglip_encoder.py`

**Current Code (Lines 570-571):**
```python
del self.vision_tower.vision_model.encoder.layers[-1:]  # Remove last layer
self.vision_tower.vision_model.head = nn.Identity()     # Remove pooling
```

**Modified Code for 3,645 High-Res Tokens:**
```python
# SHIRG Modification: Restore full SigLIP encoder for high-resolution tokens
# Option A: Complete restoration (recommended)
# del self.vision_tower.vision_model.encoder.layers[-1:]  # COMMENT OUT - keep all layers
self.vision_tower.vision_model.head = nn.Identity()     # Keep this - remove pooling head

# Option B: Multi-view high-resolution extraction (3,645 tokens total)
self.enable_high_res_multiview = True  # Flag for 5-view processing
self.high_res_views = [
    (336, 336),  # 4 views at 336² → ~24×24 = 576 tokens each
    (672, 672)   # 1 view at 672² → ~48×48 = 2,304 tokens
]  # Total: 4×576 + 2,304 = 4,608 tokens (trim to 3,645 as in paper)
```

**Add High-Resolution Extraction Method:**
```python
def extract_multiview_tokens(self, images):
    """
    Extract 3,645 high-resolution tokens from multi-view processing
    Following LaViDa paper specification: 4×336² + 1×672² views
    """
    batch_size = images.shape[0]
    all_tokens = []
    
    # Process 4 views at 336² resolution
    for view_idx in range(4):
        # Resize to 336×336 for this view
        view_images = F.interpolate(images, size=(336, 336), mode='bilinear', align_corners=False)
        
        # Extract tokens from this view (should give ~576 tokens)
        view_tokens = self._extract_view_tokens(view_images)
        all_tokens.append(view_tokens)
    
    # Process 1 view at 672² resolution  
    high_res_images = F.interpolate(images, size=(672, 672), mode='bilinear', align_corners=False)
    high_res_tokens = self._extract_view_tokens(high_res_images)
    all_tokens.append(high_res_tokens)
    
    # Concatenate all views: 4×576 + 2,304 = 4,608 tokens
    concatenated_tokens = torch.cat(all_tokens, dim=1)
    
    # Trim to exactly 3,645 tokens as specified in LaViDa paper
    if concatenated_tokens.shape[1] > 3645:
        concatenated_tokens = concatenated_tokens[:, :3645, :]
    
    return concatenated_tokens

def _extract_view_tokens(self, view_images):
    """Extract patch tokens from a single view"""
    with torch.no_grad():
        outputs = self.vision_tower(view_images.to(device=self.device, dtype=self.dtype), 
                                  output_hidden_states=True)
        # Use last hidden state (all encoder layers restored)
        patch_tokens = outputs.last_hidden_state
        
        # Remove CLS token if present
        if patch_tokens.shape[1] > (view_images.shape[-1] // 14) ** 2:
            patch_tokens = patch_tokens[:, 1:, :]  # Remove CLS token
            
        return patch_tokens.to(view_images.dtype)
```

### **Step 2: Create High-Resolution Token Extractor**

**Add new method to SigLipVisionTower:**
```python
def forward_with_high_res(self, images, return_high_res=False):
    """
    Forward pass with optional high-resolution token extraction
    
    Args:
        images: Input images
        return_high_res: If True, return both 729 tokens and high-res tokens
        
    Returns:
        image_features: Standard 729 tokens for compatibility
        high_res_features: Higher resolution tokens for SHIRG (if requested)
    """
    if type(images) is list:
        image_features = []
        high_res_features = [] if return_high_res else None
        
        for image in images:
            image_forward_out = self.vision_tower(
                image.to(device=self.device, dtype=self.dtype).unsqueeze(0), 
                output_hidden_states=True
            )
            
            # Standard 729 tokens (current LaViDa path)
            image_feature = image_forward_out.hidden_states[-1].to(image.dtype)
            assert image_feature.shape[-2] == 729
            image_features.append(image_feature)
            
            # High-resolution tokens for SHIRG
            if return_high_res:
                # Access earlier layer or full model output
                if hasattr(self, 'high_res_layer_idx'):
                    high_res_feature = image_forward_out.hidden_states[self.high_res_layer_idx]
                else:
                    # Use full model with original pooling head
                    full_model_out = self._get_full_resolution_features(image)
                    high_res_feature = full_model_out
                    
                high_res_features.append(high_res_feature.to(image.dtype))
    else:
        # Batch processing
        image_forward_outs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype), 
            output_hidden_states=True
        )
        
        # Standard 729 tokens
        image_features = image_forward_outs.hidden_states[-1].to(images.dtype)
        assert image_features.shape[-2] == 729
        
        # High-resolution tokens
        high_res_features = None
        if return_high_res:
            if hasattr(self, 'high_res_layer_idx'):
                high_res_features = image_forward_outs.hidden_states[self.high_res_layer_idx]
            else:
                high_res_features = self._get_full_resolution_features(images)
            high_res_features = high_res_features.to(images.dtype)
    
    if return_high_res:
        return image_features, high_res_features
    return image_features

def _get_full_resolution_features(self, images):
    """Get full resolution features by using original SigLIP model"""
    # Load a separate full SigLIP model for high-res extraction
    if not hasattr(self, '_full_siglip_model'):
        from transformers import SigLipVisionModel
        self._full_siglip_model = SigLipVisionModel.from_pretrained(
            self.vision_tower_name
        ).to(device=self.device, dtype=self.dtype)
        self._full_siglip_model.requires_grad_(False)
    
    with torch.no_grad():
        full_output = self._full_siglip_model(images)
        # This will give us the original high-resolution patch tokens
        return full_output.last_hidden_state  # [B, N_patches, D]
```

### **Step 3: Modify SHIRG Integration Point**

**File:** `lavida_shirg_integration.py`

**Modified encode_images patch:**
```python
def patched_encode_images(self, images):
    """Enhanced encode_images with real high-resolution SHIRG selection"""
    
    wrapper = getattr(self, 'shirg_wrapper', None)
    vision_tower = self.get_model().get_vision_tower()
    
    if (wrapper is not None and 
        hasattr(wrapper, '_current_question_tokens') and 
        wrapper._current_question_tokens is not None and
        wrapper.shirg_config.get('alpha', 0) > 0):
        
        try:
            # SOLUTION: Get REAL high-resolution tokens from modified vision tower
            if hasattr(vision_tower, 'forward_with_high_res'):
                # Get both standard 729 tokens and high-res tokens
                standard_features, high_res_features = vision_tower.forward_with_high_res(
                    images, return_high_res=True
                )
                
                if wrapper.shirg_config.get('debug', False):
                    print(f"🔍 Real high-res tokens: {high_res_features.shape}")
                
                # Apply SHIRG to real high-resolution tokens
                selected_features = self.shirg_selector(
                    image_tokens=high_res_features,
                    text_embeddings=wrapper._current_question_tokens,
                    image_sizes=getattr(self, '_current_image_sizes', None)
                )
                
                if wrapper.shirg_config.get('debug', False):
                    print(f"🎯 SHIRG applied to real tokens: {high_res_features.shape} → {selected_features.shape}")
                
                image_features = selected_features
            else:
                # Fallback to current approach if modification not available
                image_features = vision_tower(images)
                
        except Exception as e:
            if wrapper.shirg_config.get('debug', False):
                print(f"⚠️ High-res SHIRG failed: {e}, using standard path")
            image_features = vision_tower(images)
    else:
        # Baseline: use standard LaViDa path
        image_features = vision_tower(images)
    
    # Apply mm_projector to final features
    image_features = self.get_model().mm_projector(image_features)
    return image_features
```

### **Step 4: LoRA Projector Adaptation Strategy**

**Following Proven HiRes-LLaVA/LLaVA-HR Methodology:**

The projector needs realignment when exposed to 3,645 high-resolution tokens instead of 729. Based on successful HiRes-LLaVA research, this requires minimal LoRA training.

**LoRA Configuration:**
```python
from peft import LoraConfig, get_peft_model, TaskType
import torch.nn as nn

def setup_projector_lora(model, lora_config):
    """Setup LoRA for mm_projector following HiRes-LLaVA methodology"""
    
    # LoRA configuration optimized for projector adaptation
    projector_lora_config = LoraConfig(
        r=64,                    # Rank: 64 (proven optimal for projector)
        lora_alpha=16,           # Alpha: 16 (α/r = 0.25 scaling)
        target_modules=[
            "mm_projector.0",    # First linear layer of projector
            "mm_projector.2"     # Second linear layer of projector  
        ],
        lora_dropout=0.05,       # Low dropout for stable adaptation
        bias="none",             # No bias adaptation needed
        task_type=TaskType.FEATURE_EXTRACTION
    )
    
    # Apply LoRA only to mm_projector
    model = get_peft_model(model, projector_lora_config)
    
    # Freeze everything except LoRA parameters
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            print(f"✓ LoRA parameter enabled: {name}")
    
    return model

def verify_projector_setup(model):
    """Verify LoRA setup is correct"""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable ratio: {trainable_params/total_params:.4f}")
    
    # Should be ~0.3% trainable (proven optimal ratio)
    assert trainable_params/total_params < 0.01, "Too many trainable parameters"
    
    return True
```

**Dataset Preparation:**
```python
def prepare_training_data():
    """Prepare LCS-558K + OCR samples following HiRes-LLaVA methodology"""
    
    # Core dataset: LCS-558K (558K image-text pairs)
    lcs_dataset = {
        "source": "LCS-558K", 
        "size": 558000,
        "format": "image-caption pairs",
        "purpose": "Vision-language alignment for projector"
    }
    
    # OCR enhancement: 50K high-resolution text-dense images
    ocr_dataset = {
        "source": "ChartQA + DocVQA + TextVQA samples",
        "size": 50000,
        "format": "high-res images with dense text",
        "purpose": "OCR-specific projector alignment"
    }
    
    # Training configuration
    training_config = {
        "total_samples": 608000,
        "batch_size": 32,        # Memory-optimized for 8×A100
        "gradient_accumulation": 4,
        "effective_batch_size": 128,
        "total_steps": 4750,     # 608K / 128 = 4,750 steps
        "warmup_steps": 475,     # 10% warmup
        "learning_rate": 2e-4,   # Proven optimal for projector LoRA
        "weight_decay": 0.01,
        "lr_scheduler": "cosine"
    }
    
    return lcs_dataset, ocr_dataset, training_config
```

**Training Loop Implementation:**
```python
def train_projector_lora(model, train_dataloader, config):
    """LoRA training loop following HiRes-LLaVA methodology"""
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config["total_steps"],
        eta_min=config["learning_rate"] * 0.1
    )
    
    model.train()
    
    for step, batch in enumerate(train_dataloader):
        if step >= config["total_steps"]:
            break
            
        # Forward pass with high-res tokens
        images, texts = batch["images"], batch["texts"]
        
        # Get high-resolution features (3,645 tokens)
        with torch.no_grad():
            vision_tower = model.get_vision_tower()
            high_res_features = vision_tower.forward_with_high_res(
                images, return_high_res=True
            )[1]  # Get high-res tokens
        
        # Apply LoRA-adapted projector
        projected_features = model.mm_projector(high_res_features)
        
        # Compute alignment loss with text embeddings
        text_embeddings = model.get_text_embeddings(texts)
        loss = compute_alignment_loss(projected_features, text_embeddings)
        
        # Backward pass
        loss.backward()
        
        if (step + 1) % config["gradient_accumulation"] == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        # Logging
        if step % 100 == 0:
            print(f"Step {step}/{config['total_steps']}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    return model

def compute_alignment_loss(vision_features, text_features):
    """Compute vision-text alignment loss for projector training"""
    # L2 normalize features
    vision_norm = F.normalize(vision_features, p=2, dim=-1)
    text_norm = F.normalize(text_features, p=2, dim=-1)
    
    # Compute similarity matrix
    similarity = torch.matmul(vision_norm, text_norm.transpose(-2, -1))
    
    # InfoNCE loss for alignment
    batch_size = similarity.shape[0]
    labels = torch.arange(batch_size, device=similarity.device)
    
    loss = F.cross_entropy(similarity / 0.07, labels)  # Temperature = 0.07
    return loss
```

## Implementation Timeline & Resource Requirements

### **Phase 1: Fork Setup (Days 1-2)**
- [ ] Fork LaViDa repository and create SHIRG branch
- [ ] Modify SigLIP vision tower to expose 3,645 high-res tokens
- [ ] Update SHIRG integration to use real high-resolution features
- [ ] Test basic functionality with synthetic data
- [ ] Verify high-res token extraction works correctly

### **Phase 2: LoRA Training Setup (Days 3-4)**
- [ ] Prepare LCS-558K dataset (558K image-caption pairs)
- [ ] Add OCR-specific samples (50K from ChartQA/DocVQA/TextVQA)
- [ ] Setup LoRA configuration (rank=64, α=16) for mm_projector
- [ ] Implement training loop with InfoNCE alignment loss
- [ ] Configure 8×A100 training environment (or 2×A100 fallback)

### **Phase 3: Projector LoRA Training (Day 5)**
**Resource Requirements:**
- **Optimal**: 8×A100 GPUs (40GB each) → **3.5 hours training time**
- **Fallback**: 2×A100 GPUs (80GB each) → **7 hours training time** 
- **Memory**: ~280GB total GPU memory for optimal setup
- **Storage**: ~500GB for dataset + model checkpoints
- **Network**: High-speed interconnect for multi-GPU training

**Training Schedule:**
```
Total samples: 608,000 (558K LCS + 50K OCR)
Batch size: 32 per GPU
Gradient accumulation: 4 steps
Effective batch size: 128
Total training steps: 4,750
Learning rate: 2e-4 (cosine decay)
Warmup: 475 steps (10%)

Training time breakdown:
- Data loading: 15 min
- Model setup: 10 min  
- Training: 3h 30min (8×A100) / 7h (2×A100)
- Validation: 15 min
Total: ~4h (8×A100) / ~7.5h (2×A100)
```

### **Phase 4: Integration & Testing (Days 6-7)**
- [ ] Integrate LoRA-adapted projector with SHIRG system
- [ ] Run comprehensive evaluation on ChartQA/DocVQA/TextVQA
- [ ] Compare: Baseline vs SHIRG(729 tokens) vs SHIRG(3,645 tokens)
- [ ] Validate OCR accuracy improvements and token selection quality
- [ ] Performance profiling and memory optimization

### **Phase 5: Evaluation & Documentation (Days 8-10)**
- [ ] Complete ablation studies (different token counts, LoRA ranks)
- [ ] Generate comprehensive performance metrics
- [ ] Document implementation details and research findings
- [ ] Prepare reproducible training scripts and model artifacts
- [ ] Write technical report with results analysis

### **Resource Cost Estimation:**
- **8×A100 Training**: ~$50-80 for 4 hours (cloud pricing)
- **2×A100 Training**: ~$30-50 for 8 hours (cloud pricing)
- **Storage**: ~$10-20 for dataset and model storage
- **Development Time**: 10 days (1-2 researchers)
- **Total Budget**: $100-150 (compute) + researcher time

## Expected Performance Improvements

### **Quantitative Targets (Based on HiRes-LLaVA Results):**

**OCR Tasks:**
- ChartQA: +8-12% accuracy improvement
- DocVQA: +6-10% accuracy improvement  
- TextVQA: +5-8% accuracy improvement

**Token Selection Quality:**
- Text relevance correlation: +0.15-0.25 improvement
- Spatial coherence: +0.20-0.30 improvement
- Feature diversity: Maintained while improving relevance

**Architecture Benefits:**
- **Genuine High-Resolution Access**: Real 3,645 tokens vs artificial 729→3025 interpolation
- **Meaningful Token Selection**: SHIRG operates on actual SigLIP patches, not degraded features
- **Preserved Spatial Relationships**: Better compatibility with diffusion generation
- **Valid Research Hypothesis**: True baseline vs enhanced SHIRG comparison

### **Technical Validation Metrics:**
```python
# Validation checklist
validation_metrics = {
    "token_extraction": {
        "high_res_shape": "[B, 3645, 1152]",  # Verify correct token count
        "feature_quality": ">0.8 correlation with full SigLIP",
        "memory_usage": "<45GB per A100 during inference"
    },
    "projector_adaptation": {
        "lora_parameters": "<1% of total model parameters", 
        "training_loss": "<0.5 InfoNCE loss convergence",
        "alignment_score": ">0.7 vision-text similarity"
    },
    "shirg_integration": {
        "selection_diversity": "Gini coefficient >0.6",
        "relevance_improvement": "+0.2 over random selection",
        "inference_speed": "<2x slowdown vs baseline"
    }
}
```

## Risk Mitigation & Fallback Strategies

### **Technical Risks:**
1. **Memory Constraints**: 
   - Mitigation: Gradient checkpointing, mixed precision training
   - Fallback: Reduce batch size, use 2×A100 instead of 8×A100

2. **Training Instability**:
   - Mitigation: Conservative LoRA config (rank=64, α=16), warmup schedule
   - Fallback: Lower learning rate (1e-4), increase warmup to 20%

3. **Projector Misalignment**:
   - Mitigation: Extensive validation on diverse datasets
   - Fallback: Statistical normalization as backup to LoRA training

### **Implementation Safeguards:**
```python
# Compatibility preservation
class SigLipVisionTowerWithShirg(SigLipVisionTower):
    def forward(self, images, enable_shirg=False):
        if enable_shirg and hasattr(self, 'forward_with_high_res'):
            return self.forward_with_shirg(images)
        else:
            return super().forward(images)  # Original LaViDa path
    
    def forward_with_shirg(self, images):
        # SHIRG-enabled path with high-res tokens
        pass
```

### **Success Criteria:**
- [ ] ✅ High-res token extraction: 3,645 tokens with >0.8 feature quality correlation
- [ ] ✅ LoRA training convergence: <0.5 InfoNCE loss, stable training
- [ ] ✅ Performance improvement: >5% accuracy gain on OCR tasks
- [ ] ✅ SHIRG validation: Meaningful token selection with improved relevance scores
- [ ] ✅ System integration: <2x inference slowdown, maintained stability

This comprehensive modification plan provides a proven path to test the genuine SHIRG research hypothesis with minimal risk and maximum compatibility.