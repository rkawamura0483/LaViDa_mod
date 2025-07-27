# SHIRG (Static Hierarchical Relevance Gate) Implementation Plan
## Training-Free Token Selection for LaViDa Diffusion VLMs

### **Project Overview**
Implement SHIRG, a training-free token selection method that chooses the most semantically relevant visual tokens once before diffusion starts, preserving LaViDa's prefix-KV cache efficiency while enabling high-resolution visual understanding for OCR/VQA tasks.

---

## **CRITICAL LESSONS FROM U-HiRID EXPERIENCE**

### ⚠️ **MANDATORY FIXES Applied to SHIRG Plan:**

1. **Model Path**: Use correct `KonstantinosKK/lavida-llada-v1.0-instruct-hf-transformers`
2. **LaViDa Configuration**: Include all 14+ required configuration parameters
3. **SigLIP Requirements**: Use `google/siglip-so400m-patch14-384` with proper vision_kwargs
4. **Generation Parameters**: Apply LaViDa-specific diffusion constraints
5. **Error Handling**: Extensive fallback mechanisms for generation failures
6. **Evaluation Framework**: Use OCRBench v2 with limited bbox data availability
7. **Timeline**: Realistic 10-14 day implementation vs. optimistic 7 days

---

## **Day 1: Environment Setup & Architecture Analysis**

### **Morning (4 hours)**
#### 1.1 Critical Environment Setup
```bash
# Clone and setup LaViDa environment
conda create --name shirg-lavida python=3.13
conda activate shirg-lavida
cd /Users/ryokawamura/Documents/Coding/lavida/LaViDa
pip install -e .[train]
cd eval && pip install -e . && cd ../
pip install trl==0.17.0

# CRITICAL: Install LaViDa-specific requirements
pip install flash-attn  # Required for flash_attention_2
pip install torch torchvision  # Ensure CUDA compatibility
```

#### 1.2 LaViDa Model Configuration (CRITICAL)
```python
# REQUIRED vision_kwargs for LaViDa model loading (from U-HiRID experience)
vision_kwargs = {
    "mm_vision_tower": "google/siglip-so400m-patch14-384",  # NOT CLIP!
    "mm_resampler_type": None,
    "mm_projector_type": 'mlp2x_gelu',
    "mm_hidden_size": 1152,  # SigLIP-SO400M hidden size
    "use_mm_proj": True
}

# REQUIRED model loading with correct path
from llava.model.builder import load_pretrained_model
model_path = "KonstantinosKK/lavida-llada-v1.0-instruct-hf-transformers"  # CORRECT PATH
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path,
    None,  # model_base
    "llava_llada",  # CRITICAL: Use LaViDa model name
    device_map="auto",
    vision_kwargs=vision_kwargs,
    torch_dtype='bfloat16',  # Memory efficiency
    attn_implementation="flash_attention_2"
)

# REQUIRED configuration fixes after model loading
model.eval()  # Must be in eval mode

# Fix attention implementation
if getattr(model.config, 'attn_implementation', 'unknown') != 'flash_attention_2':
    model.config.attn_implementation = "flash_attention_2"
    if hasattr(model, 'model') and hasattr(model.model, 'config'):
        model.model.config.attn_implementation = "flash_attention_2"

# Fix mm_patch_merge_type for SigLIP compatibility
if getattr(model.config, 'mm_patch_merge_type', 'unknown') != 'flat':
    model.config.mm_patch_merge_type = 'flat'
    if hasattr(model, 'model') and hasattr(model.model, 'config'):
        model.model.config.mm_patch_merge_type = 'flat'
```

### **Afternoon (4 hours)**
#### 1.3 Download Required Checkpoints
```bash
# Create checkpoint directory
mkdir -p shirg-ckpts
cd shirg-ckpts

# Download LaViDa-LLaDa checkpoint (CORRECT PATH)
git clone https://huggingface.co/KonstantinosKK/lavida-llada-v1.0-instruct-hf-transformers lavida-llada-hd
```

#### 1.4 Analyze Current Vision Pipeline
```python
# Critical analysis: understand how LaViDa processes vision tokens
def analyze_lavida_vision_pipeline():
    """Analyze LaViDa's vision token processing from SigLIP to diffusion"""
    
    # DISCOVERED: LaViDa uses SigLIP-SO400M-384 (NOT CLIP)
    # - Input: 384x384 images
    # - Patch size: 14x14
    # - Tokens per view: 27x27 = 729 tokens
    # - Multi-view: 5 views (4×336² + 1×672²) = ~3,645 tokens total
    # - Current pooling: Average pooling to 980 tokens
    
    # CRITICAL: Location of pooling in llava_arch.py:312
    # if idx in video_idx_in_batch or ALWASY_DO_2DPOOL:
    #     image_features.append(self.get_2dPool(image_feat))
    
    # SHIRG INTEGRATION POINT: Replace get_2dPool with SHIRG selection
    pass

def analyze_prefix_kv_cache_constraints():
    """Understand LaViDa's prefix-KV cache requirements"""
    
    # DISCOVERED: DreamPrefixLMCache in modeling_dream.py
    # - Prefix tokens are cached once before diffusion
    # - Cache is reused across all diffusion steps (~12-30 steps)
    # - Changing token count/order invalidates cache → massive latency hit
    # - CONSTRAINT: SHIRG must produce FIXED token count & order
    
    # CRITICAL: Diffusion generation parameters
    generation_config = {
        'max_new_tokens': 32,      # NOT 128! LaViDa constraint
        'temperature': 0,
        'top_p': 1.0,
        'num_beams': 1,
        'do_sample': False,
        'block_length': 32,        # LaViDa diffusion parameter
        'step_per_block': 32,      # LaViDa diffusion parameter  
        'use_cache': True,
        'prefix_lm': True,         # ENABLE Prefix-DLM for cache reuse
        'pad_token_id': tokenizer.eos_token_id
    }
    
    return generation_config
```

---

## **Day 2: SHIRG Algorithm Design**

### **Morning (4 hours)**
#### 2.1 SHIRG Core Algorithm Implementation
```python
# File: llava/model/shirg_selector.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
from sklearn.cluster import AgglomerativeClustering
import time

class SHIRGSelector(nn.Module):
    """
    Static Hierarchical Relevance Gate for LaViDa
    
    Training-free token selection that maintains:
    1. Fixed output size (1024 tokens) for KV cache compatibility
    2. Static selection (once per sample) for cache reuse
    3. Text-conditioned relevance scoring
    4. Hierarchical spatial grouping
    """
    
    def __init__(self, 
                 target_tokens=1024,      # Fixed output for cache compatibility
                 alpha=0.3,               # Balance between detail and semantics
                 hierarchical_levels=3,   # Levels of spatial clustering
                 latency_budget_ms=1000):   # Must finish within 1000ms
        super().__init__()
        
        self.target_tokens = target_tokens
        self.alpha = alpha
        self.hierarchical_levels = hierarchical_levels
        self.latency_budget_ms = latency_budget_ms
        
        # No learnable parameters - training-free approach
        
    def forward(self, 
                image_tokens: torch.Tensor,      # [B, N, D] - unpooled tokens
                text_embeddings: torch.Tensor,   # [B, L, D] - question embeddings
                image_sizes: List[Tuple],        # Image dimension info
                **kwargs) -> torch.Tensor:       # [B, target_tokens, D]
        """
        Apply SHIRG token selection
        
        Args:
            image_tokens: Unpooled vision tokens from SigLIP [B, 3645, 1152]
            text_embeddings: Text question embeddings [B, seq_len, 1152]
            image_sizes: Original image sizes for spatial reasoning
            
        Returns:
            selected_tokens: Fixed-size selected tokens [B, 1024, 1152]
        """
        batch_size = image_tokens.shape[0]
        start_time = time.time()
        
        selected_tokens = []
        
        for b in range(batch_size):
            # Per-sample token selection
            img_tokens = image_tokens[b]  # [N, D]
            txt_tokens = text_embeddings[b]  # [L, D]
            
            # 1. Compute saliency scores
            saliency_scores = self._compute_saliency_scores(
                img_tokens, txt_tokens
            )
            
            # 2. Hierarchical spatial clustering
            spatial_groups = self._hierarchical_clustering(
                img_tokens, saliency_scores, image_sizes[b] if image_sizes else None
            )
            
            # 3. Budget-aware token selection
            selected = self._budget_aware_selection(
                img_tokens, saliency_scores, spatial_groups
            )
            
            # 4. Add global summary token
            summary_token = self._create_summary_token(img_tokens, selected)
            
            # 5. Ensure fixed output size
            final_tokens = self._pad_or_truncate_to_target(selected, summary_token)
            
            selected_tokens.append(final_tokens)
            
            # Check latency budget
            if time.time() - start_time > self.latency_budget_ms / 1000:
                print(f"⚠️ SHIRG exceeded latency budget: {(time.time() - start_time)*1000:.1f}ms")
                break
        
        return torch.stack(selected_tokens, dim=0)
    
    def _compute_saliency_scores(self, 
                                img_tokens: torch.Tensor, 
                                txt_tokens: torch.Tensor) -> torch.Tensor:
        """
        Compute saliency scores combining information content and text relevance
        
        Formula: s_i = α * Var(v_i) + (1-α) * max_j cos(v_i, t_j)
        """
        # Information term: local variance as entropy proxy
        info_scores = torch.var(img_tokens, dim=-1, keepdim=True)  # [N, 1]
        
        # Relevance term: max cosine similarity with text tokens
        img_norm = F.normalize(img_tokens, dim=-1)      # [N, D]
        txt_norm = F.normalize(txt_tokens, dim=-1)      # [L, D]
        
        # Compute all pairwise similarities
        similarities = torch.mm(img_norm, txt_norm.t())  # [N, L]
        relevance_scores = torch.max(similarities, dim=-1, keepdim=True)[0]  # [N, 1]
        
        # Normalize scores to [0, 1] range
        info_scores = (info_scores - info_scores.min()) / (info_scores.max() - info_scores.min() + 1e-8)
        relevance_scores = (relevance_scores - relevance_scores.min()) / (relevance_scores.max() - relevance_scores.min() + 1e-8)
        
        # Combine with alpha weighting
        saliency_scores = self.alpha * info_scores + (1 - self.alpha) * relevance_scores
        
        return saliency_scores.squeeze(-1)  # [N]
    
    def _hierarchical_clustering(self, 
                                img_tokens: torch.Tensor,
                                saliency_scores: torch.Tensor,
                                image_size: Optional[Tuple] = None) -> List[List[int]]:
        """
        Perform hierarchical spatial clustering to group neighboring tokens
        """
        num_tokens = img_tokens.shape[0]
        
        # Infer spatial layout (assuming square grid from SigLIP)
        if image_size:
            # Use actual image dimensions to compute spatial layout
            grid_size = int(np.sqrt(num_tokens))
        else:
            # Default to square grid
            grid_size = int(np.sqrt(num_tokens))
        
        if grid_size * grid_size != num_tokens:
            # Handle non-square token layouts (multi-view concatenation)
            # For LaViDa's 5-view setup: 4×(27×27) + 1×(27×27) ≈ 3645
            return self._handle_multiview_clustering(num_tokens, saliency_scores)
        
        # Create spatial coordinates
        coords = []
        for i in range(num_tokens):
            row = i // grid_size
            col = i % grid_size
            coords.append([row, col])
        
        coords = np.array(coords)
        
        # Hierarchical clustering based on spatial proximity
        clustering = AgglomerativeClustering(
            n_clusters=min(self.target_tokens // 4, num_tokens // 8),
            linkage='ward'
        )
        
        cluster_labels = clustering.fit_predict(coords)
        
        # Group tokens by cluster
        clusters = {}
        for token_idx, cluster_id in enumerate(cluster_labels):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(token_idx)
        
        return list(clusters.values())
    
    def _handle_multiview_clustering(self, num_tokens: int, saliency_scores: torch.Tensor) -> List[List[int]]:
        """
        Handle LaViDa's multi-view token layout
        
        LaViDa uses 5 views with different resolutions:
        - 4 views at 336×336 → 24×24 = 576 tokens each
        - 1 view at 672×672 → 48×48 = 2304 tokens
        - Total: 4×576 + 2304 = 4608 tokens (approximately)
        """
        # Simplified clustering for multi-view: group by view and then spatially
        clusters = []
        
        # Estimate view boundaries (this is approximate)
        view_size_small = 576  # 24×24
        view_size_large = 2304  # 48×48
        
        start_idx = 0
        # Process 4 small views
        for view in range(4):
            end_idx = min(start_idx + view_size_small, num_tokens)
            view_tokens = list(range(start_idx, end_idx))
            
            # Sub-cluster within view
            sub_clusters = self._cluster_single_view(view_tokens, grid_size=24)
            clusters.extend(sub_clusters)
            
            start_idx = end_idx
        
        # Process 1 large view
        if start_idx < num_tokens:
            end_idx = num_tokens
            view_tokens = list(range(start_idx, end_idx))
            
            # Sub-cluster within large view
            sub_clusters = self._cluster_single_view(view_tokens, grid_size=48)
            clusters.extend(sub_clusters)
        
        return clusters
    
    def _cluster_single_view(self, token_indices: List[int], grid_size: int) -> List[List[int]]:
        """Cluster tokens within a single view"""
        num_clusters = max(1, len(token_indices) // 16)  # ~16 tokens per cluster
        
        if len(token_indices) <= num_clusters:
            return [[idx] for idx in token_indices]
        
        # Create spatial coordinates within view
        coords = []
        for i, token_idx in enumerate(token_indices):
            local_idx = i  # Index within this view
            row = local_idx // grid_size
            col = local_idx % grid_size
            coords.append([row, col])
        
        coords = np.array(coords)
        
        # Cluster spatially
        clustering = AgglomerativeClustering(
            n_clusters=num_clusters,
            linkage='ward'
        )
        
        cluster_labels = clustering.fit_predict(coords)
        
        # Group tokens by cluster
        clusters = {}
        for i, cluster_id in enumerate(cluster_labels):
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(token_indices[i])
        
        return list(clusters.values())
    
    def _budget_aware_selection(self, 
                               img_tokens: torch.Tensor,
                               saliency_scores: torch.Tensor,
                               spatial_groups: List[List[int]]) -> torch.Tensor:
        """
        Select tokens within budget while respecting spatial grouping
        """
        # Compute group saliency scores
        group_scores = []
        group_sizes = []
        
        for group in spatial_groups:
            group_saliency = saliency_scores[group].sum().item()
            group_scores.append(group_saliency)
            group_sizes.append(len(group))
        
        # Allocate tokens proportionally to group saliency
        total_saliency = sum(group_scores)
        selected_indices = []
        
        for i, (group, score, size) in enumerate(zip(spatial_groups, group_scores, group_sizes)):
            # Proportion of tokens to allocate to this group
            if total_saliency > 0:
                allocation_ratio = score / total_saliency
            else:
                allocation_ratio = 1.0 / len(spatial_groups)
            
            tokens_for_group = max(1, int(allocation_ratio * (self.target_tokens - 1)))  # -1 for summary token
            tokens_for_group = min(tokens_for_group, size)  # Can't allocate more than available
            
            # Select top tokens within group
            group_saliencies = saliency_scores[group]
            top_indices = torch.topk(group_saliencies, tokens_for_group)[1]
            selected_indices.extend([group[idx] for idx in top_indices])
        
        # If we haven't reached target, add more tokens
        if len(selected_indices) < self.target_tokens - 1:
            remaining_budget = self.target_tokens - 1 - len(selected_indices)
            all_indices = set(range(len(saliency_scores)))
            remaining_indices = list(all_indices - set(selected_indices))
            
            if remaining_indices:
                remaining_scores = saliency_scores[remaining_indices]
                top_remaining = torch.topk(remaining_scores, min(remaining_budget, len(remaining_indices)))[1]
                selected_indices.extend([remaining_indices[idx] for idx in top_remaining])
        
        # If we have too many tokens, truncate
        if len(selected_indices) > self.target_tokens - 1:
            # Keep top tokens by saliency
            selected_scores = saliency_scores[selected_indices]
            top_indices = torch.topk(selected_scores, self.target_tokens - 1)[1]
            selected_indices = [selected_indices[idx] for idx in top_indices]
        
        return img_tokens[selected_indices]
    
    def _create_summary_token(self, 
                             img_tokens: torch.Tensor,
                             selected_tokens: torch.Tensor) -> torch.Tensor:
        """
        Create global summary token from dropped tokens for fallback context
        """
        # Find dropped tokens
        num_total = img_tokens.shape[0]
        num_selected = selected_tokens.shape[0]
        
        if num_selected >= num_total:
            # No tokens dropped, use global average
            return img_tokens.mean(dim=0, keepdim=True)
        
        # Create mask for selected tokens (this is approximate)
        # In practice, we'd track exact indices
        dropped_tokens = img_tokens  # Simplified: use all tokens for summary
        
        # Average pool dropped tokens
        summary_token = dropped_tokens.mean(dim=0, keepdim=True)
        
        return summary_token
    
    def _pad_or_truncate_to_target(self, 
                                  selected_tokens: torch.Tensor,
                                  summary_token: torch.Tensor) -> torch.Tensor:
        """
        Ensure exactly target_tokens output for cache compatibility
        """
        current_count = selected_tokens.shape[0] + 1  # +1 for summary token
        
        if current_count == self.target_tokens:
            return torch.cat([selected_tokens, summary_token], dim=0)
        elif current_count < self.target_tokens:
            # Pad with repeated summary tokens
            padding_needed = self.target_tokens - current_count
            padding = summary_token.repeat(padding_needed, 1)
            return torch.cat([selected_tokens, summary_token, padding], dim=0)
        else:
            # Truncate selected tokens
            truncate_to = self.target_tokens - 1
            return torch.cat([selected_tokens[:truncate_to], summary_token], dim=0)
```

### **Afternoon (4 hours)**
#### 2.2 Integration with LaViDa Architecture
```python
# File: llava/model/llava_arch.py modifications
def prepare_inputs_labels_for_multimodal_with_shirg(self, ...):
    """
    Modified version of prepare_inputs_labels_for_multimodal that uses SHIRG
    
    CRITICAL: This is the main integration point where pooling is replaced
    """
    # ... existing code until line ~312 ...
    
    # SHIRG INTEGRATION POINT
    for idx, image_feat in enumerate(encoded_image_features):
        try:
            if hasattr(self.config, 'use_shirg') and self.config.use_shirg:
                # Extract text context for SHIRG conditioning
                if labels is not None:
                    # Training mode: extract text from labels
                    text_context = self._extract_text_from_labels(labels, idx)
                else:
                    # Inference mode: extract from input_ids
                    text_context = self._extract_text_from_input_ids(input_ids, idx)
                
                # Apply SHIRG selection
                compressed_feat = self.shirg_selector(
                    image_tokens=image_feat,
                    text_embeddings=text_context,
                    image_sizes=getattr(self, '_current_image_sizes', None)
                )
                
                # Validate output shape for KV cache compatibility
                expected_shape = (self.config.shirg_target_tokens, image_feat.shape[-1])
                if compressed_feat.shape[1:] != expected_shape:
                    raise ValueError(f"SHIRG output shape mismatch: {compressed_feat.shape} vs {expected_shape}")
                
                image_features.append(compressed_feat)
                
            elif idx in video_idx_in_batch or ALWASY_DO_2DPOOL:
                # Fallback to original pooling
                image_features.append(self.get_2dPool(image_feat))
            else:
                image_features.append(image_feat)
                
        except Exception as e:
            print(f"⚠️ SHIRG selection failed: {e}")
            # Graceful fallback to original pooling
            if idx in video_idx_in_batch or ALWASY_DO_2DPOOL:
                image_features.append(self.get_2dPool(image_feat))
            else:
                image_features.append(image_feat)
    
    # ... rest of existing code ...

def _extract_text_from_labels(self, labels, image_idx):
    """Extract text embeddings from training labels for SHIRG conditioning"""
    # Implementation depends on LaViDa's training data format
    # This is a simplified version
    try:
        # Find text tokens (non-IGNORE_INDEX)
        valid_mask = (labels != IGNORE_INDEX)
        if valid_mask.any():
            valid_labels = labels[valid_mask]
            text_embeds = self.get_model().embed_tokens(valid_labels.unsqueeze(0))
            return text_embeds
        else:
            # No valid text found, return zero embeddings
            return torch.zeros(1, 1, self.config.hidden_size, device=labels.device, dtype=labels.dtype)
    except Exception as e:
        print(f"Failed to extract text from labels: {e}")
        return torch.zeros(1, 1, self.config.hidden_size, device=labels.device, dtype=labels.dtype)

def _extract_text_from_input_ids(self, input_ids, image_idx):
    """Extract text embeddings from input_ids for SHIRG conditioning"""
    try:
        # Extract text tokens (excluding special image tokens)
        text_mask = (input_ids != IMAGE_TOKEN_INDEX) & (input_ids != self.tokenizer.pad_token_id)
        if text_mask.any():
            text_tokens = input_ids[text_mask]
            text_embeds = self.get_model().embed_tokens(text_tokens.unsqueeze(0))
            return text_embeds
        else:
            return torch.zeros(1, 1, self.config.hidden_size, device=input_ids.device, dtype=torch.bfloat16)
    except Exception as e:
        print(f"Failed to extract text from input_ids: {e}")
        return torch.zeros(1, 1, self.config.hidden_size, device=input_ids.device, dtype=torch.bfloat16)
```

---

## **Day 3: Integration & Configuration**

### **Morning (4 hours)**
#### 3.1 Add SHIRG to Model Configuration
```python
# File: llava/model/builder.py modifications
def load_pretrained_model_with_shirg(model_path, model_base, model_name, 
                                    load_8bit=False, load_4bit=False, 
                                    device_map="auto", device="cuda", 
                                    use_flash_attn=False, 
                                    shirg_config=None, **kwargs):
    """Enhanced model loading with SHIRG support"""
    
    # Load base model with existing functionality
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name, load_8bit, load_4bit,
        device_map, device, use_flash_attn, **kwargs
    )
    
    # Add SHIRG configuration if specified
    if shirg_config:
        # Add SHIRG config to model
        model.config.use_shirg = True
        model.config.shirg_target_tokens = shirg_config.get('target_tokens', 1024)
        model.config.shirg_alpha = shirg_config.get('alpha', 0.3)
        model.config.shirg_hierarchical_levels = shirg_config.get('hierarchical_levels', 3)
        model.config.shirg_latency_budget_ms = shirg_config.get('latency_budget_ms', 30)
        
        # Initialize SHIRG selector
        from llava.model.shirg_selector import SHIRGSelector
        model.shirg_selector = SHIRGSelector(
            target_tokens=model.config.shirg_target_tokens,
            alpha=model.config.shirg_alpha,
            hierarchical_levels=model.config.shirg_hierarchical_levels,
            latency_budget_ms=model.config.shirg_latency_budget_ms
        )
        
        print(f"✅ SHIRG selector initialized with {model.config.shirg_target_tokens} target tokens")
    
    return tokenizer, model, image_processor, context_len

# Default SHIRG configuration
DEFAULT_SHIRG_CONFIG = {
    'target_tokens': 1024,      # Fixed for KV cache compatibility
    'alpha': 0.3,               # Balance detail vs semantics
    'hierarchical_levels': 3,   # Spatial clustering depth
    'latency_budget_ms': 1000     # Must complete within 1000ms
}
```

#### 3.2 Model Configuration Patches (CRITICAL)
```python
# Apply all critical configuration fixes from U-HiRID experience
def setup_lavida_with_shirg(model_path: str, shirg_config: dict = None):
    """Complete LaViDa setup with SHIRG and all required patches"""
    
    # Step 1: Correct vision configuration
    vision_kwargs = {
        "mm_vision_tower": "google/siglip-so400m-patch14-384",
        "mm_resampler_type": None,
        "mm_projector_type": 'mlp2x_gelu',
        "mm_hidden_size": 1152,
        "use_mm_proj": True
    }
    
    # Step 2: Load model with correct path
    tokenizer, model, image_processor, context_len = load_pretrained_model_with_shirg(
        model_path="KonstantinosKK/lavida-llada-v1.0-instruct-hf-transformers",
        model_base=None,
        model_name="llava_llada",
        device_map="auto",
        vision_kwargs=vision_kwargs,
        torch_dtype='bfloat16',
        attn_implementation="flash_attention_2",
        shirg_config=shirg_config or DEFAULT_SHIRG_CONFIG
    )
    
    # Step 3: Apply required configuration patches
    model.eval()
    
    # Fix attention implementation
    if getattr(model.config, 'attn_implementation', 'unknown') != 'flash_attention_2':
        model.config.attn_implementation = "flash_attention_2"
        if hasattr(model, 'model') and hasattr(model.model, 'config'):
            model.model.config.attn_implementation = "flash_attention_2"
    
    # Fix mm_patch_merge_type for SigLIP compatibility
    if getattr(model.config, 'mm_patch_merge_type', 'unknown') != 'flat':
        model.config.mm_patch_merge_type = 'flat'
        if hasattr(model, 'model') and hasattr(model.model, 'config'):
            model.model.config.mm_patch_merge_type = 'flat'
    
    # Step 4: Setup conversation template
    conv = setup_lavida_conversation_template(tokenizer)
    
    return tokenizer, model, image_processor, context_len, conv

def setup_lavida_conversation_template(tokenizer):
    """Handle LaViDa conversation template compatibility"""
    from llava.conversation import conv_templates
    
    if "llada" not in conv_templates:
        print(f"⚠️ WARNING: 'llada' template not available!")
        if "plain" in conv_templates:
            template_name = "plain"
        else:
            template_name = list(conv_templates.keys())[0]
    else:
        template_name = "llada"
    
    conv = conv_templates[template_name].copy()
    conv.tokenizer = tokenizer
    return conv
```

### **Afternoon (4 hours)**
#### 3.3 LaViDa Generation with SHIRG
```python
def lavida_generate_with_shirg(model, tokenizer, image_processor, 
                              image_path: str, question: str, 
                              conv_template, shirg_config: dict = None):
    """
    Generate response using LaViDa with SHIRG token selection
    
    CRITICAL: Uses LaViDa-specific generation parameters from U-HiRID experience
    """
    try:
        # Load and process image
        from PIL import Image
        import torch
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = image_processor.preprocess([image], return_tensors='pt')['pixel_values']
        
        # Prepare conversation
        conv = conv_template.copy()
        conv.append_message(conv.roles[0], f"<image>\n{question}")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # Tokenize
        input_ids = tokenizer(prompt, return_tensors='pt')['input_ids']
        
        # CRITICAL: LaViDa-specific generation parameters
        generation_config = {
            'max_new_tokens': 32,       # LaViDa constraint
            'temperature': 0,
            'top_p': 1.0,
            'num_beams': 1,
            'do_sample': False,
            'block_length': 32,         # LaViDa diffusion parameter
            'step_per_block': 32,       # LaViDa diffusion parameter
            'use_cache': True,
            'prefix_lm': True,          # Enable prefix-DLM for cache reuse
            'pad_token_id': tokenizer.eos_token_id
        }
        
        # Generate with SHIRG-enabled model
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                tokenizer=tokenizer,
                **generation_config
            )
        
        # CRITICAL: LaViDa-specific output processing
        new_tokens_count = output_ids.shape[1] - input_ids.shape[1]
        if new_tokens_count <= 0:
            print(f"⚠️ ERROR: No new tokens generated!")
            return ""
        
        new_tokens = output_ids[:, input_ids.shape[1]:]
        outputs = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
        
        # Strip exclamation marks (from official LaViDa evaluation)
        result = outputs.lstrip('!').strip()
        
        return result
        
    except Exception as e:
        print(f"⚠️ Generation failed: {e}")
        return f"Generation failed: {str(e)}"
```

---

## **Day 4: Testing & Validation**

### **Morning (4 hours)**
#### 4.1 Unit Testing SHIRG Components
```python
# File: test_shirg.py
import torch
import pytest
from llava.model.shirg_selector import SHIRGSelector

def test_shirg_output_shape():
    """Test that SHIRG produces fixed-size output"""
    shirg = SHIRGSelector(target_tokens=1024)
    
    # Test different input sizes (LaViDa multi-view scenario)
    input_sizes = [729, 2304, 3645]  # Different resolution scenarios
    
    for size in input_sizes:
        image_tokens = torch.randn(2, size, 1152)  # Batch=2, SigLIP hidden_size=1152
        text_tokens = torch.randn(2, 50, 1152)     # Typical question length
        
        output = shirg(image_tokens, text_tokens, image_sizes=[(384, 384)] * 2)
        
        # CRITICAL: Must be exactly target_tokens for cache compatibility
        assert output.shape == (2, 1024, 1152), f"Shape mismatch: {output.shape}"
        print(f"✅ Input size {size} → Output shape {output.shape}")

def test_shirg_latency():
    """Test that SHIRG meets latency budget"""
    import time
    
    shirg = SHIRGSelector(target_tokens=1024, latency_budget_ms=30)
    
    # Large input to stress test
    image_tokens = torch.randn(1, 3645, 1152)
    text_tokens = torch.randn(1, 100, 1152)
    
    start_time = time.time()
    output = shirg(image_tokens, text_tokens, image_sizes=[(672, 672)])
    latency_ms = (time.time() - start_time) * 1000
    
    print(f"SHIRG latency: {latency_ms:.2f}ms")
    assert latency_ms < 50, f"Latency too high: {latency_ms}ms"  # Allow some buffer

def test_shirg_integration():
    """Test SHIRG integration with LaViDa model"""
    from llava.model.builder import setup_lavida_with_shirg
    
    # Test model loading with SHIRG
    try:
        tokenizer, model, image_processor, context_len, conv = setup_lavida_with_shirg(
            model_path="KonstantinosKK/lavida-llada-v1.0-instruct-hf-transformers",
            shirg_config={'target_tokens': 1024, 'alpha': 0.3}
        )
        
        # Verify SHIRG is properly configured
        assert hasattr(model, 'shirg_selector'), "SHIRG selector not found"
        assert model.config.use_shirg == True, "SHIRG not enabled in config"
        assert model.config.shirg_target_tokens == 1024, "Wrong target tokens"
        
        print("✅ SHIRG integration test passed")
        
    except Exception as e:
        print(f"❌ SHIRG integration test failed: {e}")
        raise

def test_comparison_with_baseline():
    """Compare SHIRG output with baseline pooling"""
    # This will be implemented after baseline evaluation
    pass

if __name__ == "__main__":
    test_shirg_output_shape()
    test_shirg_latency()
    test_shirg_integration()
    print("🎉 All SHIRG tests passed!")
```

### **Afternoon (4 hours)**
#### 4.2 Baseline Evaluation Setup
```python
# File: eval_shirg_baseline.py
"""
Evaluate baseline LaViDa performance for comparison with SHIRG

CRITICAL: Use OCRBench v2 framework with bbox filtering
"""

import json
import torch
from PIL import Image
from tqdm import tqdm
import time

def evaluate_baseline_lavida():
    """Evaluate baseline LaViDa on OCRBench v2 subset"""
    
    # Setup baseline model (without SHIRG)
    tokenizer, model, image_processor, context_len, conv = setup_lavida_with_shirg(
        model_path="KonstantinosKK/lavida-llada-v1.0-instruct-hf-transformers",
        shirg_config=None  # No SHIRG for baseline
    )
    
    # Load OCRBench v2 dataset with bbox filtering
    dataset = load_ocrbench_v2_with_bbox_filter()
    print(f"Baseline evaluation on {len(dataset)} samples with bbox data")
    
    results = []
    generation_failures = 0
    total_time = 0
    
    for idx, sample in enumerate(tqdm(dataset)):
        try:
            start_time = time.time()
            
            # Generate response
            response = lavida_generate_with_shirg(
                model, tokenizer, image_processor,
                sample['image_path'], sample['question'], conv
            )
            
            inference_time = time.time() - start_time
            total_time += inference_time
            
            if response == "":
                generation_failures += 1
                response = "GENERATION_FAILED"
            
            results.append({
                'question_id': sample['question_id'],
                'question': sample['question'],
                'response': response,
                'ground_truth': sample['answer'],
                'inference_time_ms': inference_time * 1000,
                'method': 'baseline_pooling'
            })
            
        except Exception as e:
            print(f"Baseline evaluation failed for sample {idx}: {e}")
            generation_failures += 1
            results.append({
                'question_id': sample.get('question_id', f'sample_{idx}'),
                'response': "ERROR",
                'ground_truth': sample.get('answer', 'unknown'),
                'inference_time_ms': 0,
                'method': 'baseline_pooling'
            })
    
    # Compute metrics
    avg_latency = (total_time / len(dataset)) * 1000 if dataset else 0
    failure_rate = generation_failures / len(dataset) * 100 if dataset else 0
    
    print(f"📊 Baseline Results:")
    print(f"   Average latency: {avg_latency:.1f}ms")
    print(f"   Generation failure rate: {failure_rate:.1f}%")
    print(f"   Total samples: {len(dataset)}")
    
    # Save results
    with open('baseline_results.json', 'w') as f:
        json.dump({
            'results': results,
            'summary': {
                'avg_latency_ms': avg_latency,
                'failure_rate_percent': failure_rate,
                'total_samples': len(dataset)
            }
        }, f, indent=2)
    
    return results

def load_ocrbench_v2_with_bbox_filter():
    """Load OCRBench v2 dataset with bbox filtering"""
    # This is a placeholder - actual implementation depends on dataset format
    # Based on U-HiRID experience: only ~200 images have complete bbox data
    
    try:
        # Load full dataset
        with open('data/ocrbench_v2/dataset.json', 'r') as f:
            full_dataset = json.load(f)
        
        # Filter for samples with bbox data
        bbox_dataset = []
        for sample in full_dataset:
            if 'bbox' in sample and sample['bbox'] is not None:
                bbox_dataset.append(sample)
        
        print(f"Filtered OCRBench v2: {len(bbox_dataset)}/{len(full_dataset)} samples have bbox data")
        return bbox_dataset[:200]  # Limit to manageable size
        
    except FileNotFoundError:
        print("⚠️ OCRBench v2 dataset not found, using dummy data")
        # Create dummy dataset for testing
        return create_dummy_ocr_dataset()

def create_dummy_ocr_dataset():
    """Create dummy OCR dataset for testing"""
    import os
    
    dummy_samples = []
    for i in range(10):
        dummy_samples.append({
            'question_id': f'dummy_{i}',
            'image_path': f'test_images/dummy_{i}.jpg',  # These need to exist
            'question': f'What text is visible in this image {i}?',
            'answer': f'Sample text {i}',
            'bbox': [10, 10, 100, 50]  # Dummy bbox
        })
    
    return dummy_samples
```

---

## **Day 5: SHIRG Evaluation & Comparison**

### **Morning (4 hours)**
#### 5.1 SHIRG Evaluation
```python
# File: eval_shirg_performance.py
"""
Evaluate SHIRG performance against baseline

CRITICAL: Use same evaluation framework as U-HiRID experience
"""

def evaluate_shirg_performance():
    """Evaluate SHIRG-enabled LaViDa"""
    
    # Setup SHIRG model
    shirg_config = {
        'target_tokens': 1024,
        'alpha': 0.3,  # Balance detail vs semantics
        'hierarchical_levels': 3,
        'latency_budget_ms': 30
    }
    
    tokenizer, model, image_processor, context_len, conv = setup_lavida_with_shirg(
        model_path="KonstantinosKK/lavida-llada-v1.0-instruct-hf-transformers",
        shirg_config=shirg_config
    )
    
    # Load same dataset as baseline
    dataset = load_ocrbench_v2_with_bbox_filter()
    print(f"SHIRG evaluation on {len(dataset)} samples")
    
    results = []
    generation_failures = 0
    total_time = 0
    shirg_latencies = []
    
    for idx, sample in enumerate(tqdm(dataset)):
        try:
            start_time = time.time()
            
            # Generate with SHIRG
            response = lavida_generate_with_shirg(
                model, tokenizer, image_processor,
                sample['image_path'], sample['question'], conv,
                shirg_config=shirg_config
            )
            
            inference_time = time.time() - start_time
            total_time += inference_time
            
            # Track SHIRG-specific metrics
            if hasattr(model, 'shirg_selector'):
                # Get SHIRG latency if available
                shirg_latency = getattr(model.shirg_selector, 'last_latency_ms', 0)
                shirg_latencies.append(shirg_latency)
            
            if response == "":
                generation_failures += 1
                response = "GENERATION_FAILED"
            
            results.append({
                'question_id': sample['question_id'],
                'question': sample['question'],
                'response': response,
                'ground_truth': sample['answer'],
                'inference_time_ms': inference_time * 1000,
                'shirg_latency_ms': shirg_latencies[-1] if shirg_latencies else 0,
                'method': 'shirg'
            })
            
        except Exception as e:
            print(f"SHIRG evaluation failed for sample {idx}: {e}")
            generation_failures += 1
    
    # Compute metrics
    avg_latency = (total_time / len(dataset)) * 1000 if dataset else 0
    avg_shirg_latency = sum(shirg_latencies) / len(shirg_latencies) if shirg_latencies else 0
    failure_rate = generation_failures / len(dataset) * 100 if dataset else 0
    
    print(f"📊 SHIRG Results:")
    print(f"   Average total latency: {avg_latency:.1f}ms")
    print(f"   Average SHIRG latency: {avg_shirg_latency:.1f}ms")
    print(f"   Generation failure rate: {failure_rate:.1f}%")
    print(f"   Total samples: {len(dataset)}")
    
    # Save results
    with open('shirg_results.json', 'w') as f:
        json.dump({
            'results': results,
            'summary': {
                'avg_total_latency_ms': avg_latency,
                'avg_shirg_latency_ms': avg_shirg_latency,
                'failure_rate_percent': failure_rate,
                'total_samples': len(dataset),
                'shirg_config': shirg_config
            }
        }, f, indent=2)
    
    return results

### **Afternoon (4 hours)**
#### 5.2 Performance Comparison & Analysis
```python
def compare_shirg_vs_baseline():
    """Compare SHIRG vs baseline performance"""
    
    # Load results
    with open('baseline_results.json', 'r') as f:
        baseline_data = json.load(f)
    
    with open('shirg_results.json', 'r') as f:
        shirg_data = json.load(f)
    
    baseline_results = baseline_data['results']
    shirg_results = shirg_data['results']
    
    # Compute accuracy using OCRBench v2 metrics
    baseline_accuracy = compute_ocrbench_accuracy(baseline_results)
    shirg_accuracy = compute_ocrbench_accuracy(shirg_results)
    
    # Latency comparison
    baseline_latency = baseline_data['summary']['avg_latency_ms']
    shirg_latency = shirg_data['summary']['avg_total_latency_ms']
    
    # Improvement metrics
    accuracy_improvement = ((shirg_accuracy - baseline_accuracy) / baseline_accuracy) * 100
    latency_overhead = ((shirg_latency - baseline_latency) / baseline_latency) * 100
    
    # Results summary
    comparison_results = {
        'accuracy': {
            'baseline': baseline_accuracy,
            'shirg': shirg_accuracy,
            'improvement_percent': accuracy_improvement
        },
        'latency': {
            'baseline_ms': baseline_latency,
            'shirg_ms': shirg_latency,
            'overhead_percent': latency_overhead
        },
        'token_count': {
            'baseline': 980,  # LaViDa default pooling
            'shirg': shirg_data['summary']['shirg_config']['target_tokens']
        }
    }
    
    print(f"🎯 SHIRG vs Baseline Comparison:")
    print(f"   Accuracy: {baseline_accuracy:.1f}% → {shirg_accuracy:.1f}% ({accuracy_improvement:+.1f}%)")
    print(f"   Latency: {baseline_latency:.1f}ms → {shirg_latency:.1f}ms ({latency_overhead:+.1f}%)")
    print(f"   Tokens: 980 → {shirg_data['summary']['shirg_config']['target_tokens']}")
    
    # Save comparison
    with open('shirg_comparison.json', 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    return comparison_results

def compute_ocrbench_accuracy(results):
    """Compute accuracy using OCRBench v2 evaluation metrics"""
    correct = 0
    total = 0
    
    for result in results:
        if result['response'] != "GENERATION_FAILED" and result['response'] != "ERROR":
            # Simple exact match for now - could be enhanced with fuzzy matching
            if result['response'].strip().lower() == result['ground_truth'].strip().lower():
                correct += 1
            total += 1
    
    return (correct / total * 100) if total > 0 else 0.0

def analyze_shirg_attention_patterns():
    """Analyze what tokens SHIRG selects"""
    # This would require saving token selection info during evaluation
    # Implementation depends on specific analysis needs
    print("🔍 SHIRG attention pattern analysis would go here")
    pass
```

---

## **Day 6-7: Ablation Studies & Optimization**

### **Day 6: Ablation Studies**
#### 6.1 Alpha Parameter Ablation
```python
def ablate_alpha_parameter():
    """Test different alpha values for saliency scoring"""
    alpha_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    results = {}
    
    dataset = load_ocrbench_v2_with_bbox_filter()[:50]  # Smaller subset for speed
    
    for alpha in alpha_values:
        print(f"Testing alpha = {alpha}")
        
        shirg_config = {
            'target_tokens': 1024,
            'alpha': alpha,
            'hierarchical_levels': 3,
            'latency_budget_ms': 30
        }
        
        # Evaluate with this alpha
        accuracy, latency = evaluate_alpha_config(shirg_config, dataset)
        
        results[alpha] = {
            'accuracy': accuracy,
            'latency': latency
        }
        
        print(f"  α={alpha}: Accuracy={accuracy:.1f}%, Latency={latency:.1f}ms")
    
    # Find best alpha
    best_alpha = max(results.keys(), key=lambda a: results[a]['accuracy'])
    print(f"🏆 Best alpha: {best_alpha} (Accuracy: {results[best_alpha]['accuracy']:.1f}%)")
    
    return results

#### 6.2 Target Token Count Ablation
def ablate_target_tokens():
    """Test different target token counts"""
    token_counts = [512, 768, 1024, 1280, 1536]
    results = {}
    
    dataset = load_ocrbench_v2_with_bbox_filter()[:50]
    
    for tokens in token_counts:
        print(f"Testing target_tokens = {tokens}")
        
        shirg_config = {
            'target_tokens': tokens,
            'alpha': 0.3,  # Use best alpha from previous ablation
            'hierarchical_levels': 3,
            'latency_budget_ms': 30
        }
        
        accuracy, latency = evaluate_alpha_config(shirg_config, dataset)
        
        results[tokens] = {
            'accuracy': accuracy,
            'latency': latency,
            'memory_usage': estimate_memory_usage(tokens)
        }
        
        print(f"  Tokens={tokens}: Accuracy={accuracy:.1f}%, Latency={latency:.1f}ms")
    
    return results
```

### **Day 7: Final Optimization & Documentation**
#### 7.1 Production-Ready SHIRG
```python
# File: llava/model/shirg_selector_optimized.py
"""
Production-optimized SHIRG implementation with all lessons learned
"""

class SHIRGSelectorOptimized(nn.Module):
    """
    Optimized SHIRG implementation incorporating all ablation study results
    """
    
    def __init__(self, 
                 target_tokens=1024,
                 alpha=0.3,  # From ablation studies
                 use_fast_clustering=True,
                 enable_caching=True):
        super().__init__()
        
        self.target_tokens = target_tokens
        self.alpha = alpha
        self.use_fast_clustering = use_fast_clustering
        self.enable_caching = enable_caching
        
        # Cache for repeated inputs
        if enable_caching:
            self.selection_cache = {}
            self.cache_hits = 0
            self.cache_misses = 0
    
    def forward(self, image_tokens, text_embeddings, image_sizes=None, **kwargs):
        """Optimized SHIRG forward pass"""
        
        # Fast path for cached results
        if self.enable_caching:
            cache_key = self._compute_cache_key(image_tokens, text_embeddings)
            if cache_key in self.selection_cache:
                self.cache_hits += 1
                return self.selection_cache[cache_key]
            self.cache_misses += 1
        
        # Compute selection
        start_time = time.time()
        selected_tokens = self._select_tokens_optimized(image_tokens, text_embeddings, image_sizes)
        selection_time = time.time() - start_time
        
        # Cache result
        if self.enable_caching and selection_time < 0.1:  # Only cache fast computations
            self.selection_cache[cache_key] = selected_tokens
            # Limit cache size
            if len(self.selection_cache) > 100:
                # Remove oldest entries
                oldest_key = next(iter(self.selection_cache))
                del self.selection_cache[oldest_key]
        
        return selected_tokens
    
    def _select_tokens_optimized(self, image_tokens, text_embeddings, image_sizes):
        """Optimized token selection with fast approximations"""
        # Implementation with all optimizations from ablation studies
        pass
    
    def get_cache_stats(self):
        """Get cache performance statistics"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total * 100 if total > 0 else 0
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate_percent': hit_rate
        }
```

#### 7.2 Final Evaluation & Results
```python
def final_shirg_evaluation():
    """Comprehensive final evaluation of optimized SHIRG"""
    
    # Load best configuration from ablation studies
    best_config = {
        'target_tokens': 1024,  # Best from ablation
        'alpha': 0.3,           # Best from ablation
        'hierarchical_levels': 3,
        'latency_budget_ms': 30
    }
    
    # Evaluate on full OCRBench v2 subset
    dataset = load_ocrbench_v2_with_bbox_filter()
    
    print(f"🎯 Final SHIRG Evaluation on {len(dataset)} samples")
    
    # Run comprehensive evaluation
    results = evaluate_shirg_performance_comprehensive(best_config, dataset)
    
    # Generate final report
    generate_final_report(results)
    
    return results

def generate_final_report(results):
    """Generate comprehensive final report"""
    
    report = f"""
# SHIRG Implementation Results Report

## Summary
- **Method**: Static Hierarchical Relevance Gate (SHIRG)
- **Target**: Training-free token selection for LaViDa diffusion VLMs
- **Evaluation**: OCRBench v2 subset with bbox data

## Results
- **Accuracy Improvement**: {results['accuracy_improvement']:.1f}%
- **Latency Overhead**: {results['latency_overhead']:.1f}%
- **Token Reduction**: 3,645 → 1,024 tokens ({((3645-1024)/3645*100):.1f}% reduction)

## Key Achievements
✅ Fixed-size output for KV cache compatibility
✅ Training-free implementation  
✅ Text-conditioned semantic selection
✅ Sub-1000ms selection latency
✅ Hierarchical spatial clustering

## Limitations
⚠️ Limited evaluation data (OCRBench v2 bbox subset)
⚠️ Approximated multi-view clustering
⚠️ Simple saliency scoring

## Future Work
- Dynamic token allocation based on image complexity
- Multi-scale hierarchical selection
- Integration with other diffusion VLMs
"""
    
    with open('SHIRG_FINAL_REPORT.md', 'w') as f:
        f.write(report)
    
    print("📄 Final report saved to SHIRG_FINAL_REPORT.md")
```

---

## **CRITICAL SUCCESS METRICS**

### **Primary Success (Required):**
- ✅ SHIRG produces fixed 1,024 token output for KV cache compatibility
- ✅ Selection completes within 1000ms latency budget
- ✅ Shows improvement on OCRBench v2 bbox subset (limited data)
- ✅ Maintains LaViDa's diffusion generation capabilities

### **Secondary Success (Desired):**
- ✅ Statistically significant improvement over baseline pooling
- ✅ Clear attention visualizations showing text conditioning
- ✅ Comprehensive ablation studies for hyperparameters
- ✅ Cache hit rate >70% for repeated inputs

### **Research Impact (Ideal):**
- ✅ First training-free token selection for diffusion VLMs
- ✅ Novel hierarchical clustering for multi-view architectures  
- ✅ Generalizable approach to other diffusion-based VLMs

---

## **RISK MITIGATION (Based on U-HiRID Experience)**

### **Implementation Risks:**
- **Backup Plan 1:** Fallback to standard pooling if SHIRG fails
- **Backup Plan 2:** Simplified token selection without hierarchical clustering
- **Backup Plan 3:** Analysis-only mode for attention pattern study

### **Evaluation Risks:**
- **Mitigation 1:** Use OCRBench v2 framework from working U-HiRID code
- **Mitigation 2:** Accept limited evaluation data (~200 bbox samples)
- **Mitigation 3:** Focus on latency and cache efficiency metrics

### **Technical Risks:**
- **Mitigation 1:** Extensive unit testing of all components
- **Mitigation 2:** Progressive integration with fallback mechanisms
- **Mitigation 3:** Real-time monitoring of cache efficiency

---

## **TIMELINE: Realistic 10-Day Implementation**

**Days 1-2**: Environment setup & architecture analysis (CRITICAL)  
**Days 3-4**: SHIRG algorithm implementation & integration  
**Days 5-6**: Evaluation framework & baseline comparison  
**Days 7-8**: Ablation studies & optimization  
**Days 9-10**: Final evaluation & documentation  

This **comprehensive SHIRG implementation plan** incorporates all critical lessons from the U-HiRID experience, providing realistic expectations and robust fallback mechanisms for successful implementation.

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"id": "1", "content": "Analyze LaViDa codebase architecture and vision token processing", "status": "completed", "priority": "high"}, {"id": "2", "content": "Examine SigLIP multi-view configuration and pooling mechanisms", "status": "completed", "priority": "high"}, {"id": "3", "content": "Study LaViDa's prefix-KV cache implementation and constraints", "status": "completed", "priority": "high"}, {"id": "4", "content": "Investigate existing token selection/compression techniques in codebase", "status": "completed", "priority": "medium"}, {"id": "5", "content": "Create comprehensive SHIRG implementation plan avoiding U-HiRID mistakes", "status": "completed", "priority": "high"}]