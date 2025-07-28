# LaViDa_mod + SHIRG Research Project Rules
NEVER RUN CODE LOCALLY. I ONLY RUN IT IN COLAB. DO NOT TRY TO TEST LOCALLY

## Project Overview
This repository contains a **forked and modified version of LaViDa** with **SHIRG (Static-Hierarchical Relevance Gate)** research integration. The coder has full edit access to both the LaViDa codebase and the SHIRG implementation. The LaViDa codebase is kept mostly the same as original except for the modularized siglip encoder architecture.

## General Rules
- **Primary Research Focus**: Read `shirg/SHIRG_RESEARCH_IDEA.md` to understand the SHIRG research objective
- **LaViDa Understanding**: Analyze how LaViDa's bidirectional diffusion-language model works, particularly the prefix KV-cache mechanism
- **Integration Goals**: SHIRG is designed to improve LaViDa's high-resolution token selection without breaking the cache
- **Codebase Access**: You can modify both LaViDa core code (`llava/`, `scripts/`, etc.) AND SHIRG code (`shirg/`)

## SigLIP Encoder Architecture
The vision encoder has been refactored into a modular architecture for better code organization and research extensibility:

### Core Modules
- **`siglip_base.py`**: Clean base SigLIP implementation containing:
  - `SigLipVisionConfig`: Configuration class
  - `SigLipImageProcessor`: Image preprocessing
  - `SigLipVisionModel`: Core vision transformer
  - All transformer components (attention, MLP, encoder layers)
  - Position embedding interpolation for different resolutions

- **`siglip_shirg.py`**: SHIRG research extensions containing:
  - `SigLipShirgExtensions`: Mixin class with SHIRG methods
  - High-resolution token extraction (672×672 → 2304 tokens)
  - Distance-aware token selection algorithms
  - Dual-scale processing (hi-detail + lo-res scaffold)
  - Cache-compatible static selection methods

- **`siglip_encoder.py`**: Integration layer containing:
  - `SigLipVisionTower`: Main class that combines base + SHIRG
  - Backward compatibility with existing LaViDa code
  - Configurable SHIRG enable/disable functionality
  - LoRA training support for token selection
  - Gradient flow management for selective training

- **`original_siglip_encoder.py`**: Reference implementation (unchanged LaViDa original)

### Key Integration Points
- **Standard LaViDa Mode**: 384×384 → 729 tokens (maintains original behavior)
- **SHIRG Mode**: 672×672 → 1216 tokens (1152 selected + 64 scaffold)
- **Dynamic Switching**: Runtime switching between modes via `use_shirg` parameter
- **Cache Compatibility**: SHIRG maintains LaViDa's prefix KV-cache benefits
- **LoRA Integration**: Selective gradient flow for token selection training

### Usage Patterns
```python
# Standard LaViDa processing
vision_tower = SigLipVisionTower(model_name, config, delay_load=False)
tokens = vision_tower(images)  # [B, 729, D]

# SHIRG high-resolution processing  
config.enable_shirg = True
vision_tower = SigLipVisionTower(model_name, config, delay_load=False)
tokens = vision_tower(images, use_shirg=True)  # [B, 1216, D]

# Runtime switching
baseline_tokens = vision_tower(images, use_shirg=False)  # [B, 729, D]
shirg_tokens = vision_tower(images, use_shirg=True)      # [B, 1216, D]
```

## Project Setup & Environment
- **Execution Environment**: Code runs ONLY in Google Colab with GPU, never locally
- **Dependencies**: All dependencies managed by `shirg/install_dependencies.py` (includes both LaViDa and SHIRG requirements)
- **GPU Resources**: 40GB RAM GPU available - optimize code for this constraint
- **Deployment**: Code developed in VSCode → GitHub → Google Colab execution via `!python filename.py`

## Repository Structure
```
LaViDa_mod/                    # Forked LaViDa repository
├── llava/                     # Core LaViDa model code (EDITABLE)
│   └── model/multimodal_encoder/     # Vision encoder components
│       ├── siglip_base.py           # Base SigLIP implementation (clean)
│       ├── siglip_shirg.py          # SHIRG extensions and algorithms  
│       ├── siglip_encoder.py        # Integration layer (main interface)
│       └── original_siglip_encoder.py # Original LaViDa implementation
├── scripts/                   # Training/inference scripts (EDITABLE) 
├── eval/                      # Evaluation framework (EDITABLE)
├── shirg/                     # SHIRG research implementation (EDITABLE)
│   ├── SHIRG_RESEARCH_IDEA.md        # Main research documentation
│   ├── install_dependencies.py       # Unified dependency installer
│   ├── shirg_selector.py            # Core SHIRG algorithm
│   ├── lavida_shirg_integration.py  # LaViDa integration layer
│   └── shirg_evaluation.py          # SHIRG evaluation pipeline
├── data/                      # Test datasets
├── predict*.py               # Inference scripts
└── CLAUDE.md                 # This file
```

## Path Management
- **Colab Workflow**: Upload entire `LaViDa_mod` repo to Colab, then `cd` into the repo directory
- **Path Strategy**: Use relative paths consistently since we work from repo root
- **Integration Paths**: SHIRG code integrates with LaViDa using relative imports
- **Example Pattern**:
  ```python
  import os
  import sys
  
  # Check if running in Colab
  try:
      import google.colab
      IN_COLAB = True
  except ImportError:
      IN_COLAB = False
  
  # Paths from LaViDa_mod root
  BASE_PATH = './'
  DATA_PATH = './data/'
  SHIRG_PATH = './shirg/'
  LLAVA_PATH = './llava/'
  
  # Add SHIRG to Python path for imports
  sys.path.append(SHIRG_PATH)
  ```

## Research Objective Adherence
- **SHIRG Focus**: All code changes must align with SHIRG research objectives (training-free token selection for diffusion VLMs)
- **LaViDa Integration**: Ensure SHIRG modifications don't break LaViDa's prefix KV-cache mechanism
- **Modular Architecture**: Maintain clean separation between base SigLIP (`siglip_base.py`) and SHIRG extensions (`siglip_shirg.py`)
- **Backward Compatibility**: The integration layer (`siglip_encoder.py`) must maintain compatibility with existing LaViDa code
- **Error Resolution**: When fixing errors, ensure solutions maintain both LaViDa functionality and SHIRG goals
- **Feature Scope**: Prioritize features that improve high-resolution token selection without requiring training
- **Cache Preservation**: Any modifications must preserve LaViDa's bidirectional diffusion cache benefits
- **Validation**: Before implementing fixes, verify they support both LaViDa performance and SHIRG research hypothesis

## Error Handling & Documentation
- **Detailed Comments**: Write comprehensive English comments for every fix implemented
- **Knowledge Base**: Maintain documentation files for:
  - `shirg/SHIRG_PROBLEM_ANALYSIS_AND_SOLUTIONS.md` - SHIRG-specific issues and fixes
  - `shirg/LAVIDA_FORK_MODIFICATION_PLAN.md` - LaViDa codebase modifications
  - LaViDa-specific integration notes and token selection behaviors
  - Common error patterns between LaViDa and SHIRG integration
- **Error Prevention**: Update knowledge base after each fix to prevent repetitive errors
- **Integration Testing**: Document how changes affect both LaViDa inference and SHIRG performance
- **Comment Format**:
  ```python
  # SHIRG-FIX: [Date] - [Brief description]
  # ISSUE: [Original problem description]
  # SOLUTION: [Detailed explanation of fix]
  # LAVIDA IMPACT: [How this affects LaViDa functionality]
  # SHIRG IMPACT: [How this supports SHIRG research objective]
  ```

## Code Quality Standards
- **Type Hints**: Use type hints for all function parameters and return values
- **Docstrings**: Follow Google-style docstrings for all functions and classes
- **Error Handling**: Implement proper exception handling with informative error messages
- **Memory Management**: Be mindful of GPU memory constraints (40GB limit), especially for high-res token processing
- **Modular Organization**: 
  - Keep base SigLIP functionality in `siglip_base.py` (no SHIRG imports)
  - Place all SHIRG extensions in `siglip_shirg.py` (imports base components)
  - Use `siglip_encoder.py` only for integration and external API
  - Maintain `original_siglip_encoder.py` as reference implementation
- **Integration Points**: Clearly document where SHIRG interfaces with LaViDa components
- **Import Hygiene**: Base module should not import SHIRG, SHIRG can import base, integration imports both
- **Backwards Compatibility**: Ensure LaViDa can still run without SHIRG if needed
- **Mixin Pattern**: Use `SigLipShirgExtensions` as mixin to avoid complex inheritance

## GPU & Memory Optimization
- **Memory Monitoring**: Include GPU memory tracking, especially during SHIRG token selection
- **Batch Processing**: Implement batch processing for large datasets and multi-view images
- **Memory Cleanup**: Explicitly clear GPU memory when switching between LaViDa and SHIRG operations
- **Resource Checks**: Validate GPU availability before running GPU-intensive code
- **Token Selection Efficiency**: SHIRG selection must complete in < 30ms to preserve LaViDa's speed benefits
- **Example Pattern**:
  ```python
  import torch
  
  def check_gpu_memory_shirg():
      if torch.cuda.is_available():
          print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
          print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
          # SHIRG-specific memory tracking
          print(f"Available for token selection: {torch.cuda.memory_reserved() / 1e9:.1f}GB")
  ```

## File Organization
- **Structure**: Maintain clean project structure with clear LaViDa/SHIRG separation
  ```
  LaViDa_mod/
  ├── shirg/install_dependencies.py    # Unified installer
  ├── predict*.py                      # Modified inference scripts
  ├── llava/                          # LaViDa core (editable)
  │   ├── model/                      # Model definitions
  │   ├── train/                      # Training scripts
  │   └── eval/                       # Evaluation code
  ├── shirg/                          # SHIRG implementation
  │   ├── SHIRG_RESEARCH_IDEA.md      # Research documentation
  │   ├── shirg_selector.py           # Core algorithm
  │   ├── lavida_shirg_integration.py # Integration layer
  │   ├── shirg_evaluation.py         # Evaluation pipeline
  │   └── test_shirg_fix.py          # Testing utilities
  ├── scripts/                        # LaViDa training scripts (editable)
  ├── data/                          # Test datasets
  ├── eval/                          # Extended evaluation framework
  └── docs/                          # Documentation
  ```
- **Naming**: Use descriptive, consistent naming conventions
- **Imports**: Organize imports (stdlib, third-party, LaViDa, SHIRG) with proper separation
- **Modularity**: Keep SHIRG code modular so it can be easily integrated or disabled

## Security & Best Practices
- **Secrets Management**: Never commit API keys, tokens, or sensitive data
- **Input Validation**: Validate all inputs, especially file paths and model parameters
- **Error Logging**: Implement proper logging for debugging
- **Code Review**: Self-review code before committing

## Communication & Documentation
- **Progress Updates**: Document both LaViDa modification progress and SHIRG research findings
- **Issue Tracking**: Maintain clear issue descriptions for both integration and algorithm development
- **Code Comments**: Explain complex logic, research-specific decisions, and LaViDa-SHIRG integrations
- **README Updates**: Keep setup and execution instructions current for the integrated repository
- **Research Documentation**: Update SHIRG research files with implementation findings

## Debugging Guidelines
- **Colab Debugging**: Use Colab-specific debugging techniques for both LaViDa and SHIRG
- **GPU Debugging**: Include CUDA error checking and memory debugging for token processing
- **LaViDa-SHIRG Integration Debugging**: Debug interactions between LaViDa's cache and SHIRG's selection
- **Cache Validation**: Ensure SHIRG doesn't invalidate LaViDa's prefix KV-cache
- **Error Reproduction**: Create minimal reproducible examples for complex integration issues
- **Performance Profiling**: Profile SHIRG token selection latency to maintain LaViDa's speed advantage

## Research-Specific Rules
- **SHIRG Experiment Tracking**: Log all hyperparameters, token selection strategies, and performance metrics
- **LaViDa Baseline Preservation**: Ensure original LaViDa performance can be reproduced without SHIRG
- **Integration Reproducibility**: Ensure all SHIRG-enhanced experiments can be reproduced
- **Data Integrity**: Validate data preprocessing, especially for high-resolution multi-view images
- **Model Validation**: Implement proper evaluation metrics for both OCR/VQA tasks and token selection efficiency
- **Result Documentation**: Save and version all experimental results comparing LaViDa vs LaViDa+SHIRG
- **Cache Impact Analysis**: Measure and document how SHIRG affects LaViDa's diffusion cache performance

## Performance Monitoring
- **Execution Time**: Track and optimize slow operations, especially SHIRG token selection overhead
- **Memory Usage**: Monitor both system and GPU memory during multi-view image processing
- **Resource Utilization**: Ensure efficient use of available 40GB GPU memory for high-res tokens
- **Bottleneck Identification**: Profile code to identify performance issues in LaViDa-SHIRG integration
- **Latency Budgets**: 
  - SHIRG selection: < 30ms to preserve LaViDa's speed advantage
  - Overall inference: maintain LaViDa's ~1.9x speedup over autoregressive VLMs
- **Cache Performance**: Monitor prefix KV-cache hit rates and memory efficiency

## SHIRG-Specific Guidelines
- **Training-Free Constraint**: All SHIRG implementations must work without fine-tuning LaViDa
- **Token Selection Quality**: Prioritize keeping high-information tokens for OCR/VQA tasks
- **Multi-View Handling**: Handle LaViDa's 5-view image representation (4×336² + 1×672²) appropriately
- **Hierarchical Selection**: Implement hierarchical relevance gating as described in research documentation
- **Integration Testing**: Test SHIRG with various LaViDa model configurations and inference modes