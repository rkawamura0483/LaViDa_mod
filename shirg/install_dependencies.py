#!/usr/bin/env python3
"""
Fast, one-shot installer for LaViDa on Google Colab.

Usage (in a Colab cell):
!python install_dependencies.py

If installation gets stuck, interrupt and try:
!python install_minimal.py

For minimal install from the start:
import os
os.environ['LAVIDA_MINIMAL_INSTALL'] = 'true'
!python install_dependencies.py
"""

import subprocess
import sys
import re
from pathlib import Path

FLASH_VERSION = "2.8.0"        # wheels exist for torch 2.6 + cu124 + cp311
WHEEL_REPO_TAG = "v0.3.12"     # release that contains those wheels

################################################################################
# Helpers
################################################################################


def sh(cmd: str, desc: str, timeout: int = 300) -> bool:
    "Run shell command with timeout; return True on success."
    print(f"\n🔄 {desc}\n$ {cmd}")
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=True, 
            text=True, 
            timeout=timeout,
            capture_output=True
        )
        print("✅ OK")
        return True
    except subprocess.TimeoutExpired:
        print(f"⏰ {desc} timed out after {timeout}s")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ {desc} failed\n{e.stderr or e.stdout}")
        return False


def build_wheel_url() -> str:
    """
    Compose the Flash-Attention wheel URL that matches the current runtime.
    Example --> flash_attn-2.8.0+cu124torch2.6-cp311-cp311-linux_x86_64.whl
    """
    import torch

    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    torch_mm = ".".join(torch.__version__.split(".")[:2])      # e.g. "2.6"
    cuda_raw = torch.version.cuda or "12.4"                    # fallback
    cuda_tag = re.sub(r"\.", "", cuda_raw)                     # "12.4"→"124"

    wheel = (f"flash_attn-{FLASH_VERSION}+cu{cuda_tag}torch{torch_mm}-"
             f"{py_tag}-{py_tag}-linux_x86_64.whl")

    url = (f"https://github.com/mjun0812/flash-attention-prebuild-wheels/"
           f"releases/download/{WHEEL_REPO_TAG}/{wheel}")
    return url


################################################################################
# Main routine
################################################################################

def install_minimal():
    """Minimal installation for when the full install gets stuck."""
    print("🔄 Installing minimal LaViDa requirements...")
    
    # Essential packages only
    essential = [
        "transformers>=4.37.0",
        "accelerate>=0.21.0",
        "torch",
        "torchvision", 
        "einops>=0.6.0",
        "Pillow>=9.0.0",
        "opencv-python",
        "requests"
    ]
    
    for package in essential:
        sh(f"pip install -q {package}", f"Installing {package}", timeout=120)
    
    print("✅ Minimal installation complete!")

def main():
    print("🚀 Fast install for LaViDa (Colab)")

    # Check if we should do minimal install
    import os
    if os.environ.get('LAVIDA_MINIMAL_INSTALL', '').lower() == 'true':
        return install_minimal()
    
    # Check environment
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

    # ------------------------------------------------------------------ step 0
    sh("pip install -q --upgrade pip setuptools wheel packaging ninja",
       "Upgrading pip tooling", timeout=180)

    # ------------------------------------------------------------------ step 1
    TORCH_VERS = "2.6.0+cu124"
    TORCH_INDEX = "https://download.pytorch.org/whl/cu124"
    sh(f"pip install -q torch=={TORCH_VERS} torchvision --index-url {TORCH_INDEX}",
       f"Installing PyTorch {TORCH_VERS}", timeout=600)

    # ------------------------------------------------------------------ step 2 : Flash-Attention wheel
    wheel_url = build_wheel_url()
    print(f"\n🔄 Attempting wheel install:\n    {wheel_url}")
    flash_ok = sh(f"pip install -q --no-deps {wheel_url}",
                  "Installing Flash-Attention wheel", timeout=300)

    if not flash_ok:  # fallback → source build (slow)
        print("⚠️  Pre-built wheel not found – falling back to source build (slow)…")
        flash_ok = sh("pip install flash-attn==2.5.8 --no-build-isolation",
                      "Compiling Flash-Attention from source", timeout=900)

    if not flash_ok:
        print("💣  Flash-Attention failed completely; continuing without it.")

    # ------------------------------------------------------------------ step 3 : DeepSpeed
    ds_ok = sh("pip install -q deepspeed==0.17.2", "Installing DeepSpeed", timeout=600)
    if not ds_ok:
        sh("DS_BUILD_OPS=0 pip install -q deepspeed==0.17.2",
           "Installing DeepSpeed (CUDA ops disabled)", timeout=300)

    # ------------------------------------------------------------------ step 4 : the rest (split into batches)
    
    # Batch 1: Core ML libraries (usually pre-installed or fast)
    core_ml = [
        "transformers>=4.37.0",
        "accelerate>=0.21.0", 
        "tokenizers>=0.15.0",
        "safetensors>=0.3.2",
        "huggingface_hub",
        "datasets>=2.14.7"
    ]
    sh(f"pip install -q {' '.join(core_ml)}", "Installing core ML libraries", timeout=600)
    
    # Batch 2: Basic utilities (fast installs)
    utilities = [
        "einops>=0.6.0",
        "einops-exts>=0.0.4", 
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "shortuuid",
        "httpx>=0.24.0",
        "ftfy",
        "requests",
        "tyro"
    ]
    sh(f"pip install -q {' '.join(utilities)}", "Installing utilities", timeout=300)
    
    # Batch 3: Image/Video processing (can be slow)
    media_libs = [
        "opencv-python",
        "Pillow>=9.0.0", 
        "av",
        "decord"
    ]
    sh(f"pip install -q {' '.join(media_libs)}", "Installing media processing libraries", timeout=400)
    
    # Batch 4: Optional/heavy packages that might cause issues
    optional_heavy = []
    
    # Try bitsandbytes with fallback
    if not sh("pip install -q bitsandbytes>=0.41.0", "Installing bitsandbytes", timeout=300):
        print("⚠️ bitsandbytes failed, trying CPU-only version")
        sh("pip install -q bitsandbytes --force-reinstall --no-deps", "Installing bitsandbytes (fallback)", timeout=180)
    
    # Try the rest with individual error handling
    remaining_packages = [
        ("peft>=0.4.0", "PEFT"),
        ("gradio>=4.0.0", "Gradio"), 
        ("nvidia-ml-py3", "NVIDIA ML"),
        ("sentencepiece>=0.1.99", "SentencePiece"),
        ("protobuf>=4.21.0", "Protobuf"),
        ("open_clip_torch==2.32.0", "OpenCLIP"),
        ("timm", "TIMM"),
        ("hf_transfer", "HF Transfer"),
        ("fastapi", "FastAPI"),
        ("markdown2[all]", "Markdown2"),
        ("uvicorn", "Uvicorn"),
        ("wandb", "Weights & Biases"),
        ("ipdb", "IPDB"),
        ("distance", "Distance"),
        ("Levenshtein", "Levenshtein"),
        ("apted", "APTED"),
        # SHIRG-specific dependencies
        ("scikit-learn>=1.3.0", "Scikit-learn (for SHIRG clustering)"),
        ("networkx", "NetworkX (for graph operations)"),
    ]
    
    for package, name in remaining_packages:
        if not sh(f"pip install -q {package}", f"Installing {name}", timeout=200):
            print(f"⚠️ {name} installation failed, continuing...")
    
    # Evaluation packages (often problematic)
    eval_packages = [
        ("lmms-eval", "LMMS Eval"),
        ("pycocoevalcap", "COCO Eval"),
        ("rouge-score", "ROUGE Score"), 
        ("sacrebleu", "SacreBLEU"),
        ("bert-score", "BERT Score")
    ]
    
    print("\n🔄 Installing evaluation packages (these may take time or fail)...")
    for package, name in eval_packages:
        if not sh(f"pip install -q {package}", f"Installing {name}", timeout=300):
            print(f"⚠️ {name} installation failed, skipping...")
    
    # Data science packages that are usually pre-installed in Colab
    print("\n🔄 Checking/installing data science packages...")
    data_science = [
        ("numpy>=1.24.0,<2.0", "NumPy"),
        ("scipy>=1.10.0", "SciPy"), 
        ("pandas>=2.0.0", "Pandas"),
        ("matplotlib>=3.7.0", "Matplotlib"),
        ("seaborn>=0.12.0", "Seaborn"),
        ("scikit-learn>=1.3.0", "Scikit-learn")
    ]
    
    for package, name in data_science:
        if not sh(f"pip install -q --upgrade {package}", f"Upgrading {name}", timeout=200):
            print(f"⚠️ {name} upgrade failed, using existing version...")

    # ------------------------------------------------------------------ step 5 : Install LaViDa
    print("\n🔄 Installing LaViDa package...")
    
    # Check if LaViDa exists in current directory
    # In Colab, we're already in /content/repo-name/ so LaViDa is local
    import os
    lavida_path = "./LaViDa"
    
    if os.path.exists(lavida_path):
        print(f"   Found LaViDa at: {lavida_path}")
        # Install LaViDa in editable mode
        lavida_ok = sh(f"pip install -e {lavida_path}", "Installing LaViDa package", timeout=300)
        if lavida_ok:
            print("✅ LaViDa package installed successfully!")
        else:
            print("⚠️ LaViDa package installation failed, trying standalone install...")
            sh(f"pip install -e {lavida_path}[standalone]", "Installing LaViDa (standalone)", timeout=300)
    else:
        print("⚠️ LaViDa directory not found - you may need to clone it manually")
    
    print("\n🎉 All done – enjoy your fast Flash-Attention build!")


if __name__ == "__main__":
    main()
