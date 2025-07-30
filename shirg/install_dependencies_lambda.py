#!/usr/bin/env python3
"""
Fast installer for LaViDa SHIRG on Lambda Cloud.
Adapted from Colab version for Lambda GPU Cloud instances.

Usage:
python install_dependencies_lambda.py

For minimal install:
python install_dependencies_lambda.py --minimal
"""

import subprocess
import sys
import re
import argparse
import os
from pathlib import Path

FLASH_VERSION = "2.5.8"  # More stable version for Lambda Cloud

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


def detect_cuda_version():
    """Detect CUDA version on Lambda Cloud"""
    try:
        result = subprocess.run(
            "nvidia-smi | grep 'CUDA Version' | awk '{print $9}'",
            shell=True,
            capture_output=True,
            text=True
        )
        cuda_version = result.stdout.strip()
        print(f"Detected CUDA version: {cuda_version}")
        return cuda_version
    except:
        return "12.1"  # Default for Lambda Cloud


################################################################################
# Main routine
################################################################################

def install_minimal():
    """Minimal installation for quick setup."""
    print("🔄 Installing minimal LaViDa SHIRG requirements...")
    
    # Essential packages only
    essential = [
        "torch>=2.1.0",
        "torchvision",
        "torchaudio",
        "transformers>=4.37.0",
        "accelerate>=0.21.0",
        "peft>=0.4.0",
        "datasets>=2.14.0",
        "einops>=0.6.0",
        "Pillow>=9.0.0",
        "opencv-python",
        "requests",
        "wandb",
        "tqdm",
        "pyyaml",
        "safetensors",
        "scikit-learn>=1.3.0",
        "GPUtil",  # For batch size optimization
        "psutil",  # For system monitoring
    ]
    
    for package in essential:
        sh(f"pip install -q {package}", f"Installing {package}", timeout=120)
    
    print("✅ Minimal installation complete!")


def main():
    parser = argparse.ArgumentParser(description="Install LaViDa SHIRG on Lambda Cloud")
    parser.add_argument("--minimal", action="store_true", help="Minimal installation only")
    parser.add_argument("--cuda", type=str, default=None, help="CUDA version (e.g., 12.1)")
    args = parser.parse_args()
    
    print("🚀 Fast install for LaViDa SHIRG (Lambda Cloud)")
    
    if args.minimal:
        return install_minimal()
    
    # Detect CUDA version
    cuda_version = args.cuda or detect_cuda_version()
    cuda_major = cuda_version.split('.')[0]
    
    # ------------------------------------------------------------------ step 0
    sh("pip install -q --upgrade pip setuptools wheel packaging ninja",
       "Upgrading pip tooling", timeout=180)

    # ------------------------------------------------------------------ step 1: PyTorch
    # Lambda Cloud usually has PyTorch pre-installed, but let's ensure correct version
    if cuda_major == "12":
        TORCH_INDEX = "https://download.pytorch.org/whl/cu121"
    elif cuda_major == "11":
        TORCH_INDEX = "https://download.pytorch.org/whl/cu118"
    else:
        TORCH_INDEX = "https://download.pytorch.org/whl/cu121"  # Default
    
    sh(f"pip install torch torchvision torchaudio --index-url {TORCH_INDEX}",
       f"Installing PyTorch for CUDA {cuda_version}", timeout=600)

    # ------------------------------------------------------------------ step 2: Flash-Attention
    # Try pre-built wheel first, then source
    flash_ok = sh(f"pip install flash-attn=={FLASH_VERSION} --no-build-isolation",
                  "Installing Flash-Attention", timeout=900)
    
    if not flash_ok:
        print("⚠️ Flash-Attention failed; continuing without it.")

    # ------------------------------------------------------------------ step 3: Core ML libraries
    core_ml = [
        "transformers>=4.37.0",
        "accelerate>=0.21.0",
        "tokenizers>=0.15.0",
        "safetensors>=0.3.2",
        "huggingface_hub",
        "datasets>=2.14.7",
        "peft>=0.4.0",
        "bitsandbytes>=0.41.0",
    ]
    sh(f"pip install -q {' '.join(core_ml)}", "Installing core ML libraries", timeout=600)
    
    # ------------------------------------------------------------------ step 4: SHIRG-specific
    shirg_deps = [
        "einops>=0.6.0",
        "einops-exts>=0.0.4",
        "scikit-learn>=1.3.0",
        "networkx",
        "wandb",
        "GPUtil",
        "psutil",
        "ipdb",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "requests",
        "Pillow>=9.0.0",
        "opencv-python",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ]
    sh(f"pip install -q {' '.join(shirg_deps)}", "Installing SHIRG dependencies", timeout=400)
    
    # ------------------------------------------------------------------ step 5: Evaluation tools
    eval_packages = [
        ("lmms-eval", "LMMS Eval"),
        ("pycocoevalcap", "COCO Eval"),
        ("rouge-score", "ROUGE Score"),
    ]
    
    print("\n🔄 Installing evaluation packages...")
    for package, name in eval_packages:
        if not sh(f"pip install -q {package}", f"Installing {name}", timeout=300):
            print(f"⚠️ {name} installation failed, skipping...")
    
    # ------------------------------------------------------------------ step 6: Install LaViDa
    print("\n🔄 Installing LaViDa package...")
    
    # Check if LaViDa exists
    lavida_path = "./LaViDa"
    if os.path.exists(lavida_path):
        print(f"   Found LaViDa at: {lavida_path}")
        lavida_ok = sh(f"pip install -e {lavida_path}", "Installing LaViDa package", timeout=300)
        if lavida_ok:
            print("✅ LaViDa package installed successfully!")
    else:
        print("⚠️ LaViDa directory not found - please clone the repository first")
    
    # ------------------------------------------------------------------ step 7: Verify installation
    print("\n🔍 Verifying installation...")
    
    # Test imports
    test_imports = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("accelerate", "Accelerate"),
        ("peft", "PEFT"),
        ("wandb", "Weights & Biases"),
    ]
    
    all_good = True
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} not available")
            all_good = False
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            print("❌ No GPU detected")
            all_good = False
    except:
        print("❌ Could not check GPU")
    
    if all_good:
        print("\n🎉 All done - ready for SHIRG LoRA training on Lambda Cloud!")
    else:
        print("\n⚠️ Some components missing - please check the errors above")


if __name__ == "__main__":
    main()