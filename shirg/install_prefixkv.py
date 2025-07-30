#!/usr/bin/env python3
"""
Install PrefixKV dependency for SHIRG
"""

import subprocess
import sys

def install_prefixkv():
    """Install PrefixKV package"""
    print("📦 Installing PrefixKV...")
    
    try:
        # Try to install from PyPI first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "prefixkv"])
        print("✅ Successfully installed PrefixKV from PyPI")
        return True
    except subprocess.CalledProcessError:
        print("⚠️ Failed to install from PyPI, trying from source...")
        
    try:
        # If PyPI fails, try installing from GitHub
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "git+https://github.com/prefixkv/prefixkv.git"
        ])
        print("✅ Successfully installed PrefixKV from GitHub")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install PrefixKV: {e}")
        print("\n💡 You can try manual installation:")
        print("   pip install prefixkv")
        print("   or")
        print("   pip install git+https://github.com/prefixkv/prefixkv.git")
        return False

if __name__ == "__main__":
    success = install_prefixkv()
    sys.exit(0 if success else 1)