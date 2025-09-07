#!/usr/bin/env python3
"""
Setup script for Transformer models (BERT, RoBERTa, DistilBERT)
Installs required dependencies for advanced text parsing
"""

import subprocess
import sys
import os

def install_transformers():
    """Install transformers and torch dependencies"""
    print("🚀 Installing Transformer dependencies...")

    # List of packages to install
    packages = [
        "transformers>=4.21.0",
        "torch>=1.12.0",
        "torchvision",
        "torchaudio",
        "Pillow>=9.0.0",  # For image processing (DONUT)
        "datasets",  # Optional, for some model training
    ]

    try:
        # Install packages
        for package in packages:
            print(f"📦 Installing {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])

        print("✅ Transformer dependencies installed successfully!")

        # Test the installation
        print("🧪 Testing installation...")
        try:
            import transformers
            import torch
            print(f"✅ Transformers version: {transformers.__version__}")
            print(f"✅ PyTorch version: {torch.__version__}")
            print(f"✅ CUDA available: {torch.cuda.is_available()}")

            if torch.cuda.is_available():
                print(f"✅ CUDA devices: {torch.cuda.device_count()}")

        except ImportError as e:
            print(f"❌ Import test failed: {e}")
            return False

        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def download_models():
    """Download and cache common transformer models"""
    print("📥 Downloading common transformer models...")

    models = [
        "bert-base-uncased",
        "roberta-base",
        "distilbert-base-uncased"
    ]

    try:
        from transformers import AutoTokenizer, AutoModel

        for model_name in models:
            print(f"📥 Downloading {model_name}...")
            try:
                # Download tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                # Download model
                model = AutoModel.from_pretrained(model_name)
                print(f"✅ {model_name} downloaded successfully")
            except Exception as e:
                print(f"⚠️ Failed to download {model_name}: {e}")
                continue

        print("✅ Model downloads completed!")
        return True

    except ImportError:
        print("❌ Transformers not available for model downloads")
        return False

def main():
    """Main setup function"""
    print("🤖 Transformer Setup for Spend Categorization Model")
    print("=" * 50)

    # Check if running in virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in virtual environment")
    else:
        print("⚠️ Not running in virtual environment")
        print("Consider using a virtual environment for better dependency management")

    # Install dependencies
    if not install_transformers():
        print("❌ Setup failed during dependency installation")
        sys.exit(1)

    # Optionally download models
    download = input("\n📥 Download common models now? (y/n) [y]: ").lower().strip()
    if download in ['', 'y', 'yes']:
        download_models()
    else:
        print("ℹ️ Skipping model downloads. Models will be downloaded on first use.")

    print("\n🎉 Transformer setup completed!")
    print("\nUsage:")
    print("  Set text_parser_type to 'bert', 'roberta', or 'distilbert' in config.py")
    print("  Example: config.text_parser_type = 'bert'")
    print("\nFor GPU support, ensure CUDA is properly installed on your system.")

if __name__ == "__main__":
    main()
