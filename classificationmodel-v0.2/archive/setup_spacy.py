#!/usr/bin/env python3
"""
Setup script for spaCy parser
Installs spaCy and downloads the required language model
"""

import subprocess
import sys
import os

def install_spacy():
    """Install spaCy and download language model"""
    print("🚀 Setting up spaCy for enhanced text processing...")
    print("="*60)

    try:
        # Check if spaCy is already installed
        import spacy
        version = getattr(spacy, '__version__', 'unknown')
        print(f"✅ spaCy is already installed (version {version})")
    except ImportError:
        print("📦 Installing spaCy...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy"])
            print("✅ spaCy installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install spaCy: {e}")
            return False

    # Download the language model
    model_name = "en_core_web_sm"
    print(f"📥 Downloading spaCy language model: {model_name}")

    try:
        import spacy
        # Use subprocess to download the model instead of spacy.cli.download
        subprocess.run([sys.executable, '-m', 'spacy', 'download', model_name], check=True)
        print(f"✅ Language model '{model_name}' downloaded successfully!")

        # Test the model
        print("🧪 Testing spaCy model...")
        nlp = spacy.load(model_name)
        doc = nlp("This is a test sentence for spaCy.")
        print(f"✅ Model working! Processed {len(doc)} tokens.")

    except Exception as e:
        print(f"❌ Failed to download or test model: {e}")
        print("💡 Try running: python -m spacy download en_core_web_sm")
        return False

    print("="*60)
    print("🎉 spaCy setup complete!")
    print("You can now use the spaCy parser with:")
    print("  python train_classification_model.py --parser spacy")
    print("  python train_classification_model.py --preset spacy")
    print("="*60)

    return True

def main():
    """Main setup function"""
    print("SPEND PLATFORM - spaCy Setup")
    print("This will install spaCy and download the English language model.")
    print()

    # Ask for confirmation
    response = input("Continue with installation? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Installation cancelled.")
        return

    # Run installation
    success = install_spacy()

    if success:
        print("\n🎯 Ready to use spaCy parser!")
        print("Expected performance improvement: ~56-58% accuracy")
    else:
        print("\n⚠️  spaCy setup failed.")
        print("You can still use the basic or NLTK parsers.")

if __name__ == "__main__":
    main()
