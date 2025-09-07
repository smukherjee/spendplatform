#!/usr/bin/env python3
"""
Test script to verify that LayoutLMv2 and DONUT parsers work with lazy loading
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ModelConfig

def test_lazy_loading():
    """Test that parsers can be created without transformers installed"""

    print("🧪 Testing Lazy Loading for Transformer Parsers")
    print("=" * 50)

    # Test parsers that should work without transformers
    test_parsers = ['bert', 'roberta', 'distilbert', 'layoutlmv2', 'donut']

    for parser_type in test_parsers:
        try:
            config = ModelConfig()
            config.text_parser_type = parser_type
            parser = config.get_text_parser()

            # Type assertion to help Pylance understand parser is not None
            assert parser is not None, f"Parser creation failed for {parser_type}"

            print(f"✅ {parser_type.upper()} parser created successfully: {type(parser).__name__}")

            # Test basic methods that don't require model loading
            info = parser.get_parser_info()
            print(f"   Model name: {info.get('model_name', 'N/A')}")
            print(f"   Max length: {info.get('max_length', 'N/A')}")

        except Exception as e:
            print(f"❌ {parser_type.upper()} parser failed: {e}")

    print("\n📝 Note: Embeddings will only be generated when transformers are installed")
    print("   Run: pip install transformers torch torchvision torchaudio Pillow")

if __name__ == "__main__":
    test_lazy_loading()
