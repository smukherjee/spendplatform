#!/usr/bin/env python3
"""
Demo script for Transformer-based text parsers
Shows how to use BERT, RoBERTa, and DistilBERT for text preprocessing
"""

import sys
import os
import numpy as np

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ModelConfig
from text_parsers import TRANSFORMERS_AVAILABLE

def demo_transformer_parsers():
    """Demonstrate transformer parser capabilities"""

    if not TRANSFORMERS_AVAILABLE:
        print("❌ Transformers not available. Run setup_transformers.py first")
        print("Install with: pip install transformers torch")
        return

    print("🤖 Transformer Parser Demo")
    print("=" * 40)

    # Sample spend data
    sample_texts = [
        "Purchase of office supplies including pens, paper, and printer ink",
        "IT equipment maintenance and software license renewal",
        "Travel expenses for business meeting in New York",
        "Marketing campaign materials and promotional items",
        "Facility maintenance and cleaning supplies",
        "Professional development training for staff",
        "Office furniture and equipment setup",
        "Communication services and internet connectivity"
    ]

    # Test each transformer model
    models_to_test = ['bert', 'roberta', 'distilbert', 'layoutlmv2', 'donut']

    for model_type in models_to_test:
        print(f"\n🔍 Testing {model_type.upper()} Parser")
        print("-" * 30)

        try:
            # Create config for this model
            config = ModelConfig()
            config.text_parser_type = model_type
            config.transformer_max_length = 64  # Shorter for demo
            config.transformer_batch_size = 4

            parser = config.get_text_parser()

            # Ensure we have a transformer parser
            if not hasattr(parser, 'get_embeddings'):
                print(f"❌ {model_type} parser doesn't support embeddings")
                continue

            print(f"✅ {model_type.upper()} parser initialized")
            print(f"Model: {parser.model_name}")  # type: ignore
            print(f"Max length: {parser.max_length}")  # type: ignore
            print(f"GPU available: {parser.use_gpu}")  # type: ignore

            # Preprocess texts
            print("\n📝 Preprocessing texts...")
            preprocessed = [parser.preprocess_text(text) for text in sample_texts[:3]]  # type: ignore
            for i, text in enumerate(preprocessed):
                print(f"  {i+1}. {text}")

            # Extract features
            print("\n📊 Extracting features...")
            features = parser.extract_pos_features(sample_texts[:3])  # type: ignore
            for i, feat in enumerate(features):
                print(f"  {i+1}. Length: {feat['text_length']}, Words: {feat['word_count']}")

            # Generate embeddings
            print("\n🧠 Generating embeddings...")
            embeddings = parser.get_embeddings(sample_texts[:3])  # type: ignore
            print(f"Embeddings shape: {embeddings.shape}")
            print(f"Sample embedding (first 5 values): {embeddings[0][:5]}")

            # Get parser info
            info = parser.get_parser_info()  # type: ignore
            print("\nℹ️ Parser Info:")
            for key, value in info.items():
                if key != 'tokenizer_info':  # Skip detailed tokenizer info
                    print(f"  {key}: {value}")

        except Exception as e:
            print(f"❌ Error with {model_type}: {e}")
            continue

def benchmark_parsers():
    """Benchmark different parsers for speed comparison"""

    if not TRANSFORMERS_AVAILABLE:
        print("❌ Transformers not available for benchmarking")
        return

    print("\n⚡ Parser Benchmark")
    print("=" * 40)

    import time

    # Larger sample for benchmarking
    sample_texts = [
        "Purchase of office supplies including pens, paper, and printer ink",
        "IT equipment maintenance and software license renewal",
        "Travel expenses for business meeting in New York",
        "Marketing campaign materials and promotional items",
        "Facility maintenance and cleaning supplies",
        "Professional development training for staff",
        "Office furniture and equipment setup",
        "Communication services and internet connectivity"
    ]
    benchmark_texts = sample_texts * 10  # 80 texts

    parsers_to_benchmark = ['bert', 'roberta', 'distilbert', 'layoutlmv2', 'donut']

    for model_type in parsers_to_benchmark:
        try:
            config = ModelConfig()
            config.text_parser_type = model_type
            config.transformer_max_length = 128
            config.transformer_batch_size = 8

            parser = config.get_text_parser()

            # Time preprocessing
            start_time = time.time()
            preprocessed = [parser.preprocess_text(text) for text in benchmark_texts]  # type: ignore
            preprocess_time = time.time() - start_time

            # Time embedding generation
            start_time = time.time()
            embeddings = parser.get_embeddings(benchmark_texts)  # type: ignore
            embedding_time = time.time() - start_time

            print(f"{model_type.upper():10} | Preprocess: {preprocess_time:.3f}s | Embeddings: {embedding_time:.3f}s")

        except Exception as e:
            print(f"{model_type.upper():10} | Error: {str(e)[:30]}...")

def main():
    """Main demo function"""
    print("🚀 Spend Categorization - Transformer Parser Demo")
    print("=" * 55)

    # Check availability
    if not TRANSFORMERS_AVAILABLE:
        print("⚠️ Transformer dependencies not installed.")
        print("Run: python setup_transformers.py")
        print("Or install manually: pip install transformers torch")
        return

    # Run demos
    demo_transformer_parsers()
    benchmark_parsers()

    print("\n🎉 Demo completed!")
    print("\n💡 Tips:")
    print("  • BERT: Balanced performance and accuracy")
    print("  • RoBERTa: Often better accuracy than BERT")
    print("  • DistilBERT: Faster, lighter version of BERT")
    print("  • Use GPU for better performance with large datasets")

if __name__ == "__main__":
    main()
