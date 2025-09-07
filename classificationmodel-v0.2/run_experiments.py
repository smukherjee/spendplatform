#!/usr/bin/env python3
"""
Experiment Runner for Spend Platform Categorization Model
Allows running experiments with different parsers and configurations
"""

import os
import sys
from pathlib import Path

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ModelConfig, ConfigPresets
from text_parsers import BasicTextParser, NLTKTextParser
from train_classification_model import MultiLevelSpendCategorizationModel


def run_experiment(config, experiment_name):
    """Run a single experiment with the given configuration"""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {experiment_name}")
    print('='*60)

    # Get text parser from config
    text_parser = config.get_text_parser()

    # Initialize model
    model = MultiLevelSpendCategorizationModel(
        target_levels=[config.target_level],
        text_parser=text_parser,
        config=config
    )

    try:
        # Load data
        train_data, test_data = model.load_preprocessed_data(config.train_file, config.test_file)

        # Extract features
        print("\nPreparing training data...")
        X_train = model.extract_features(train_data, is_training=True)
        y_train = model.prepare_labels(train_data, is_training=True)

        print("\nPreparing test data...")
        X_test = model.extract_features(test_data, is_training=False)
        y_test = model.prepare_labels(test_data, is_training=False)

        # Train model
        model.train_model(X_train, y_train, model_type=config.model_type)

        # Evaluate model
        evaluation_results = model.evaluate_model(X_test, y_test)

        # Save model with experiment name
        model_filename = f"spend_categorization_model_{experiment_name.lower().replace(' ', '_')}.pkl"
        model.save_model(model_filename)

        print(f"\n✅ Experiment '{experiment_name}' completed!")
        print(f"📊 Accuracy: {evaluation_results['accuracy']:.2%}")
        print(f"📁 Model saved as: {model_filename}")

        return evaluation_results

    except Exception as e:
        print(f"❌ Experiment '{experiment_name}' failed: {str(e)}")
        return None


def main():
    """Run multiple experiments with different configurations"""
    print("SPEND PLATFORM CATEGORIZATION - EXPERIMENT RUNNER")
    print("="*80)

    experiments = [
        ("Basic Parser + Fast Training", ConfigPresets.fast_training()),
        ("NLTK Parser + Balanced", ConfigPresets.balanced()),
        ("NLTK Parser + High Accuracy", ConfigPresets.high_accuracy()),
        ("spaCy Parser + Optimized", ConfigPresets.spacy_optimized()),
        ("BERT Parser + Optimized", ConfigPresets.bert_optimized()),
        ("RoBERTa Parser + Optimized", ConfigPresets.roberta_optimized()),
        ("DistilBERT Parser + Fast", ConfigPresets.distilbert_fast()),
        ("LayoutLMv2 Parser + Optimized", ConfigPresets.layoutlmv2_optimized()),
        ("DONUT Parser + Optimized", ConfigPresets.donut_optimized()),
        ("Custom: Basic + SVM", create_custom_config('basic', 'svm')),
        ("Custom: NLTK + Logistic Regression", create_custom_config('nltk', 'logistic_regression')),
    ]

    results = {}

    for experiment_name, config in experiments:
        result = run_experiment(config, experiment_name)
        if result:
            results[experiment_name] = result

    # Print summary
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print('='*80)

    for experiment_name, result in results.items():
        print(f"{experiment_name:<35}: {result['accuracy']:.2%}")

    print(f"\n✅ Completed {len(results)}/{len(experiments)} experiments")


def create_custom_config(parser_type, model_type):
    """Create a custom configuration"""
    config = ModelConfig()
    config.text_parser_type = parser_type
    config.model_type = model_type
    return config


def create_transformer_config(parser_type, fast=False):
    """Create a transformer-based configuration"""
    config = ModelConfig()
    config.text_parser_type = parser_type

    if fast:
        # Fast configuration for DistilBERT
        config.tfidf_max_features = 1000
        config.rf_n_estimators = 100
        config.rf_max_depth = 10
        config.transformer_max_length = 64  # Shorter sequences for speed
    else:
        # High accuracy configuration
        config.tfidf_max_features = 2000
        config.rf_n_estimators = 200
        config.rf_max_depth = 15
        config.transformer_max_length = 128

    return config


if __name__ == "__main__":
    main()
