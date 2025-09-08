"""
Configuration file for Spend Platform Categorization Model
Allows easy switching between different parsers and algorithms
"""

from typing import Optional, Any, Dict
from text_parsers import BasicTextParser, NLTKTextParser
from text_parsers.base_parser import BaseTextParser

# Try to import spaCy parser
try:
    from text_parsers import SpacyTextParser, SPACY_AVAILABLE
except ImportError:
    SpacyTextParser = None
    SPACY_AVAILABLE = False

# Try to import transformer parsers
try:
    from text_parsers import (
        BERTTextParser,
        RoBERTaTextParser,
        DistilBERTTextParser,
        LayoutLMv2TextParser,
        DONUTTextParser,
        TRANSFORMERS_AVAILABLE
    )
except ImportError:
    BERTTextParser = None
    RoBERTaTextParser = None
    DistilBERTTextParser = None
    LayoutLMv2TextParser = None
    DONUTTextParser = None
    TRANSFORMERS_AVAILABLE = False
    TRANSFORMERS_AVAILABLE = False


class ModelConfig:
    """Configuration class for the spend categorization model"""

    def __init__(self):
        # Data files
        self.train_file = 'preprocessed_data/train_data.xlsx'
        self.test_file = 'preprocessed_data/test_data.xlsx'

        # Text parser settings
        self.text_parser_type = 'nltk'  # 'nltk', 'basic', 'spacy', 'bert', 'roberta', 'distilbert'

        # spaCy-specific settings
        self.spacy_model_name = 'en_core_web_sm'  # spaCy model to use
        self.spacy_use_lemmatization = True
        self.spacy_use_pos_filtering = True
        self.spacy_remove_stop_words = True
        self.spacy_min_token_length = 2
        self.spacy_keep_pos_tags = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']

        # Transformer-specific settings
        self.transformer_model_name = 'bert-base-uncased'  # Default transformer model
        self.transformer_max_length = 128  # Maximum sequence length
        self.transformer_use_gpu = False  # Use GPU if available
        self.transformer_batch_size = 16  # Batch size for processing

        # TF-IDF settings
        self.tfidf_max_features = 1500
        self.tfidf_ngram_range = (1, 3)
        self.tfidf_min_df = 2
        self.tfidf_max_df = 0.95

        # Random Forest settings
        self.rf_n_estimators = 200
        self.rf_max_depth = 15
        self.rf_min_samples_split = 5
        self.rf_min_samples_leaf = 2

        # Model settings
        self.target_level = 'Category L2'  # Can be L1, L2, L3, L4, or L5
        self.model_type = 'random_forest'  # 'random_forest', 'logistic_regression', 'svm', or 'catboost'
        # Model save path will be generated dynamically based on parser type

    @property
    def model_save_path(self):
        """Generate parser-specific model file name dynamically"""
        return f'spend_categorization_model_{self.text_parser_type}.pkl'

    def get_model_path_for_parser(self, parser_type):
        """Get the model file path for a specific parser type"""
        return f'spend_categorization_model_{parser_type}.pkl'

    def get_parser_from_model_path(self, model_path):
        """Extract parser type from model file path"""
        import re
        match = re.search(r'spend_categorization_model_(.+)\.pkl', model_path)
        if match:
            return match.group(1)
        return 'nltk'  # Default fallback

        # Text parser settings
        self.text_parser_type = 'nltk'  # 'nltk', 'basic', 'spacy', 'bert', 'roberta', 'distilbert'

        # spaCy-specific settings
        self.spacy_model_name = 'en_core_web_sm'  # spaCy model to use
        self.spacy_use_lemmatization = True
        self.spacy_use_pos_filtering = True
        self.spacy_remove_stop_words = True
        self.spacy_min_token_length = 2
        self.spacy_keep_pos_tags = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']

        # Transformer-specific settings
        self.transformer_model_name = 'bert-base-uncased'  # Default transformer model
        self.transformer_max_length = 128  # Maximum sequence length
        self.transformer_use_gpu = False  # Use GPU if available
        self.transformer_batch_size = 16  # Batch size for processing

        # TF-IDF settings
        self.tfidf_max_features = 1500
        self.tfidf_ngram_range = (1, 3)
        self.tfidf_min_df = 2
        self.tfidf_max_df = 0.95

        # Random Forest settings
        self.rf_n_estimators = 200
        self.rf_max_depth = 15
        self.rf_min_samples_split = 5
        self.rf_min_samples_leaf = 2

    def get_text_parser(self) -> 'BaseTextParser':
        """Get the configured text parser"""
        if self.text_parser_type.lower() == 'basic':
            return BasicTextParser()
        elif self.text_parser_type.lower() == 'nltk':
            return NLTKTextParser()
        elif self.text_parser_type.lower() == 'spacy':
            if not SPACY_AVAILABLE or SpacyTextParser is None:
                print("⚠️ spaCy not available. Install with: pip install spacy")
                print("Falling back to NLTK parser...")
                return NLTKTextParser()
            else:
                # Create spaCy config
                spacy_config = {
                    'model_name': self.spacy_model_name,
                    'use_lemmatization': self.spacy_use_lemmatization,
                    'use_pos_filtering': self.spacy_use_pos_filtering,
                    'remove_stop_words': self.spacy_remove_stop_words,
                    'min_token_length': self.spacy_min_token_length,
                    'keep_pos_tags': set(self.spacy_keep_pos_tags),
                    'custom_stop_words': ['make', 'model', 'type', 'size', 'grade', 'class', 'item', 'product']
                }
                return SpacyTextParser(spacy_config)
        elif self.text_parser_type.lower() in ['bert', 'roberta', 'distilbert', 'layoutlmv2', 'donut']:
            if not TRANSFORMERS_AVAILABLE:
                print("⚠️ Transformers not available. Install with: pip install transformers torch")
                print("Falling back to NLTK parser...")
                return NLTKTextParser()
            else:
                # Type guard: ensure transformer classes are available
                assert BERTTextParser is not None, "BERTTextParser should be available when TRANSFORMERS_AVAILABLE is True"
                assert RoBERTaTextParser is not None, "RoBERTaTextParser should be available when TRANSFORMERS_AVAILABLE is True"
                assert DistilBERTTextParser is not None, "DistilBERTTextParser should be available when TRANSFORMERS_AVAILABLE is True"
                assert LayoutLMv2TextParser is not None, "LayoutLMv2TextParser should be available when TRANSFORMERS_AVAILABLE is True"
                assert DONUTTextParser is not None, "DONUTTextParser should be available when TRANSFORMERS_AVAILABLE is True"

                # Create transformer config
                transformer_config = {
                    'model_name': self.transformer_model_name,
                    'max_length': self.transformer_max_length,
                    'use_gpu': self.transformer_use_gpu,
                    'batch_size': self.transformer_batch_size
                }

                if self.text_parser_type.lower() == 'bert':
                    return BERTTextParser(transformer_config)
                elif self.text_parser_type.lower() == 'roberta':
                    return RoBERTaTextParser(transformer_config)
                elif self.text_parser_type.lower() == 'distilbert':
                    return DistilBERTTextParser(transformer_config)
                elif self.text_parser_type.lower() == 'layoutlmv2':
                    return LayoutLMv2TextParser(transformer_config)
                elif self.text_parser_type.lower() == 'donut':
                    return DONUTTextParser(transformer_config)
        else:
            print(f"⚠️ Unknown parser type '{self.text_parser_type}', using NLTK parser")
            return NLTKTextParser()

        # This should never be reached, but ensures type safety
        return NLTKTextParser()

    def print_config(self):
        """Print current configuration"""
        print("="*60)
        print("MODEL CONFIGURATION")
        print("="*60)
        print(f"Target Level: {self.target_level}")
        print(f"Model Type: {self.model_type}")
        print(f"Text Parser: {self.text_parser_type}")
        if self.text_parser_type.lower() == 'spacy':
            print(f"spaCy Model: {self.spacy_model_name}")
            print(f"spaCy Features: Lemmatization={self.spacy_use_lemmatization}, POS Filtering={self.spacy_use_pos_filtering}")
        print(f"Train File: {self.train_file}")
        print(f"Test File: {self.test_file}")
        print(f"Model Save Path: {self.model_save_path}")
        print("="*60)


# Predefined configurations for different scenarios
class ConfigPresets:
    """Predefined configuration presets"""

    @staticmethod
    def fast_training():
        """Configuration for fast training (basic parser, smaller model)"""
        config = ModelConfig()
        config.text_parser_type = 'basic'
        config.tfidf_max_features = 1000
        config.rf_n_estimators = 100
        config.rf_max_depth = 10
        return config

    @staticmethod
    def high_accuracy():
        """Configuration for high accuracy (NLTK parser, larger model)"""
        config = ModelConfig()
        config.text_parser_type = 'nltk'
        config.tfidf_max_features = 2000
        config.rf_n_estimators = 300
        config.rf_max_depth = 20
        return config

    @staticmethod
    def spacy_optimized():
        """Configuration optimized for spaCy parser with high accuracy"""
        config = ModelConfig()
        config.text_parser_type = 'spacy'
        config.spacy_model_name = 'en_core_web_sm'
        config.spacy_use_lemmatization = True
        config.spacy_use_pos_filtering = True
        config.spacy_remove_stop_words = True
        config.tfidf_max_features = 2000
        config.rf_n_estimators = 300
        config.rf_max_depth = 20
        return config

    @staticmethod
    def bert_optimized():
        """Configuration optimized for BERT parser"""
        config = ModelConfig()
        config.text_parser_type = 'bert'
        config.transformer_model_name = 'bert-base-uncased'
        config.transformer_max_length = 128
        config.tfidf_max_features = 2000
        config.rf_n_estimators = 200
        config.rf_max_depth = 15
        return config

    @staticmethod
    def roberta_optimized():
        """Configuration optimized for RoBERTa parser"""
        config = ModelConfig()
        config.text_parser_type = 'roberta'
        config.transformer_model_name = 'roberta-base'
        config.transformer_max_length = 128
        config.tfidf_max_features = 2000
        config.rf_n_estimators = 200
        config.rf_max_depth = 15
        return config

    @staticmethod
    def distilbert_fast():
        """Fast configuration using DistilBERT"""
        config = ModelConfig()
        config.text_parser_type = 'distilbert'
        config.transformer_model_name = 'distilbert-base-uncased'
        config.transformer_max_length = 64  # Shorter for speed
        config.tfidf_max_features = 1000
        config.rf_n_estimators = 100
        config.rf_max_depth = 10
        return config

    @staticmethod
    def layoutlmv2_optimized():
        """Configuration optimized for LayoutLMv2 parser"""
        config = ModelConfig()
        config.text_parser_type = 'layoutlmv2'
        config.transformer_model_name = 'microsoft/layoutlmv2-base-uncased'
        config.transformer_max_length = 128
        config.tfidf_max_features = 2000
        config.rf_n_estimators = 200
        config.rf_max_depth = 15
        return config

    @staticmethod
    def donut_optimized():
        """Configuration optimized for DONUT parser"""
        config = ModelConfig()
        config.text_parser_type = 'donut'
        config.transformer_model_name = 'naver-clova-ix/donut-base'
        config.transformer_max_length = 128
        config.tfidf_max_features = 2000
        config.rf_n_estimators = 200
        config.rf_max_depth = 15
        return config

    @staticmethod
    def balanced():
        """Balanced configuration (default settings)"""
        return ModelConfig()


def get_config_from_args():
    """Parse command line arguments for configuration (future enhancement)"""
    import sys

    config = ModelConfig()

    # Simple argument parsing
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == '--parser' and i + 1 < len(args):
            config.text_parser_type = args[i + 1]
        elif arg == '--model' and i + 1 < len(args):
            config.model_type = args[i + 1]
        elif arg == '--target' and i + 1 < len(args):
            config.target_level = args[i + 1]
        elif arg == '--spacy-model' and i + 1 < len(args):
            config.spacy_model_name = args[i + 1]
        elif arg == '--preset' and i + 1 < len(args):
            preset = args[i + 1]
            if preset == 'fast':
                config = ConfigPresets.fast_training()
            elif preset == 'accurate':
                config = ConfigPresets.high_accuracy()
            elif preset == 'balanced':
                config = ConfigPresets.balanced()
            elif preset == 'spacy':
                config = ConfigPresets.spacy_optimized()

    return config


if __name__ == "__main__":
    # Example usage
    config = get_config_from_args()
    config.print_config()
