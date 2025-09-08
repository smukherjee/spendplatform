"""
Text Parsers Package
Provides various text preprocessing parsers for the spend categorization model
"""

from .base_parser import BaseTextParser, BasicTextParser
from .nltk_parser import NLTKTextParser

# Try to import spaCy parser (optional dependency)
try:
    from .spacy_parser import SpacyTextParser
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    SpacyTextParser = None

# Try to import transformer parsers (optional dependency)
try:
    from .transformer_parser import (
        TransformerTextParser,
        BERTTextParser,
        RoBERTaTextParser,
        DistilBERTTextParser,
        LayoutLMv2TextParser,
        DONUTTextParser
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    TransformerTextParser = None
    BERTTextParser = None
    RoBERTaTextParser = None
    DistilBERTTextParser = None
    LayoutLMv2TextParser = None
    DONUTTextParser = None

__all__ = [
    'BaseTextParser',
    'BasicTextParser',
    'NLTKTextParser',
]

if SPACY_AVAILABLE:
    __all__.append('SpacyTextParser')

if TRANSFORMERS_AVAILABLE:
    __all__.extend([
        'TransformerTextParser',
        'BERTTextParser',
        'RoBERTaTextParser',
        'DistilBERTTextParser',
        'LayoutLMv2TextParser',
        'DONUTTextParser'
    ])
