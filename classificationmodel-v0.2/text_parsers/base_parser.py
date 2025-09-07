"""
Base Text Parser Module
Provides the base class for all text preprocessing parsers
"""

import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseTextParser(ABC):
    """Abstract base class for text parsers"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.name = self.config.get('name', self.__class__.__name__)

    @abstractmethod
    def preprocess_text(self, text: str) -> str:
        """Preprocess a single text string"""
        pass

    @abstractmethod
    def extract_pos_features(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Extract POS-based features from a list of texts"""
        pass

    def batch_preprocess(self, texts: List[str]) -> List[str]:
        """Batch preprocess multiple texts"""
        return [self.preprocess_text(text) for text in texts]

    def get_parser_info(self) -> Dict[str, Any]:
        """Get information about this parser"""
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'config': self.config
        }


class BasicTextParser(BaseTextParser):
    """Basic text parser without external dependencies"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.stop_words = set(self.config.get('stop_words', [
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'make', 'model', 'type', 'size',
            'grade', 'class', 'item', 'product'
        ]))

    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing without NLTK"""
        if not isinstance(text, str) or text == '':
            return ''

        # Convert to lowercase
        text = text.lower()

        # Remove special characters but keep spaces and alphanumeric
        text = re.sub(r'[^\w\s]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # Simple tokenization (split by space)
        tokens = text.split()

        # Remove stop words and short tokens
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]

        # Simple stemming (remove common suffixes)
        tokens = [self._simple_stem(token) for token in tokens]

        # Join tokens back
        processed_text = ' '.join(tokens)

        return processed_text

    def _simple_stem(self, word: str) -> str:
        """Simple stemming without NLTK"""
        # Remove common suffixes
        suffixes = ['ing', 'ly', 'ed', 'ies', 'ied', 'ies', 'ied', 's']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 1:
                return word[:-len(suffix)]
        return word

    def extract_pos_features(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Extract basic features without POS tagging"""
        features = []

        for text in texts:
            if not text or text == '':
                features.append({
                    'noun_count': 0, 'verb_count': 0, 'adj_count': 0,
                    'total_tokens': 0, 'avg_token_length': 0, 'has_capitals': 0
                })
                continue

            tokens = text.split()

            # Simple rule-based feature extraction
            noun_count = sum(1 for token in tokens if len(token) > 3 and not self._is_verb_like(token))
            verb_count = sum(1 for token in tokens if self._is_verb_like(token))
            adj_count = sum(1 for token in tokens if len(token) > 3 and token.endswith(('al', 'ic', 'ous', 'ent', 'ant')))

            features.append({
                'noun_count': noun_count,
                'verb_count': verb_count,
                'adj_count': adj_count,
                'total_tokens': len(tokens),
                'avg_token_length': sum(len(t) for t in tokens) / len(tokens) if tokens else 0,
                'has_capitals': 1 if any(t.isupper() for t in tokens) else 0
            })

        return features

    def _is_verb_like(self, token: str) -> bool:
        """Simple verb detection"""
        verb_indicators = ['ing', 'ed', 'run', 'make', 'do', 'get', 'set', 'use', 'work']
        return any(indicator in token for indicator in verb_indicators)
