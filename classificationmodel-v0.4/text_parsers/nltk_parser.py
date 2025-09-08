"""
NLTK-based Text Parser Module
Provides advanced text preprocessing using NLTK library
"""

import re
from typing import List, Dict, Any, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from .base_parser import BaseTextParser


class NLTKTextParser(BaseTextParser):
    """Advanced text parser using NLTK for preprocessing"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt')

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Downloading NLTK stopwords...")
            nltk.download('stopwords')

        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("Downloading NLTK wordnet...")
            nltk.download('wordnet')

        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            print("Downloading NLTK averaged perceptron tagger...")
            nltk.download('averaged_perceptron_tagger')

        # Initialize NLTK components
        self.stop_words = set(stopwords.words('english'))
        # Add domain-specific stop words
        domain_stops = self.config.get('domain_stop_words', {
            'make', 'model', 'type', 'size', 'grade', 'class', 'item', 'product'
        })
        self.stop_words.update(domain_stops)

        self.lemmatizer = WordNetLemmatizer()
        self.use_pos_tagging = self.config.get('use_pos_tagging', True)

    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing using NLTK"""
        if not isinstance(text, str) or text == '':
            return ''

        # Convert to lowercase
        text = text.lower()

        # Remove special characters but keep spaces and alphanumeric
        text = re.sub(r'[^\w\s]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # Tokenize using NLTK
        tokens = word_tokenize(text)

        # Remove stop words and short tokens
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]

        # Lemmatization
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

        # Join tokens back
        processed_text = ' '.join(tokens)

        return processed_text

    def extract_pos_features(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Extract POS-based features using NLTK"""
        features = []

        for text in texts:
            if not text or text == '':
                features.append({
                    'noun_count': 0, 'verb_count': 0, 'adj_count': 0,
                    'total_tokens': 0, 'avg_token_length': 0, 'has_capitals': 0
                })
                continue

            tokens = word_tokenize(text)

            if self.use_pos_tagging and tokens:
                # Full POS tagging
                pos_tags = pos_tag(tokens)

                # Count POS categories
                noun_count = sum(1 for _, tag in pos_tags if tag.startswith('NN'))
                verb_count = sum(1 for _, tag in pos_tags if tag.startswith('VB'))
                adj_count = sum(1 for _, tag in pos_tags if tag.startswith('JJ'))
            else:
                # Fallback to rule-based detection
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
        """Simple verb detection for fallback"""
        verb_indicators = ['ing', 'ed', 'run', 'make', 'do', 'get', 'set', 'use', 'work']
        return any(indicator in token for indicator in verb_indicators)

    def get_parser_info(self) -> Dict[str, Any]:
        """Get detailed information about this NLTK parser"""
        info = super().get_parser_info()
        info.update({
            'nltk_version': nltk.__version__,
            'available_corpora': ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'],
            'use_pos_tagging': self.use_pos_tagging
        })
        return info
