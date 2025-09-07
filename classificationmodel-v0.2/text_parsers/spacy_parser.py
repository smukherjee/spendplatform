"""
spaCy-based Text Parser Module
Provides advanced text preprocessing using spaCy library for superior performance
"""

import re
from typing import List, Dict, Any, Optional
import spacy
from spacy.lang.en import English

from .base_parser import BaseTextParser


class SpacyTextParser(BaseTextParser):
    """Advanced text parser using spaCy for high-performance NLP processing"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # Initialize spaCy model
        model_name = self.config.get('model_name', 'en_core_web_sm')
        try:
            self.nlp = spacy.load(model_name)
            print(f"✅ Loaded spaCy model: {model_name}")
        except OSError:
            print(f"⚠️ spaCy model '{model_name}' not found. Downloading...")
            try:
                # Try to download the model
                spacy.cli.download(model_name)
                self.nlp = spacy.load(model_name)
                print(f"✅ Downloaded and loaded spaCy model: {model_name}")
            except Exception as e:
                print(f"❌ Failed to download spaCy model: {e}")
                print("Falling back to basic English model...")
                # Fallback to basic English model
                self.nlp = English()
                print("✅ Using basic English model")

        # Configure processing options
        self.use_lemmatization = self.config.get('use_lemmatization', True)
        self.use_pos_filtering = self.config.get('use_pos_filtering', True)
        self.remove_stop_words = self.config.get('remove_stop_words', True)
        self.min_token_length = self.config.get('min_token_length', 2)

        # POS tags to keep (if using POS filtering)
        self.keep_pos_tags = self.config.get('keep_pos_tags', {
            'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'
        })

        # Custom stop words
        self.custom_stop_words = set(self.config.get('custom_stop_words', [
            'make', 'model', 'type', 'size', 'grade', 'class', 'item', 'product'
        ]))

    def preprocess_text(self, text: str) -> str:
        """Advanced text preprocessing using spaCy"""
        if not isinstance(text, str) or text == '':
            return ''

        # Convert to lowercase
        text = text.lower()

        # Remove special characters but keep spaces and alphanumeric
        text = re.sub(r'[^\w\s]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        # Process with spaCy
        doc = self.nlp(text)

        # Extract tokens based on configuration
        processed_tokens = []

        for token in doc:
            # Skip tokens that don't meet criteria
            if len(token.text) < self.min_token_length:
                continue

            if self.remove_stop_words and (token.is_stop or token.text in self.custom_stop_words):
                continue

            if self.use_pos_filtering and token.pos_ not in self.keep_pos_tags:
                continue

            # Use lemma if lemmatization is enabled
            if self.use_lemmatization:
                processed_tokens.append(token.lemma_)
            else:
                processed_tokens.append(token.text)

        # Join tokens back
        processed_text = ' '.join(processed_tokens)

        return processed_text

    def extract_pos_features(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Extract advanced POS-based features using spaCy"""
        features = []

        for text in texts:
            if not text or text == '':
                features.append({
                    'noun_count': 0, 'verb_count': 0, 'adj_count': 0,
                    'total_tokens': 0, 'avg_token_length': 0, 'has_capitals': 0,
                    'entity_count': 0, 'sentence_count': 0
                })
                continue

            # Process with spaCy
            doc = self.nlp(text)

            # Count POS categories
            pos_counts = {}
            for token in doc:
                pos = token.pos_
                pos_counts[pos] = pos_counts.get(pos, 0) + 1

            noun_count = pos_counts.get('NOUN', 0) + pos_counts.get('PROPN', 0)
            verb_count = pos_counts.get('VERB', 0) + pos_counts.get('AUX', 0)
            adj_count = pos_counts.get('ADJ', 0)

            # Additional features
            tokens = [token.text for token in doc if len(token.text) >= self.min_token_length]
            total_tokens = len(tokens)
            avg_token_length = sum(len(t) for t in tokens) / total_tokens if tokens else 0
            has_capitals = 1 if any(t.isupper() for t in tokens) else 0

            # Named entities
            entity_count = len(doc.ents)

            # Sentence count
            sentence_count = len(list(doc.sents))

            features.append({
                'noun_count': noun_count,
                'verb_count': verb_count,
                'adj_count': adj_count,
                'total_tokens': total_tokens,
                'avg_token_length': avg_token_length,
                'has_capitals': has_capitals,
                'entity_count': entity_count,
                'sentence_count': sentence_count
            })

        return features

    def get_parser_info(self) -> Dict[str, Any]:
        """Get detailed information about this spaCy parser"""
        info = super().get_parser_info()
        info.update({
            'spacy_version': spacy.__version__,
            'model_name': self.config.get('model_name', 'en_core_web_sm'),
            'use_lemmatization': self.use_lemmatization,
            'use_pos_filtering': self.use_pos_filtering,
            'remove_stop_words': self.remove_stop_words,
            'min_token_length': self.min_token_length,
            'keep_pos_tags': list(self.keep_pos_tags)
        })
        return info

    def get_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text (bonus feature)"""
        if not text or text == '':
            return []

        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })

        return entities

    def get_dependency_tree(self, text: str) -> List[Dict[str, Any]]:
        """Extract dependency parsing information (bonus feature)"""
        if not text or text == '':
            return []

        doc = self.nlp(text)
        dependencies = []

        for token in doc:
            dependencies.append({
                'text': token.text,
                'dep': token.dep_,
                'head': token.head.text,
                'pos': token.pos_,
                'tag': token.tag_
            })

        return dependencies
