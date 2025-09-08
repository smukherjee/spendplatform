#!/usr/bin/env python3
"""
Enhanced Feature Engineering for Spend Platform Categorization
Implements advanced feature extraction techniques to improve L2 categorization accuracy
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import warnings

warnings.filterwarnings('ignore')

class EnhancedFeatureExtractor:
    """Advanced feature extraction for spend categorization with focus on improving L2 accuracy"""
    
    def __init__(self, config=None):
        self.config = config
        self.vectorizer = None
        self.char_vectorizer = None
        self.domain_keywords = self._initialize_domain_keywords()
        self.brand_patterns = self._initialize_brand_patterns()
        self.unit_patterns = self._initialize_unit_patterns()
        self.material_keywords = self._initialize_material_keywords()
        
    def _initialize_domain_keywords(self) -> Dict[str, List[str]]:
        """Initialize domain-specific keywords for each L2 category"""
        return {
            'Electrical': [
                'cable', 'wire', 'voltage', 'current', 'power', 'electrical', 'electric',
                'transformer', 'circuit', 'switch', 'panel', 'motor', 'generator',
                'wiring', 'connector', 'plug', 'socket', 'breaker', 'fuse'
            ],
            'Manufacturing Components & Supplies': [
                'bearing', 'gear', 'shaft', 'bolt', 'screw', 'nut', 'washer',
                'fastener', 'gasket', 'seal', 'spring', 'bracket', 'mount',
                'coupling', 'bushing', 'component', 'part', 'assembly'
            ],
            'Industrial Manufacturing & Processing Machinery & Accessories': [
                'machine', 'machinery', 'equipment', 'pump', 'compressor', 'valve',
                'motor', 'drive', 'conveyor', 'industrial', 'processing', 'manufacturing',
                'automation', 'control', 'hydraulic', 'pneumatic', 'mechanical'
            ],
            'Tools & General Machinery': [
                'tool', 'wrench', 'drill', 'hammer', 'saw', 'cutting', 'grinding',
                'machining', 'lathe', 'mill', 'workshop', 'hand', 'power', 'portable'
            ],
            'Chemicals & Lubes': [
                'oil', 'lubricant', 'grease', 'fluid', 'chemical', 'solvent',
                'cleaner', 'adhesive', 'sealant', 'coating', 'paint', 'fuel',
                'hydraulic', 'synthetic', 'mineral', 'additive'
            ],
            'Filtration': [
                'filter', 'filtration', 'strainer', 'separator', 'purifier',
                'cartridge', 'element', 'media', 'membrane', 'screen'
            ],
            'Office Equipment, Furniture, & Supplies': [
                'office', 'desk', 'chair', 'cabinet', 'paper', 'pen', 'printer',
                'computer', 'keyboard', 'mouse', 'monitor', 'furniture', 'supplies',
                'stationery', 'file', 'folder', 'toner', 'ink'
            ],
            'Pipes, Valves & Fittings': [
                'pipe', 'tube', 'valve', 'fitting', 'elbow', 'tee', 'reducer',
                'coupling', 'flange', 'union', 'adapter', 'nipple', 'cap',
                'plug', 'drain', 'flow', 'pressure'
            ],
            'Power Generation & Distribution Machinery & Accessories': [
                'generator', 'alternator', 'battery', 'charger', 'inverter',
                'transformer', 'distribution', 'power', 'energy', 'electrical',
                'grid', 'supply', 'backup', 'emergency'
            ],
            'Material H&ling, Storage & Packaging': [
                'storage', 'container', 'box', 'crate', 'pallet', 'rack',
                'shelf', 'bin', 'handling', 'lifting', 'transport', 'packaging',
                'wrap', 'tape', 'label', 'material'
            ]
        }
    
    def _initialize_brand_patterns(self) -> List[str]:
        """Initialize common brand/manufacturer patterns"""
        return [
            r'\b[A-Z]{2,}\b',  # All caps words (likely brands)
            r'\b[A-Z][a-z]+\s*[A-Z][a-z]*\b',  # Title case combinations
            r'\b\w*[0-9]+[A-Z]+\w*\b',  # Alphanumeric codes
        ]
    
    def _initialize_unit_patterns(self) -> List[str]:
        """Initialize measurement unit patterns"""
        return [
            r'\d+\s*(mm|cm|m|km|inch|in|ft|yard)',  # Length units
            r'\d+\s*(kg|g|lb|oz|ton)',  # Weight units
            r'\d+\s*(l|ml|gal|qt|pt)',  # Volume units
            r'\d+\s*(v|volt|amp|watt|hp)',  # Electrical units
            r'\d+\s*(psi|bar|pa|mpa)',  # Pressure units
            r'\d+\s*(rpm|hz|khz|mhz)',  # Frequency units
        ]
    
    def _initialize_material_keywords(self) -> List[str]:
        """Initialize material-related keywords"""
        return [
            'steel', 'iron', 'aluminum', 'copper', 'brass', 'bronze',
            'plastic', 'rubber', 'vinyl', 'polymer', 'ceramic', 'glass',
            'carbon', 'stainless', 'galvanized', 'coated', 'treated'
        ]
    
    def extract_enhanced_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Extract comprehensive enhanced features for improved L2 categorization"""
        
        print("🔧 Extracting enhanced features for improved L2 categorization...")
        
        df_processed = df.copy().reset_index(drop=True)
        
        # Ensure we have the description column
        if 'processed_description' not in df_processed.columns:
            if 'Item_Descripton' in df_processed.columns:
                df_processed['processed_description'] = df_processed['Item_Descripton'].fillna('').astype(str)
            else:
                raise ValueError("No description column found")
        
        # Create feature list to store all features
        all_features = []
        
        # 1. Enhanced Text Features
        text_features = self._extract_text_features(df_processed, is_training)
        all_features.append(text_features)
        
        # 2. Domain-Specific Features
        domain_features = self._extract_domain_features(df_processed)
        all_features.append(domain_features)
        
        # 3. Structural Features
        structural_features = self._extract_structural_features(df_processed)
        all_features.append(structural_features)
        
        # 4. Semantic Features
        semantic_features = self._extract_semantic_features(df_processed)
        all_features.append(semantic_features)
        
        # 5. Statistical Features
        statistical_features = self._extract_statistical_features(df_processed)
        all_features.append(statistical_features)
        
        # Combine all features
        feature_df = pd.concat(all_features, axis=1)
        
        print(f"✅ Enhanced feature extraction complete: {feature_df.shape[1]} features")
        return feature_df
    
    def _extract_text_features(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Extract enhanced text-based features"""
        
        descriptions = df['processed_description'].fillna('').astype(str)
        
        # Enhanced TF-IDF features with optimized parameters
        if is_training:
            self.vectorizer = TfidfVectorizer(
                max_features=2000,  # Increased from 1500
                ngram_range=(1, 4),  # Increased to capture more phrases
                min_df=2,
                max_df=0.92,  # Slightly lower to capture more specific terms
                sublinear_tf=True,
                use_idf=True,
                smooth_idf=True,
                analyzer='word',
                lowercase=True,
                token_pattern=r'\b\w{2,}\b'  # Minimum 2 characters
            )
            tfidf_matrix = self.vectorizer.fit_transform(descriptions)
        else:
            if self.vectorizer is None:
                raise ValueError("TF-IDF vectorizer not fitted")
            tfidf_matrix = self.vectorizer.transform(descriptions)
        
        # Convert sparse matrix to DataFrame
        tfidf_feature_names = [f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
        
        # Convert to dense array using numpy
        try:
            import scipy.sparse as sp
            if sp.issparse(tfidf_matrix):
                tfidf_array = np.array(tfidf_matrix.todense())
            else:
                tfidf_array = np.array(tfidf_matrix)
            tfidf_df = pd.DataFrame(tfidf_array, columns=tfidf_feature_names, index=df.index)
        except Exception as e:
            print(f"Warning: Could not convert TF-IDF matrix to DataFrame: {e}")
            # Fallback: create empty DataFrame with correct shape
            tfidf_df = pd.DataFrame(np.zeros((len(df), len(tfidf_feature_names))), 
                                   columns=tfidf_feature_names, index=df.index)
        
        # Character n-grams for capturing patterns
        if is_training:
            self.char_vectorizer = TfidfVectorizer(
                max_features=500,
                ngram_range=(2, 4),
                analyzer='char',
                min_df=3,
                max_df=0.9
            )
            char_matrix = self.char_vectorizer.fit_transform(descriptions)
        else:
            if self.char_vectorizer is None:
                raise ValueError("Character vectorizer not fitted")
            char_matrix = self.char_vectorizer.transform(descriptions)
        
        # Convert character n-gram sparse matrix to DataFrame
        char_feature_names = [f'char_{i}' for i in range(char_matrix.shape[1])]
        
        try:
            import scipy.sparse as sp
            if sp.issparse(char_matrix):
                char_array = np.array(char_matrix.todense())
            else:
                char_array = np.array(char_matrix)
            char_df = pd.DataFrame(char_array, columns=char_feature_names, index=df.index)
        except Exception as e:
            print(f"Warning: Could not convert character matrix to DataFrame: {e}")
            # Fallback: create empty DataFrame with correct shape
            char_df = pd.DataFrame(np.zeros((len(df), len(char_feature_names))), 
                                  columns=char_feature_names, index=df.index)
        
        # Combine text features
        return pd.concat([tfidf_df, char_df], axis=1)
    
    def _extract_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract domain-specific features based on L2 categories"""
        
        features = {}
        descriptions = df['processed_description'].fillna('').astype(str).str.lower()
        
        # Category-specific keyword features
        for category, keywords in self.domain_keywords.items():
            category_name = category.lower().replace(' ', '_').replace('&', 'and')
            
            # Count of category keywords
            keyword_counts = []
            keyword_present = []
            keyword_ratios = []
            
            for desc in descriptions:
                count = sum(1 for keyword in keywords if keyword in desc)
                keyword_counts.append(count)
                keyword_present.append(1 if count > 0 else 0)
                word_len = max(len(desc.split()), 1)
                keyword_ratios.append(count / word_len)
            
            features[f'domain_{category_name}_count'] = keyword_counts
            features[f'domain_{category_name}_present'] = keyword_present
            features[f'domain_{category_name}_ratio'] = keyword_ratios
        
        # Brand/manufacturer detection
        brand_counts = [self._count_brands(desc) for desc in descriptions]
        features['brand_count'] = brand_counts
        features['has_brand'] = [1 if count > 0 else 0 for count in brand_counts]
        
        # Material detection
        material_counts = [sum(1 for material in self.material_keywords if material in desc) for desc in descriptions]
        features['material_count'] = material_counts
        features['has_material'] = [1 if count > 0 else 0 for count in material_counts]
        
        # Unit/measurement detection
        unit_counts = [self._count_units(desc) for desc in descriptions]
        features['unit_count'] = unit_counts
        features['has_units'] = [1 if count > 0 else 0 for count in unit_counts]
        
        return pd.DataFrame(features, index=df.index)
    
    def _extract_structural_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract structural/format features"""
        
        features = {}
        descriptions = df['processed_description'].fillna('').astype(str)
        original_descriptions = df.get('Item_Descripton', descriptions).fillna('').astype(str)
        
        # Basic length features
        features['desc_length'] = original_descriptions.str.len()
        features['word_count'] = descriptions.str.split().str.len().fillna(0)
        features['char_count'] = descriptions.str.len()
        
        # Word length statistics
        word_lengths_data = []
        avg_word_lengths = []
        max_word_lengths = []
        min_word_lengths = []
        
        for desc in descriptions:
            words = desc.split() if desc else []
            if words:
                lengths = [len(word) for word in words]
                avg_word_lengths.append(np.mean(lengths))
                max_word_lengths.append(max(lengths))
                min_word_lengths.append(min(lengths))
            else:
                avg_word_lengths.append(0)
                max_word_lengths.append(0)
                min_word_lengths.append(0)
        
        features['avg_word_length'] = avg_word_lengths
        features['max_word_length'] = max_word_lengths
        features['min_word_length'] = min_word_lengths
        
        # Unique word features
        unique_words = [len(set(desc.split())) if desc else 0 for desc in descriptions]
        features['unique_words'] = unique_words
        
        # Unique word ratio
        word_counts = features['word_count']
        unique_word_ratios = []
        for i in range(len(word_counts)):
            if word_counts.iloc[i] > 0:
                unique_word_ratios.append(unique_words[i] / word_counts.iloc[i])
            else:
                unique_word_ratios.append(0)
        features['unique_word_ratio'] = unique_word_ratios
        
        # Character pattern features
        features['digit_count'] = original_descriptions.str.count(r'\d')
        features['alpha_count'] = original_descriptions.str.count(r'[a-zA-Z]')
        features['special_char_count'] = original_descriptions.str.count(r'[^\w\s]')
        features['uppercase_count'] = original_descriptions.str.count(r'[A-Z]')
        features['lowercase_count'] = original_descriptions.str.count(r'[a-z]')
        
        # Ratio features
        char_counts = features['char_count']
        
        digit_ratios = []
        alpha_ratios = []
        special_ratios = []
        upper_ratios = []
        
        for i in range(len(char_counts)):
            total = char_counts.iloc[i]
            if total > 0:
                digit_ratios.append(features['digit_count'].iloc[i] / total)
                alpha_ratios.append(features['alpha_count'].iloc[i] / total)
                special_ratios.append(features['special_char_count'].iloc[i] / total)
                upper_ratios.append(features['uppercase_count'].iloc[i] / total)
            else:
                digit_ratios.append(0)
                alpha_ratios.append(0)
                special_ratios.append(0)
                upper_ratios.append(0)
        
        features['digit_ratio'] = digit_ratios
        features['alpha_ratio'] = alpha_ratios
        features['special_ratio'] = special_ratios
        features['upper_ratio'] = upper_ratios
        
        # Format pattern features
        features['has_numbers'] = [1 if count > 0 else 0 for count in features['digit_count']]
        features['has_special_chars'] = [1 if count > 0 else 0 for count in features['special_char_count']]
        
        has_mixed_case = []
        for i in range(len(features['uppercase_count'])):
            upper = features['uppercase_count'].iloc[i]
            lower = features['lowercase_count'].iloc[i]
            has_mixed_case.append(1 if (upper > 0 and lower > 0) else 0)
        features['has_mixed_case'] = has_mixed_case
        
        features['starts_with_number'] = original_descriptions.str.match(r'^\d').fillna(False).astype(int)
        features['ends_with_number'] = original_descriptions.str.match(r'.*\d$').fillna(False).astype(int)
        
        return pd.DataFrame(features, index=df.index)
    
    def _extract_semantic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract semantic and contextual features"""
        
        features = {}
        descriptions = df['processed_description'].fillna('').astype(str).str.lower()
        
        # Action words (verbs that might indicate function)
        action_words = ['replace', 'repair', 'maintain', 'install', 'connect', 'operate', 'control']
        features['action_word_count'] = [sum(1 for word in action_words if word in desc) for desc in descriptions]
        
        # Technical specification patterns
        features['has_model_number'] = descriptions.str.contains(r'\b\w*\d+\w*\b').astype(int)
        features['has_part_number'] = descriptions.str.contains(r'\b(part|p/n|pn|model|mod)\b').astype(int)
        features['has_size_spec'] = descriptions.str.contains(r'\d+\s*(mm|cm|inch|in|\")').astype(int)
        features['has_voltage_spec'] = descriptions.str.contains(r'\d+\s*(v|volt|kv)').astype(int)
        features['has_pressure_spec'] = descriptions.str.contains(r'\d+\s*(psi|bar|pa)').astype(int)
        
        # Quality/grade indicators
        quality_words = ['premium', 'standard', 'heavy', 'duty', 'grade', 'quality', 'industrial', 'commercial']
        features['quality_indicator_count'] = [sum(1 for word in quality_words if word in desc) for desc in descriptions]
        
        # Color mentions (might be relevant for certain categories)
        color_words = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'gray', 'silver']
        features['color_mention_count'] = [sum(1 for color in color_words if color in desc) for desc in descriptions]
        
        return pd.DataFrame(features, index=df.index)
    
    def _extract_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract statistical features from text"""
        
        features = {}
        descriptions = df['processed_description'].fillna('').astype(str)
        
        # Word frequency features
        max_word_freqs = []
        singleton_words = []
        vocabulary_richness = []
        word_repetition_ratios = []
        
        for desc in descriptions:
            if desc:
                words = desc.split()
                if words:
                    word_counter = Counter(words)
                    max_word_freqs.append(word_counter.most_common(1)[0][1] if word_counter else 0)
                    singleton_words.append(sum(1 for count in word_counter.values() if count == 1))
                    unique_words = len(set(words))
                    total_words = len(words)
                    vocabulary_richness.append(unique_words / max(total_words, 1))
                    word_repetition_ratios.append((total_words - unique_words) / max(total_words, 1))
                else:
                    max_word_freqs.append(0)
                    singleton_words.append(0)
                    vocabulary_richness.append(0)
                    word_repetition_ratios.append(0)
            else:
                max_word_freqs.append(0)
                singleton_words.append(0)
                vocabulary_richness.append(0)
                word_repetition_ratios.append(0)
        
        features['max_word_freq'] = max_word_freqs
        features['singleton_words'] = singleton_words
        features['vocabulary_richness'] = vocabulary_richness
        features['word_repetition_ratio'] = word_repetition_ratios
        
        # Punctuation features
        features['comma_count'] = descriptions.str.count(',')
        features['dash_count'] = descriptions.str.count('-')
        features['slash_count'] = descriptions.str.count('/')
        features['bracket_count'] = descriptions.str.count(r'[\(\)\[\]]')
        
        return pd.DataFrame(features, index=df.index)
    
    def _count_brands(self, text: str) -> int:
        """Count potential brand mentions in text"""
        count = 0
        for pattern in self.brand_patterns:
            count += len(re.findall(pattern, text))
        return count
    
    def _count_units(self, text: str) -> int:
        """Count measurement units in text"""
        count = 0
        for pattern in self.unit_patterns:
            count += len(re.findall(pattern, text, re.IGNORECASE))
        return count
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Get feature groups for importance analysis"""
        return {
            'text_features': ['tfidf_', 'char_'],
            'domain_features': ['domain_'],
            'structural_features': ['desc_length', 'word_count', 'char_count', '_ratio', '_count'],
            'semantic_features': ['has_', 'action_', 'quality_', 'color_'],
            'statistical_features': ['max_word_freq', 'singleton_', 'vocabulary_', 'repetition_']
        }
