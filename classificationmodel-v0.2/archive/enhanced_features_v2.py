#!/usr/bin/env python3
"""
Optimized Enhanced Feature Engineering v2 for Spend Platform Categorization
Implements optimized feature extraction based on performance analysis to improve L2 accuracy
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
import warnings

warnings.filterwarnings('ignore')

class OptimizedEnhancedFeatureExtractor:
    """Optimized advanced feature extraction for spend categorization with focus on improving L2 accuracy"""
    
    def __init__(self, config=None):
        self.config = config or self._get_default_config()
        self.vectorizer = None
        self.char_vectorizer = None
        self.feature_selector = None
        self.selected_feature_names = None
        self.domain_keywords = self._initialize_optimized_domain_keywords()
        self.brand_patterns = self._initialize_brand_patterns()
        self.unit_patterns = self._initialize_unit_patterns()
        self.material_keywords = self._initialize_material_keywords()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get optimized default configuration"""
        return {
            "tfidf_config": {
                "max_features": 1300,  # Reduced from 2000
                "ngram_range": (1, 3),  # Reduced from (1, 4)
                "min_df": 4,  # Increased from 2
                "max_df": 0.90,  # Reduced from 0.92
                "sublinear_tf": True,
                "use_idf": True,
                "smooth_idf": True,
                "analyzer": 'word',
                "lowercase": True,
                "token_pattern": r'\b\w{2,}\b'
            },
            "char_ngram_config": {
                "max_features": 250,  # Reduced from 500
                "ngram_range": (3, 4),  # More focused on 3-4 char patterns
                "min_df": 5,  # Increased from 3
                "max_df": 0.85,
                "analyzer": 'char'
            },
            "feature_selection": {
                "method": "SelectKBest",
                "k": 1000,  # Total features to keep
                "score_func": "chi2"  # or "mutual_info_classif"
            },
            "domain_keywords": {
                "max_keywords_per_category": 15,  # Limit to most relevant
                "min_keyword_frequency": 5,  # Only frequent keywords
                "use_l3_keywords": True,  # Add L3-specific terms
                "weight_by_category_size": True
            }
        }
        
    def _initialize_optimized_domain_keywords(self) -> Dict[str, List[str]]:
        """Initialize optimized domain-specific keywords for each L2 category"""
        return {
            'Electrical': [
                'electrical', 'electric', 'cable', 'wire', 'voltage', 'current', 'power',
                'transformer', 'motor', 'switch', 'panel', 'circuit', 'connector', 'breaker'
            ],
            'Manufacturing Components & Supplies': [
                'bearing', 'gear', 'bolt', 'screw', 'fastener', 'gasket', 'seal',
                'component', 'part', 'assembly', 'coupling', 'bushing', 'spring', 'washer'
            ],
            'Industrial Manufacturing & Processing Machinery & Accessories': [
                'machinery', 'equipment', 'pump', 'compressor', 'valve', 'industrial',
                'processing', 'manufacturing', 'automation', 'hydraulic', 'pneumatic', 'conveyor',
                'drive', 'control', 'mechanical'
            ],
            'Tools & General Machinery': [
                'tool', 'drill', 'cutting', 'grinding', 'machining', 'wrench',
                'hammer', 'saw', 'lathe', 'mill', 'workshop', 'hand', 'power', 'portable'
            ],
            'Chemicals & Lubes': [
                'oil', 'lubricant', 'grease', 'chemical', 'fluid', 'solvent',
                'cleaner', 'adhesive', 'coating', 'paint', 'hydraulic', 'synthetic', 'additive'
            ],
            'Filtration': [
                'filter', 'filtration', 'strainer', 'separator', 'cartridge',
                'element', 'media', 'membrane', 'screen', 'purifier'
            ],
            'Office Equipment, Furniture, & Supplies': [
                'office', 'furniture', 'desk', 'chair', 'paper', 'printer',
                'computer', 'supplies', 'stationery', 'toner', 'ink', 'cabinet', 'file'
            ],
            'Pipes, Valves & Fittings': [
                'pipe', 'valve', 'fitting', 'tube', 'elbow', 'flange',
                'coupling', 'union', 'adapter', 'pressure', 'flow', 'drain'
            ],
            'Power Generation & Distribution Machinery & Accessories': [
                'generator', 'power', 'energy', 'battery', 'distribution',
                'electrical', 'backup', 'supply', 'inverter', 'charger', 'alternator'
            ],
            'Material Handling, Storage & Packaging': [
                'storage', 'handling', 'container', 'pallet', 'packaging',
                'transport', 'lifting', 'material', 'box', 'rack', 'bin'
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
            'steel', 'aluminum', 'copper', 'brass', 'plastic', 'rubber',
            'stainless', 'carbon', 'ceramic', 'glass', 'vinyl', 'polymer'
        ]
    
    def extract_enhanced_features(self, df: pd.DataFrame, y_labels=None, is_training: bool = True) -> pd.DataFrame:
        """Extract optimized enhanced features for improved L2 categorization"""
        
        print("🔧 Extracting optimized enhanced features for improved L2 categorization...")
        
        df_processed = df.copy().reset_index(drop=True)
        
        # Ensure we have the description column
        if 'processed_description' not in df_processed.columns:
            if 'Item_Descripton' in df_processed.columns:
                df_processed['processed_description'] = df_processed['Item_Descripton'].fillna('').astype(str)
            else:
                raise ValueError("No description column found")
        
        # Create feature list to store all features
        all_features = []
        
        # 1. Optimized Text Features
        text_features = self._extract_optimized_text_features(df_processed, is_training)
        all_features.append(text_features)
        
        # 2. Refined Domain-Specific Features
        domain_features = self._extract_refined_domain_features(df_processed)
        all_features.append(domain_features)
        
        # 3. Keep Best Structural Features
        structural_features = self._extract_optimized_structural_features(df_processed)
        all_features.append(structural_features)
        
        # 4. Focused Semantic Features
        semantic_features = self._extract_focused_semantic_features(df_processed)
        all_features.append(semantic_features)
        
        # Combine all features
        feature_df = pd.concat(all_features, axis=1)
        
        # Apply feature selection if training and labels are provided
        if is_training and y_labels is not None:
            feature_df = self._apply_feature_selection(feature_df, y_labels)
        elif not is_training and self.feature_selector is not None:
            # Apply already fitted feature selector
            feature_df = self._apply_fitted_feature_selection(feature_df)
        
        print(f"✅ Optimized enhanced feature extraction complete: {feature_df.shape[1]} features")
        return feature_df
    
    def _extract_optimized_text_features(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Extract optimized text-based features with reduced dimensionality"""
        
        descriptions = df['processed_description'].fillna('').astype(str)
        
        # Optimized TF-IDF features
        if is_training:
            tfidf_config = self.config["tfidf_config"]
            self.vectorizer = TfidfVectorizer(**tfidf_config)
            tfidf_matrix = self.vectorizer.fit_transform(descriptions)
        else:
            if self.vectorizer is None:
                raise ValueError("TF-IDF vectorizer not fitted")
            tfidf_matrix = self.vectorizer.transform(descriptions)
        
        # Convert to DataFrame
        tfidf_feature_names = [f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
        
        try:
            import scipy.sparse as sp
            if sp.issparse(tfidf_matrix):
                tfidf_array = np.array(tfidf_matrix.todense())
            else:
                tfidf_array = np.array(tfidf_matrix)
            tfidf_df = pd.DataFrame(tfidf_array, columns=tfidf_feature_names, index=df.index)
        except Exception as e:
            print(f"Warning: Could not convert TF-IDF matrix to DataFrame: {e}")
            tfidf_df = pd.DataFrame(np.zeros((len(df), len(tfidf_feature_names))), 
                                   columns=tfidf_feature_names, index=df.index)
        
        # Optimized character n-grams
        if is_training:
            char_config = self.config["char_ngram_config"]
            self.char_vectorizer = TfidfVectorizer(**char_config)
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
            char_df = pd.DataFrame(np.zeros((len(df), len(char_feature_names))), 
                                  columns=char_feature_names, index=df.index)
        
        # Combine text features
        return pd.concat([tfidf_df, char_df], axis=1)
    
    def _extract_refined_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract refined domain-specific features with better keywords"""
        
        features = {}
        descriptions = df['processed_description'].fillna('').astype(str).str.lower()
        
        # Category-specific keyword features (refined)
        for category, keywords in self.domain_keywords.items():
            category_name = category.lower().replace(' ', '_').replace('&', 'and')
            
            # Only keep most important features per category
            keyword_counts = []
            keyword_present = []
            
            for desc in descriptions:
                count = sum(1 for keyword in keywords if keyword in desc)
                keyword_counts.append(count)
                keyword_present.append(1 if count > 0 else 0)
            
            features[f'domain_{category_name}_count'] = keyword_counts
            features[f'domain_{category_name}_present'] = keyword_present
            
            # Only add ratio if category has significant keyword presence
            if sum(keyword_present) >= 5:  # At least 5 instances
                word_counts = descriptions.str.split().str.len().fillna(1)
                keyword_ratios = [count / max(wc, 1) for count, wc in zip(keyword_counts, word_counts)]
                features[f'domain_{category_name}_ratio'] = keyword_ratios
        
        # Technical indicators (refined)
        features['has_brand'] = [1 if self._count_brands(desc) > 0 else 0 for desc in descriptions]
        features['has_material'] = [1 if sum(1 for material in self.material_keywords if material in desc) > 0 else 0 for desc in descriptions]
        features['has_units'] = [1 if self._count_units(desc) > 0 else 0 for desc in descriptions]
        
        return pd.DataFrame(features, index=df.index)
    
    def _extract_optimized_structural_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract the most important structural features only"""
        
        features = {}
        descriptions = df['processed_description'].fillna('').astype(str)
        original_descriptions = df.get('Item_Descripton', descriptions).fillna('').astype(str)
        
        # Keep only high-importance structural features
        features['desc_length'] = original_descriptions.str.len()
        features['word_count'] = descriptions.str.split().str.len().fillna(0)
        features['char_count'] = descriptions.str.len()
        
        # Key ratio features
        char_counts = features['char_count']
        digit_counts = original_descriptions.str.count(r'\d')
        alpha_counts = original_descriptions.str.count(r'[a-zA-Z]')
        upper_counts = original_descriptions.str.count(r'[A-Z]')
        
        features['digit_count'] = digit_counts
        features['alpha_count'] = alpha_counts
        features['uppercase_count'] = upper_counts
        
        # Important ratios
        digit_ratios = []
        alpha_ratios = []
        upper_ratios = []
        
        for i in range(len(char_counts)):
            total = char_counts.iloc[i]
            if total > 0:
                digit_ratios.append(digit_counts.iloc[i] / total)
                alpha_ratios.append(alpha_counts.iloc[i] / total)
                upper_ratios.append(upper_counts.iloc[i] / total)
            else:
                digit_ratios.append(0)
                alpha_ratios.append(0)
                upper_ratios.append(0)
        
        features['digit_ratio'] = digit_ratios
        features['alpha_ratio'] = alpha_ratios
        features['upper_ratio'] = upper_ratios
        
        # Word statistics
        word_lengths_data = []
        avg_word_lengths = []
        max_word_lengths = []
        
        for desc in descriptions:
            words = desc.split() if desc else []
            if words:
                lengths = [len(word) for word in words]
                avg_word_lengths.append(np.mean(lengths))
                max_word_lengths.append(max(lengths))
            else:
                avg_word_lengths.append(0)
                max_word_lengths.append(0)
        
        features['avg_word_length'] = avg_word_lengths
        features['max_word_length'] = max_word_lengths
        
        # Binary indicators
        features['has_numbers'] = [1 if count > 0 else 0 for count in digit_counts]
        features['has_mixed_case'] = [1 if (upper > 0 and alpha - upper > 0) else 0 
                                     for upper, alpha in zip(upper_counts, alpha_counts)]
        
        return pd.DataFrame(features, index=df.index)
    
    def _extract_focused_semantic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract only the most relevant semantic features"""
        
        features = {}
        descriptions = df['processed_description'].fillna('').astype(str).str.lower()
        
        # Technical specification patterns (most important)
        features['has_model_number'] = descriptions.str.contains(r'\b\w*\d+\w*\b').astype(int)
        features['has_part_number'] = descriptions.str.contains(r'\b(part|p/n|pn|model|mod)\b').astype(int)
        features['has_size_spec'] = descriptions.str.contains(r'\d+\s*(mm|cm|inch|in|\")').astype(int)
        features['has_voltage_spec'] = descriptions.str.contains(r'\d+\s*(v|volt|kv)').astype(int)
        
        # Quality indicators (simplified)
        quality_words = ['premium', 'heavy', 'duty', 'industrial', 'commercial']
        features['quality_indicator_count'] = [sum(1 for word in quality_words if word in desc) for desc in descriptions]
        
        return pd.DataFrame(features, index=df.index)
    
    def _apply_feature_selection(self, feature_df: pd.DataFrame, y_labels: pd.Series) -> pd.DataFrame:
        """Apply feature selection to keep only the most important features"""
        
        selection_config = self.config["feature_selection"]
        k = min(selection_config["k"], feature_df.shape[1])  # Don't exceed available features
        
        # Handle categorical labels
        y_labels_encoded = pd.Categorical(y_labels).codes
        
        # Apply feature selection
        if selection_config["score_func"] == "chi2":
            # Ensure non-negative features for chi2
            feature_df_nonneg = feature_df.clip(lower=0)
            self.feature_selector = SelectKBest(score_func=chi2, k=k)
            selected_features = self.feature_selector.fit_transform(feature_df_nonneg, y_labels_encoded)
        else:
            self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=k)
            selected_features = self.feature_selector.fit_transform(feature_df, y_labels_encoded)
        
        # Get selected feature names
        selected_mask = self.feature_selector.get_support()
        self.selected_feature_names = feature_df.columns[selected_mask].tolist()
        
        # Create DataFrame with selected features
        selected_df = pd.DataFrame(
            selected_features, 
            columns=self.selected_feature_names, 
            index=feature_df.index
        )
        
        print(f"🎯 Feature selection: {len(self.selected_feature_names)} features selected from {feature_df.shape[1]}")
        return selected_df
    
    def _apply_fitted_feature_selection(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Apply already fitted feature selector"""
        
        if self.feature_selector is None or self.selected_feature_names is None:
            print("⚠️ Feature selector not fitted, returning all features")
            return feature_df
        
        # Apply the fitted selector
        try:
            # Check if we used chi2 by looking at the config
            if self.config["feature_selection"]["score_func"] == "chi2":
                feature_df_nonneg = feature_df.clip(lower=0)
                selected_features = self.feature_selector.transform(feature_df_nonneg)
            else:
                selected_features = self.feature_selector.transform(feature_df)
            
            # Convert to numpy array properly using try-except for safety
            try:
                # First check if it's already a numpy array
                if isinstance(selected_features, np.ndarray):
                    pass  # Already numpy array
                else:
                    import scipy.sparse as sp
                    if sp.issparse(selected_features):
                        selected_features = selected_features.toarray()
                    else:
                        selected_features = np.asarray(selected_features)
            except Exception as e:
                print(f"Warning: Feature conversion issue: {e}")
                # Fallback to direct conversion
                selected_features = np.asarray(selected_features)
            
            selected_df = pd.DataFrame(
                selected_features, 
                columns=self.selected_feature_names, 
                index=feature_df.index
            )
            return selected_df
            
        except Exception as e:
            print(f"⚠️ Error applying feature selection: {e}")
            return feature_df
    
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
            'semantic_features': ['has_', 'quality_'],
        }
    
    def get_feature_selection_info(self) -> Dict[str, Any]:
        """Get information about feature selection"""
        if self.feature_selector is None:
            return {"status": "not_fitted"}
        
        return {
            "status": "fitted",
            "total_features_before": "available" if hasattr(self.feature_selector, 'scores_') else "unknown",
            "selected_features": len(self.selected_feature_names) if self.selected_feature_names else 0,
            "selected_feature_names": self.selected_feature_names[:20] if self.selected_feature_names else [],  # Top 20 for display
            "method": self.config["feature_selection"]["method"],
            "score_function": self.config["feature_selection"]["score_func"]
        }
