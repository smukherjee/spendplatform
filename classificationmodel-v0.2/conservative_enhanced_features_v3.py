#!/usr/bin/env python3
"""
Conservative Enhanced Feature Engineering v3 for Spend Platform Categorization
Implements conservative feature enhancement starting with existing proven parameters
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

class ConservativeEnhancedFeatureExtractor:
    """Conservative enhanced feature extraction that builds incrementally on proven baseline"""
    
    def __init__(self, config=None):
        self.config = config or self._get_default_config()
        self.vectorizer = None
        self.feature_selector = None
        self.scaler = None
        self.selected_feature_names = None
        self.domain_keywords = self._initialize_validated_domain_keywords()
        self.technical_patterns = self._initialize_technical_patterns()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get conservative configuration starting with existing proven parameters"""
        return {
            "approach": "conservative_incremental",
            "tfidf_config": {
                # Keep existing proven parameters
                "max_features": 1500,  # Same as existing
                "ngram_range": (1, 3),  # Same as existing
                "min_df": 2,  # Same as existing
                "max_df": 0.95,  # Same as existing
                "sublinear_tf": True,
                "use_idf": True,
                "smooth_idf": True,
                "analyzer": 'word',
                "lowercase": True,
                "token_pattern": r'\b\w{2,}\b'
            },
            "add_domain_features": True,
            "add_structural_features": True,
            "add_technical_features": True,
            "feature_selection": {
                "enabled": True,
                "method": "SelectKBest",
                "k": 1200,  # Conservative selection - keep most features
                "score_func": "mutual_info_classif"  # Better for mixed feature types
            },
            "feature_scaling": {
                "enabled": True,
                "method": "StandardScaler"
            },
            "correlation_removal": {
                "enabled": True,
                "threshold": 0.95
            }
        }
        
    def _initialize_validated_domain_keywords(self) -> Dict[str, List[str]]:
        """Initialize validated domain-specific keywords - only most discriminative ones"""
        return {
            'Electrical': [
                'electrical', 'electric', 'cable', 'wire', 'voltage', 'motor', 'switch', 'circuit'
            ],
            'Manufacturing Components & Supplies': [
                'bearing', 'gear', 'bolt', 'screw', 'fastener', 'gasket', 'component', 'assembly'
            ],
            'Industrial Manufacturing & Processing Machinery & Accessories': [
                'machinery', 'pump', 'compressor', 'valve', 'industrial', 'processing', 'hydraulic'
            ],
            'Tools & General Machinery': [
                'tool', 'drill', 'cutting', 'grinding', 'wrench', 'hammer', 'lathe'
            ],
            'Chemicals & Lubes': [
                'oil', 'lubricant', 'grease', 'chemical', 'fluid', 'hydraulic'
            ],
            'Filtration': [
                'filter', 'filtration', 'strainer', 'cartridge', 'element'
            ],
            'Office Equipment, Furniture, & Supplies': [
                'office', 'furniture', 'desk', 'paper', 'printer', 'supplies'
            ],
            'Pipes, Valves & Fittings': [
                'pipe', 'valve', 'fitting', 'tube', 'pressure', 'flow'
            ],
            'Power Generation & Distribution Machinery & Accessories': [
                'generator', 'power', 'battery', 'electrical', 'energy'
            ],
            'Material Handling, Storage & Packaging': [
                'storage', 'handling', 'container', 'packaging', 'material'
            ]
        }
    
    def _initialize_technical_patterns(self) -> Dict[str, str]:
        """Initialize technical specification patterns"""
        return {
            'model_number': r'\b\w*\d+\w*\b',
            'part_number': r'\b(part|p/n|pn|model|mod)\b',
            'size_spec': r'\d+\s*(mm|cm|m|inch|in|ft)',
            'voltage_spec': r'\d+\s*(v|volt|kv)',
            'pressure_spec': r'\d+\s*(psi|bar|pa|mpa)',
            'weight_spec': r'\d+\s*(kg|g|lb|oz)',
            'capacity_spec': r'\d+\s*(l|ml|gal|qt)'
        }
    
    def extract_enhanced_features(self, df: pd.DataFrame, y_labels=None, is_training: bool = True) -> pd.DataFrame:
        """Extract conservative enhanced features"""
        
        print("🔧 Extracting conservative enhanced features (v3)...")
        
        df_processed = df.copy().reset_index(drop=True)
        
        # Ensure we have the description column
        if 'processed_description' not in df_processed.columns:
            if 'Item_Descripton' in df_processed.columns:
                df_processed['processed_description'] = df_processed['Item_Descripton'].fillna('').astype(str)
            else:
                raise ValueError("No description column found")
        
        # 1. Start with proven TF-IDF baseline
        baseline_features = self._extract_baseline_tfidf_features(df_processed, is_training)
        print(f"   ✅ Baseline TF-IDF features: {baseline_features.shape[1]}")
        
        # 2. Add validated domain features
        if self.config["add_domain_features"]:
            domain_features = self._extract_validated_domain_features(df_processed)
            print(f"   ✅ Domain features: {domain_features.shape[1]}")
            baseline_features = pd.concat([baseline_features, domain_features], axis=1)
        
        # 3. Add key structural features
        if self.config["add_structural_features"]:
            structural_features = self._extract_key_structural_features(df_processed)
            print(f"   ✅ Structural features: {structural_features.shape[1]}")
            baseline_features = pd.concat([baseline_features, structural_features], axis=1)
        
        # 4. Add technical specification features
        if self.config["add_technical_features"]:
            technical_features = self._extract_technical_features(df_processed)
            print(f"   ✅ Technical features: {technical_features.shape[1]}")
            baseline_features = pd.concat([baseline_features, technical_features], axis=1)
        
        print(f"   📊 Combined features before processing: {baseline_features.shape[1]}")
        
        # 5. Remove highly correlated features
        if self.config["correlation_removal"]["enabled"] and is_training:
            baseline_features = self._remove_correlated_features(baseline_features)
            print(f"   🔄 After correlation removal: {baseline_features.shape[1]}")
        
        # 5b. Ensure feature consistency between train and test
        if not is_training:
            baseline_features = self._align_features_to_training(baseline_features)
            print(f"   🔄 After feature alignment: {baseline_features.shape[1]}")
        
        # 6. Apply feature scaling
        if self.config["feature_scaling"]["enabled"]:
            baseline_features = self._apply_feature_scaling(baseline_features, is_training)
            print(f"   📏 Features scaled using {self.config['feature_scaling']['method']}")
        
        # 7. Apply feature selection
        if self.config["feature_selection"]["enabled"] and is_training and y_labels is not None:
            baseline_features = self._apply_conservative_feature_selection(baseline_features, y_labels)
            print(f"   🎯 Feature selection: {baseline_features.shape[1]} features selected")
        elif not is_training and self.feature_selector is not None:
            baseline_features = self._apply_fitted_feature_selection(baseline_features)
        
        print(f"✅ Conservative enhanced feature extraction complete: {baseline_features.shape[1]} features")
        return baseline_features
    
    def _extract_baseline_tfidf_features(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Extract baseline TF-IDF features using proven existing parameters"""
        
        descriptions = df['processed_description'].fillna('').astype(str)
        
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
        
        return tfidf_df
    
    def _extract_validated_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract only validated domain-specific features"""
        
        features = {}
        descriptions = df['processed_description'].fillna('').astype(str).str.lower()
        
        # Always create all domain features to ensure consistency between train/test
        for category, keywords in self.domain_keywords.items():
            category_name = category.lower().replace(' ', '_').replace('&', 'and')
            
            # Binary presence of category keywords
            keyword_present = [1 if any(keyword in desc for keyword in keywords) else 0 for desc in descriptions]
            features[f'domain_{category_name}_present'] = keyword_present
            
            # Keyword count
            keyword_counts = [sum(1 for keyword in keywords if keyword in desc) for desc in descriptions]
            features[f'domain_{category_name}_count'] = keyword_counts
        
        return pd.DataFrame(features, index=df.index)
    
    def _extract_key_structural_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract only the most important structural features"""
        
        features = {}
        descriptions = df['processed_description'].fillna('').astype(str)
        original_descriptions = df.get('Item_Descripton', descriptions).fillna('').astype(str)
        
        # Core length features
        features['desc_length'] = original_descriptions.str.len()
        features['word_count'] = descriptions.str.split().str.len().fillna(0)
        
        # Character composition features
        char_counts = features['desc_length']
        digit_counts = original_descriptions.str.count(r'\d')
        alpha_counts = original_descriptions.str.count(r'[a-zA-Z]')
        upper_counts = original_descriptions.str.count(r'[A-Z]')
        
        # Important ratios (from diagnostic analysis - these had high importance)
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
        
        # Word length statistics
        avg_word_lengths = []
        for desc in descriptions:
            words = desc.split() if desc else []
            if words:
                avg_word_lengths.append(np.mean([len(word) for word in words]))
            else:
                avg_word_lengths.append(0)
        
        features['avg_word_length'] = avg_word_lengths
        
        # Binary indicators
        features['has_numbers'] = [1 if count > 0 else 0 for count in digit_counts]
        
        return pd.DataFrame(features, index=df.index)
    
    def _extract_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract technical specification features"""
        
        features = {}
        descriptions = df['processed_description'].fillna('').astype(str).str.lower()
        
        # Technical specification patterns
        for pattern_name, pattern in self.technical_patterns.items():
            features[f'has_{pattern_name}'] = descriptions.str.contains(pattern, regex=True).astype(int)
        
        return pd.DataFrame(features, index=df.index)
    
    def _remove_correlated_features(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features"""
        
        threshold = self.config["correlation_removal"]["threshold"]
        
        # Calculate correlation matrix
        corr_matrix = feature_df.corr().abs()
        
        # Find pairs of highly correlated features
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        
        if to_drop:
            print(f"   🔄 Removing {len(to_drop)} highly correlated features (threshold: {threshold})")
            feature_df = feature_df.drop(columns=to_drop)
        
        return feature_df
    
    def _align_features_to_training(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Align test features to match training features"""
        
        if self.scaler is None:
            print("⚠️ Scaler not fitted, cannot align features")
            return feature_df
        
        # Get the feature names that the scaler was trained on
        try:
            training_features = self.scaler.feature_names_in_
        except AttributeError:
            print("⚠️ Scaler doesn't have feature_names_in_, using all features")
            return feature_df
        
        # Align features
        aligned_df = pd.DataFrame(index=feature_df.index)
        
        for feature_name in training_features:
            if feature_name in feature_df.columns:
                aligned_df[feature_name] = feature_df[feature_name]
            else:
                # Add missing features with zero values
                aligned_df[feature_name] = 0
        
        print(f"   🔧 Aligned {len(aligned_df.columns)} features to training set (added {len(training_features) - len(feature_df.columns)} missing features)")
        
        return aligned_df
    
    def _apply_feature_scaling(self, feature_df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Apply feature scaling to normalize different feature types"""
        
        if is_training:
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(feature_df)
        else:
            if self.scaler is None:
                print("⚠️ Scaler not fitted, returning unscaled features")
                return feature_df
            scaled_features = self.scaler.transform(feature_df)
        
        # Convert back to DataFrame
        scaled_df = pd.DataFrame(
            scaled_features, 
            columns=feature_df.columns, 
            index=feature_df.index
        )
        
        return scaled_df
    
    def _apply_conservative_feature_selection(self, feature_df: pd.DataFrame, y_labels: pd.Series) -> pd.DataFrame:
        """Apply conservative feature selection"""
        
        selection_config = self.config["feature_selection"]
        k = min(selection_config["k"], feature_df.shape[1])
        
        # Handle categorical labels
        y_labels_encoded = pd.Categorical(y_labels).codes
        
        # Apply feature selection with mutual information (better for mixed features)
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
        
        return selected_df
    
    def _apply_fitted_feature_selection(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Apply already fitted feature selector"""
        
        if self.feature_selector is None or self.selected_feature_names is None:
            print("⚠️ Feature selector not fitted, returning all features")
            return feature_df
        
        try:
            # Ensure we have the same features as during training
            # Fill missing features with zeros
            for feature_name in self.selected_feature_names:
                if feature_name not in feature_df.columns:
                    feature_df[feature_name] = 0
            
            # Select only the features that were selected during training
            feature_df_aligned = feature_df[self.selected_feature_names]
            
            # Apply the feature selector transformation
            selected_features = self.feature_selector.transform(feature_df_aligned)
            
            # Convert to dense array if sparse
            try:
                import scipy.sparse as sp
                if sp.issparse(selected_features):
                    selected_features = selected_features.toarray()  # type: ignore
            except:
                pass
            
            # Convert to DataFrame
            selected_features_array = np.asarray(selected_features)
            selected_df = pd.DataFrame(
                selected_features_array, 
                columns=self.selected_feature_names, 
                index=feature_df.index
            )
            return selected_df
            
        except Exception as e:
            print(f"⚠️ Error applying feature selection: {e}")
            # Return features aligned to training set
            aligned_features = pd.DataFrame(index=feature_df.index)
            for feature_name in self.selected_feature_names:
                if feature_name in feature_df.columns:
                    aligned_features[feature_name] = feature_df[feature_name]
                else:
                    aligned_features[feature_name] = 0
            return aligned_features
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about the feature extraction process"""
        
        info = {
            "approach": self.config["approach"],
            "tfidf_features": self.vectorizer.get_feature_names_out().shape[0] if self.vectorizer else 0,
            "total_features_before_selection": "unknown",
            "selected_features": len(self.selected_feature_names) if self.selected_feature_names else 0,
            "feature_scaling": self.config["feature_scaling"]["enabled"],
            "correlation_removal": self.config["correlation_removal"]["enabled"],
            "feature_selection_method": self.config["feature_selection"]["method"] if self.config["feature_selection"]["enabled"] else "none"
        }
        
        return info
    
    def get_feature_importance_analysis(self, feature_importance_scores: np.ndarray | None = None) -> Dict[str, Any]:
        """Analyze feature importance by type"""
        
        if feature_importance_scores is None or self.selected_feature_names is None:
            return {"status": "no_importance_data"}
        
        # Group features by type
        feature_groups = {
            'tfidf': [name for name in self.selected_feature_names if name.startswith('tfidf_')],
            'domain': [name for name in self.selected_feature_names if name.startswith('domain_')],
            'structural': [name for name in self.selected_feature_names if any(x in name for x in ['_ratio', '_length', '_count', 'avg_', 'has_'])],
            'technical': [name for name in self.selected_feature_names if name.startswith('has_')]
        }
        
        # Calculate importance by group
        group_importance = {}
        for group_name, feature_names in feature_groups.items():
            if feature_names:
                indices = [self.selected_feature_names.index(name) for name in feature_names]
                group_scores = [feature_importance_scores[i] for i in indices]
                group_importance[group_name] = {
                    'total_importance': sum(group_scores),
                    'avg_importance': np.mean(group_scores),
                    'feature_count': len(feature_names),
                    'top_features': [(name, feature_importance_scores[self.selected_feature_names.index(name)]) 
                                   for name in feature_names][:5]
                }
        
        return {
            "status": "analyzed",
            "group_importance": group_importance,
            "total_features": len(self.selected_feature_names)
        }
