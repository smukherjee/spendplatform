#!/usr/bin/env python3
"""
Improved Conservative Enhanced Features v4 for Spend Platform Categorization
Builds on optimized baseline (57.94%) with incremental conservative enhancements
"""

import pandas as pd
import numpy as np
import pickle
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
import time

warnings.filterwarnings('ignore')

class ImprovedConservativeFeatureExtractor:
    """Improved conservative feature extraction building on optimized baseline"""
    
    def __init__(self, baseline_config=None):
        # Load optimized baseline configuration
        self.baseline_config = baseline_config or self._load_optimized_baseline()
        self.vectorizer = None
        self.feature_selector = None
        self.scaler = None
        self.selected_feature_names = None
        self.domain_keywords = self._initialize_domain_keywords()
        
    def _load_optimized_baseline(self) -> Dict[str, Any]:
        """Load the optimized baseline configuration"""
        try:
            with open('optimized_baseline_config.pkl', 'rb') as f:
                config = pickle.load(f)
            print(f"✅ Loaded optimized baseline config: {config['best_params']}")
            return config['best_params']
        except:
            # Fallback to discovered optimal parameters
            return {
                'max_features': 1500,
                'ngram_range': (1, 2),
                'min_df': 1,
                'max_df': 0.9
            }
    
    def _initialize_domain_keywords(self) -> Dict[str, List[str]]:
        """Initialize domain keywords focused on the 10 categories from pipeline data"""
        return {
            'Electrical': [
                'electrical', 'electric', 'cable', 'wire', 'voltage', 'motor', 'switch', 
                'circuit', 'relay', 'transformer', 'conductor', 'insulator'
            ],
            'Manufacturing Components & Supplies': [
                'bearing', 'gear', 'bolt', 'screw', 'fastener', 'gasket', 'component', 
                'assembly', 'seal', 'washer', 'nut', 'rivet'
            ],
            'Industrial Manufacturing & Processing Machinery & Accessories': [
                'machinery', 'pump', 'compressor', 'valve', 'industrial', 'processing', 
                'hydraulic', 'pneumatic', 'mechanical', 'equipment'
            ],
            'Tools & General Machinery': [
                'tool', 'drill', 'cutting', 'grinding', 'wrench', 'hammer', 'lathe',
                'saw', 'driver', 'blade', 'bit'
            ],
            'Power Generation & Distribution Machinery & Accessories': [
                'generator', 'power', 'battery', 'electrical', 'energy', 'voltage',
                'distribution', 'generation', 'turbine'
            ],
            'Chemicals & Lubes': [
                'oil', 'lubricant', 'grease', 'chemical', 'fluid', 'hydraulic',
                'coolant', 'solvent', 'additive'
            ],
            'Pipes, Valves & Fittings': [
                'pipe', 'valve', 'fitting', 'tube', 'pressure', 'flow',
                'coupling', 'elbow', 'tee', 'reducer'
            ],
            'Office Equipment, Furniture, & Supplies': [
                'office', 'furniture', 'desk', 'paper', 'printer', 'supplies',
                'chair', 'cabinet', 'filing'
            ],
            'Filtration': [
                'filter', 'filtration', 'strainer', 'cartridge', 'element',
                'membrane', 'screen'
            ],
            'Material H&ling, Storage & Packaging': [
                'storage', 'handling', 'container', 'packaging', 'material',
                'box', 'pallet', 'crate'
            ]
        }
    
    def extract_enhanced_features(self, df: pd.DataFrame, y_labels=None, is_training: bool = True) -> pd.DataFrame:
        """Extract improved conservative enhanced features"""
        
        print("🔧 Extracting improved conservative enhanced features (v4)...")
        
        df_processed = df.copy().reset_index(drop=True)
        
        # Ensure we have the description column
        if 'processed_description' not in df_processed.columns:
            if 'Item_Descripton' in df_processed.columns:
                df_processed['processed_description'] = df_processed['Item_Descripton'].fillna('').astype(str)
            else:
                raise ValueError("No description column found")
        
        # 1. Start with optimized baseline TF-IDF
        baseline_features = self._extract_optimized_baseline_features(df_processed, is_training)
        print(f"   ✅ Optimized baseline TF-IDF: {baseline_features.shape[1]} features")
        
        # 2. Add domain features (conservative)
        domain_features = self._extract_domain_features(df_processed)
        print(f"   ✅ Domain features: {domain_features.shape[1]} features")
        
        # 3. Add structural features (selective)
        structural_features = self._extract_structural_features(df_processed)
        print(f"   ✅ Structural features: {structural_features.shape[1]} features")
        
        # Combine all features
        all_features = pd.concat([baseline_features, domain_features, structural_features], axis=1)
        print(f"   📊 Combined features: {all_features.shape[1]}")
        
        # 4. Apply conservative feature scaling (only non-TF-IDF features)
        if is_training:
            all_features = self._apply_selective_scaling(all_features, is_training)
        else:
            all_features = self._apply_selective_scaling(all_features, is_training)
        
        # 5. Apply conservative feature selection
        if is_training and y_labels is not None:
            all_features = self._apply_conservative_selection(all_features, y_labels)
            print(f"   🎯 Conservative feature selection: {all_features.shape[1]} features")
        elif not is_training and self.feature_selector is not None:
            all_features = self._apply_fitted_selection(all_features)
        
        print(f"✅ Improved conservative feature extraction complete: {all_features.shape[1]} features")
        return all_features
    
    def _extract_optimized_baseline_features(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Extract baseline TF-IDF features using optimized parameters"""
        
        descriptions = df['processed_description'].fillna('').astype(str)
        
        if is_training:
            self.vectorizer = TfidfVectorizer(
                **self.baseline_config,
                sublinear_tf=True,
                use_idf=True,
                lowercase=True,
                analyzer='word'
            )
            tfidf_matrix = self.vectorizer.fit_transform(descriptions)
        else:
            if self.vectorizer is None:
                raise ValueError("TF-IDF vectorizer not fitted")
            tfidf_matrix = self.vectorizer.transform(descriptions)
        
        # Convert to DataFrame with proper handling
        try:
            import scipy.sparse as sp
            if sp.issparse(tfidf_matrix):
                tfidf_array = tfidf_matrix.toarray()
            else:
                tfidf_array = np.array(tfidf_matrix)
        except:
            tfidf_array = np.array(tfidf_matrix.todense())
        
        tfidf_feature_names = [f'tfidf_{i}' for i in range(tfidf_array.shape[1])]
        tfidf_df = pd.DataFrame(tfidf_array, columns=tfidf_feature_names, index=df.index)
        
        return tfidf_df
    
    def _extract_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract conservative domain-specific features"""
        
        features = {}
        descriptions = df['processed_description'].fillna('').astype(str).str.lower()
        
        for category, keywords in self.domain_keywords.items():
            category_name = category.lower().replace(' ', '_').replace('&', 'and').replace(',', '')
            
            # Binary presence (most reliable signal)
            keyword_matches = [any(keyword in desc for keyword in keywords) for desc in descriptions]
            features[f'domain_{category_name}_present'] = [1 if match else 0 for match in keyword_matches]
            
            # Keyword density (normalized by description length)
            keyword_densities = []
            for desc in descriptions:
                desc_words = desc.split()
                if len(desc_words) > 0:
                    match_count = sum(1 for keyword in keywords if keyword in desc)
                    density = match_count / len(desc_words)
                    keyword_densities.append(density)
                else:
                    keyword_densities.append(0)
            
            features[f'domain_{category_name}_density'] = keyword_densities
        
        return pd.DataFrame(features, index=df.index)
    
    def _extract_structural_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract selective structural features that add value"""
        
        features = {}
        descriptions = df['processed_description'].fillna('').astype(str)
        
        # Length-based features (proven important)
        features['desc_length'] = descriptions.str.len()
        features['word_count'] = descriptions.str.split().str.len().fillna(0)
        
        # Character type ratios (important from previous analysis)
        char_counts = features['desc_length']
        digit_counts = descriptions.str.count(r'\d')
        
        features['digit_ratio'] = [digit_counts.iloc[i] / max(char_counts.iloc[i], 1) for i in range(len(char_counts))]
        features['has_numbers'] = [1 if count > 0 else 0 for count in digit_counts]
        
        # Technical patterns (selective)
        features['has_technical_pattern'] = descriptions.str.contains(r'\b\w*\d+\w*\b', regex=True).astype(int)
        
        return pd.DataFrame(features, index=df.index)
    
    def _apply_selective_scaling(self, feature_df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Apply scaling only to non-TF-IDF features"""
        
        # Separate TF-IDF and other features
        tfidf_cols = [col for col in feature_df.columns if col.startswith('tfidf_')]
        other_cols = [col for col in feature_df.columns if not col.startswith('tfidf_')]
        
        if not other_cols:
            return feature_df
        
        tfidf_features = feature_df[tfidf_cols]
        other_features = feature_df[other_cols]
        
        # Scale only non-TF-IDF features
        if is_training:
            self.scaler = StandardScaler()
            scaled_other = self.scaler.fit_transform(other_features)
        else:
            if self.scaler is None:
                return feature_df
            scaled_other = self.scaler.transform(other_features)
        
        # Combine back
        scaled_other_df = pd.DataFrame(scaled_other, columns=other_cols, index=feature_df.index)
        result = pd.concat([tfidf_features, scaled_other_df], axis=1)
        
        print(f"   📏 Scaled {len(other_cols)} non-TF-IDF features")
        return result
    
    def _apply_conservative_selection(self, feature_df: pd.DataFrame, y_labels: pd.Series) -> pd.DataFrame:
        """Apply conservative feature selection that preserves most features"""
        
        # Conservative selection - keep 90% of features
        k = int(feature_df.shape[1] * 0.9)
        k = max(k, min(1000, feature_df.shape[1]))  # At least 1000 or all features
        
        # Encode labels
        y_labels_encoded = pd.Categorical(y_labels).codes
        
        # Use chi2 for TF-IDF features and mutual info for mixed features
        tfidf_cols = [col for col in feature_df.columns if col.startswith('tfidf_')]
        
        if len(tfidf_cols) > 0 and len(tfidf_cols) == len(feature_df.columns):
            # All TF-IDF features - use chi2
            score_func = chi2
        else:
            # Mixed features - use mutual info
            score_func = mutual_info_classif
        
        self.feature_selector = SelectKBest(score_func=score_func, k=k)
        selected_features = self.feature_selector.fit_transform(feature_df, y_labels_encoded)
        
        # Get selected feature names
        selected_mask = self.feature_selector.get_support()
        self.selected_feature_names = feature_df.columns[selected_mask].tolist()
        
        # Convert to DataFrame
        selected_df = pd.DataFrame(
            selected_features,
            columns=self.selected_feature_names,
            index=feature_df.index
        )
        
        return selected_df
    
    def _apply_fitted_selection(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted feature selector to test data"""
        
        if self.feature_selector is None or self.selected_feature_names is None:
            print("⚠️ Feature selector not fitted, returning all features")
            return feature_df
        
        try:
            # Get the original column order from training
            training_columns = self.selected_feature_names
            
            # Create aligned feature matrix with exact same columns as training
            aligned_features = pd.DataFrame(0.0, index=feature_df.index, columns=training_columns)
            
            # Fill in available features
            for col in training_columns:
                if col in feature_df.columns:
                    aligned_features[col] = feature_df[col]
                # Missing features remain as 0
            
            print(f"   🔧 Aligned {len(aligned_features.columns)} features for test set")
            return aligned_features
            
        except Exception as e:
            print(f"⚠️ Error in feature selection: {e}")
            # Fallback - return subset of features that exist
            common_features = [col for col in self.selected_feature_names if col in feature_df.columns]
            if common_features:
                return feature_df[common_features]
            else:
                return feature_df
    
    def get_enhancement_info(self) -> Dict[str, Any]:
        """Get information about the enhancement approach"""
        
        return {
            'approach': 'improved_conservative_v4',
            'baseline_config': self.baseline_config,
            'domain_categories': len(self.domain_keywords),
            'selected_features': len(self.selected_feature_names) if self.selected_feature_names else 0,
            'feature_scaling': 'selective (non-TF-IDF only)',
            'feature_selection': 'conservative (90% retention)'
        }

def test_improved_conservative_features():
    """Test the improved conservative enhanced features"""
    
    print("🧪 Testing Improved Conservative Enhanced Features v4")
    print("=" * 60)
    
    # Load the optimal dataset identified by baseline validation
    dataset_path = '/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.2/pipeline_run_20250907_143517/preprocessed_data/test_data.xlsx'
    
    print(f"📁 Loading optimal dataset...")
    df = pd.read_excel(dataset_path)
    
    # Clean data
    df_clean = df.dropna(subset=['Item_Descripton', 'Category L2']).copy()
    print(f"   ✅ Loaded {len(df_clean)} records")
    
    # Basic text preprocessing
    df_clean['processed_description'] = df_clean['Item_Descripton'].astype(str).str.lower()
    df_clean['processed_description'] = df_clean['processed_description'].str.replace(r'[^\w\s]', ' ', regex=True)
    df_clean['processed_description'] = df_clean['processed_description'].str.replace(r'\s+', ' ', regex=True)
    df_clean['processed_description'] = df_clean['processed_description'].str.strip()
    
    # Filter categories with sufficient samples
    category_counts = df_clean['Category L2'].value_counts()
    valid_categories = category_counts[category_counts >= 5].index
    df_filtered = df_clean[df_clean['Category L2'].isin(valid_categories)].copy()
    
    print(f"   📊 Categories: {len(valid_categories)}, Records: {len(df_filtered)}")
    
    # Split data
    X = df_filtered[['Item_Descripton', 'processed_description']].copy()
    y = df_filtered['Category L2'].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   🔄 Split: {len(X_train)} train, {len(X_test)} test")
    
    # Test baseline first
    print(f"\n📊 Testing Optimized Baseline...")
    baseline_accuracy = test_baseline_performance(X_train, X_test, y_train, y_test)
    
    # Test improved conservative features
    print(f"\n🚀 Testing Improved Conservative Features v4...")
    enhanced_accuracy = test_enhanced_performance(X_train, X_test, y_train, y_test)
    
    # Summary
    print(f"\n📋 Performance Comparison")
    print("=" * 40)
    print(f"Optimized Baseline: {baseline_accuracy:.2f}%")
    print(f"Enhanced v4: {enhanced_accuracy:.2f}%")
    improvement = enhanced_accuracy - baseline_accuracy
    print(f"Improvement: {improvement:+.2f} percentage points")
    
    if enhanced_accuracy >= 60:
        print(f"✅ TARGET ACHIEVED! ({enhanced_accuracy:.2f}% ≥ 60%)")
    elif improvement > 0:
        print(f"📈 POSITIVE IMPROVEMENT! (+{improvement:.2f}%)")
    else:
        print(f"⚠️ Enhancement needs refinement ({improvement:+.2f}%)")
    
    return {
        'baseline_accuracy': baseline_accuracy,
        'enhanced_accuracy': enhanced_accuracy,
        'improvement': improvement
    }

def test_baseline_performance(X_train, X_test, y_train, y_test):
    """Test baseline performance with optimized parameters"""
    
    # Load optimized baseline config
    try:
        with open('optimized_baseline_config.pkl', 'rb') as f:
            config = pickle.load(f)
        best_params = config['best_params']
    except:
        best_params = {'max_features': 1500, 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 0.9}
    
    # Create vectorizer and classifier
    vectorizer = TfidfVectorizer(**best_params, sublinear_tf=True)
    classifier = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    
    # Train and evaluate
    X_train_vec = vectorizer.fit_transform(X_train['processed_description'])
    X_test_vec = vectorizer.transform(X_test['processed_description'])
    
    classifier.fit(X_train_vec, y_train)
    y_pred = classifier.predict(X_test_vec)
    
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f"   Baseline accuracy: {accuracy:.2f}%")
    
    return accuracy

def test_enhanced_performance(X_train, X_test, y_train, y_test):
    """Test enhanced features performance"""
    
    # Create feature extractor
    extractor = ImprovedConservativeFeatureExtractor()
    
    # Extract features
    X_train_features = extractor.extract_enhanced_features(X_train, y_train, is_training=True)
    X_test_features = extractor.extract_enhanced_features(X_test, is_training=False)
    
    # Train classifier
    classifier = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    classifier.fit(X_train_features, y_train)
    
    # Evaluate
    y_pred = classifier.predict(X_test_features)
    accuracy = accuracy_score(y_test, y_pred) * 100
    
    print(f"   Enhanced accuracy: {accuracy:.2f}%")
    
    # Feature importance analysis
    feature_importance = classifier.feature_importances_
    enhancement_info = extractor.get_enhancement_info()
    
    print(f"   📊 Enhancement info: {enhancement_info}")
    
    return accuracy

if __name__ == "__main__":
    results = test_improved_conservative_features()
