#!/usr/bin/env python3
"""
Test Enhanced Feature Engineering Integration
Tests the new enhanced features and compares performance with existing approach
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings('ignore')

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import local modules
from enhanced_features import EnhancedFeatureExtractor
from text_parsers import NLTKTextParser
from config import ModelConfig

def load_training_data():
    """Load the training data"""
    print("📊 Loading training data...")
    
    # Try different possible paths for the training data
    data_paths = [
        project_root / "preprocessed_data" / "train_data.xlsx",
        project_root / "context" / "archive" / "sample spend data- filled.xlsx",
        project_root.parent / "context" / "archive" / "sample spend data- filled.xlsx",
        project_root / "context" / "sample spend data- filled.xlsx",
        project_root / "sample spend data- filled.xlsx",
        project_root / "training_data.xlsx"
    ]
    
    training_data = None
    for path in data_paths:
        if path.exists():
            print(f"Found training data at: {path}")
            training_data = pd.read_excel(path)
            break
    
    if training_data is None:
        raise FileNotFoundError("Could not find training data file")
    
    print(f"✅ Loaded {len(training_data)} training records")
    return training_data

def test_existing_features(df):
    """Test the existing feature extraction approach"""
    print("\n🔍 Testing existing feature extraction approach...")
    
    # Initialize model config and text processor
    model_config = ModelConfig()
    text_processor = model_config.get_text_parser()
    
    # Process text and extract features (existing approach)
    processed_descriptions = []
    for desc in df['Item_Descripton'].fillna(''):
        processed = text_processor.preprocess_text(str(desc))
        processed_descriptions.append(processed)
    
    df_processed = df.copy()
    df_processed['processed_description'] = processed_descriptions
    
    # Extract existing features using TF-IDF from config
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    vectorizer = TfidfVectorizer(
        max_features=model_config.tfidf_max_features,
        ngram_range=model_config.tfidf_ngram_range,
        min_df=model_config.tfidf_min_df,
        max_df=model_config.tfidf_max_df
    )
    
    tfidf_features = vectorizer.fit_transform(df_processed['processed_description'])
    feature_names = [f'tfidf_{i}' for i in range(tfidf_features.shape[1])]
    
    # Convert to DataFrame
    import scipy.sparse as sp
    if sp.issparse(tfidf_features):
        tfidf_array = np.array(tfidf_features.todense())
    else:
        tfidf_array = np.array(tfidf_features)
    
    existing_features = pd.DataFrame(tfidf_array, columns=feature_names, index=df.index)
    
    print(f"✅ Existing approach: {existing_features.shape[1]} features")
    return existing_features, vectorizer

def test_enhanced_features(df):
    """Test the new enhanced feature extraction approach"""
    print("\n🚀 Testing enhanced feature extraction approach...")
    
    # Initialize enhanced feature extractor
    enhanced_extractor = EnhancedFeatureExtractor()
    
    # Initialize model config and text processor
    model_config = ModelConfig()
    text_processor = model_config.get_text_parser()
    
    # Process text first
    processed_descriptions = []
    for desc in df['Item_Descripton'].fillna(''):
        processed = text_processor.preprocess_text(str(desc))
        processed_descriptions.append(processed)
    
    df_processed = df.copy()
    df_processed['processed_description'] = processed_descriptions
    
    # Extract enhanced features
    enhanced_features = enhanced_extractor.extract_enhanced_features(df_processed, is_training=True)
    
    print(f"✅ Enhanced approach: {enhanced_features.shape[1]} features")
    return enhanced_features, enhanced_extractor

def compare_performance(df, existing_features, enhanced_features):
    """Compare performance between existing and enhanced approaches"""
    print("\n📈 Comparing model performance...")
    
    # Ensure all DataFrames have the same index
    df = df.reset_index(drop=True)
    existing_features = existing_features.reset_index(drop=True)
    enhanced_features = enhanced_features.reset_index(drop=True)
    
    # Prepare target variables
    y_l2 = df['Category L2'].fillna('Unknown')
    
    # Filter out samples with missing L2 categories for fair comparison
    valid_indices = y_l2 != 'Unknown'
    if not valid_indices.any():
        print("⚠️ No valid L2 categories found for training")
        return
    
    y_l2_filtered = y_l2[valid_indices]
    existing_features_filtered = existing_features[valid_indices]
    enhanced_features_filtered = enhanced_features[valid_indices]
    
    print(f"Training on {len(y_l2_filtered)} samples with valid L2 categories")
    
    # Split data
    test_size = 0.3
    random_state = 42
    
    # Test existing approach
    print("\n🔄 Testing existing feature approach...")
    X_train_existing, X_test_existing, y_train, y_test = train_test_split(
        existing_features_filtered, y_l2_filtered, 
        test_size=test_size, random_state=random_state, stratify=y_l2_filtered
    )
    
    # Train model with existing features
    rf_existing = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf_existing.fit(X_train_existing, y_train)
    
    # Predict and evaluate existing approach
    y_pred_existing = rf_existing.predict(X_test_existing)
    accuracy_existing = accuracy_score(y_test, y_pred_existing)
    
    print(f"Existing approach accuracy: {accuracy_existing:.4f}")
    
    # Test enhanced approach
    print("\n🚀 Testing enhanced feature approach...")
    X_train_enhanced, X_test_enhanced, y_train_enh, y_test_enh = train_test_split(
        enhanced_features_filtered, y_l2_filtered, 
        test_size=test_size, random_state=random_state, stratify=y_l2_filtered
    )
    
    # Train model with enhanced features
    rf_enhanced = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf_enhanced.fit(X_train_enhanced, y_train_enh)
    
    # Predict and evaluate enhanced approach
    y_pred_enhanced = rf_enhanced.predict(X_test_enhanced)
    accuracy_enhanced = accuracy_score(y_test_enh, y_pred_enhanced)
    
    print(f"Enhanced approach accuracy: {accuracy_enhanced:.4f}")
    
    # Calculate improvement
    improvement = ((accuracy_enhanced - accuracy_existing) / accuracy_existing) * 100
    print(f"\n📊 Performance Improvement: {improvement:+.2f}%")
    
    # Detailed classification reports
    print("\n📋 Detailed Classification Report - Existing Approach:")
    print(classification_report(y_test, y_pred_existing, zero_division=0))
    
    print("\n📋 Detailed Classification Report - Enhanced Approach:")
    print(classification_report(y_test_enh, y_pred_enhanced, zero_division=0))
    
    # Feature importance analysis for enhanced approach
    print("\n🎯 Top 20 Most Important Enhanced Features:")
    feature_importance = pd.DataFrame({
        'feature': enhanced_features.columns,
        'importance': rf_enhanced.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(20).to_string(index=False))
    
    return {
        'existing_accuracy': accuracy_existing,
        'enhanced_accuracy': accuracy_enhanced,
        'improvement_percent': improvement,
        'feature_importance': feature_importance
    }

def analyze_feature_groups(enhanced_extractor, feature_importance_df):
    """Analyze performance by feature groups"""
    print("\n🔍 Analyzing feature group contributions...")
    
    feature_groups = enhanced_extractor.get_feature_importance_groups()
    group_importance = {}
    
    for group_name, group_patterns in feature_groups.items():
        group_features = []
        for pattern in group_patterns:
            matching_features = feature_importance_df[
                feature_importance_df['feature'].str.contains(pattern, case=False)
            ]
            group_features.extend(matching_features['feature'].tolist())
        
        if group_features:
            group_total_importance = feature_importance_df[
                feature_importance_df['feature'].isin(group_features)
            ]['importance'].sum()
            group_importance[group_name] = {
                'total_importance': group_total_importance,
                'feature_count': len(group_features),
                'avg_importance': group_total_importance / len(group_features) if group_features else 0
            }
    
    # Display group analysis
    print("\n📊 Feature Group Analysis:")
    for group_name, stats in sorted(group_importance.items(), 
                                   key=lambda x: x[1]['total_importance'], reverse=True):
        print(f"{group_name}:")
        print(f"  Total Importance: {stats['total_importance']:.4f}")
        print(f"  Feature Count: {stats['feature_count']}")
        print(f"  Average Importance: {stats['avg_importance']:.6f}")
        print()

def main():
    """Main testing function"""
    print("🧪 Enhanced Features Testing Suite")
    print("=" * 50)
    
    try:
        # Load training data
        training_data = load_training_data()
        
        # Take a sample for testing (to speed up the process)
        if len(training_data) > 1000:
            print(f"📝 Using sample of 1000 records for testing (from {len(training_data)} total)")
            sample_data = training_data.sample(n=1000, random_state=42)
        else:
            sample_data = training_data
        
        # Test existing approach
        existing_features, existing_vectorizer = test_existing_features(sample_data)
        
        # Test enhanced approach
        enhanced_features, enhanced_extractor = test_enhanced_features(sample_data)
        
        # Compare performance
        results = compare_performance(sample_data, existing_features, enhanced_features)
        
        if results:
            # Analyze feature groups
            analyze_feature_groups(enhanced_extractor, results['feature_importance'])
            
            # Summary
            print("\n" + "=" * 50)
            print("🎯 TESTING SUMMARY")
            print("=" * 50)
            print(f"Existing Features: {existing_features.shape[1]}")
            print(f"Enhanced Features: {enhanced_features.shape[1]}")
            print(f"Existing Accuracy: {results['existing_accuracy']:.4f}")
            print(f"Enhanced Accuracy: {results['enhanced_accuracy']:.4f}")
            print(f"Improvement: {results['improvement_percent']:+.2f}%")
            
            if results['improvement_percent'] > 0:
                print("✅ Enhanced features show improvement!")
            else:
                print("⚠️ Enhanced features need further optimization")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
