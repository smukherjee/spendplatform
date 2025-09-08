#!/usr/bin/env python3
"""
Test Optimized Enhanced Features v2
Tests the new optimized enhanced features and compares performance with v1 and existing approach
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
from enhanced_features_v2 import OptimizedEnhancedFeatureExtractor
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

def test_enhanced_features_v1(df):
    """Test the original enhanced feature extraction approach (v1)"""
    print("\n🔧 Testing enhanced features v1...")
    
    # Initialize enhanced feature extractor v1
    enhanced_extractor_v1 = EnhancedFeatureExtractor()
    
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
    
    # Extract enhanced features v1
    enhanced_features_v1 = enhanced_extractor_v1.extract_enhanced_features(df_processed, is_training=True)
    
    print(f"✅ Enhanced v1 approach: {enhanced_features_v1.shape[1]} features")
    return enhanced_features_v1, enhanced_extractor_v1

def test_optimized_enhanced_features_v2(df, y_labels):
    """Test the new optimized enhanced feature extraction approach (v2)"""
    print("\n🚀 Testing optimized enhanced features v2...")
    
    # Initialize optimized enhanced feature extractor v2
    optimized_extractor = OptimizedEnhancedFeatureExtractor()
    
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
    
    # Extract optimized enhanced features v2 with feature selection
    optimized_features = optimized_extractor.extract_enhanced_features(
        df_processed, 
        y_labels=y_labels, 
        is_training=True
    )
    
    print(f"✅ Optimized v2 approach: {optimized_features.shape[1]} features")
    
    # Print feature selection info
    selection_info = optimized_extractor.get_feature_selection_info()
    print(f"🎯 Feature selection: {selection_info['status']}")
    if selection_info['status'] == 'fitted':
        print(f"   Selected {selection_info['selected_features']} features using {selection_info['method']}")
    
    return optimized_features, optimized_extractor

def compare_all_approaches(df, existing_features, enhanced_v1_features, optimized_v2_features):
    """Compare performance between all three approaches"""
    print("\n📈 Comparing all approaches...")
    
    # Ensure all DataFrames have the same index
    df = df.reset_index(drop=True)
    existing_features = existing_features.reset_index(drop=True)
    enhanced_v1_features = enhanced_v1_features.reset_index(drop=True)
    optimized_v2_features = optimized_v2_features.reset_index(drop=True)
    
    # Prepare target variables
    y_l2 = df['Category L2'].fillna('Unknown')
    
    # Filter out samples with missing L2 categories for fair comparison
    valid_indices = y_l2 != 'Unknown'
    if not valid_indices.any():
        print("⚠️ No valid L2 categories found for training")
        return
    
    y_l2_filtered = y_l2[valid_indices]
    existing_features_filtered = existing_features[valid_indices]
    enhanced_v1_features_filtered = enhanced_v1_features[valid_indices]
    optimized_v2_features_filtered = optimized_v2_features[valid_indices]
    
    print(f"Training on {len(y_l2_filtered)} samples with valid L2 categories")
    
    # Split data
    test_size = 0.3
    random_state = 42
    
    results = {}
    
    # Test existing approach
    print("\n🔄 Testing existing feature approach...")
    X_train_existing, X_test_existing, y_train, y_test = train_test_split(
        existing_features_filtered, y_l2_filtered, 
        test_size=test_size, random_state=random_state, stratify=y_l2_filtered
    )
    
    rf_existing = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf_existing.fit(X_train_existing, y_train)
    y_pred_existing = rf_existing.predict(X_test_existing)
    accuracy_existing = accuracy_score(y_test, y_pred_existing)
    
    print(f"Existing approach accuracy: {accuracy_existing:.4f}")
    results['existing'] = {'accuracy': accuracy_existing, 'features': existing_features.shape[1]}
    
    # Test enhanced v1 approach
    print("\n🔧 Testing enhanced v1 feature approach...")
    X_train_v1, X_test_v1, y_train_v1, y_test_v1 = train_test_split(
        enhanced_v1_features_filtered, y_l2_filtered, 
        test_size=test_size, random_state=random_state, stratify=y_l2_filtered
    )
    
    rf_v1 = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf_v1.fit(X_train_v1, y_train_v1)
    y_pred_v1 = rf_v1.predict(X_test_v1)
    accuracy_v1 = accuracy_score(y_test_v1, y_pred_v1)
    
    print(f"Enhanced v1 approach accuracy: {accuracy_v1:.4f}")
    results['enhanced_v1'] = {'accuracy': accuracy_v1, 'features': enhanced_v1_features.shape[1]}
    
    # Test optimized v2 approach
    print("\n🚀 Testing optimized v2 feature approach...")
    X_train_v2, X_test_v2, y_train_v2, y_test_v2 = train_test_split(
        optimized_v2_features_filtered, y_l2_filtered, 
        test_size=test_size, random_state=random_state, stratify=y_l2_filtered
    )
    
    rf_v2 = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf_v2.fit(X_train_v2, y_train_v2)
    y_pred_v2 = rf_v2.predict(X_test_v2)
    accuracy_v2 = accuracy_score(y_test_v2, y_pred_v2)
    
    print(f"Optimized v2 approach accuracy: {accuracy_v2:.4f}")
    results['optimized_v2'] = {'accuracy': accuracy_v2, 'features': optimized_v2_features.shape[1]}
    
    # Calculate improvements
    improvement_v1 = ((accuracy_v1 - accuracy_existing) / accuracy_existing) * 100
    improvement_v2 = ((accuracy_v2 - accuracy_existing) / accuracy_existing) * 100
    v2_vs_v1_improvement = ((accuracy_v2 - accuracy_v1) / accuracy_v1) * 100
    
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"Existing vs Enhanced v1: {improvement_v1:+.2f}%")
    print(f"Existing vs Optimized v2: {improvement_v2:+.2f}%")
    print(f"Enhanced v1 vs Optimized v2: {v2_vs_v1_improvement:+.2f}%")
    
    # Detailed classification reports
    print("\n📋 Detailed Classification Report - Existing Approach:")
    print(classification_report(y_test, y_pred_existing, zero_division=0))
    
    print("\n📋 Detailed Classification Report - Optimized v2 Approach:")
    print(classification_report(y_test_v2, y_pred_v2, zero_division=0))
    
    # Feature importance analysis for optimized v2
    print("\n🎯 Top 20 Most Important Optimized v2 Features:")
    feature_importance = pd.DataFrame({
        'feature': optimized_v2_features.columns,
        'importance': rf_v2.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(20).to_string(index=False))
    
    return {
        'results': results,
        'improvements': {
            'v1_vs_existing': improvement_v1,
            'v2_vs_existing': improvement_v2,
            'v2_vs_v1': v2_vs_v1_improvement
        },
        'feature_importance': feature_importance
    }

def main():
    """Main testing function"""
    print("🧪 Optimized Enhanced Features v2 Testing Suite")
    print("=" * 60)
    
    try:
        # Load training data
        training_data = load_training_data()
        
        # Take a sample for testing (to speed up the process)
        if len(training_data) > 1000:
            print(f"📝 Using sample of 1000 records for testing (from {len(training_data)} total)")
            sample_data = training_data.sample(n=1000, random_state=42)
        else:
            sample_data = training_data
        
        sample_data = sample_data.reset_index(drop=True)
        
        # Prepare labels for feature selection
        y_labels = sample_data['Category L2'].fillna('Unknown')
        valid_labels = y_labels[y_labels != 'Unknown']
        
        # Test existing approach
        existing_features, existing_vectorizer = test_existing_features(sample_data)
        
        # Test enhanced v1 approach
        enhanced_v1_features, enhanced_v1_extractor = test_enhanced_features_v1(sample_data)
        
        # Test optimized v2 approach with feature selection
        optimized_v2_features, optimized_v2_extractor = test_optimized_enhanced_features_v2(sample_data, y_labels)
        
        # Compare all approaches
        comparison_results = compare_all_approaches(
            sample_data, existing_features, enhanced_v1_features, optimized_v2_features
        )
        
        if comparison_results:
            # Summary
            print("\n" + "=" * 60)
            print("🎯 COMPREHENSIVE TESTING SUMMARY")
            print("=" * 60)
            
            results = comparison_results['results']
            improvements = comparison_results['improvements']
            
            print(f"Existing Features: {results['existing']['features']}")
            print(f"Enhanced v1 Features: {results['enhanced_v1']['features']}")
            print(f"Optimized v2 Features: {results['optimized_v2']['features']}")
            print()
            print(f"Existing Accuracy: {results['existing']['accuracy']:.4f}")
            print(f"Enhanced v1 Accuracy: {results['enhanced_v1']['accuracy']:.4f}")
            print(f"Optimized v2 Accuracy: {results['optimized_v2']['accuracy']:.4f}")
            print()
            print(f"Improvement v1 vs Existing: {improvements['v1_vs_existing']:+.2f}%")
            print(f"Improvement v2 vs Existing: {improvements['v2_vs_existing']:+.2f}%")
            print(f"Improvement v2 vs v1: {improvements['v2_vs_v1']:+.2f}%")
            
            # Determine best approach
            best_accuracy = max(results['existing']['accuracy'], 
                              results['enhanced_v1']['accuracy'], 
                              results['optimized_v2']['accuracy'])
            
            if results['optimized_v2']['accuracy'] == best_accuracy:
                print("\n✅ Optimized v2 features show the best performance!")
            elif results['enhanced_v1']['accuracy'] == best_accuracy:
                print("\n⚠️ Enhanced v1 features still perform best - v2 needs more optimization")
            else:
                print("\n⚠️ Existing features still perform best - enhanced features need more work")
            
            # Feature selection summary
            selection_info = optimized_v2_extractor.get_feature_selection_info()
            print(f"\n🎯 Feature Selection Summary:")
            print(f"   Method: {selection_info['method']}")
            print(f"   Score Function: {selection_info['score_function']}")
            print(f"   Selected Features: {selection_info['selected_features']}")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
