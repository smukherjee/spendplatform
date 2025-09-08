#!/usr/bin/env python3
"""
Test Conservative Enhanced Features v3 for Spend Platform Categorization
Tests the conservative incremental enhancement approach
"""

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append('/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.2')

from conservative_enhanced_features_v3 import ConservativeEnhancedFeatureExtractor

def load_and_prepare_data():
    """Load and prepare the dataset"""
    print("📁 Loading spend data...")
    
    # Use the original predicted training data that was used in previous tests
    file_path = '/Users/sujoymukherjee/code/spendplatform/context/predicted_original_training_data.xlsx'
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Original training dataset not found at {file_path}")
    
    df = pd.read_excel(file_path)
    print(f"   ✅ Loaded {len(df)} records from {file_path}")
    print(f"   📝 Columns: {list(df.columns)}")
    
    # Use the proper column names from original dataset
    description_col = 'Descripton'  # Note: typo in original
    category_col = 'Category L2'
    
    # Basic data preparation
    df = df.dropna(subset=[description_col, category_col]).copy()
    print(f"   ✅ After removing missing values: {len(df)} records")
    
    # Standardize column names for compatibility
    df['Item_Descripton'] = df[description_col]
    df['L2 Category'] = df[category_col]
    
    # Basic text preprocessing
    df['processed_description'] = df['Item_Descripton'].astype(str).str.lower()
    df['processed_description'] = df['processed_description'].str.replace(r'[^\w\s]', ' ', regex=True)
    df['processed_description'] = df['processed_description'].str.replace(r'\s+', ' ', regex=True)
    df['processed_description'] = df['processed_description'].str.strip()
    
    return df

def test_conservative_enhanced_features():
    """Test conservative enhanced features approach"""
    
    print("🧪 Testing Conservative Enhanced Features v3")
    print("=" * 60)
    
    # Load data
    df = load_and_prepare_data()
    
    # Check category distribution
    category_counts = df['L2 Category'].value_counts()
    print(f"\n📊 Category Distribution (top 10):")
    for cat, count in category_counts.head(10).items():
        print(f"   {cat}: {count}")
    
    # Filter categories with sufficient samples (minimum 10 for testing)
    min_samples = 10
    valid_categories = category_counts[category_counts >= min_samples].index
    df_filtered = df[df['L2 Category'].isin(valid_categories)].copy()
    
    print(f"\n🔧 Filtered to {len(valid_categories)} categories with ≥{min_samples} samples")
    print(f"   Total records for testing: {len(df_filtered)}")
    
    # Split data
    X = df_filtered[['Item_Descripton', 'processed_description']].copy()
    y = df_filtered['L2 Category'].copy()
    
    # Use stratified split to ensure all categories are represented
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Data split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing: {len(X_test)} samples")
    
    # Test conservative enhanced features
    print(f"\n🚀 Testing Conservative Enhanced Features v3...")
    print("-" * 50)
    
    start_time = time.time()
    
    # Initialize feature extractor
    feature_extractor = ConservativeEnhancedFeatureExtractor()
    
    # Extract training features
    print("🔧 Extracting training features...")
    X_train_features = feature_extractor.extract_enhanced_features(
        X_train, y_labels=y_train, is_training=True
    )
    
    # Extract test features
    print("🔧 Extracting test features...")
    X_test_features = feature_extractor.extract_enhanced_features(
        X_test, y_labels=None, is_training=False
    )
    
    feature_extraction_time = time.time() - start_time
    
    print(f"⏱️ Feature extraction time: {feature_extraction_time:.2f}s")
    print(f"📊 Training features shape: {X_train_features.shape}")
    print(f"📊 Test features shape: {X_test_features.shape}")
    
    # Train classifier
    print("\n🎯 Training RandomForest classifier...")
    start_time = time.time()
    
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    classifier.fit(X_train_features, y_train)
    training_time = time.time() - start_time
    
    print(f"⏱️ Training time: {training_time:.2f}s")
    
    # Make predictions
    print("\n🔮 Making predictions...")
    start_time = time.time()
    
    y_train_pred = classifier.predict(X_train_features)
    y_test_pred = classifier.predict(X_test_features)
    
    prediction_time = time.time() - start_time
    print(f"⏱️ Prediction time: {prediction_time:.2f}s")
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\n📈 Conservative Enhanced Features v3 Results:")
    print(f"   Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Feature importance analysis
    print(f"\n🔍 Feature Importance Analysis:")
    feature_importance = classifier.feature_importances_
    importance_analysis = feature_extractor.get_feature_importance_analysis(feature_importance)
    
    if importance_analysis["status"] == "analyzed":
        print("   Feature Group Importance:")
        for group_name, group_info in importance_analysis["group_importance"].items():
            print(f"   📊 {group_name.upper()}: {group_info['total_importance']:.4f} "
                  f"(avg: {group_info['avg_importance']:.4f}, features: {group_info['feature_count']})")
            
            # Show top features for each group
            if group_info['top_features']:
                print(f"      Top features:")
                for feat_name, feat_score in group_info['top_features'][:3]:
                    print(f"        - {feat_name}: {feat_score:.4f}")
    
    # Detailed classification report for test set
    print(f"\n📊 Detailed Test Set Classification Report:")
    print("-" * 60)
    report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
    
    # Show overall metrics
    print(f"Overall Test Metrics:")
    if isinstance(report, dict):
        accuracy = report.get('accuracy', 0)
        macro_avg = report.get('macro avg', {})
        print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        if isinstance(macro_avg, dict):
            print(f"   Macro Avg Precision: {macro_avg.get('precision', 0):.4f}")
            print(f"   Macro Avg Recall: {macro_avg.get('recall', 0):.4f}")
            print(f"   Macro Avg F1-Score: {macro_avg.get('f1-score', 0):.4f}")
    
    # Show per-category results (top performing categories)
    category_results = []
    if isinstance(report, dict):
        for category, metrics in report.items():
            if isinstance(metrics, dict) and 'f1-score' in metrics:
                category_results.append((category, metrics['f1-score'], metrics['support']))
    
    category_results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 10 Category Results (by F1-Score):")
    for i, (category, f1_score, support) in enumerate(category_results[:10]):
        print(f"   {i+1:2d}. {category[:40]:<40} F1: {f1_score:.3f} (n={support})")
    
    # Get feature extractor info
    feature_info = feature_extractor.get_feature_info()
    print(f"\n🔧 Feature Extraction Summary:")
    print(f"   Approach: {feature_info['approach']}")
    print(f"   TF-IDF Features: {feature_info['tfidf_features']}")
    print(f"   Selected Features: {feature_info['selected_features']}")
    print(f"   Feature Scaling: {feature_info['feature_scaling']}")
    print(f"   Correlation Removal: {feature_info['correlation_removal']}")
    print(f"   Selection Method: {feature_info['feature_selection_method']}")
    
    return {
        'test_accuracy': test_accuracy,
        'train_accuracy': train_accuracy,
        'feature_count': X_test_features.shape[1],
        'feature_extraction_time': feature_extraction_time,
        'training_time': training_time,
        'prediction_time': prediction_time,
        'classification_report': report,
        'feature_info': feature_info
    }

def performance_comparison():
    """Compare with diagnostic analysis baseline"""
    
    print("\n🔄 Performance Comparison with Previous Results")
    print("=" * 60)
    
    # Previous results from diagnostic analysis
    baseline_results = {
        'existing_approach': 62.67,
        'enhanced_v1': 57.33,
        'enhanced_v2': 55.33
    }
    
    print("Previous Results:")
    for approach, accuracy in baseline_results.items():
        print(f"   {approach}: {accuracy:.2f}%")
    
    # Run current test
    results = test_conservative_enhanced_features()
    current_accuracy = results['test_accuracy'] * 100
    
    print(f"\nConservative Enhanced v3: {current_accuracy:.2f}%")
    
    # Calculate improvements
    print(f"\nComparison vs Previous:")
    for approach, prev_accuracy in baseline_results.items():
        improvement = current_accuracy - prev_accuracy
        print(f"   vs {approach}: {improvement:+.2f} percentage points")
    
    # Target achievement
    target_accuracy = 70.0
    target_gap = target_accuracy - current_accuracy
    print(f"\nTarget Achievement:")
    print(f"   Target: {target_accuracy:.1f}%")
    print(f"   Current: {current_accuracy:.2f}%")
    print(f"   Gap to target: {target_gap:.2f} percentage points")
    
    if current_accuracy >= target_accuracy:
        print("   🎯 TARGET ACHIEVED! ✅")
    else:
        print(f"   📈 Need {target_gap:.2f}% improvement to reach target")
    
    return results

if __name__ == "__main__":
    try:
        results = performance_comparison()
        print(f"\n✅ Conservative enhanced features v3 testing completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
