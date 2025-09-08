#!/usr/bin/env python3
"""
Analysis script to understand the performance issues with conservative enhanced features v3
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import sys
import os

# Add current directory to path for imports
sys.path.append('/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.2')

def load_original_training_data():
    """Load the dataset used in previous tests"""
    print("📁 Loading original training data...")
    
    # Try to find the original dataset used in previous tests
    possible_paths = [
        '/Users/sujoymukherjee/code/spendplatform/context/predicted_original_training_data.xlsx',
        '/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.2/pipeline_run_20250907_143517/preprocessed_data/train_data.xlsx',
        '/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.2/pipeline_run_20250907_143517/preprocessed_data/test_data.xlsx'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"   Found dataset: {path}")
            df = pd.read_excel(path)
            print(f"   Columns: {list(df.columns)}")
            print(f"   Shape: {df.shape}")
            
            # Check for category distribution
            category_cols = [col for col in df.columns if 'category' in col.lower() or 'l2' in col.lower()]
            for col in category_cols:
                print(f"   {col} unique values: {df[col].nunique()}")
                if df[col].nunique() > 0:
                    print(f"     Sample values: {df[col].value_counts().head()}")
            
            return df, path
    
    print("❌ No original training dataset found")
    return None, None

def test_baseline_tfidf():
    """Test basic TF-IDF approach on current dataset"""
    print("\n🧪 Testing Baseline TF-IDF on Current Dataset")
    print("=" * 50)
    
    # Load current dataset
    file_path = '/Users/sujoymukherjee/code/spendplatform/context/Copy of Catergorization Working Sheet_Labelled Data for Training.xlsx'
    df = pd.read_excel(file_path, header=1)
    df = df.drop(columns=[df.columns[0]])
    
    column_names = ['Item Code', 'Item_Descripton', 'Category L1', 'Category L2', 'Category L3', 'Category L4', 'Category L5']
    df.columns = column_names[:len(df.columns)]
    
    # Clean data
    df = df.dropna(subset=['Item_Descripton', 'Category L2']).copy()
    print(f"   ✅ Loaded {len(df)} records")
    
    # Filter categories with sufficient samples
    category_counts = df['Category L2'].value_counts()
    valid_categories = category_counts[category_counts >= 10].index
    df_filtered = df[df['Category L2'].isin(valid_categories)].copy()
    
    print(f"   ✅ Using {len(valid_categories)} categories with ≥10 samples")
    print(f"   ✅ Total records: {len(df_filtered)}")
    
    # Prepare text data
    df_filtered['processed_description'] = df_filtered['Item_Descripton'].astype(str).str.lower()
    df_filtered['processed_description'] = df_filtered['processed_description'].str.replace(r'[^\w\s]', ' ', regex=True)
    df_filtered['processed_description'] = df_filtered['processed_description'].str.replace(r'\s+', ' ', regex=True)
    df_filtered['processed_description'] = df_filtered['processed_description'].str.strip()
    
    # Split data
    X = df_filtered['processed_description']
    y = df_filtered['Category L2']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Test basic TF-IDF with existing parameters
    print("\n🔧 Testing Basic TF-IDF (existing parameters)...")
    vectorizer = TfidfVectorizer(
        max_features=1500,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"   TF-IDF features: {X_train_tfidf.shape[1]}")
    
    # Train classifier
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    classifier.fit(X_train_tfidf, y_train)
    
    # Evaluate
    y_train_pred = classifier.predict(X_train_tfidf)
    y_test_pred = classifier.predict(X_test_tfidf)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"   📊 Basic TF-IDF Results:")
    print(f"   Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    return test_accuracy

def analyze_data_quality():
    """Analyze data quality issues"""
    print("\n🔍 Analyzing Data Quality")
    print("=" * 50)
    
    file_path = '/Users/sujoymukherjee/code/spendplatform/context/Copy of Catergorization Working Sheet_Labelled Data for Training.xlsx'
    df = pd.read_excel(file_path, header=1)
    df = df.drop(columns=[df.columns[0]])
    
    column_names = ['Item Code', 'Item_Descripton', 'Category L1', 'Category L2', 'Category L3', 'Category L4', 'Category L5']
    df.columns = column_names[:len(df.columns)]
    
    df_clean = df.dropna(subset=['Item_Descripton', 'Category L2']).copy()
    
    print(f"📊 Dataset Analysis:")
    print(f"   Total records: {len(df_clean)}")
    print(f"   Unique L2 categories: {df_clean['Category L2'].nunique()}")
    
    # Category distribution analysis
    category_counts = df_clean['Category L2'].value_counts()
    print(f"   Categories with ≥10 samples: {len(category_counts[category_counts >= 10])}")
    print(f"   Categories with ≥20 samples: {len(category_counts[category_counts >= 20])}")
    print(f"   Categories with ≥50 samples: {len(category_counts[category_counts >= 50])}")
    
    # Text quality analysis
    descriptions = df_clean['Item_Descripton'].astype(str)
    avg_length = descriptions.str.len().mean()
    avg_words = descriptions.str.split().str.len().mean()
    
    print(f"   📝 Text Quality:")
    print(f"   Average description length: {avg_length:.1f} characters")
    print(f"   Average word count: {avg_words:.1f} words")
    
    # Check for very short or long descriptions
    short_desc = len(descriptions[descriptions.str.len() < 10])
    long_desc = len(descriptions[descriptions.str.len() > 100])
    
    print(f"   Very short descriptions (<10 chars): {short_desc} ({short_desc/len(descriptions)*100:.1f}%)")
    print(f"   Long descriptions (>100 chars): {long_desc} ({long_desc/len(descriptions)*100:.1f}%)")
    
    # Sample descriptions
    print(f"\n📝 Sample Descriptions:")
    for i, (desc, cat) in enumerate(zip(descriptions.head(5), df_clean['Category L2'].head(5))):
        print(f"   {i+1}. {desc[:60]}... → {cat}")
    
    return df_clean

def compare_with_original():
    """Compare with original dataset if available"""
    print("\n🔄 Comparing with Original Dataset")
    print("=" * 50)
    
    original_df, original_path = load_original_training_data()
    if original_df is None:
        print("❌ Cannot compare - original dataset not found")
        return
    
    current_df = analyze_data_quality()
    
    print(f"\n📊 Dataset Comparison:")
    print(f"   Original dataset: {len(original_df)} records")
    print(f"   Current dataset: {len(current_df)} records")
    
    # If original has L2 categories, compare them
    l2_cols = [col for col in original_df.columns if 'l2' in col.lower() or ('category' in col.lower() and '2' in col)]
    if l2_cols:
        original_l2_col = l2_cols[0]
        original_categories = set(original_df[original_l2_col].dropna().unique())
        current_categories = set(current_df['Category L2'].dropna().unique())
        
        print(f"   Original L2 categories: {len(original_categories)}")
        print(f"   Current L2 categories: {len(current_categories)}")
        print(f"   Categories overlap: {len(original_categories & current_categories)}")

if __name__ == "__main__":
    print("🔍 Performance Analysis for Conservative Enhanced Features v3")
    print("=" * 70)
    
    # Analyze data quality
    analyze_data_quality()
    
    # Test baseline performance
    baseline_accuracy = test_baseline_tfidf()
    
    # Compare with original if possible
    compare_with_original()
    
    print(f"\n📋 Summary:")
    print(f"   Baseline TF-IDF accuracy: {baseline_accuracy*100:.2f}%")
    print(f"   Conservative enhanced v3: 45.56%")
    print(f"   Expected existing approach: 62.67%")
    
    if baseline_accuracy * 100 < 50:
        print(f"\n⚠️  ISSUE IDENTIFIED: Dataset appears different from previous tests")
        print(f"   • Current dataset baseline: {baseline_accuracy*100:.2f}%")
        print(f"   • Expected baseline: ~62.67%")
        print(f"   • Recommendation: Use original training dataset or investigate data preprocessing")
    else:
        print(f"\n✅ Dataset appears consistent - investigating feature engineering issues")
