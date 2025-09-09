#!/usr/bin/env python3
"""
L4/L5 Category Analysis and Model Development
Analyze correlations between item descriptions and L4/L5 categories
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_and_examine_data():
    """Load and examine the categorization data"""
    print("🔍 LOADING AND EXAMINING L4/L5 DATA")
    print("=" * 60)
    
    file_path = '/Users/sujoymukherjee/code/spendplatform/context/Copy of Catergorization Working Sheet_Labelled Data for Training.xlsx'
    
    try:
        # Load the data with correct parameters from examination
        df = pd.read_excel(file_path, sheet_name='Working File_KGP', header=2)
        
        print(f"✅ Data loaded successfully!")
        print(f"📊 Dataset info:")
        print(f"   • Shape: {df.shape}")
        print(f"   • Columns: {list(df.columns)}")
        
        # Show first few rows
        print(f"\n📋 First 5 rows:")
        print(df[['Item_Descripton', 'Category L4', 'Category L5']].head())
        
        # Set L4 and L5 columns
        l4_columns = ['Category L4']
        l5_columns = ['Category L5']
        
        print(f"\n🏷️ L4 columns found: {l4_columns}")
        print(f"🏷️ L5 columns found: {l5_columns}")
        
        # Check data availability
        for col in ['Category L4', 'Category L5']:
            if col in df.columns:
                non_null_count = df[col].notna().sum()
                print(f"   {col}: {non_null_count}/{len(df)} ({non_null_count/len(df)*100:.1f}%) non-null")
        
        return df, l4_columns, l5_columns
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, [], []

def analyze_l4_l5_structure(df, l4_columns, l5_columns):
    """Analyze the structure and distribution of L4/L5 categories"""
    print(f"\n📊 L4/L5 CATEGORY STRUCTURE ANALYSIS")
    print("=" * 60)
    
    # Use the known column names
    main_l4_col = 'Category L4'
    main_l5_col = 'Category L5'
    
    print(f"🎯 Using columns:")
    print(f"   • L4: {main_l4_col}")
    print(f"   • L5: {main_l5_col}")
    
    results = {}
    
    # Analyze L4 categories
    if main_l4_col in df.columns:
        l4_data = df[df[main_l4_col].notna()]
        l4_counts = l4_data[main_l4_col].value_counts()
        
        print(f"\n📈 L4 Category Analysis:")
        print(f"   • Total L4 categories: {len(l4_counts)}")
        print(f"   • Records with L4: {len(l4_data)}")
        print(f"   • Coverage: {len(l4_data)/len(df)*100:.1f}%")
        
        print(f"\n🏆 Top 15 L4 Categories:")
        for i, (category, count) in enumerate(l4_counts.head(15).items(), 1):
            print(f"   {i:2d}. {category}: {count}")
        
        results['l4_data'] = l4_data
        results['l4_counts'] = l4_counts
        results['l4_col'] = main_l4_col
    
    # Analyze L5 categories
    if main_l5_col in df.columns:
        l5_data = df[df[main_l5_col].notna()]
        l5_counts = l5_data[main_l5_col].value_counts()
        
        print(f"\n📈 L5 Category Analysis:")
        print(f"   • Total L5 categories: {len(l5_counts)}")
        print(f"   • Records with L5: {len(l5_data)}")
        print(f"   • Coverage: {len(l5_data)/len(df)*100:.1f}%")
        
        print(f"\n🏆 Top 15 L5 Categories:")
        for i, (category, count) in enumerate(l5_counts.head(15).items(), 1):
            print(f"   {i:2d}. {category}: {count}")
        
        results['l5_data'] = l5_data
        results['l5_counts'] = l5_counts
        results['l5_col'] = main_l5_col
    
    return results

def analyze_description_patterns(df, category_col, level_name):
    """Analyze description patterns for specific category level"""
    print(f"\n🔤 DESCRIPTION PATTERN ANALYSIS - {level_name}")
    print("=" * 60)
    
    # Get data with descriptions and categories
    data = df[df[category_col].notna() & df['Item_Descripton'].notna()].copy()
    
    print(f"📊 Analysis dataset:")
    print(f"   • Records: {len(data)}")
    print(f"   • Categories: {data[category_col].nunique()}")
    
    # Analyze description lengths by category
    data['desc_length'] = data['Item_Descripton'].str.len()
    data['word_count'] = data['Item_Descripton'].str.split().str.len()
    
    print(f"\n📝 Description Statistics:")
    print(f"   • Avg length: {data['desc_length'].mean():.1f} chars")
    print(f"   • Avg words: {data['word_count'].mean():.1f} words")
    
    # Category-wise description analysis
    category_stats = data.groupby(category_col).agg({
        'desc_length': ['mean', 'std'],
        'word_count': ['mean', 'std'],
        'Item_Descripton': 'count'
    }).round(2)
    
    category_stats.columns = ['avg_length', 'std_length', 'avg_words', 'std_words', 'count']
    category_stats = category_stats.sort_values('count', ascending=False)
    
    print(f"\n📊 Top Categories by Description Patterns:")
    for i, (category, stats) in enumerate(category_stats.head(10).iterrows(), 1):
        print(f"   {i:2d}. {category} ({stats['count']} items)")
        print(f"       Avg length: {stats['avg_length']:.1f} ± {stats['std_length']:.1f}")
        print(f"       Avg words: {stats['avg_words']:.1f} ± {stats['std_words']:.1f}")
    
    return data, category_stats

def find_keyword_correlations(df, category_col, level_name):
    """Find keyword correlations for categories"""
    print(f"\n🔍 KEYWORD CORRELATION ANALYSIS - {level_name}")
    print("=" * 60)
    
    data = df[df[category_col].notna() & df['Item_Descripton'].notna()].copy()
    
    # Extract keywords for top categories
    top_categories = data[category_col].value_counts().head(10).index
    
    keyword_analysis = {}
    
    for category in top_categories:
        cat_descriptions = data[data[category_col] == category]['Item_Descripton'].str.lower()
        
        # Simple keyword extraction (most common words)
        all_words = ' '.join(cat_descriptions).split()
        word_freq = pd.Series(all_words).value_counts()
        
        # Filter out common words
        stop_words = {'and', 'or', 'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
        meaningful_words = word_freq[~word_freq.index.isin(stop_words)]
        
        keyword_analysis[category] = meaningful_words.head(10)
        
        print(f"\n🔤 Keywords for '{category}':")
        for word, freq in meaningful_words.head(10).items():
            print(f"   {word}: {freq}")
    
    return keyword_analysis

def create_l4_model(data, category_col):
    """Create and train L4 classification model"""
    print(f"\n🚀 TRAINING L4 CLASSIFICATION MODEL")
    print("=" * 60)
    
    # Prepare data
    clean_data = data[data[category_col].notna() & data['Item_Descripton'].notna()].copy()
    
    # Filter categories with sufficient samples
    category_counts = clean_data[category_col].value_counts()
    min_samples = 5
    valid_categories = category_counts[category_counts >= min_samples].index
    clean_data = clean_data[clean_data[category_col].isin(valid_categories)]
    
    print(f"📊 Training data:")
    print(f"   • Records: {len(clean_data)}")
    print(f"   • Categories: {clean_data[category_col].nunique()}")
    print(f"   • Min samples per category: {min_samples}")
    
    # Split data
    X = clean_data['Item_Descripton'].str.lower()
    y = clean_data[category_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create features
    vectorizer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2,
        max_df=0.95
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    print(f"🔤 Features: {X_train_vec.shape[1]}")
    print(f"🏷️ Categories: {len(label_encoder.classes_)}")
    
    # Train models
    models = {}
    
    # SVM Model
    print(f"\n🎯 Training SVM model...")
    svm_model = SVC(kernel='rbf', C=2.0, gamma='scale', random_state=42)
    svm_model.fit(X_train_vec, y_train_encoded)
    svm_pred = svm_model.predict(X_test_vec)
    svm_accuracy = accuracy_score(y_test_encoded, svm_pred)
    
    models['svm'] = {
        'model': svm_model,
        'accuracy': svm_accuracy,
        'predictions': svm_pred
    }
    
    # Random Forest Model
    print(f"🌲 Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_vec, y_train_encoded)
    rf_pred = rf_model.predict(X_test_vec)
    rf_accuracy = accuracy_score(y_test_encoded, rf_pred)
    
    models['rf'] = {
        'model': rf_model,
        'accuracy': rf_accuracy,
        'predictions': rf_pred
    }
    
    print(f"\n📊 Model Performance:")
    print(f"   • SVM Accuracy: {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")
    print(f"   • Random Forest Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
    
    # Choose best model
    best_model_name = 'svm' if svm_accuracy > rf_accuracy else 'rf'
    best_model = models[best_model_name]
    
    print(f"\n🏆 Best Model: {best_model_name.upper()}")
    
    # Detailed evaluation
    best_pred = best_model['predictions']
    report = classification_report(
        y_test_encoded, 
        best_pred, 
        target_names=label_encoder.classes_,
        output_dict=True
    )
    
    print(f"\n📋 Detailed Performance:")
    accuracy = report.get('accuracy', 0) if isinstance(report, dict) else 0
    macro_avg = report.get('macro avg', {}) if isinstance(report, dict) else {}
    weighted_avg = report.get('weighted avg', {}) if isinstance(report, dict) else {}
    
    print(f"   • Accuracy: {accuracy:.4f}")
    print(f"   • Macro Avg F1: {macro_avg.get('f1-score', 0):.4f}")
    print(f"   • Weighted Avg F1: {weighted_avg.get('f1-score', 0):.4f}")
    
    # Save model
    model_package = {
        'model': best_model['model'],
        'vectorizer': vectorizer,
        'label_encoder': label_encoder,
        'accuracy': best_model['accuracy'],
        'categories': len(label_encoder.classes_),
        'model_type': f'L4 {best_model_name.upper()}',
        'training_samples': len(clean_data)
    }
    
    model_path = '/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.5/L4L5/l4_classification_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"💾 Model saved: {model_path}")
    
    return model_package, report

def create_l5_model(data, category_col):
    """Create and train L5 classification model"""
    print(f"\n🚀 TRAINING L5 CLASSIFICATION MODEL")
    print("=" * 60)
    
    # Prepare data
    clean_data = data[data[category_col].notna() & data['Item_Descripton'].notna()].copy()
    
    # Filter categories with sufficient samples
    category_counts = clean_data[category_col].value_counts()
    min_samples = 3  # Lower threshold for L5 as it's more granular
    valid_categories = category_counts[category_counts >= min_samples].index
    clean_data = clean_data[clean_data[category_col].isin(valid_categories)]
    
    print(f"📊 Training data:")
    print(f"   • Records: {len(clean_data)}")
    print(f"   • Categories: {clean_data[category_col].nunique()}")
    print(f"   • Min samples per category: {min_samples}")
    
    if len(clean_data) < 50:
        print(f"⚠️ Insufficient data for L5 model training")
        return None, None
    
    # Split data
    X = clean_data['Item_Descripton'].str.lower()
    y = clean_data[category_col]
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        print(f"⚠️ Cannot stratify L5 data - using random split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    # Create features
    vectorizer = TfidfVectorizer(
        max_features=1500,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=1,
        max_df=0.95
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    print(f"🔤 Features: {X_train_vec.shape[1]}")
    print(f"🏷️ Categories: {len(label_encoder.classes_)}")
    
    # Train SVM model (simpler for L5)
    print(f"\n🎯 Training SVM model...")
    model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    model.fit(X_train_vec, y_train_encoded)
    pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test_encoded, pred)
    
    print(f"\n📊 Model Performance:")
    print(f"   • SVM Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Detailed evaluation
    report = classification_report(
        y_test_encoded, 
        pred, 
        target_names=label_encoder.classes_,
        output_dict=True
    )
    
    print(f"\n📋 Detailed Performance:")
    accuracy = report.get('accuracy', 0) if isinstance(report, dict) else 0
    macro_avg = report.get('macro avg', {}) if isinstance(report, dict) else {}
    weighted_avg = report.get('weighted avg', {}) if isinstance(report, dict) else {}
    
    print(f"   • Accuracy: {accuracy:.4f}")
    print(f"   • Macro Avg F1: {macro_avg.get('f1-score', 0):.4f}")
    print(f"   • Weighted Avg F1: {weighted_avg.get('f1-score', 0):.4f}")
    
    # Save model
    model_package = {
        'model': model,
        'vectorizer': vectorizer,
        'label_encoder': label_encoder,
        'accuracy': accuracy,
        'categories': len(label_encoder.classes_),
        'model_type': 'L5 SVM',
        'training_samples': len(clean_data)
    }
    
    model_path = '/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.5/L4L5/l5_classification_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"💾 Model saved: {model_path}")
    
    return model_package, report

def compare_with_l2_performance():
    """Compare L4/L5 performance with L2 model"""
    print(f"\n📈 PERFORMANCE COMPARISON WITH L2 MODEL")
    print("=" * 60)
    
    # Load L2 model for comparison
    try:
        l2_model_path = '/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.5/models/production_svm_enhanced.pkl'
        with open(l2_model_path, 'rb') as f:
            l2_model = pickle.load(f)
        
        print(f"✅ L2 Model loaded:")
        print(f"   • Accuracy: {l2_model.get('accuracy', 0):.4f}")
        print(f"   • Categories: {l2_model.get('categories', 0)}")
        
    except Exception as e:
        print(f"⚠️ Could not load L2 model: {e}")
        l2_model = None
    
    # Load L4/L5 models
    models_performance = {}
    
    try:
        l4_model_path = '/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.5/L4L5/l4_classification_model.pkl'
        with open(l4_model_path, 'rb') as f:
            l4_model = pickle.load(f)
        models_performance['L4'] = l4_model
    except:
        l4_model = None
    
    try:
        l5_model_path = '/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.5/L4L5/l5_classification_model.pkl'
        with open(l5_model_path, 'rb') as f:
            l5_model = pickle.load(f)
        models_performance['L5'] = l5_model
    except:
        l5_model = None
    
    print(f"\n📊 Model Comparison Summary:")
    print(f"{'Level':<6} {'Accuracy':<10} {'Categories':<12} {'Samples':<10} {'Granularity'}")
    print("-" * 60)
    
    if l2_model:
        print(f"{'L2':<6} {l2_model.get('accuracy', 0):<10.4f} {l2_model.get('categories', 0):<12} {'-':<10} {'Broad'}")
    
    for level, model in models_performance.items():
        if model:
            granularity = 'Fine' if level == 'L5' else 'Medium'
            print(f"{level:<6} {model.get('accuracy', 0):<10.4f} {model.get('categories', 0):<12} {model.get('training_samples', 0):<10} {granularity}")
    
    return models_performance

def main():
    """Main analysis function"""
    print("🔍 L4/L5 CATEGORY ANALYSIS AND MODEL DEVELOPMENT")
    print("=" * 70)
    
    # Load and examine data
    df, l4_columns, l5_columns = load_and_examine_data()
    if df is None:
        return
    
    # Analyze structure
    structure_results = analyze_l4_l5_structure(df, l4_columns, l5_columns)
    
    # Train models if data is available
    l4_model = None
    l5_model = None
    
    if 'l4_data' in structure_results and len(structure_results['l4_data']) > 50:
        l4_data, l4_stats = analyze_description_patterns(
            structure_results['l4_data'], 
            structure_results['l4_col'], 
            'L4'
        )
        l4_keywords = find_keyword_correlations(
            structure_results['l4_data'], 
            structure_results['l4_col'], 
            'L4'
        )
        l4_model, l4_report = create_l4_model(structure_results['l4_data'], structure_results['l4_col'])
    
    if 'l5_data' in structure_results and len(structure_results['l5_data']) > 30:
        l5_data, l5_stats = analyze_description_patterns(
            structure_results['l5_data'], 
            structure_results['l5_col'], 
            'L5'
        )
        l5_keywords = find_keyword_correlations(
            structure_results['l5_data'], 
            structure_results['l5_col'], 
            'L5'
        )
        l5_model, l5_report = create_l5_model(structure_results['l5_data'], structure_results['l5_col'])
    
    # Compare performance
    performance_comparison = compare_with_l2_performance()
    
    print(f"\n" + "=" * 70)
    print(f"🎉 L4/L5 ANALYSIS COMPLETE!")
    if l4_model:
        print(f"✅ L4 Model: {l4_model['accuracy']*100:.2f}% accuracy ({l4_model['categories']} categories)")
    if l5_model:
        print(f"✅ L5 Model: {l5_model['accuracy']*100:.2f}% accuracy ({l5_model['categories']} categories)")
    print(f"📁 Models saved in: /L4L5/ directory")
    print("=" * 70)

if __name__ == "__main__":
    main()
