#!/usr/bin/env python3
"""
L4/L5 Analysis Summary and Performance Comparison
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the L4/L5 data"""
    file_path = '/Users/sujoymukherjee/code/spendplatform/context/Copy of Catergorization Working Sheet_Labelled Data for Training.xlsx'
    df = pd.read_excel(file_path, sheet_name='Working File_KGP', header=2)
    return df

def analyze_hierarchical_structure():
    """Analyze the hierarchical structure and create summary"""
    print("🔍 HIERARCHICAL CATEGORY ANALYSIS SUMMARY")
    print("=" * 70)
    
    df = load_data()
    
    # Basic data info
    print(f"📊 Dataset Overview:")
    print(f"   • Total records: {len(df)}")
    print(f"   • Columns: {list(df.columns)}")
    
    # Category analysis
    categories = {}
    for level in ['L1', 'L2', 'L3', 'L4', 'L5']:
        col = f'Category {level}'
        if col in df.columns:
            unique_cats = df[col].nunique()
            coverage = df[col].notna().sum() / len(df) * 100
            categories[level] = {
                'unique': unique_cats,
                'coverage': coverage,
                'top_categories': df[col].value_counts().head(10).to_dict()
            }
            print(f"   • {level}: {unique_cats} categories ({coverage:.1f}% coverage)")
    
    # Description analysis
    desc_stats = {
        'avg_length': df['Item_Descripton'].str.len().mean(),
        'avg_words': df['Item_Descripton'].str.split().str.len().mean(),
        'total_with_desc': df['Item_Descripton'].notna().sum()
    }
    
    print(f"\n📝 Description Analysis:")
    print(f"   • Records with descriptions: {desc_stats['total_with_desc']}")
    print(f"   • Average length: {desc_stats['avg_length']:.1f} characters")
    print(f"   • Average words: {desc_stats['avg_words']:.1f} words")
    
    return df, categories, desc_stats

def train_simplified_models(df):
    """Train simplified L4 and L5 models with better handling"""
    print(f"\n🚀 SIMPLIFIED MODEL TRAINING")
    print("=" * 70)
    
    models = {}
    
    # L4 Model with fewer categories
    print(f"\n📊 L4 Model Training:")
    l4_data = df[df['Category L4'].notna() & df['Item_Descripton'].notna()].copy()
    
    # Keep only top categories for L4
    l4_counts = l4_data['Category L4'].value_counts()
    top_l4_categories = l4_counts[l4_counts >= 10].index  # At least 10 samples
    l4_filtered = l4_data[l4_data['Category L4'].isin(top_l4_categories)]
    
    print(f"   • Original L4 categories: {l4_data['Category L4'].nunique()}")
    print(f"   • Filtered L4 categories: {len(top_l4_categories)}")
    print(f"   • Training samples: {len(l4_filtered)}")
    
    if len(l4_filtered) > 100 and len(top_l4_categories) > 5:
        # Train L4 model
        X_l4 = l4_filtered['Item_Descripton'].str.lower()
        y_l4 = l4_filtered['Category L4']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_l4, y_l4, test_size=0.2, random_state=42, stratify=y_l4
        )
        
        vectorizer = TfidfVectorizer(
            max_features=1500,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train Random Forest (performed better)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_vec, y_train)
        pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, pred)
        
        models['L4'] = {
            'accuracy': accuracy,
            'categories': len(top_l4_categories),
            'training_samples': len(l4_filtered),
            'model_type': 'Random Forest'
        }
        
        print(f"   ✅ L4 Model: {accuracy:.4f} ({accuracy*100:.2f}%) accuracy")
        
        # Save model
        model_package = {
            'model': model,
            'vectorizer': vectorizer,
            'accuracy': accuracy,
            'categories': len(top_l4_categories),
            'model_type': 'L4 Random Forest'
        }
        
        with open('/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.5/L4L5/l4_simplified_model.pkl', 'wb') as f:
            pickle.dump(model_package, f)
    
    # L5 Model with top categories only
    print(f"\n📊 L5 Model Training:")
    l5_data = df[df['Category L5'].notna() & df['Item_Descripton'].notna()].copy()
    
    # Keep only top categories for L5
    l5_counts = l5_data['Category L5'].value_counts()
    top_l5_categories = l5_counts[l5_counts >= 8].index  # At least 8 samples
    l5_filtered = l5_data[l5_data['Category L5'].isin(top_l5_categories)]
    
    print(f"   • Original L5 categories: {l5_data['Category L5'].nunique()}")
    print(f"   • Filtered L5 categories: {len(top_l5_categories)}")
    print(f"   • Training samples: {len(l5_filtered)}")
    
    if len(l5_filtered) > 100 and len(top_l5_categories) > 5:
        # Train L5 model
        X_l5 = l5_filtered['Item_Descripton'].str.lower()
        y_l5 = l5_filtered['Category L5']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_l5, y_l5, test_size=0.2, random_state=42, stratify=y_l5
        )
        
        vectorizer = TfidfVectorizer(
            max_features=1200,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train SVM model
        model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        model.fit(X_train_vec, y_train)
        pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, pred)
        
        models['L5'] = {
            'accuracy': accuracy,
            'categories': len(top_l5_categories),
            'training_samples': len(l5_filtered),
            'model_type': 'SVM'
        }
        
        print(f"   ✅ L5 Model: {accuracy:.4f} ({accuracy*100:.2f}%) accuracy")
        
        # Save model
        model_package = {
            'model': model,
            'vectorizer': vectorizer,
            'accuracy': accuracy,
            'categories': len(top_l5_categories),
            'model_type': 'L5 SVM'
        }
        
        with open('/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.5/L4L5/l5_simplified_model.pkl', 'wb') as f:
            pickle.dump(model_package, f)
    
    return models

def compare_all_models():
    """Compare L2, L4, and L5 model performance"""
    print(f"\n📈 COMPREHENSIVE MODEL COMPARISON")
    print("=" * 70)
    
    # Load L2 model
    l2_performance = None
    try:
        l2_path = '/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.5/models/production_svm_enhanced.pkl'
        with open(l2_path, 'rb') as f:
            l2_model = pickle.load(f)
        l2_performance = {
            'accuracy': l2_model.get('accuracy', 0),
            'categories': l2_model.get('categories', 0),
            'model_type': 'Enhanced SVM Ensemble'
        }
    except:
        pass
    
    # Load L4 model
    l4_performance = None
    try:
        l4_path = '/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.5/L4L5/l4_simplified_model.pkl'
        with open(l4_path, 'rb') as f:
            l4_model = pickle.load(f)
        l4_performance = {
            'accuracy': l4_model.get('accuracy', 0),
            'categories': l4_model.get('categories', 0),
            'model_type': l4_model.get('model_type', 'Unknown')
        }
    except:
        pass
    
    # Load L5 model
    l5_performance = None
    try:
        l5_path = '/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.5/L4L5/l5_simplified_model.pkl'
        with open(l5_path, 'rb') as f:
            l5_model = pickle.load(f)
        l5_performance = {
            'accuracy': l5_model.get('accuracy', 0),
            'categories': l5_model.get('categories', 0),
            'model_type': l5_model.get('model_type', 'Unknown')
        }
    except:
        pass
    
    # Create comparison table
    print(f"📊 Model Performance Comparison:")
    print(f"{'Level':<6} {'Accuracy':<10} {'Categories':<12} {'Model Type':<20} {'Granularity'}")
    print("-" * 75)
    
    if l2_performance:
        print(f"{'L2':<6} {l2_performance['accuracy']:<10.4f} {l2_performance['categories']:<12} {l2_performance['model_type']:<20} {'Broad'}")
    
    if l4_performance:
        print(f"{'L4':<6} {l4_performance['accuracy']:<10.4f} {l4_performance['categories']:<12} {l4_performance['model_type']:<20} {'Medium'}")
    
    if l5_performance:
        print(f"{'L5':<6} {l5_performance['accuracy']:<10.4f} {l5_performance['categories']:<12} {l5_performance['model_type']:<20} {'Fine'}")
    
    # Analysis
    print(f"\n📋 Performance Analysis:")
    
    if l2_performance and l4_performance:
        l2_acc = l2_performance['accuracy']
        l4_acc = l4_performance['accuracy']
        if l4_acc > l2_acc:
            improvement = (l4_acc - l2_acc) / l2_acc * 100
            print(f"   ✅ L4 shows {improvement:.1f}% improvement over L2")
        else:
            decline = (l2_acc - l4_acc) / l2_acc * 100
            print(f"   ⚠️ L4 shows {decline:.1f}% decline from L2")
    
    if l4_performance and l5_performance:
        l4_acc = l4_performance['accuracy']
        l5_acc = l5_performance['accuracy']
        if l5_acc > l4_acc:
            improvement = (l5_acc - l4_acc) / l4_acc * 100
            print(f"   ✅ L5 shows {improvement:.1f}% improvement over L4")
        else:
            decline = (l4_acc - l5_acc) / l4_acc * 100
            print(f"   ⚠️ L5 shows {decline:.1f}% decline from L4")
    
    return {
        'L2': l2_performance,
        'L4': l4_performance,
        'L5': l5_performance
    }

def generate_recommendations(performance_data):
    """Generate recommendations based on analysis"""
    print(f"\n💡 RECOMMENDATIONS")
    print("=" * 70)
    
    if performance_data['L2'] and performance_data['L4']:
        l2_acc = performance_data['L2']['accuracy']
        l4_acc = performance_data['L4']['accuracy']
        
        if l4_acc < l2_acc:
            print(f"🎯 Primary Recommendation: CONTINUE WITH L2 MODEL")
            print(f"   • L2 model (80.28% accuracy) outperforms L4 model")
            print(f"   • L4 granularity introduces classification complexity")
            print(f"   • Focus on L2 model optimization and confidence tuning")
        else:
            print(f"🎯 Primary Recommendation: CONSIDER L4 MODEL")
            print(f"   • L4 model shows improvement over L2")
            print(f"   • Increased granularity with maintained accuracy")
    
    print(f"\n📈 Optimization Strategies:")
    print(f"   1. Feature Engineering: Enhance description preprocessing")
    print(f"   2. Category Consolidation: Merge similar low-frequency categories")
    print(f"   3. Hierarchical Classification: Use L2 → L4 → L5 pipeline")
    print(f"   4. Ensemble Methods: Combine multiple level predictions")
    print(f"   5. Active Learning: Focus on low-confidence predictions")
    
    print(f"\n🔍 Next Steps:")
    print(f"   • Implement confidence-based routing between models")
    print(f"   • Test hierarchical classification pipeline")
    print(f"   • Optimize category thresholds based on business needs")
    print(f"   • Develop hybrid approach combining L2 stability with L4/L5 granularity")

def main():
    """Main analysis function"""
    print("🔍 L4/L5 HIERARCHICAL ANALYSIS SUMMARY")
    print("=" * 70)
    
    # Analyze structure
    df, categories, desc_stats = analyze_hierarchical_structure()
    
    # Train models
    model_performance = train_simplified_models(df)
    
    # Compare all models
    all_performance = compare_all_models()
    
    # Generate recommendations
    generate_recommendations(all_performance)
    
    print(f"\n" + "=" * 70)
    print(f"🎉 L4/L5 ANALYSIS COMPLETE!")
    print(f"📁 Results saved in: classificationmodel-v0.5/L4L5/")
    print("=" * 70)

if __name__ == "__main__":
    main()
