#!/usr/bin/env python3
"""
Performance Analysis and Model Optimization
Understanding the results and improving the enhanced model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def analyze_results():
    """Analyze the training results and understand the performance patterns"""
    print("🔍 ENHANCED SVM PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    print("""
📊 KEY FINDINGS FROM TRAINING:

🎯 PERFORMANCE COMPARISON:
   • Baseline Model: 69.04% accuracy (10 categories)
   • Enhanced Model: 66.11% accuracy (40 categories)
   • Performance Drop: -2.93 percentage points

📈 SCALE EXPANSION:
   • Data Volume: 9.4x increase (2,131 → 19,971 samples)
   • Categories: 4.0x increase (10 → 40 categories)
   • Training Time: 33x increase (0.19s → 6.15s)

🎭 CATEGORY PERFORMANCE PATTERNS:
   • Top Enhanced Categories: >90% F1-score
   • Original Categories: Generally maintained performance
   • New Categories: Mixed performance (0-70% F1-score)

🤔 ANALYSIS INSIGHTS:
""")
    
    # Load data for analysis
    orig_path = '/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.5/preprocessed_data/top_10_l2_categories_data.xlsx'
    combined_path = '/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.5/syntheticdata/combined_spend_data.xlsx'
    
    original_df = pd.read_excel(orig_path)
    combined_df = pd.read_excel(combined_path, sheet_name='Combined_Data')
    
    print(f"✅ Original categories: {sorted(original_df['Category L2'].unique())}")
    print(f"✅ Enhanced categories: {len(combined_df['Category L2'].unique())} total")
    
    # Identify new categories
    original_cats = set(original_df['Category L2'].unique())
    all_cats = set(combined_df['Category L2'].unique())
    new_cats = all_cats - original_cats
    
    print(f"\n🆕 NEW CATEGORIES ADDED ({len(new_cats)}):")
    for i, cat in enumerate(sorted(new_cats), 1):
        count = len(combined_df[combined_df['Category L2'] == cat])
        print(f"   {i:2d}. {cat} ({count} samples)")
    
    print(f"\n🎯 PERFORMANCE ANALYSIS:")
    
    # Category-wise analysis
    category_stats = combined_df['Category L2'].value_counts()
    
    print(f"\n📊 CATEGORY DISTRIBUTION:")
    print(f"   • Categories with >1000 samples: {sum(category_stats > 1000)}")
    print(f"   • Categories with 500-1000 samples: {sum((category_stats >= 500) & (category_stats <= 1000))}")
    print(f"   • Categories with 200-500 samples: {sum((category_stats >= 200) & (category_stats < 500))}")
    print(f"   • Categories with <200 samples: {sum(category_stats < 200)}")
    
    # Text analysis
    print(f"\n📝 TEXT COMPLEXITY ANALYSIS:")
    combined_df['desc_length'] = combined_df['Item_Descripton'].str.len()
    combined_df['word_count'] = combined_df['Item_Descripton'].str.split().str.len()
    
    print(f"   • Average description length: {combined_df['desc_length'].mean():.1f} chars")
    print(f"   • Average word count: {combined_df['word_count'].mean():.1f} words")
    print(f"   • Unique descriptions: {combined_df['Item_Descripton'].nunique():,}")
    
    # Check for potential overfitting
    print(f"\n🎯 POTENTIAL ISSUES IDENTIFIED:")
    print(f"   1. Category Imbalance: Large variance in sample sizes")
    print(f"   2. Complexity Increase: 4x more categories to distinguish")
    print(f"   3. New Categories: Synthetic data may lack real-world complexity")
    print(f"   4. Feature Dilution: Same 1000 features across 4x categories")

def optimize_enhanced_model():
    """Try optimizations for the enhanced model"""
    print(f"\n🚀 MODEL OPTIMIZATION STRATEGIES")
    print("=" * 70)
    
    # Load combined dataset
    combined_path = '/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.5/syntheticdata/combined_spend_data.xlsx'
    combined_df = pd.read_excel(combined_path, sheet_name='Combined_Data')
    
    # Clean data
    clean_df = combined_df[['Item_Descripton', 'Category L2']].copy()
    clean_df = clean_df.dropna()
    clean_df['Item_Descripton'] = clean_df['Item_Descripton'].astype(str).str.lower().str.strip()
    
    # Filter categories with sufficient samples
    category_counts = clean_df['Category L2'].value_counts()
    valid_categories = category_counts[category_counts >= 50].index  # More lenient threshold
    clean_df = clean_df[clean_df['Category L2'].isin(valid_categories)]
    
    print(f"📊 Optimization Dataset:")
    print(f"   • Records: {len(clean_df):,}")
    print(f"   • Categories: {clean_df['Category L2'].nunique()}")
    
    # Split data
    train_df, test_df = train_test_split(clean_df, test_size=0.2, random_state=42, stratify=clean_df['Category L2'])
    
    print(f"\n🔧 OPTIMIZATION 1: INCREASED FEATURES")
    print("=" * 50)
    
    # Try with more features
    vectorizer_opt1 = TfidfVectorizer(
        max_features=2000,  # Double the features
        ngram_range=(1, 2), # Add bigrams
        stop_words='english',
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    
    X_train_opt1 = vectorizer_opt1.fit_transform(train_df['Item_Descripton'])
    X_test_opt1 = vectorizer_opt1.transform(test_df['Item_Descripton'])
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df['Category L2'])
    y_test = label_encoder.transform(test_df['Category L2'])
    
    # Train optimized model
    model_opt1 = SVC(kernel='rbf', C=2.0, gamma='scale', random_state=42)
    model_opt1.fit(X_train_opt1, y_train)
    
    accuracy_opt1 = model_opt1.score(X_test_opt1, y_test)
    print(f"✅ Optimization 1 Results:")
    print(f"   • Features: 2000 (unigrams + bigrams)")
    print(f"   • Accuracy: {accuracy_opt1:.4f} ({accuracy_opt1*100:.2f}%)")
    
    print(f"\n🔧 OPTIMIZATION 2: HYPERPARAMETER TUNING")
    print("=" * 50)
    
    # Quick hyperparameter search
    param_grid = {
        'C': [1.0, 2.0, 5.0],
        'gamma': ['scale', 'auto', 0.001, 0.01]
    }
    
    # Use smaller sample for speed
    sample_size = min(5000, len(train_df))
    train_sample = train_df.sample(n=sample_size, random_state=42)
    
    X_train_sample = vectorizer_opt1.transform(train_sample['Item_Descripton'])
    y_train_sample = label_encoder.transform(train_sample['Category L2'])
    
    grid_search = GridSearchCV(
        SVC(kernel='rbf', random_state=42),
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X_train_sample, y_train_sample)
    
    print(f"✅ Best parameters: {grid_search.best_params_}")
    print(f"✅ Best CV score: {grid_search.best_score_:.4f}")
    
    # Train final optimized model
    best_model = SVC(
        kernel='rbf',
        C=grid_search.best_params_['C'],
        gamma=grid_search.best_params_['gamma'],
        random_state=42
    )
    
    best_model.fit(X_train_opt1, y_train)
    accuracy_opt2 = best_model.score(X_test_opt1, y_test)
    
    print(f"✅ Optimization 2 Results:")
    print(f"   • Best Parameters: C={grid_search.best_params_['C']}, gamma={grid_search.best_params_['gamma']}")
    print(f"   • Accuracy: {accuracy_opt2:.4f} ({accuracy_opt2*100:.2f}%)")
    
    print(f"\n🔧 OPTIMIZATION 3: BALANCED SAMPLING")
    print("=" * 50)
    
    # Create balanced dataset
    min_samples_per_category = 200
    balanced_dfs = []
    
    for category in clean_df['Category L2'].unique():
        cat_data = clean_df[clean_df['Category L2'] == category]
        if len(cat_data) >= min_samples_per_category:
            # Take exactly min_samples_per_category samples
            if len(cat_data) > min_samples_per_category:
                cat_sample = cat_data.sample(n=min_samples_per_category, random_state=42)
            else:
                cat_sample = cat_data
            balanced_dfs.append(cat_sample)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    print(f"✅ Balanced Dataset:")
    print(f"   • Records: {len(balanced_df):,}")
    print(f"   • Categories: {balanced_df['Category L2'].nunique()}")
    print(f"   • Samples per category: ~{min_samples_per_category}")
    
    # Train on balanced data
    train_bal, test_bal = train_test_split(balanced_df, test_size=0.2, random_state=42, stratify=balanced_df['Category L2'])
    
    X_train_bal = vectorizer_opt1.transform(train_bal['Item_Descripton'])
    X_test_bal = vectorizer_opt1.transform(test_bal['Item_Descripton'])
    
    y_train_bal = label_encoder.fit_transform(train_bal['Category L2'])
    y_test_bal = label_encoder.transform(test_bal['Category L2'])
    
    model_bal = SVC(
        kernel='rbf',
        C=grid_search.best_params_['C'],
        gamma=grid_search.best_params_['gamma'],
        random_state=42
    )
    
    model_bal.fit(X_train_bal, y_train_bal)
    accuracy_bal = model_bal.score(X_test_bal, y_test_bal)
    
    print(f"✅ Optimization 3 Results:")
    print(f"   • Balanced sampling: {min_samples_per_category} per category")
    print(f"   • Accuracy: {accuracy_bal:.4f} ({accuracy_bal*100:.2f}%)")
    
    # Summary
    print(f"\n📈 OPTIMIZATION SUMMARY")
    print("=" * 50)
    print(f"   • Original Enhanced Model: 66.11%")
    print(f"   • Optimization 1 (More Features): {accuracy_opt1*100:.2f}%")
    print(f"   • Optimization 2 (Hyperparameters): {accuracy_opt2*100:.2f}%")
    print(f"   • Optimization 3 (Balanced Data): {accuracy_bal*100:.2f}%")
    
    # Determine best approach
    accuracies = {
        'More Features': accuracy_opt1,
        'Hyperparameters': accuracy_opt2,
        'Balanced Data': accuracy_bal
    }
    
    best_accuracy = max(accuracy_opt1, accuracy_opt2, accuracy_bal)
    best_approach = "Multiple optimizations"
    for name, acc in accuracies.items():
        if acc == best_accuracy:
            best_approach = name
            break
    
    print(f"\n🏆 BEST OPTIMIZATION: {best_approach}")
    print(f"   • Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"   • Improvement: {(best_accuracy - 0.6611)*100:.2f} percentage points")
    
    return best_model, vectorizer_opt1, label_encoder, best_accuracy

def recommendations():
    """Provide recommendations based on analysis"""
    print(f"\n💡 RECOMMENDATIONS & NEXT STEPS")
    print("=" * 70)
    
    print(f"""
🎯 IMMEDIATE IMPROVEMENTS:
   1. Feature Engineering:
      • Increase TF-IDF features to 2000+
      • Include bigrams for better context
      • Consider domain-specific feature extraction
   
   2. Data Balance:
      • Ensure minimum 200 samples per category
      • Consider data augmentation for underrepresented categories
      • Remove categories with insufficient samples
   
   3. Model Optimization:
      • Hyperparameter tuning with larger grid search
      • Consider ensemble methods
      • Experiment with other kernels (linear, polynomial)

📊 PERFORMANCE EXPECTATIONS:
   • Current Enhanced Model: 66.11% → Expected: 70-75%
   • Baseline Comparison: 69.04% (10 categories)
   • Target: Match baseline performance with 4x categories

🚀 ADVANCED STRATEGIES:
   1. Hierarchical Classification:
      • First classify into major groups
      • Then sub-classify within groups
   
   2. Multi-label Classification:
      • Consider L2, L3, L4 level predictions
      • Use category hierarchy information
   
   3. Deep Learning:
      • BERT-based text classification
      • Custom embeddings for domain terms

🎉 SUCCESS METRICS:
   • Enhanced model accuracy > 70%
   • Performance per category > baseline
   • Practical deployment readiness
""")

def main():
    """Main analysis function"""
    analyze_results()
    best_model, vectorizer, label_encoder, best_accuracy = optimize_enhanced_model()
    recommendations()
    
    print(f"\n" + "=" * 70)
    print(f"🎯 ANALYSIS COMPLETE!")
    print(f"✅ Current Performance: 66.11% → Optimized: {best_accuracy*100:.2f}%")
    print(f"📈 Enhancement Potential: {(best_accuracy - 0.6611)*100:.2f} percentage points")
    print(f"🚀 Ready for production optimization!")
    print("=" * 70)

if __name__ == "__main__":
    main()
