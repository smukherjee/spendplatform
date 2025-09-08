#!/usr/bin/env python3
"""
Comprehensive Model Comparison: Random Forest vs CatBoost vs SVM vs Logistic Regression
Tests all available models with the same dataset and configuration
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from catboost import CatBoostClassifier
import warnings
import time
import pickle
from pathlib import Path

warnings.filterwarnings('ignore')

def load_data():
    """Load the preprocessed data"""
    print("Loading preprocessed data...")
    
    train_file = 'preprocessed_data/train_data.xlsx'
    test_file = 'preprocessed_data/test_data.xlsx'
    
    train_data = pd.read_excel(train_file)
    test_data = pd.read_excel(test_file)
    
    # Clean column names
    train_data.columns = train_data.columns.str.strip()
    test_data.columns = test_data.columns.str.strip()
    
    # Fill missing values
    train_data['Item_Descripton'] = train_data['Item_Descripton'].fillna('').astype(str)
    test_data['Item_Descripton'] = test_data['Item_Descripton'].fillna('').astype(str)
    
    print(f"✅ Loaded {len(train_data)} training samples")
    print(f"✅ Loaded {len(test_data)} test samples")
    
    return train_data, test_data

def preprocess_text(text):
    """Simple text preprocessing"""
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text).lower().strip()
    text = ' '.join(text.split())
    
    return text

def extract_features(train_data, test_data):
    """Extract TF-IDF features from text data"""
    print("Extracting TF-IDF features...")
    
    # Preprocess text
    train_texts = train_data['Item_Descripton'].apply(preprocess_text)
    test_texts = test_data['Item_Descripton'].apply(preprocess_text)
    
    # Create TF-IDF vectorizer with optimized parameters
    vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )
    
    # Fit and transform training data
    X_train = vectorizer.fit_transform(train_texts)
    
    # Transform test data
    X_test = vectorizer.transform(test_texts)
    
    print(f"✅ Feature matrix shape - Train: {X_train.shape}, Test: {X_test.shape}")
    
    return X_train, X_test, vectorizer

def prepare_labels(train_data, test_data):
    """Prepare target labels"""
    print("Preparing target labels...")
    
    target_column = 'Category L2'
    
    # Create label encoder
    label_encoder = LabelEncoder()
    
    # Fit on training data
    y_train = label_encoder.fit_transform(train_data[target_column])
    
    # Transform test data
    y_test = label_encoder.transform(test_data[target_column])
    
    print(f"✅ Label encoding - Classes: {len(label_encoder.classes_)}")
    
    return y_train, y_test, label_encoder

def train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test):
    """Train a model and evaluate its performance"""
    print(f"\n🔄 Training {model_name}...")
    start_time = time.time()
    
    # Train model
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation (use smaller CV for SVM to save time)
    cv_folds = 3 if model_name == 'SVM' else 5
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"✅ {model_name} Results:")
    print(f"   • Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   • CV Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"   • Training Time: {training_time:.2f}s")
    
    return {
        'model': model,
        'name': model_name,
        'accuracy': accuracy,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'training_time': training_time,
        'predictions': y_pred
    }

def main():
    """Main execution function"""
    print("🚀 Comprehensive Model Comparison for Spend Categorization")
    print("="*70)
    
    # Load data
    train_data, test_data = load_data()
    
    # Extract features
    X_train, X_test, vectorizer = extract_features(train_data, test_data)
    
    # Prepare labels
    y_train, y_test, label_encoder = prepare_labels(train_data, test_data)
    
    print("\n" + "="*70)
    print("TRAINING AND EVALUATING ALL MODELS")
    print("="*70)
    
    # Define models to test
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ),
        'CatBoost': CatBoostClassifier(
            iterations=500,
            learning_rate=0.1,
            depth=6,
            random_state=42,
            verbose=False,
            auto_class_weights='Balanced'
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0,
            class_weight='balanced'
        ),
        'SVM': SVC(
            kernel='linear',
            C=1.0,
            random_state=42,
            class_weight='balanced'
        )
    }
    
    # Train and evaluate each model
    results = {}
    for model_name, model in models.items():
        results[model_name] = train_and_evaluate_model(
            model, model_name, X_train, y_train, X_test, y_test
        )
    
    # Sort results by accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    print("\n" + "="*70)
    print("FINAL RANKING BY ACCURACY")
    print("="*70)
    
    print(f"{'Rank':<4} {'Model':<20} {'Accuracy':<10} {'CV Score':<15} {'Time (s)':<10}")
    print("-" * 70)
    
    for rank, (model_name, result) in enumerate(sorted_results, 1):
        print(f"{rank:<4} {model_name:<20} {result['accuracy']:.4f}    "
              f"{result['cv_mean']:.4f}±{result['cv_std']:.3f}    {result['training_time']:.2f}")
    
    # Best model analysis
    best_model_name, best_result = sorted_results[0]
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   • Test Accuracy: {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)")
    print(f"   • CV Accuracy: {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f}")
    print(f"   • Training Time: {best_result['training_time']:.2f}s")
    
    # Performance comparison matrix
    print(f"\n📊 DETAILED PERFORMANCE MATRIX")
    print("="*70)
    print(f"{'Model':<20} {'Test Acc':<10} {'CV Mean':<10} {'CV Std':<10} {'Time':<10}")
    print("-" * 70)
    
    for model_name, result in results.items():
        print(f"{model_name:<20} {result['accuracy']:.4f}    "
              f"{result['cv_mean']:.4f}    {result['cv_std']:.4f}    {result['training_time']:.2f}s")
    
    # Speed vs Accuracy Analysis
    print(f"\n⚡ SPEED VS ACCURACY ANALYSIS")
    print("="*40)
    
    fastest_model = min(results.items(), key=lambda x: x[1]['training_time'])
    most_accurate_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    
    print(f"Fastest: {fastest_model[0]} ({fastest_model[1]['training_time']:.2f}s, "
          f"{fastest_model[1]['accuracy']*100:.2f}% accuracy)")
    print(f"Most Accurate: {most_accurate_model[0]} ({most_accurate_model[1]['accuracy']*100:.2f}% accuracy, "
          f"{most_accurate_model[1]['training_time']:.2f}s)")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("="*40)
    
    if best_result['accuracy'] > 0.60:
        print(f"✅ {best_model_name} shows excellent performance (>60% accuracy)")
    elif best_result['accuracy'] > 0.55:
        print(f"✅ {best_model_name} shows good performance (>55% accuracy)")
    else:
        print(f"⚠️  {best_model_name} shows moderate performance (<55% accuracy)")
    
    # Check if there's a significant difference between models
    second_best = sorted_results[1][1]['accuracy']
    accuracy_gap = best_result['accuracy'] - second_best
    
    if accuracy_gap > 0.05:
        print(f"🎯 {best_model_name} is significantly better than other models (+{accuracy_gap*100:.2f}%)")
    elif accuracy_gap > 0.02:
        print(f"📈 {best_model_name} is moderately better than other models (+{accuracy_gap*100:.2f}%)")
    else:
        print(f"⚖️  Models show similar performance (difference: {accuracy_gap*100:.2f}%)")
    
    # Save best model
    print(f"\n💾 Saving best model ({best_model_name})...")
    
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = model_dir / f'best_model_{best_model_name.lower().replace(" ", "_")}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(best_result['model'], f)
    
    print(f"✅ Best model saved to: {model_path}")
    
    print(f"\n🎯 Model comparison completed!")
    print(f"Winner: {best_model_name} with {best_result['accuracy']*100:.2f}% accuracy")

if __name__ == "__main__":
    main()
