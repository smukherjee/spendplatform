#!/usr/bin/env python3
"""
Final Model Performance Summary
Comprehensive comparison of all model improvements including enhanced LR features
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
import time
import warnings

warnings.filterwarnings('ignore')

def load_data():
    """Load and clean data"""
    train_data = pd.read_excel('preprocessed_data/train_data.xlsx')
    test_data = pd.read_excel('preprocessed_data/test_data.xlsx')
    
    # Clean column names
    train_data.columns = train_data.columns.str.strip()
    test_data.columns = test_data.columns.str.strip()
    
    # Fill missing values
    train_data['Item_Descripton'] = train_data['Item_Descripton'].fillna('').astype(str)
    test_data['Item_Descripton'] = test_data['Item_Descripton'].fillna('').astype(str)
    
    return train_data, test_data

def test_basic_models(train_data, test_data):
    """Test basic models with simple TF-IDF"""
    print("🔄 Testing Basic Models with Simple TF-IDF")
    print("-" * 50)
    
    # Basic TF-IDF
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words='english')
    X_train = vectorizer.fit_transform(train_data['Item_Descripton'])
    X_test = vectorizer.transform(test_data['Item_Descripton'])
    
    # Labels
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_data['Category L2'])
    y_test = encoder.transform(test_data['Category L2'])
    
    results = {}
    
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'SVM': SVC(class_weight='balanced', random_state=42),
        'CatBoost': CatBoostClassifier(iterations=300, verbose=False, auto_class_weights='Balanced', random_state=42)
    }
    
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'training_time': training_time,
            'features': 'Basic TF-IDF (500)'
        }
        
        print(f"  {name}: {accuracy:.4f} ({accuracy*100:.2f}%) - {training_time:.2f}s")
    
    return results

def test_enhanced_lr(train_data, test_data):
    """Test enhanced Logistic Regression"""
    print("\\n🚀 Testing Enhanced Logistic Regression")
    print("-" * 50)
    
    from improved_lr_model import ImprovedLRCategorizer
    
    improved_lr = ImprovedLRCategorizer()
    
    start_time = time.time()
    improved_lr.train(train_data)
    training_time = time.time() - start_time
    
    results = improved_lr.evaluate(test_data)
    
    return {
        'Enhanced LR': {
            'accuracy': results['accuracy'],
            'training_time': training_time,
            'features': 'Enhanced (TF-IDF + Char + Text Stats)'
        }
    }

def main():
    """Main comparison function"""
    print("🎯 COMPREHENSIVE MODEL PERFORMANCE SUMMARY")
    print("="*60)
    print("Testing all models with enhanced feature engineering improvements")
    print("="*60)
    
    # Load data
    print("📁 Loading data...")
    train_data, test_data = load_data()
    print(f"✅ Training: {len(train_data)} samples, Test: {len(test_data)} samples")
    
    # Test basic models
    basic_results = test_basic_models(train_data, test_data)
    
    # Test enhanced LR
    enhanced_results = test_enhanced_lr(train_data, test_data)
    
    # Combine results
    all_results = {**basic_results, **enhanced_results}
    
    # Final ranking
    print("\\n🏆 FINAL PERFORMANCE RANKING")
    print("="*60)
    
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    print(f"{'Rank':<4} {'Model':<25} {'Accuracy':<12} {'Time':<8} {'Features'}")
    print("-" * 80)
    
    for rank, (model_name, result) in enumerate(sorted_results, 1):
        accuracy = result['accuracy']
        time_taken = result['training_time']
        features = result['features']
        
        print(f"{rank:<4} {model_name:<25} {accuracy:.4f} ({accuracy*100:.2f}%) {time_taken:<8.2f}s {features}")
    
    # Analysis
    best_model, best_result = sorted_results[0]
    baseline_model = 'Random Forest'
    baseline_accuracy = basic_results[baseline_model]['accuracy']
    
    print("\\n📊 IMPROVEMENT ANALYSIS")
    print("="*40)
    print(f"🥇 Best Model: {best_model}")
    print(f"   • Accuracy: {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)")
    print(f"   • Training Time: {best_result['training_time']:.2f}s")
    print(f"   • Features: {best_result['features']}")
    
    improvement_over_baseline = best_result['accuracy'] - baseline_accuracy
    print(f"\\n📈 Improvement over Random Forest baseline:")
    print(f"   • Absolute: +{improvement_over_baseline:.4f} ({improvement_over_baseline*100:.2f}%)")
    print(f"   • Relative: {improvement_over_baseline/baseline_accuracy*100:.2f}% better")
    
    # Speed vs Accuracy Analysis
    print("\\n⚡ SPEED vs ACCURACY TRADE-OFFS")
    print("="*40)
    
    fastest_model = min(all_results.items(), key=lambda x: x[1]['training_time'])
    most_accurate_model = max(all_results.items(), key=lambda x: x[1]['accuracy'])
    
    print(f"⚡ Fastest: {fastest_model[0]}")
    print(f"   • Time: {fastest_model[1]['training_time']:.2f}s")
    print(f"   • Accuracy: {fastest_model[1]['accuracy']*100:.2f}%")
    
    print(f"\\n🎯 Most Accurate: {most_accurate_model[0]}")
    print(f"   • Accuracy: {most_accurate_model[1]['accuracy']*100:.2f}%")
    print(f"   • Time: {most_accurate_model[1]['training_time']:.2f}s")
    
    # Feature Engineering Impact
    print("\\n🔧 FEATURE ENGINEERING IMPACT")
    print("="*40)
    
    basic_lr_accuracy = basic_results['Logistic Regression']['accuracy']
    enhanced_lr_accuracy = enhanced_results['Enhanced LR']['accuracy']
    feature_improvement = enhanced_lr_accuracy - basic_lr_accuracy
    
    print(f"📊 Logistic Regression Comparison:")
    print(f"   • Basic LR: {basic_lr_accuracy:.4f} ({basic_lr_accuracy*100:.2f}%)")
    print(f"   • Enhanced LR: {enhanced_lr_accuracy:.4f} ({enhanced_lr_accuracy*100:.2f}%)")
    print(f"   • Improvement: +{feature_improvement:.4f} ({feature_improvement*100:.2f}%)")
    print(f"   • Relative gain: {feature_improvement/basic_lr_accuracy*100:.2f}%")
    
    # CatBoost Impact
    catboost_accuracy = basic_results['CatBoost']['accuracy']
    rf_accuracy = basic_results['Random Forest']['accuracy']
    catboost_improvement = catboost_accuracy - rf_accuracy
    
    print(f"\\n🚀 CatBoost Addition Impact:")
    print(f"   • Random Forest: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
    print(f"   • CatBoost: {catboost_accuracy:.4f} ({catboost_accuracy*100:.2f}%)")
    print(f"   • Improvement: +{catboost_improvement:.4f} ({catboost_improvement*100:.2f}%)")
    print(f"   • Relative gain: {catboost_improvement/rf_accuracy*100:.2f}%")
    
    # Final Recommendations
    print("\\n💡 FINAL RECOMMENDATIONS")
    print("="*40)
    
    if best_result['accuracy'] > 0.65:
        performance_level = "excellent"
    elif best_result['accuracy'] > 0.60:
        performance_level = "good"
    else:
        performance_level = "moderate"
    
    print(f"✅ Best overall model: {best_model} with {performance_level} performance")
    
    if best_result['training_time'] < 1.0:
        print(f"⚡ Fast training time ({best_result['training_time']:.2f}s) suitable for production")
    else:
        print(f"⏱️  Moderate training time ({best_result['training_time']:.2f}s)")
    
    # Production recommendations
    print(f"\\n🚀 Production Deployment Recommendations:")
    if best_model == 'Enhanced LR':
        print(f"   • Use Enhanced Logistic Regression for best accuracy")
        print(f"   • Features: TF-IDF + Character n-grams + Text statistics")
        print(f"   • Fast training and prediction")
    else:
        print(f"   • {best_model} recommended for production")
        print(f"   • Consider Enhanced LR for better feature engineering")
    
    print(f"\\n🎯 Summary: Successfully improved from {baseline_accuracy*100:.2f}% to {best_result['accuracy']*100:.2f}% accuracy!")

if __name__ == "__main__":
    main()
