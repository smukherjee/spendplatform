#!/usr/bin/env python3
"""
Test CatBoost Model for Spend Platform Categorization
Trains and evaluates a CatBoost classifier specifically
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
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
    
    # Convert to lowercase and strip
    text = str(text).lower().strip()
    
    # Remove extra whitespace
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
    print(f"✅ Class distribution in training:")
    unique, counts = np.unique(y_train, return_counts=True)
    for i, (class_idx, count) in enumerate(zip(unique, counts)):
        class_name = label_encoder.classes_[class_idx]
        print(f"   {class_name}: {count} samples ({count/len(y_train)*100:.1f}%)")
    
    return y_train, y_test, label_encoder

def train_catboost_models(X_train, y_train, X_test, y_test):
    """Train multiple CatBoost configurations and compare"""
    print("\n" + "="*60)
    print("TRAINING CATBOOST MODELS")
    print("="*60)
    
    # Different CatBoost configurations to test
    configs = {
        'default': {
            'iterations': 500,
            'learning_rate': 0.1,
            'depth': 6,
            'random_state': 42,
            'verbose': False,
            'auto_class_weights': 'Balanced'
        },
        'fast': {
            'iterations': 300,
            'learning_rate': 0.15,
            'depth': 4,
            'random_state': 42,
            'verbose': False,
            'auto_class_weights': 'Balanced'
        },
        'deep': {
            'iterations': 700,
            'learning_rate': 0.08,
            'depth': 8,
            'random_state': 42,
            'verbose': False,
            'auto_class_weights': 'Balanced'
        },
        'optimized': {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_state': 42,
            'verbose': False,
            'auto_class_weights': 'Balanced',
            'eval_metric': 'Accuracy'
        }
    }
    
    results = {}
    
    for config_name, config_params in configs.items():
        print(f"\n🔄 Training CatBoost ({config_name})...")
        start_time = time.time()
        
        # Create and train model
        model = CatBoostClassifier(**config_params)
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # Store results
        results[config_name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'training_time': training_time,
            'predictions': y_pred
        }
        
        print(f"✅ {config_name.capitalize()} CatBoost:")
        print(f"   • Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   • CV Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"   • Training Time: {training_time:.2f}s")
    
    return results

def compare_with_baseline(X_train, y_train, X_test, y_test):
    """Compare CatBoost with baseline Random Forest"""
    print("\n" + "="*60)
    print("BASELINE COMPARISON")
    print("="*60)
    
    from sklearn.ensemble import RandomForestClassifier
    
    print("🔄 Training Random Forest baseline...")
    start_time = time.time()
    
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    rf_model.fit(X_train, y_train)
    rf_training_time = time.time() - start_time
    
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
    rf_cv_mean = rf_cv_scores.mean()
    rf_cv_std = rf_cv_scores.std()
    
    print(f"✅ Random Forest Baseline:")
    print(f"   • Test Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
    print(f"   • CV Accuracy: {rf_cv_mean:.4f} ± {rf_cv_std:.4f}")
    print(f"   • Training Time: {rf_training_time:.2f}s")
    
    return {
        'model': rf_model,
        'accuracy': rf_accuracy,
        'cv_mean': rf_cv_mean,
        'cv_std': rf_cv_std,
        'training_time': rf_training_time,
        'predictions': rf_pred
    }

def detailed_evaluation(y_test, predictions, label_encoder, model_name):
    """Provide detailed evaluation metrics"""
    print(f"\n📊 Detailed Evaluation for {model_name}:")
    print("="*40)
    
    # Classification report
    report = classification_report(y_test, predictions, 
                                   target_names=label_encoder.classes_,
                                   output_dict=True)
    
    print("\nPer-class Performance:")
    for class_name in label_encoder.classes_:
        if class_name in report:
            metrics = report[class_name]
            print(f"  • {class_name}:")
            print(f"    - Precision: {metrics['precision']:.3f}")
            print(f"    - Recall: {metrics['recall']:.3f}")
            print(f"    - F1-Score: {metrics['f1-score']:.3f}")
            print(f"    - Support: {int(metrics['support'])}")
    
    # Overall metrics
    print(f"\nOverall Metrics:")
    print(f"  • Accuracy: {report['accuracy']:.4f}")
    print(f"  • Macro Avg F1: {report['macro avg']['f1-score']:.4f}")
    print(f"  • Weighted Avg F1: {report['weighted avg']['f1-score']:.4f}")

def save_best_model(best_result, vectorizer, label_encoder):
    """Save the best performing model"""
    print("\n💾 Saving best model...")
    
    # Create model directory if it doesn't exist
    model_dir = Path('models')
    model_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = model_dir / 'catboost_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(best_result['model'], f)
    
    # Save vectorizer
    vectorizer_path = model_dir / 'catboost_vectorizer.pkl'
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Save label encoder
    encoder_path = model_dir / 'catboost_label_encoder.pkl'
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Vectorizer saved to: {vectorizer_path}")
    print(f"✅ Label encoder saved to: {encoder_path}")

def main():
    """Main execution function"""
    print("🚀 CatBoost Model Testing for Spend Categorization")
    print("="*60)
    
    # Load data
    train_data, test_data = load_data()
    
    # Extract features
    X_train, X_test, vectorizer = extract_features(train_data, test_data)
    
    # Prepare labels
    y_train, y_test, label_encoder = prepare_labels(train_data, test_data)
    
    # Train CatBoost models
    catboost_results = train_catboost_models(X_train, y_train, X_test, y_test)
    
    # Compare with baseline
    rf_result = compare_with_baseline(X_train, y_train, X_test, y_test)
    
    # Find best CatBoost configuration
    best_config = max(catboost_results.keys(), 
                      key=lambda k: catboost_results[k]['accuracy'])
    best_result = catboost_results[best_config]
    
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    print(f"\n🏆 Best CatBoost Configuration: {best_config}")
    print(f"   • Test Accuracy: {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)")
    print(f"   • CV Accuracy: {best_result['cv_mean']:.4f} ± {best_result['cv_std']:.4f}")
    
    print(f"\n📊 Baseline Random Forest:")
    print(f"   • Test Accuracy: {rf_result['accuracy']:.4f} ({rf_result['accuracy']*100:.2f}%)")
    print(f"   • CV Accuracy: {rf_result['cv_mean']:.4f} ± {rf_result['cv_std']:.4f}")
    
    # Performance comparison
    improvement = best_result['accuracy'] - rf_result['accuracy']
    print(f"\n🔍 Performance Comparison:")
    if improvement > 0:
        print(f"   • CatBoost improves accuracy by {improvement:.4f} ({improvement*100:.2f}%)")
        print(f"   • CatBoost is {improvement/rf_result['accuracy']*100:.2f}% relatively better")
    else:
        print(f"   • Random Forest is better by {abs(improvement):.4f} ({abs(improvement)*100:.2f}%)")
    
    # Speed comparison
    speed_ratio = rf_result['training_time'] / best_result['training_time']
    print(f"   • Training speed: CatBoost {speed_ratio:.1f}x {'faster' if speed_ratio > 1 else 'slower'} than RF")
    
    # Detailed evaluation for best model
    detailed_evaluation(y_test, best_result['predictions'], label_encoder, 
                       f"Best CatBoost ({best_config})")
    
    # Save best model
    save_best_model(best_result, vectorizer, label_encoder)
    
    print("\n🎯 CatBoost testing completed!")
    print(f"Best configuration: {best_config} with {best_result['accuracy']*100:.2f}% accuracy")

if __name__ == "__main__":
    main()
