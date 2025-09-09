#!/usr/bin/env python3
"""
Test different n-gram configurations for SVM model performance
"""

import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from production_svm_model import ProductionSVMCategorizer
import warnings
warnings.filterwarnings('ignore')

class NgramTestSVM:
    """SVM model for testing different n-gram configurations"""
    
    def __init__(self, ngram_range=(1, 1), max_features=500):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.vectorizer = None
        self.model = None
        self.label_encoder = None
        self.is_trained = False
    
    def preprocess_text(self, text):
        """Basic text preprocessing"""
        if pd.isna(text):
            return ""
        
        import re
        # Convert to lowercase and remove special characters
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def train(self, train_data, target_column='Category L2'):
        """Train the SVM model with specified n-gram configuration"""
        print(f"📊 Training SVM with n-gram range: {self.ngram_range}")
        
        # Preprocess text data
        texts = train_data['Item_Descripton'].apply(self.preprocess_text)
        
        # Extract TF-IDF features
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words='english',
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        
        X_train = self.vectorizer.fit_transform(texts)
        
        # Prepare labels
        self.label_encoder = LabelEncoder()
        y_train = self.label_encoder.fit_transform(train_data[target_column])
        
        # Train SVM model with optimized C=0.5
        start_time = time.time()
        self.model = SVC(
            kernel='linear',
            C=0.5,
            class_weight='balanced',
            random_state=42,
            probability=True
        )
        
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        self.is_trained = True
        
        print(f"   ✅ Feature matrix: {X_train.shape}")
        print(f"   ✅ Training time: {training_time:.2f}s")
        
        return self
    
    def evaluate(self, test_data, target_column='Category L2'):
        """Evaluate the model"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if self.vectorizer is None or self.label_encoder is None or self.model is None:
            raise ValueError("Model components not properly initialized")
        
        # Preprocess test data
        texts = test_data['Item_Descripton'].apply(self.preprocess_text)
        X_test = self.vectorizer.transform(texts)
        y_test = self.label_encoder.transform(test_data[target_column])
        
        # Predict
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'true_labels': y_test
        }

def test_ngram_configurations():
    """Test different n-gram configurations"""
    print("🔬 N-gram Configuration Performance Analysis")
    print("="*60)
    
    # Load data
    train_data = pd.read_excel('preprocessed_data/train_data.xlsx')
    test_data = pd.read_excel('preprocessed_data/test_data.xlsx')
    
    print(f"📊 Data loaded:")
    print(f"   Training: {len(train_data)} records")
    print(f"   Testing: {len(test_data)} records")
    
    # Define different n-gram configurations to test
    ngram_configs = [
        {'range': (1, 1), 'name': 'Unigrams Only', 'features': 500},
        {'range': (1, 2), 'name': 'Unigrams + Bigrams (Current)', 'features': 500},
        {'range': (1, 3), 'name': 'Unigrams + Bigrams + Trigrams', 'features': 500},
        {'range': (2, 2), 'name': 'Bigrams Only', 'features': 500},
        {'range': (2, 3), 'name': 'Bigrams + Trigrams', 'features': 500},
        {'range': (1, 2), 'name': 'Unigrams + Bigrams (Extended)', 'features': 1000},
        {'range': (1, 3), 'name': 'Unigrams + Bigrams + Trigrams (Extended)', 'features': 1000},
    ]
    
    results = []
    
    for config in ngram_configs:
        print(f"\n{'-'*50}")
        print(f"🧪 Testing: {config['name']}")
        print(f"   N-gram range: {config['range']}")
        print(f"   Max features: {config['features']}")
        print('-'*50)
        
        try:
            # Train model
            model = NgramTestSVM(
                ngram_range=config['range'], 
                max_features=config['features']
            )
            
            start_time = time.time()
            model.train(train_data)
            
            # Evaluate
            test_results = model.evaluate(test_data)
            total_time = time.time() - start_time
            
            accuracy = test_results['accuracy']
            
            print(f"   ✅ Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"   ✅ Total Time: {total_time:.2f}s")
            
            # Store results
            results.append({
                'name': config['name'],
                'ngram_range': config['range'],
                'max_features': config['features'],
                'accuracy': accuracy,
                'time': total_time
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                'name': config['name'],
                'ngram_range': config['range'], 
                'max_features': config['features'],
                'accuracy': 0.0,
                'time': 0.0,
                'error': str(e)
            })
    
    # Summary results
    print(f"\n{'='*60}")
    print("📈 N-GRAM CONFIGURATION COMPARISON")
    print('='*60)
    
    # Sort by accuracy
    valid_results = [r for r in results if r['accuracy'] > 0]
    valid_results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print(f"{'Rank':<4} {'Configuration':<35} {'Accuracy':<10} {'Time':<8}")
    print('-'*60)
    
    for i, result in enumerate(valid_results, 1):
        accuracy_str = f"{result['accuracy']:.4f}"
        time_str = f"{result['time']:.1f}s"
        print(f"{i:<4} {result['name']:<35} {accuracy_str:<10} {time_str:<8}")
    
    # Find best configuration
    if valid_results:
        best = valid_results[0]
        current_baseline = next((r for r in valid_results if 'Current' in r['name']), None)
        
        print(f"\n🏆 BEST CONFIGURATION:")
        print(f"   {best['name']}")
        print(f"   N-gram range: {best['ngram_range']}")
        print(f"   Max features: {best['max_features']}")
        print(f"   Accuracy: {best['accuracy']:.4f} ({best['accuracy']*100:.2f}%)")
        
        if current_baseline and best != current_baseline:
            improvement = best['accuracy'] - current_baseline['accuracy']
            improvement_pct = (improvement / current_baseline['accuracy']) * 100
            
            print(f"\n📊 IMPROVEMENT OVER CURRENT:")
            if improvement > 0:
                print(f"   ✅ +{improvement:.4f} ({improvement_pct:+.2f}%)")
            else:
                print(f"   ❌ {improvement:.4f} ({improvement_pct:+.2f}%)")
        
    return valid_results

def test_optimal_ngram_detailed():
    """Test the optimal n-gram configuration with detailed analysis"""
    print(f"\n{'='*60}")
    print("🔍 DETAILED ANALYSIS - OPTIMAL N-GRAM CONFIGURATION")
    print('='*60)
    
    # Load data
    train_data = pd.read_excel('preprocessed_data/train_data.xlsx')
    test_data = pd.read_excel('preprocessed_data/test_data.xlsx')
    
    # Test the best performing configuration
    print("🧪 Testing: Unigrams + Bigrams + Trigrams with 1000 features")
    
    model = NgramTestSVM(ngram_range=(1, 3), max_features=1000)
    model.train(train_data)
    results = model.evaluate(test_data)
    
    print(f"\n📋 Detailed Results:")
    print(f"   Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    
    # Get classification report
    y_pred = results['predictions']
    y_test = results['true_labels']
    
    print(f"\n📊 Classification Report:")
    if model.label_encoder is not None:
        class_names = model.label_encoder.classes_
        report = classification_report(y_test, y_pred, target_names=class_names)
        print(report)
    else:
        print("   ❌ Label encoder not available")

if __name__ == "__main__":
    try:
        # Run n-gram comparison
        results = test_ngram_configurations()
        
        # Detailed analysis of best configuration
        if results:
            test_optimal_ngram_detailed()
            
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()
