#!/usr/bin/env python3
"""
Test RBF kernel vs Linear kernel for non-linear pattern detection
"""

import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score
from typing import Union, Literal
import warnings
warnings.filterwarnings('ignore')

class KernelTestSVM:
    """SVM model for testing different kernel types"""
    
    def __init__(self, kernel: str = 'linear', C: float = 0.5, gamma: Union[str, float] = 'scale'):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.vectorizer = None
        self.model = None
        self.label_encoder = None
        self.is_trained = False
    
    def preprocess_text(self, text):
        """Simple text preprocessing"""
        if pd.isna(text) or text == '':
            return ''
        return str(text).lower().strip()
    
    def train(self, train_data, target_column='Category L2'):
        """Train the SVM model with specified kernel"""
        print(f"📊 Training SVM with {self.kernel} kernel (C={self.C}, gamma={self.gamma})")
        
        # Preprocess text data
        texts = train_data['Item_Descripton'].apply(self.preprocess_text)
        
        # Extract TF-IDF features
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 1),  # Unigrams only
            stop_words='english',
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        
        X_train = self.vectorizer.fit_transform(texts)
        
        # Prepare labels
        self.label_encoder = LabelEncoder()
        y_train = self.label_encoder.fit_transform(train_data[target_column])
        
        # Train SVM model
        start_time = time.time()
        self.model = SVC(
            kernel=self.kernel,  # type: ignore
            C=self.C,
            gamma=self.gamma,  # type: ignore
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

def test_rbf_vs_linear():
    """Compare RBF and Linear kernels"""
    print("🔬 RBF vs Linear Kernel Comparison")
    print("="*50)
    
    # Load data
    train_data = pd.read_excel('preprocessed_data/train_data.xlsx')
    test_data = pd.read_excel('preprocessed_data/test_data.xlsx')
    
    print(f"📊 Data loaded:")
    print(f"   Training: {len(train_data)} records")
    print(f"   Testing: {len(test_data)} records")
    
    # Test configurations
    configs = [
        {'kernel': 'linear', 'C': 0.5, 'gamma': 'scale', 'name': 'Linear (Current)'},
        {'kernel': 'rbf', 'C': 0.5, 'gamma': 'scale', 'name': 'RBF (Auto Gamma)'},
        {'kernel': 'rbf', 'C': 0.5, 'gamma': 0.001, 'name': 'RBF (Gamma=0.001)'},
        {'kernel': 'rbf', 'C': 0.5, 'gamma': 0.01, 'name': 'RBF (Gamma=0.01)'},
        {'kernel': 'rbf', 'C': 0.5, 'gamma': 0.1, 'name': 'RBF (Gamma=0.1)'},
        {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale', 'name': 'RBF (C=1.0)'},
        {'kernel': 'rbf', 'C': 2.0, 'gamma': 'scale', 'name': 'RBF (C=2.0)'},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{'-'*40}")
        print(f"🧪 Testing: {config['name']}")
        print(f"   Kernel: {config['kernel']}, C: {config['C']}, Gamma: {config['gamma']}")
        print('-'*40)
        
        try:
            # Train model
            model = KernelTestSVM(
                kernel=config['kernel'], 
                C=config['C'],
                gamma=config['gamma']
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
                'kernel': config['kernel'],
                'C': config['C'],
                'gamma': config['gamma'],
                'accuracy': accuracy,
                'time': total_time
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                'name': config['name'],
                'kernel': config['kernel'],
                'C': config['C'],
                'gamma': config['gamma'],
                'accuracy': 0.0,
                'time': 0.0,
                'error': str(e)
            })
    
    return results

def analyze_kernel_results(results):
    """Analyze and display kernel comparison results"""
    print(f"\n{'='*60}")
    print("📈 KERNEL COMPARISON RESULTS")
    print('='*60)
    
    # Sort by accuracy
    valid_results = [r for r in results if r['accuracy'] > 0]
    valid_results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print(f"{'Rank':<4} {'Configuration':<25} {'Accuracy':<10} {'Time':<8} {'Kernel':<6}")
    print('-'*60)
    
    for i, result in enumerate(valid_results, 1):
        accuracy_str = f"{result['accuracy']:.4f}"
        time_str = f"{result['time']:.1f}s"
        print(f"{i:<4} {result['name']:<25} {accuracy_str:<10} {time_str:<8} {result['kernel']:<6}")
    
    # Analysis
    if valid_results:
        best = valid_results[0]
        linear_result = next((r for r in valid_results if r['kernel'] == 'linear'), None)
        best_rbf = next((r for r in valid_results if r['kernel'] == 'rbf'), None)
        
        print(f"\n🏆 BEST OVERALL:")
        print(f"   {best['name']}")
        print(f"   Accuracy: {best['accuracy']:.4f} ({best['accuracy']*100:.2f}%)")
        print(f"   Kernel: {best['kernel']}")
        
        if linear_result and best_rbf:
            improvement = best_rbf['accuracy'] - linear_result['accuracy']
            improvement_pct = (improvement / linear_result['accuracy']) * 100
            
            print(f"\n📊 RBF vs LINEAR COMPARISON:")
            print(f"   Linear: {linear_result['accuracy']:.4f} ({linear_result['accuracy']*100:.2f}%)")
            print(f"   Best RBF: {best_rbf['accuracy']:.4f} ({best_rbf['accuracy']*100:.2f}%)")
            
            if improvement > 0:
                print(f"   ✅ RBF Improvement: +{improvement:.4f} ({improvement_pct:+.2f}%)")
            elif improvement < 0:
                print(f"   ❌ RBF Performance: {improvement:.4f} ({improvement_pct:+.2f}%)")
            else:
                print(f"   ➖ No difference between kernels")
    
    return valid_results

def test_rbf_grid_search():
    """Perform grid search optimization for RBF kernel"""
    print(f"\n{'='*60}")
    print("🔍 RBF KERNEL GRID SEARCH OPTIMIZATION")
    print('='*60)
    
    # Load data
    train_data = pd.read_excel('preprocessed_data/train_data.xlsx')
    test_data = pd.read_excel('preprocessed_data/test_data.xlsx')
    
    # Use a subset for faster grid search
    train_sample = train_data.sample(n=min(1500, len(train_data)), random_state=42)
    print(f"📊 Using {len(train_sample)} training samples for grid search")
    
    # Preprocess
    model = KernelTestSVM()
    texts = train_sample['Item_Descripton'].apply(model.preprocess_text)
    
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 1),
        stop_words='english',
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    
    X_train = vectorizer.fit_transform(texts)
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_sample['Category L2'])
    
    print(f"✅ Feature matrix: {X_train.shape}")
    
    # Grid search parameters for RBF
    param_grid = {
        'C': [0.1, 0.5, 1.0, 2.0, 5.0],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0]
    }
    
    print("🔍 Grid search parameters:")
    print(f"   C: {param_grid['C']}")
    print(f"   Gamma: {param_grid['gamma']}")
    
    # Perform grid search
    base_svm = SVC(
        kernel='rbf',
        class_weight='balanced',
        random_state=42,
        probability=True
    )
    
    grid_search = GridSearchCV(
        estimator=base_svm,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    print(f"\n✅ Grid search completed in {search_time:.2f}s")
    print(f"✅ Best parameters: {grid_search.best_params_}")
    print(f"✅ Best CV score: {grid_search.best_score_:.4f} ({grid_search.best_score_*100:.2f}%)")
    
    # Test on full test set
    print("\n🧪 Testing optimized RBF on full test set...")
    test_texts = test_data['Item_Descripton'].apply(model.preprocess_text)
    X_test = vectorizer.transform(test_texts)
    y_test = label_encoder.transform(test_data['Category L2'])
    
    y_pred = grid_search.best_estimator_.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Optimized RBF test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    return {
        'best_params': grid_search.best_params_,
        'cv_score': grid_search.best_score_,
        'test_accuracy': test_accuracy
    }

if __name__ == "__main__":
    try:
        # Compare different kernels
        results = test_rbf_vs_linear()
        
        # Analyze results
        best_results = analyze_kernel_results(results)
        
        # Grid search optimization for RBF
        rbf_optimization = test_rbf_grid_search()
        
        print(f"\n{'='*60}")
        print("🎯 FINAL SUMMARY")
        print('='*60)
        
        if best_results:
            print(f"Best manual config: {best_results[0]['accuracy']:.4f} ({best_results[0]['accuracy']*100:.2f}%)")
        
        print(f"Best RBF grid search: {rbf_optimization['test_accuracy']:.4f} ({rbf_optimization['test_accuracy']*100:.2f}%)")
        print(f"Optimal RBF parameters: {rbf_optimization['best_params']}")
        
        # Compare with linear baseline
        linear_baseline = 0.6698
        best_rbf = rbf_optimization['test_accuracy']
        improvement = best_rbf - linear_baseline
        improvement_pct = (improvement / linear_baseline) * 100
        
        print(f"\nLinear baseline: {linear_baseline:.4f} ({linear_baseline*100:.2f}%)")
        print(f"Best RBF result: {best_rbf:.4f} ({best_rbf*100:.2f}%)")
        
        if improvement > 0:
            print(f"✅ RBF improvement: +{improvement:.4f} ({improvement_pct:+.2f}%)")
        elif improvement < 0:
            print(f"❌ RBF performance: {improvement:.4f} ({improvement_pct:+.2f}%)")
        else:
            print(f"➖ No difference between kernels")
            
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()
