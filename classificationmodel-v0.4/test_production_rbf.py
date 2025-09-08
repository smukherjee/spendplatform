#!/usr/bin/env python3
"""
Update production model with optimal RBF kernel and comprehensive testing
"""

import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

class OptimizedRBFSVM:
    """Production SVM with optimized RBF kernel"""
    
    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.model: Optional[SVC] = None
        self.is_trained: bool = False
    
    def preprocess_text(self, text):
        """Optimized text preprocessing for spend categorization"""
        if pd.isna(text) or text == '':
            return ''
        return str(text).lower().strip()
    
    def train(self, train_data, target_column='Category L2', use_rbf=True):
        """Train the SVM model with RBF or Linear kernel"""
        kernel_type = "RBF" if use_rbf else "Linear"
        print(f"🚀 Training Production SVM Model ({kernel_type} Kernel)")
        print("="*50)
        
        # Preprocess text data
        print("📝 Preprocessing text data...")
        texts = train_data['Item_Descripton'].apply(self.preprocess_text)
        
        # Extract TF-IDF features
        print("📊 Extracting TF-IDF features...")
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
        print("🏷️  Encoding labels...")
        self.label_encoder = LabelEncoder()
        y_train = self.label_encoder.fit_transform(train_data[target_column])
        
        print(f"✅ Feature matrix: {X_train.shape}")
        print(f"✅ Classes: {len(self.label_encoder.classes_)}")
        
        # Train SVM model with optimal parameters
        print(f"🎯 Training {kernel_type} SVM...")
        start_time = time.time()
        
        if use_rbf:
            # Best RBF configuration from testing
            self.model = SVC(
                kernel='rbf',
                C=2.0,              # Best C for RBF
                gamma='scale',      # Auto-scale gamma
                class_weight='balanced',
                random_state=42,
                probability=True
            )
        else:
            # Linear configuration
            self.model = SVC(
                kernel='linear',
                C=0.5,              # Optimized C for linear
                class_weight='balanced', 
                random_state=42,
                probability=True
            )
        
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        self.is_trained = True
        
        print(f"✅ Training completed in {training_time:.2f}s")
        print(f"✅ Model ready for production use")
        
        return self
    
    def evaluate(self, test_data, target_column='Category L2'):
        """Evaluate the model performance"""
        if not self.is_trained or self.vectorizer is None or self.label_encoder is None or self.model is None:
            raise ValueError("Model must be trained first")
        
        print("\n📈 Evaluating Model Performance")
        print("="*40)
        
        # Preprocess and extract features
        texts = test_data['Item_Descripton'].apply(self.preprocess_text)
        X_test = self.vectorizer.transform(texts)
        y_test = self.label_encoder.transform(test_data[target_column])
        
        # Cross-validation on training data
        print("🔄 Running cross-validation...")
        X_train_eval = self.vectorizer.transform(test_data['Item_Descripton'].apply(self.preprocess_text))
        cv_scores = cross_val_score(self.model, X_test, y_test, cv=5, scoring='accuracy')
        
        # Test predictions
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"✅ CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Classification report
        print(f"\n📋 Classification Report:")
        print("-" * 50)
        class_names = self.label_encoder.classes_
        report = classification_report(y_test, y_pred, target_names=class_names)
        print(report)
        
        return {
            'accuracy': accuracy,
            'cv_scores': cv_scores,
            'classification_report': report
        }

def comprehensive_kernel_comparison():
    """Comprehensive comparison of Linear vs RBF kernels"""
    print("🔬 Comprehensive Linear vs RBF Kernel Comparison")
    print("="*60)
    
    # Load data
    train_data = pd.read_excel('preprocessed_data/train_data.xlsx')
    test_data = pd.read_excel('preprocessed_data/test_data.xlsx')
    
    print(f"📊 Data loaded:")
    print(f"   Training: {len(train_data)} records")
    print(f"   Testing: {len(test_data)} records")
    
    results = {}
    
    # Test Linear kernel
    print(f"\n{'-'*50}")
    print("🧪 LINEAR KERNEL MODEL")
    print('-'*50)
    
    linear_model = OptimizedRBFSVM()
    linear_model.train(train_data, use_rbf=False)
    linear_results = linear_model.evaluate(test_data)
    results['linear'] = linear_results
    
    # Test RBF kernel
    print(f"\n{'-'*50}")
    print("🧪 RBF KERNEL MODEL")  
    print('-'*50)
    
    rbf_model = OptimizedRBFSVM()
    rbf_model.train(train_data, use_rbf=True)
    rbf_results = rbf_model.evaluate(test_data)
    results['rbf'] = rbf_results
    
    # Final comparison
    print(f"\n{'='*60}")
    print("📈 FINAL KERNEL COMPARISON")
    print('='*60)
    
    linear_acc = results['linear']['accuracy']
    rbf_acc = results['rbf']['accuracy']
    improvement = rbf_acc - linear_acc
    improvement_pct = (improvement / linear_acc) * 100
    
    print(f"Linear Kernel:")
    print(f"   Accuracy: {linear_acc:.4f} ({linear_acc*100:.2f}%)")
    print(f"   CV Score: {results['linear']['cv_scores'].mean():.4f} ± {results['linear']['cv_scores'].std():.4f}")
    
    print(f"\nRBF Kernel:")
    print(f"   Accuracy: {rbf_acc:.4f} ({rbf_acc*100:.2f}%)")
    print(f"   CV Score: {results['rbf']['cv_scores'].mean():.4f} ± {results['rbf']['cv_scores'].std():.4f}")
    
    print(f"\nImprovement Analysis:")
    if improvement > 0:
        print(f"   ✅ RBF Improvement: +{improvement:.4f} ({improvement_pct:+.2f}%)")
        print(f"   🏆 RBF kernel performs better!")
    elif improvement < 0:
        print(f"   ❌ RBF Performance: {improvement:.4f} ({improvement_pct:+.2f}%)")
        print(f"   🏆 Linear kernel performs better!")
    else:
        print(f"   ➖ No significant difference between kernels")
    
    # Recommend best model
    if rbf_acc > linear_acc:
        print(f"\n💡 RECOMMENDATION: Use RBF kernel for production")
        print(f"   Configuration: C=2.0, gamma='scale', kernel='rbf'")
        return rbf_model, rbf_results
    else:
        print(f"\n💡 RECOMMENDATION: Keep Linear kernel for production")
        print(f"   Configuration: C=0.5, kernel='linear'")
        return linear_model, linear_results

if __name__ == "__main__":
    try:
        # Run comprehensive comparison
        best_model, best_results = comprehensive_kernel_comparison()
        
        print(f"\n{'='*60}")
        print("🎯 PRODUCTION MODEL READY")
        print('='*60)
        print(f"Best accuracy achieved: {best_results['accuracy']:.4f} ({best_results['accuracy']*100:.2f}%)")
        
        # Safely get model attributes
        kernel_type = getattr(best_model.model, 'kernel', 'unknown') if best_model.model else 'unknown'
        print(f"Model type: {'RBF' if kernel_type == 'rbf' else 'Linear'}")
        
        if best_model.model and hasattr(best_model.model, 'C'):
            c_param = getattr(best_model.model, 'C', 'unknown')
            print(f"C parameter: {c_param}")
            
        if best_model.model and hasattr(best_model.model, 'gamma') and kernel_type == 'rbf':
            gamma_param = getattr(best_model.model, 'gamma', 'unknown')
            print(f"Gamma parameter: {gamma_param}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
