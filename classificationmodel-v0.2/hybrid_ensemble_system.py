#!/usr/bin/env python3
"""
Hybrid Advanced Categorization System
Combines optimized TF-IDF baseline with selective advanced features and ensemble voting
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# Core ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# XGBoost with fallback
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings('ignore')

class SelectiveFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract only the most valuable features for categorization"""
    
    def __init__(self):
        # Focus on proven patterns from domain analysis
        self.domain_patterns = {
            'electrical': [r'\b\d+v\b', r'\bvolt\b', r'\bamp\b', r'\bwatt\b', r'\bcurrent\b', 
                          r'\belectric\b', r'\bwire\b', r'\bcable\b'],
            'mechanical': [r'\b\d+mm\b', r'\b\d+cm\b', r'\bmetal\b', r'\bsteel\b', 
                          r'\bgear\b', r'\bbearing\b', r'\bbolt\b'],
            'manufacturing': [r'\bmachine\b', r'\btool\b', r'\bequipment\b', r'\bmanufactur\b'],
            'measurements': [r'\b\d+x\d+\b', r'\b\d+\.\d+\b', r'\b\d+/\d+\b']
        }
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Extract selective high-value features"""
        features = []
        for text in X:
            text_str = str(text).lower() if pd.notna(text) else ""
            feature_vector = self._extract_features(text_str)
            features.append(feature_vector)
        
        return np.array(features)
    
    def _extract_features(self, text: str) -> List[float]:
        """Extract 20 most valuable features"""
        if not text:
            return [0] * 20
        
        features = []
        words = text.split()
        
        # 1-5: Basic text statistics (proven valuable)
        features.extend([
            len(text),                          # Character count
            len(words),                         # Word count  
            len(set(words)) / max(len(words), 1),  # Lexical diversity
            sum(len(w) for w in words) / max(len(words), 1),  # Avg word length
            len([c for c in text if c.isdigit()])  # Digit count
        ])
        
        # 6-9: Domain pattern matching (high precision)
        for domain, patterns in self.domain_patterns.items():
            import re
            count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in patterns)
            features.append(min(count, 10))  # Cap to prevent outliers
        
        # 10-15: Technical indicators (manufacturing focus)
        features.extend([
            text.count('make'),                 # Brand indicator
            text.count('model'),                # Model indicator  
            text.count('type'),                 # Type indicator
            len([w for w in words if w.isalpha()]),  # Alpha words
            len([w for w in words if len(w) > 8]),   # Long technical terms
            text.count(' ')                     # Complexity indicator
        ])
        
        # 16-20: Material and specification indicators
        materials = ['steel', 'aluminum', 'plastic', 'rubber', 'copper']
        features.extend([
            sum(text.count(mat) for mat in materials),  # Material mentions
            text.count('mm') + text.count('cm'),        # Metric measurements
            text.count('grade') + text.count('class'),  # Quality indicators
            len([w for w in words if w.endswith('ed')]), # Past tense (specifications)
            min(max([len(w) for w in words] + [0]), 20)  # Max word length (capped)
        ])
        
        return features

class HybridEnsembleClassifier:
    """Hybrid system combining optimized baseline with selective enhancement"""
    
    def __init__(self):
        # Use proven optimal TF-IDF configuration
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1500,      # Optimal from baseline
            ngram_range=(1, 2),     # Optimal from baseline  
            min_df=1,               # Optimal from baseline
            max_df=0.9,             # Optimal from baseline
            sublinear_tf=True,
            use_idf=True,
            lowercase=True,
            analyzer='word'
        )
        
        self.feature_extractor = SelectiveFeatureExtractor()
        self.scaler = StandardScaler()
        self.ensemble_model = None
        self.is_trained = False
        
    def _create_optimized_ensemble(self):
        """Create ensemble with optimized hyperparameters"""
        
        # Optimized Random Forest (primary model)
        rf_classifier = RandomForestClassifier(
            n_estimators=150,       # Balanced performance/speed
            max_depth=15,           # Optimal from baseline
            min_samples_split=5,    # Optimal from baseline
            min_samples_leaf=2,     # Optimal from baseline
            max_features='sqrt',    # Reduce overfitting
            random_state=42,
            n_jobs=-1
        )
        
        # Optimized SVM for different perspective
        svm_classifier = SVC(
            kernel='rbf',
            C=0.8,                  # Slightly regularized
            gamma='scale',
            probability=True,
            random_state=42
        )
        
        estimators = [
            ('rf', rf_classifier),
            ('svm', svm_classifier)
        ]
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            xgb_classifier = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                eval_metric='mlogloss',
                verbosity=0
            )
            estimators.append(('xgb', xgb_classifier))
        
        # Use hard voting for better performance with fewer models
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='hard' if len(estimators) <= 2 else 'soft',
            n_jobs=-1
        )
        
        return ensemble
    
    def prepare_hybrid_features(self, descriptions: pd.Series, fit_transform: bool = True):
        """Prepare optimized feature combination"""
        
        # Primary: Proven TF-IDF features
        if fit_transform:
            tfidf_features = self.tfidf_vectorizer.fit_transform(descriptions)
        else:
            tfidf_features = self.tfidf_vectorizer.transform(descriptions)
        
        # Secondary: Selective engineered features (only if they add value)
        if fit_transform:
            engineered_features = self.feature_extractor.fit_transform(descriptions)
            engineered_features = self.scaler.fit_transform(engineered_features)
        else:
            engineered_features = self.feature_extractor.transform(descriptions)
            engineered_features = self.scaler.transform(engineered_features)
        
        # Combine with TF-IDF weighted higher (80/20 ratio)
        tfidf_array = tfidf_features.toarray()
        
        # Weight TF-IDF features higher since they're proven better
        weighted_tfidf = tfidf_array * 1.0
        weighted_engineered = engineered_features * 0.3  # Lower weight for engineered features
        
        combined_features = np.hstack([weighted_tfidf, weighted_engineered])
        
        return combined_features
    
    def train(self, X: pd.Series, y: pd.Series, validate: bool = True):
        """Train the hybrid ensemble model"""
        
        start_time = datetime.now()
        print(f"🚀 Training Hybrid Ensemble Model")
        print(f"   📊 Training data: {len(X)} samples, {y.nunique()} categories")
        
        # Split for validation
        if validate:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, y_train = X, y
            X_val = y_val = None
        
        # Prepare hybrid features
        print("\n📈 Hybrid Feature Engineering...")
        X_train_features = self.prepare_hybrid_features(X_train, fit_transform=True)
        
        print(f"   ✅ TF-IDF: {self.tfidf_vectorizer.max_features} features")
        print(f"   ✅ Engineered: 20 selective features")
        print(f"   ✅ Total: {X_train_features.shape[1]} features")
        
        # Create and train ensemble
        print("\n🤖 Training Optimized Ensemble...")
        self.ensemble_model = self._create_optimized_ensemble()
        
        model_names = [name for name, _ in self.ensemble_model.estimators]
        print(f"   Models: {' + '.join(model_names).upper()}")
        
        self.ensemble_model.fit(X_train_features, y_train)
        
        # Training accuracy
        train_pred = self.ensemble_model.predict(X_train_features)
        train_accuracy = accuracy_score(y_train, train_pred) * 100
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        results = {
            'train_accuracy': train_accuracy,
            'training_time': training_time,
            'n_features': X_train_features.shape[1],
            'n_categories': y_train.nunique(),
            'n_samples': len(X_train)
        }
        
        print(f"\n📊 Training Results:")
        print(f"   🎯 Training accuracy: {train_accuracy:.2f}%")
        print(f"   ⏱️ Training time: {training_time:.2f}s")
        
        # Validation
        if validate and X_val is not None:
            print("\n🔍 Validation Phase...")
            X_val_features = self.prepare_hybrid_features(X_val, fit_transform=False)
            val_pred = self.ensemble_model.predict(X_val_features)
            val_accuracy = accuracy_score(y_val, val_pred) * 100
            
            results['val_accuracy'] = val_accuracy
            print(f"   ✅ Validation accuracy: {val_accuracy:.2f}%")
            
            # Compare to targets
            if val_accuracy >= 60.0:
                print(f"   🎉 TARGET ACHIEVED! ({val_accuracy:.2f}% ≥ 60%)")
            elif val_accuracy >= 58.0:
                print(f"   📈 NEAR TARGET! ({val_accuracy:.2f}%)")
            elif val_accuracy >= 55.0:
                print(f"   ✅ ACCEPTABLE PERFORMANCE ({val_accuracy:.2f}%)")
            else:
                print(f"   ⚠️ BELOW BASELINE ({val_accuracy:.2f}%)")
        
        self.is_trained = True
        return results
    
    def predict(self, descriptions: pd.Series, return_probabilities: bool = False):
        """Make hybrid predictions"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Prepare features
        X_features = self.prepare_hybrid_features(descriptions, fit_transform=False)
        
        # Predict
        predictions = self.ensemble_model.predict(X_features)
        
        results = pd.DataFrame({
            'predicted_category': predictions
        }, index=descriptions.index)
        
        if return_probabilities:
            if hasattr(self.ensemble_model, 'predict_proba'):
                probabilities = self.ensemble_model.predict_proba(X_features)
                max_probs = np.max(probabilities, axis=1)
                results['confidence'] = max_probs
            else:
                results['confidence'] = 1.0  # Hard voting doesn't provide probabilities
        
        return results
    
    def save_model(self, filepath: Optional[str] = None):
        """Save the hybrid model"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f'hybrid_ensemble_model_{timestamp}.pkl'
        
        model_data = {
            'ensemble_model': self.ensemble_model,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'feature_extractor': self.feature_extractor,
            'scaler': self.scaler,
            'timestamp': datetime.now().isoformat(),
            'xgboost_available': XGBOOST_AVAILABLE
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Hybrid ensemble model saved to {filepath}")
        return filepath

def main_hybrid_demo():
    """Demonstrate the hybrid ensemble system"""
    
    print("🌟 Hybrid Advanced Categorization System")
    print("🎯 Optimized TF-IDF + Selective Features + Smart Ensemble")
    print("=" * 70)
    
    # Load data
    dataset_path = '/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.2/pipeline_run_20250907_143517/preprocessed_data/test_data.xlsx'
    
    print(f"1️⃣ Loading dataset...")
    df = pd.read_excel(dataset_path)
    
    # Prepare data
    print(f"\n2️⃣ Preparing data...")
    df_clean = df.dropna(subset=['Item_Descripton', 'Category L2']).copy()
    X = df_clean['Item_Descripton']
    y = df_clean['Category L2']
    
    print(f"   📊 Dataset: {len(X)} samples, {y.nunique()} categories")
    
    # Initialize and train hybrid ensemble
    print(f"\n3️⃣ Training Hybrid Ensemble...")
    hybrid = HybridEnsembleClassifier()
    results = hybrid.train(X, y, validate=True)
    
    # Save model
    print(f"\n4️⃣ Saving model...")
    model_path = hybrid.save_model()
    
    # Demo predictions
    print(f"\n5️⃣ Demo predictions...")
    sample_X = X.sample(5)
    predictions = hybrid.predict(sample_X, return_probabilities=True)
    
    print(f"\nSample Predictions:")
    for idx, (desc_idx, desc) in enumerate(sample_X.items()):
        pred = predictions.loc[desc_idx, 'predicted_category']
        conf = predictions.loc[desc_idx, 'confidence'] if 'confidence' in predictions.columns else 1.0
        print(f"   {idx+1}. '{str(desc)[:50]}...'")
        print(f"       → {pred} (confidence: {conf:.3f})")
    
    # Summary
    print(f"\n🎯 Hybrid Ensemble Results:")
    val_acc = results.get('val_accuracy', 0)
    print(f"   ✅ Validation Accuracy: {val_acc:.2f}%")
    print(f"   🎯 Feature Engineering: TF-IDF (1500) + Selective (20)")
    print(f"   ⏱️ Training Time: {results['training_time']:.2f}s")
    print(f"   💾 Model saved: {model_path}")
    
    # Performance comparison
    print(f"\n📊 Performance Analysis:")
    print(f"   📈 vs. Baseline (57.94%): {val_acc - 57.94:+.2f} points")
    print(f"   📈 vs. Advanced Ensemble (54.21%): {val_acc - 54.21:+.2f} points")
    
    gap_to_target = 60.0 - val_acc
    print(f"   🎯 Gap to 60% target: {gap_to_target:.2f} points")
    
    return hybrid, results

if __name__ == "__main__":
    hybrid, results = main_hybrid_demo()
    
    print(f"\n🏆 Hybrid Ensemble Implementation Complete!")
    val_acc = results.get('val_accuracy', 0)
    
    if val_acc >= 60:
        print(f"🎉 TARGET ACHIEVED! ({val_acc:.2f}% ≥ 60%)")
        print(f"🚀 Ready for production deployment!")
    elif val_acc >= 58:
        print(f"📈 NEAR TARGET PERFORMANCE! ({val_acc:.2f}%)")
        print(f"🔧 Consider final tuning for production")
    elif val_acc >= 55:
        print(f"✅ SOLID PERFORMANCE! ({val_acc:.2f}%)")
        print(f"📊 Competitive with baseline, ready for testing")
    else:
        print(f"⚠️ BELOW EXPECTATIONS ({val_acc:.2f}%)")
        print(f"🔄 May need additional optimization")
