#!/usr/bin/env python3
"""
Advanced Ensemble Categorization System
Combines Random Forest + XGBoost + SVM with 50+ engineered features
"""

import pandas as pd
import numpy as np
import pickle
import re
import warnings
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter

# Core ML imports
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️ XGBoost not available, will use alternative")
    XGBOOST_AVAILABLE = False

# NLP features
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

warnings.filterwarnings('ignore')

class AdvancedFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract 50+ advanced features for categorization"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Domain-specific patterns
        self.electrical_patterns = [
            r'\b\d+v\b', r'\b\d+volt\b', r'\bamp\b', r'\bwatt\b', r'\bcurrent\b',
            r'\bvoltage\b', r'\belectric\b', r'\bwire\b', r'\bcable\b', r'\bswitch\b'
        ]
        
        self.mechanical_patterns = [
            r'\b\d+mm\b', r'\b\d+cm\b', r'\b\d+inch\b', r'\bmetal\b', r'\bsteel\b',
            r'\baluminum\b', r'\bgear\b', r'\bbearing\b', r'\bbolt\b', r'\bnut\b'
        ]
        
        self.manufacturing_patterns = [
            r'\bmachine\b', r'\btool\b', r'\bequipment\b', r'\bproduction\b',
            r'\bassembly\b', r'\bmanufactur\b', r'\bindustrial\b', r'\bprocess\b'
        ]
        
        self.safety_patterns = [
            r'\bsafety\b', r'\bprotection\b', r'\bguard\b', r'\bhelmet\b',
            r'\bglove\b', r'\bppe\b', r'\bemergency\b', r'\bfirst aid\b'
        ]
        
        # Technical measurement patterns
        self.measurement_patterns = [
            r'\b\d+x\d+\b', r'\b\d+\.\d+\b', r'\b\d+/\d+\b',
            r'\b\d+°\b', r'\b\d+deg\b', r'\b\d+rpm\b'
        ]
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Extract all advanced features"""
        print("🔧 Extracting 50+ advanced features...")
        
        features = []
        for text in X:
            text_str = str(text).lower() if pd.notna(text) else ""
            feature_vector = self._extract_single_features(text_str)
            features.append(feature_vector)
        
        feature_matrix = np.array(features)
        print(f"   ✅ Extracted {feature_matrix.shape[1]} features")
        return feature_matrix
    
    def _extract_single_features(self, text: str) -> List[float]:
        """Extract features for a single text"""
        features = []
        
        # 1. Basic Text Statistics (10 features)
        features.extend(self._get_basic_stats(text))
        
        # 2. Lexical Features (10 features)
        features.extend(self._get_lexical_features(text))
        
        # 3. Syntactic Features (8 features)
        features.extend(self._get_syntactic_features(text))
        
        # 4. Semantic Features (8 features)
        features.extend(self._get_semantic_features(text))
        
        # 5. Domain-Specific Features (15 features)
        features.extend(self._get_domain_features(text))
        
        # 6. Advanced NLP Features (5 features)
        features.extend(self._get_nlp_features(text))
        
        return features
    
    def _get_basic_stats(self, text: str) -> List[float]:
        """Basic text statistics (10 features)"""
        if not text:
            return [0] * 10
        
        words = text.split()
        chars = list(text)
        
        return [
            len(text),                          # 1. Character count
            len(words),                         # 2. Word count
            len([w for w in words if len(w) > 6]),  # 3. Long words
            len(set(words)),                    # 4. Unique words
            len(set(words)) / max(len(words), 1),   # 5. Lexical diversity
            sum(len(w) for w in words) / max(len(words), 1),  # 6. Avg word length
            len([c for c in chars if c.isdigit()]),  # 7. Digit count
            len([c for c in chars if c.isupper()]),  # 8. Uppercase count
            text.count(' '),                    # 9. Space count
            len([w for w in words if w.isalpha()])  # 10. Alpha words
        ]
    
    def _get_lexical_features(self, text: str) -> List[float]:
        """Lexical diversity features (10 features)"""
        if not text:
            return [0] * 10
        
        words = word_tokenize(text)
        words_clean = [w.lower() for w in words if w.isalpha()]
        
        # Stemmed and lemmatized versions
        stemmed = [self.stemmer.stem(w) for w in words_clean]
        lemmatized = [self.lemmatizer.lemmatize(w) for w in words_clean]
        
        # Stop words
        stop_word_count = len([w for w in words_clean if w in self.stop_words])
        content_words = [w for w in words_clean if w not in self.stop_words]
        
        return [
            len(words_clean),                   # 1. Clean word count
            len(set(words_clean)),             # 2. Unique clean words
            len(set(stemmed)),                 # 3. Unique stems
            len(set(lemmatized)),              # 4. Unique lemmas
            stop_word_count,                   # 5. Stop word count
            len(content_words),                # 6. Content word count
            len([w for w in words_clean if len(w) == 1]),  # 7. Single char words
            len([w for w in words_clean if len(w) >= 10]), # 8. Very long words
            max([len(w) for w in words_clean] + [0]),      # 9. Max word length
            min([len(w) for w in words_clean] + [100])     # 10. Min word length
        ]
    
    def _get_syntactic_features(self, text: str) -> List[float]:
        """Syntactic pattern features (8 features)"""
        if not text:
            return [0] * 8
        
        return [
            len(re.findall(r'[.!?]', text)),    # 1. Sentence endings
            len(re.findall(r'[,;:]', text)),    # 2. Punctuation
            len(re.findall(r'[-_]', text)),     # 3. Separators
            len(re.findall(r'[()]', text)),     # 4. Brackets
            len(re.findall(r'\d+', text)),      # 5. Number sequences
            len(re.findall(r'[A-Z]{2,}', text)), # 6. Abbreviations
            len(re.findall(r'\b\w+ed\b', text)), # 7. Past tense words
            len(re.findall(r'\b\w+ing\b', text)) # 8. Present participle
        ]
    
    def _get_semantic_features(self, text: str) -> List[float]:
        """Semantic content features (8 features)"""
        if not text:
            return [0] * 8
        
        words = word_tokenize(text.lower())
        
        try:
            # POS tagging
            pos_tags = pos_tag(words)
            pos_counts = Counter([tag for word, tag in pos_tags])
            
            return [
                pos_counts.get('NN', 0) + pos_counts.get('NNS', 0),    # 1. Nouns
                pos_counts.get('VB', 0) + pos_counts.get('VBG', 0),    # 2. Verbs
                pos_counts.get('JJ', 0) + pos_counts.get('JJR', 0),    # 3. Adjectives
                pos_counts.get('RB', 0) + pos_counts.get('RBR', 0),    # 4. Adverbs
                pos_counts.get('IN', 0),                                # 5. Prepositions
                pos_counts.get('DT', 0),                                # 6. Determiners
                pos_counts.get('CD', 0),                                # 7. Numbers
                len([w for w in words if w.startswith('un')])          # 8. Negative prefix
            ]
        except:
            return [0] * 8
    
    def _get_domain_features(self, text: str) -> List[float]:
        """Domain-specific features (15 features)"""
        if not text:
            return [0] * 15
        
        features = []
        
        # Pattern matching for different domains
        for patterns in [self.electrical_patterns, self.mechanical_patterns, 
                        self.manufacturing_patterns, self.safety_patterns]:
            count = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                       for pattern in patterns)
            features.append(count)
        
        # Measurement patterns
        features.append(len(re.findall(r'\b\d+x\d+x\d+\b', text)))  # 5. 3D measurements
        features.append(len(re.findall(r'\b\d+mm\b', text)))        # 6. Millimeters
        features.append(len(re.findall(r'\b\d+inch\b', text)))      # 7. Inches
        features.append(len(re.findall(r'\b\d+kg\b', text)))        # 8. Weight
        features.append(len(re.findall(r'\b\d+°c\b', text)))        # 9. Temperature
        
        # Material indicators
        materials = ['steel', 'aluminum', 'plastic', 'rubber', 'copper', 'iron']
        features.append(sum(text.lower().count(mat) for mat in materials))  # 10. Materials
        
        # Brand/manufacturer indicators
        features.append(len(re.findall(r'\bmake\b', text)))         # 11. Make indicator
        features.append(len(re.findall(r'\bmodel\b', text)))        # 12. Model indicator
        features.append(len(re.findall(r'\btype\b', text)))         # 13. Type indicator
        features.append(len(re.findall(r'\bgrade\b', text)))        # 14. Grade indicator
        features.append(len(re.findall(r'\bsize\b', text)))         # 15. Size indicator
        
        return features
    
    def _get_nlp_features(self, text: str) -> List[float]:
        """Advanced NLP features (5 features)"""
        if not text:
            return [0] * 5
        
        words = word_tokenize(text.lower())
        
        try:
            # Named entity recognition
            pos_tags = pos_tag(words)
            tree = ne_chunk(pos_tags)
            entities = [chunk.label() for chunk in tree if hasattr(chunk, 'label')]
            
            return [
                len(entities),                              # 1. Named entities
                len([e for e in entities if e == 'ORGANIZATION']),  # 2. Organizations
                len([e for e in entities if e == 'PERSON']),        # 3. Persons
                text.lower().count('and') + text.lower().count('or'), # 4. Conjunctions
                len([w for w in words if w.endswith('ly')])         # 5. Adverb endings
            ]
        except:
            return [0] * 5

class EnsembleClassifier:
    """Advanced ensemble classifier with RF + XGBoost + SVM"""
    
    def __init__(self):
        self.feature_extractor = AdvancedFeatureExtractor()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8,
            sublinear_tf=True
        )
        self.scaler = StandardScaler()
        self.ensemble_model = None
        self.is_trained = False
        
    def _create_ensemble(self):
        """Create the ensemble voting classifier"""
        
        # Random Forest
        rf_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # SVM
        svm_classifier = SVC(
            kernel='rbf',
            C=1.0,
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
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='mlogloss'
            )
            estimators.append(('xgb', xgb_classifier))
            print("✅ XGBoost included in ensemble")
        else:
            print("⚠️ XGBoost not available, using RF + SVM ensemble")
        
        # Create voting classifier
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Use probability-based voting
            n_jobs=-1
        )
        
        return ensemble
    
    def prepare_features(self, descriptions: pd.Series, fit_transform: bool = True):
        """Prepare combined features"""
        print(f"🔧 Preparing features for {len(descriptions)} samples...")
        
        # TF-IDF features
        if fit_transform:
            tfidf_features = self.tfidf_vectorizer.fit_transform(descriptions)
        else:
            tfidf_features = self.tfidf_vectorizer.transform(descriptions)
        
        print(f"   📊 TF-IDF features: {tfidf_features.shape[1]}")
        
        # Advanced engineered features
        if fit_transform:
            engineered_features = self.feature_extractor.fit_transform(descriptions)
            # Fit scaler on engineered features
            engineered_features = self.scaler.fit_transform(engineered_features)
        else:
            engineered_features = self.feature_extractor.transform(descriptions)
            engineered_features = self.scaler.transform(engineered_features)
        
        print(f"   🎯 Engineered features: {engineered_features.shape[1]}")
        
        # Combine features
        combined_features = np.hstack([
            tfidf_features.toarray(),
            engineered_features
        ])
        
        print(f"   ✅ Total features: {combined_features.shape[1]}")
        return combined_features
    
    def train(self, X: pd.Series, y: pd.Series, validate: bool = True):
        """Train the ensemble model"""
        
        start_time = datetime.now()
        print(f"🚀 Training Advanced Ensemble Model")
        print(f"   📊 Training data: {len(X)} samples, {y.nunique()} categories")
        
        # Split for validation
        if validate:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, y_train = X, y
            X_val = y_val = None
        
        # Prepare features
        print("\n📈 Feature Engineering Phase...")
        X_train_features = self.prepare_features(X_train, fit_transform=True)
        
        # Create and train ensemble
        print("\n🤖 Ensemble Training Phase...")
        self.ensemble_model = self._create_ensemble()
        
        print("   Training Random Forest...")
        print("   Training SVM...")
        if XGBOOST_AVAILABLE:
            print("   Training XGBoost...")
        print("   Combining models with soft voting...")
        
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
        print(f"   📈 Features used: {X_train_features.shape[1]}")
        
        # Validation
        if validate and X_val is not None:
            print("\n🔍 Validation Phase...")
            X_val_features = self.prepare_features(X_val, fit_transform=False)
            val_pred = self.ensemble_model.predict(X_val_features)
            val_accuracy = accuracy_score(y_val, val_pred) * 100
            
            results['val_accuracy'] = val_accuracy
            print(f"   ✅ Validation accuracy: {val_accuracy:.2f}%")
            
            # Individual model performance
            print(f"\n🔍 Individual Model Performance:")
            try:
                for name in self.ensemble_model.named_estimators_:
                    model = self.ensemble_model.named_estimators_[name]
                    individual_pred = model.predict(X_val_features)
                    individual_acc = accuracy_score(y_val, individual_pred) * 100
                    print(f"   {name.upper()}: {individual_acc:.2f}%")
            except Exception as e:
                print(f"   ⚠️ Could not analyze individual models: {e}")
        
        self.is_trained = True
        return results
    
    def predict(self, descriptions: pd.Series, return_probabilities: bool = False):
        """Make predictions"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        print(f"🔮 Making ensemble predictions on {len(descriptions)} samples...")
        
        # Prepare features
        X_features = self.prepare_features(descriptions, fit_transform=False)
        
        # Predict
        predictions = self.ensemble_model.predict(X_features)
        
        results = pd.DataFrame({
            'predicted_category': predictions
        }, index=descriptions.index)
        
        if return_probabilities:
            probabilities = self.ensemble_model.predict_proba(X_features)
            max_probs = np.max(probabilities, axis=1)
            results['confidence'] = max_probs
            
            # Individual model predictions for analysis
            try:
                for name in self.ensemble_model.named_estimators_:
                    model = self.ensemble_model.named_estimators_[name]
                    model_pred = model.predict(X_features)
                    results[f'{name}_prediction'] = model_pred
            except Exception as e:
                print(f"   ⚠️ Could not get individual predictions: {e}")
        
        print(f"   ✅ Ensemble predictions complete")
        return results
    
    def save_model(self, filepath: str = None):
        """Save the trained ensemble model"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f'advanced_ensemble_model_{timestamp}.pkl'
        
        model_data = {
            'ensemble_model': self.ensemble_model,
            'feature_extractor': self.feature_extractor,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'scaler': self.scaler,
            'timestamp': datetime.now().isoformat(),
            'xgboost_available': XGBOOST_AVAILABLE
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Advanced ensemble model saved to {filepath}")
        return filepath

def main_advanced_demo():
    """Demonstrate the advanced ensemble system"""
    
    print("🌟 Advanced Ensemble Categorization System")
    print("🤖 Random Forest + XGBoost + SVM with 50+ Features")
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
    
    # Initialize and train ensemble
    print(f"\n3️⃣ Training Advanced Ensemble...")
    ensemble = EnsembleClassifier()
    results = ensemble.train(X, y, validate=True)
    
    # Save model
    print(f"\n4️⃣ Saving model...")
    model_path = ensemble.save_model()
    
    # Demo predictions
    print(f"\n5️⃣ Demo predictions...")
    sample_X = X.sample(5)
    predictions = ensemble.predict(sample_X, return_probabilities=True)
    
    print(f"\nSample Predictions:")
    for idx, (desc_idx, desc) in enumerate(sample_X.items()):
        pred_row = predictions.loc[desc_idx]
        conf = pred_row.get('confidence', 0.0)
        print(f"   {idx+1}. '{str(desc)[:50]}...'")
        print(f"       → {pred_row['predicted_category']} (confidence: {conf:.3f})")
    
    # Summary
    print(f"\n🎯 Advanced Ensemble Results:")
    print(f"   ✅ Validation Accuracy: {results.get('val_accuracy', 0):.2f}%")
    print(f"   🎯 Total Features: {results['n_features']}")
    print(f"   ⏱️ Training Time: {results['training_time']:.2f}s")
    print(f"   🤖 Models: RF + SVM" + (" + XGBoost" if XGBOOST_AVAILABLE else ""))
    print(f"   💾 Model saved: {model_path}")
    
    return ensemble, results

if __name__ == "__main__":
    ensemble, results = main_advanced_demo()
    
    print(f"\n🏆 Advanced Ensemble Implementation Complete!")
    val_acc = results.get('val_accuracy', 0)
    if val_acc > 60:
        print(f"🎉 TARGET EXCEEDED! ({val_acc:.2f}% > 60%)")
    elif val_acc > 55:
        print(f"📈 STRONG PERFORMANCE! ({val_acc:.2f}%)")
    else:
        print(f"🔧 BASELINE PERFORMANCE ({val_acc:.2f}%)")
