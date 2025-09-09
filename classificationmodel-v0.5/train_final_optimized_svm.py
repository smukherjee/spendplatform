#!/usr/bin/env python3
"""
Final Optimized Enhanced SVM Model
Implementing best practices discovered from analysis
"""

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

class FinalOptimizedSVM:
    def __init__(self):
        self.vectorizer = None
        self.label_encoder = None
        self.model = None
        
    def create_optimized_dataset(self):
        """Create the most optimized dataset based on our analysis"""
        print("🔧 CREATING OPTIMIZED DATASET")
        print("=" * 50)
        
        # Load datasets
        orig_path = '/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.5/preprocessed_data/top_10_l2_categories_data.xlsx'
        combined_path = '/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.5/syntheticdata/combined_spend_data.xlsx'
        
        original_df = pd.read_excel(orig_path)
        combined_df = pd.read_excel(combined_path, sheet_name='Combined_Data')
        
        # Strategy: Focus on categories with good representation
        clean_df = combined_df[['Item_Descripton', 'Category L2']].copy()
        clean_df = clean_df.dropna()
        clean_df['Item_Descripton'] = clean_df['Item_Descripton'].astype(str).str.lower().str.strip()
        
        # Category filtering based on analysis
        category_counts = clean_df['Category L2'].value_counts()
        
        # Keep original categories (proven to work well)
        original_cats = set(original_df['Category L2'].unique())
        
        # Add well-represented new categories (>= 500 samples)
        well_represented_cats = set(category_counts[category_counts >= 500].index)
        
        # Final category set
        selected_categories = original_cats.union(well_represented_cats)
        
        # Filter to selected categories
        optimized_df = clean_df[clean_df['Category L2'].isin(selected_categories)]
        
        print(f"✅ Optimized Dataset:")
        print(f"   • Total Records: {len(optimized_df):,}")
        print(f"   • Categories: {optimized_df['Category L2'].nunique()}")
        print(f"   • Original Categories: {len(original_cats)}")
        print(f"   • New Categories: {optimized_df['Category L2'].nunique() - len(original_cats)}")
        
        # Category distribution
        final_counts = optimized_df['Category L2'].value_counts()
        print(f"\n📊 Final Category Distribution:")
        for cat, count in final_counts.head(10).items():
            is_original = "🔵" if cat in original_cats else "🟢"
            print(f"   {is_original} {cat}: {count}")
        
        return optimized_df
    
    def create_advanced_features(self, train_texts, test_texts):
        """Create advanced TF-IDF features based on our analysis"""
        print(f"\n🔤 CREATING ADVANCED FEATURES")
        print("=" * 50)
        
        # Optimized parameters from analysis
        self.vectorizer = TfidfVectorizer(
            max_features=3000,           # Increased from analysis
            ngram_range=(1, 2),          # Include bigrams
            stop_words='english',
            min_df=3,                    # Slightly higher for quality
            max_df=0.9,                  # More restrictive
            sublinear_tf=True,
            lowercase=True,
            strip_accents='ascii',
            use_idf=True,
            smooth_idf=True,
            norm='l2'
        )
        
        print(f"🎯 Advanced TF-IDF Configuration:")
        print(f"   • Max Features: 3000")
        print(f"   • N-grams: unigrams + bigrams")
        print(f"   • Min/Max DF: 3 / 0.9")
        print(f"   • Normalization: L2")
        
        # Fit and transform
        X_train = self.vectorizer.fit_transform(train_texts)
        X_test = self.vectorizer.transform(test_texts)
        
        print(f"✅ Feature matrices:")
        print(f"   • Training: {X_train.shape}")
        print(f"   • Test: {X_test.shape}")
        print(f"   • Sparsity: {1 - X_train.nnz / X_train.size:.4f}")
        
        return X_train, X_test
    
    def train_final_model(self, df):
        """Train the final optimized model"""
        print(f"\n🚀 TRAINING FINAL OPTIMIZED MODEL")
        print("=" * 50)
        
        # Split data
        train_df, test_df = train_test_split(
            df, 
            test_size=0.2, 
            random_state=42, 
            stratify=df['Category L2']
        )
        
        # Create advanced features
        X_train, X_test = self.create_advanced_features(
            train_df['Item_Descripton'].tolist(),
            test_df['Item_Descripton'].tolist()
        )
        
        # Encode labels
        print(f"\n🏷️ ENCODING LABELS")
        self.label_encoder = LabelEncoder()
        y_train = self.label_encoder.fit_transform(train_df['Category L2'])
        y_test = self.label_encoder.transform(test_df['Category L2'])
        
        print(f"✅ Encoded {len(self.label_encoder.classes_)} categories")
        
        # Train ensemble of optimized SVMs
        print(f"\n🎯 Training Optimized SVM Ensemble...")
        start_time = time.time()
        
        # Best performing SVM configurations from analysis
        svm1 = SVC(
            kernel='rbf',
            C=2.0,
            gamma='scale',
            random_state=42,
            probability=True  # For ensemble
        )
        
        svm2 = SVC(
            kernel='rbf',
            C=5.0,              # Slightly different C
            gamma='auto',       # Different gamma
            random_state=43,
            probability=True
        )
        
        svm3 = SVC(
            kernel='linear',    # Different kernel
            C=1.0,
            random_state=44,
            probability=True
        )
        
        # Create ensemble
        self.model = VotingClassifier(
            estimators=[
                ('rbf_scale', svm1),
                ('rbf_auto', svm2),
                ('linear', svm3)
            ],
            voting='soft'  # Use probabilities
        )
        
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"✅ Final Model Results:")
        print(f"   • Model Type: SVM Ensemble (RBF + Linear)")
        print(f"   • Training Time: {training_time:.2f}s")
        print(f"   • Training Accuracy: {train_score:.4f} ({train_score*100:.2f}%)")
        print(f"   • Test Accuracy: {test_score:.4f} ({test_score*100:.2f}%)")
        print(f"   • Categories: {len(self.label_encoder.classes_)}")
        print(f"   • Features: {X_train.shape[1]:,}")
        
        # Cross-validation
        print(f"\n🔄 Cross-Validation (5-fold)...")
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"   • CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"   • CV Scores: {[f'{s:.3f}' for s in cv_scores]}")
        
        # Detailed evaluation
        y_pred = self.model.predict(X_test)
        report = classification_report(
            y_test, 
            y_pred, 
            target_names=self.label_encoder.classes_, 
            output_dict=True
        )
        
        print(f"\n📊 DETAILED PERFORMANCE:")
        print(f"   • Accuracy: {report['accuracy']:.4f}")
        print(f"   • Macro Avg F1: {report['macro avg']['f1-score']:.4f}")
        print(f"   • Weighted Avg F1: {report['weighted avg']['f1-score']:.4f}")
        
        return {
            'model': self.model,
            'accuracy': test_score,
            'cv_accuracy': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_time': training_time,
            'categories': len(self.label_encoder.classes_),
            'features': X_train.shape[1],
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'report': report
        }
    
    def save_final_model(self, results):
        """Save the final optimized model"""
        print(f"\n💾 SAVING FINAL OPTIMIZED MODEL")
        print("=" * 50)
        
        model_package = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'accuracy': results['accuracy'],
            'cv_accuracy': results['cv_accuracy'],
            'cv_std': results['cv_std'],
            'categories': results['categories'],
            'features': results['features'],
            'training_samples': results['train_samples'],
            'model_type': 'Optimized SVM Ensemble',
            'optimization_version': 'v2.0',
            'training_date': '2025-09-09',
            'performance_report': results['report']
        }
        
        # Save multiple versions
        base_path = '/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.5/models'
        
        # Production model
        prod_path = f'{base_path}/production_svm_enhanced.pkl'
        with open(prod_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        # Versioned backup
        version_path = f'{base_path}/svm_enhanced_v2.0.pkl'
        with open(version_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"✅ Saved models:")
        print(f"   • Production: production_svm_enhanced.pkl")
        print(f"   • Versioned: svm_enhanced_v2.0.pkl")
        print(f"📊 Model specifications:")
        print(f"   • Type: SVM Ensemble (RBF + Linear)")
        print(f"   • Accuracy: {results['accuracy']:.4f}")
        print(f"   • Categories: {results['categories']}")
        print(f"   • Features: {results['features']:,}")
        
        return prod_path, version_path

def main():
    """Main function for final optimized training"""
    print("🎯 FINAL OPTIMIZED SVM TRAINING")
    print("=" * 70)
    
    trainer = FinalOptimizedSVM()
    
    # Create optimized dataset
    optimized_df = trainer.create_optimized_dataset()
    
    # Train final model
    results = trainer.train_final_model(optimized_df)
    
    # Save model
    prod_path, version_path = trainer.save_final_model(results)
    
    # Final summary
    print(f"\n" + "=" * 70)
    print(f"🎉 FINAL OPTIMIZED TRAINING COMPLETE!")
    print(f"✅ Model Performance:")
    print(f"   • Test Accuracy: {results['accuracy']*100:.2f}%")
    print(f"   • CV Accuracy: {results['cv_accuracy']*100:.2f}% ± {results['cv_std']*100:.2f}%")
    print(f"   • Categories: {results['categories']}")
    print(f"   • Features: {results['features']:,}")
    print(f"💾 Saved Models:")
    print(f"   • Production: {prod_path.split('/')[-1]}")
    print(f"   • Version: {version_path.split('/')[-1]}")
    print(f"🚀 Ready for production deployment!")
    print("=" * 70)
    
    return results

if __name__ == "__main__":
    final_results = main()
