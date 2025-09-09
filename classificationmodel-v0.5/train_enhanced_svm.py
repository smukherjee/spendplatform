#!/usr/bin/env python3
"""
Enhanced SVM Training on Combined Dataset (Original + Synthetic)
Evaluate performance improvement with expanded data and categories
"""

from typing import Optional, Dict, Any, Tuple, List
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

class EnhancedSVMTrainer:
    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.model: Optional[SVC] = None
        self.training_stats: Dict[str, Any] = {}
        
    def load_datasets(self):
        """Load original and combined datasets for comparison"""
        print("📊 LOADING DATASETS")
        print("=" * 50)
        
        # Load original dataset
        orig_path = '/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.5/preprocessed_data/top_10_l2_categories_data.xlsx'
        self.original_df = pd.read_excel(orig_path)
        
        # Load combined dataset (original + synthetic)
        combined_path = '/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.5/syntheticdata/combined_spend_data.xlsx'
        self.combined_df = pd.read_excel(combined_path, sheet_name='Combined_Data')
        
        print(f"✅ Original Dataset: {len(self.original_df):,} records, {self.original_df['Category L2'].nunique()} categories")
        print(f"✅ Combined Dataset: {len(self.combined_df):,} records, {self.combined_df['Category L2'].nunique()} categories")
        print(f"📈 Enhancement: {len(self.combined_df)/len(self.original_df):.1f}x records, {self.combined_df['Category L2'].nunique()/self.original_df['Category L2'].nunique():.1f}x categories")
        
        return self.original_df, self.combined_df
    
    def preprocess_data(self, df, is_combined=False):
        """Preprocess data for training"""
        print(f"\n🔧 PREPROCESSING {'COMBINED' if is_combined else 'ORIGINAL'} DATASET")
        print("=" * 50)
        
        # Clean data
        clean_df = df[['Item_Descripton', 'Category L2']].copy()
        clean_df = clean_df.dropna()
        
        # Simple text preprocessing
        clean_df['Item_Descripton'] = clean_df['Item_Descripton'].astype(str).str.lower().str.strip()
        
        print(f"✅ Cleaned data: {len(clean_df):,} records")
        print(f"📊 Categories: {clean_df['Category L2'].nunique()}")
        
        # Show category distribution
        category_counts = clean_df['Category L2'].value_counts()
        print(f"\n📋 Top 10 categories by count:")
        for cat, count in category_counts.head(10).items():
            print(f"   {cat}: {count}")
        
        # Check for categories with very few samples
        min_samples = 5
        low_count_cats = category_counts[category_counts < min_samples]
        if len(low_count_cats) > 0:
            print(f"\n⚠️ Categories with < {min_samples} samples (will be removed):")
            for cat, count in low_count_cats.items():
                print(f"   {cat}: {count}")
            
            # Filter out low-count categories
            valid_categories = category_counts[category_counts >= min_samples].index
            clean_df = clean_df[clean_df['Category L2'].isin(valid_categories)]
            print(f"✅ Filtered to {len(clean_df):,} records with {clean_df['Category L2'].nunique()} categories")
        
        return clean_df
    
    def create_optimized_features(self, train_texts, test_texts=None):
        """Create optimized TF-IDF features based on previous optimization results"""
        print(f"\n🔤 CREATING OPTIMIZED TF-IDF FEATURES")
        print("=" * 50)
        
        # Use optimized parameters from previous experiments
        self.vectorizer = TfidfVectorizer(
            max_features=1000,          # Optimal from previous testing
            ngram_range=(1, 1),         # Unigrams performed best
            stop_words='english',       # Remove common words
            min_df=2,                   # Minimum document frequency
            max_df=0.95,               # Maximum document frequency  
            sublinear_tf=True,         # Apply sublinear tf scaling
            lowercase=True,            # Convert to lowercase
            strip_accents='ascii'      # Remove accents
        )
        
        print(f"🎯 TF-IDF Configuration:")
        print(f"   Max Features: 1000")
        print(f"   N-grams: unigrams only")
        print(f"   Stop Words: English")
        print(f"   Min/Max DF: 2 / 0.95")
        
        # Fit and transform training data
        X_train = self.vectorizer.fit_transform(train_texts)
        print(f"✅ Training features: {X_train.shape}")
        
        # Transform test data if provided
        X_test = None
        if test_texts is not None:
            X_test = self.vectorizer.transform(test_texts)
            print(f"✅ Test features: {X_test.shape}")
        
        return X_train, X_test
    
    def encode_labels(self, train_labels, test_labels=None):
        """Encode category labels"""
        print(f"\n🏷️ ENCODING LABELS")
        print("=" * 50)
        
        self.label_encoder = LabelEncoder()
        y_train = self.label_encoder.fit_transform(train_labels)
        
        print(f"✅ Encoded {len(self.label_encoder.classes_)} unique categories")
        print(f"📊 Label distribution: {np.bincount(y_train)[:10]}...")  # Show first 10
        
        y_test = None
        if test_labels is not None:
            y_test = self.label_encoder.transform(test_labels)
            
        return y_train, y_test
    
    def train_baseline_model(self, df):
        """Train baseline model for comparison"""
        print(f"\n🔥 TRAINING BASELINE MODEL")
        print("=" * 50)
        
        # Split data
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Category L2'])
        
        # Create features
        X_train, X_test = self.create_optimized_features(
            train_df['Item_Descripton'].tolist(),
            test_df['Item_Descripton'].tolist()
        )
        
        # Encode labels
        y_train, y_test = self.encode_labels(
            train_df['Category L2'].tolist(),
            test_df['Category L2'].tolist()
        )
        
        # Train optimized RBF SVM (from previous optimization)
        print(f"\n🎯 Training RBF SVM with optimized parameters...")
        start_time = time.time()
        
        baseline_model = SVC(
            kernel='rbf',
            C=2.0,                    # Optimized from previous experiments
            gamma='scale',            # Optimized gamma setting
            random_state=42
        )
        
        baseline_model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate
        train_score = baseline_model.score(X_train, y_train)
        test_score = baseline_model.score(X_test, y_test)
        
        print(f"✅ Baseline Model Results:")
        print(f"   Training Time: {training_time:.2f}s")
        print(f"   Training Accuracy: {train_score:.4f} ({train_score*100:.2f}%)")
        print(f"   Test Accuracy: {test_score:.4f} ({test_score*100:.2f}%)")
        print(f"   Categories: {len(self.label_encoder.classes_)}")
        print(f"   Training Samples: {len(train_df):,}")
        print(f"   Test Samples: {len(test_df):,}")
        
        return {
            'model': baseline_model,
            'accuracy': test_score,
            'training_time': training_time,
            'categories': len(self.label_encoder.classes_),
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'X_test': X_test,
            'y_test': y_test,
            'test_df': test_df
        }
    
    def train_enhanced_model(self, df):
        """Train enhanced model on combined dataset"""
        print(f"\n🚀 TRAINING ENHANCED MODEL")
        print("=" * 50)
        
        # Split data
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Category L2'])
        
        # Create features
        X_train, X_test = self.create_optimized_features(
            train_df['Item_Descripton'].tolist(),
            test_df['Item_Descripton'].tolist()
        )
        
        # Encode labels
        y_train, y_test = self.encode_labels(
            train_df['Category L2'].tolist(),
            test_df['Category L2'].tolist()
        )
        
        # Train enhanced RBF SVM
        print(f"\n🎯 Training Enhanced RBF SVM...")
        start_time = time.time()
        
        self.model = SVC(
            kernel='rbf',
            C=2.0,                    # Keep optimized parameters
            gamma='scale',
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"✅ Enhanced Model Results:")
        print(f"   Training Time: {training_time:.2f}s")
        print(f"   Training Accuracy: {train_score:.4f} ({train_score*100:.2f}%)")
        print(f"   Test Accuracy: {test_score:.4f} ({test_score*100:.2f}%)")
        print(f"   Categories: {len(self.label_encoder.classes_)}")
        print(f"   Training Samples: {len(train_df):,}")
        print(f"   Test Samples: {len(test_df):,}")
        
        # Cross-validation for robust evaluation
        print(f"\n🔄 Cross-Validation (5-fold)...")
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"   CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return {
            'model': self.model,
            'accuracy': test_score,
            'cv_accuracy': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_time': training_time,
            'categories': len(self.label_encoder.classes_),
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'X_test': X_test,
            'y_test': y_test,
            'test_df': test_df
        }
    
    def detailed_evaluation(self, model_results, model_name):
        """Perform detailed evaluation of model"""
        print(f"\n📊 DETAILED EVALUATION: {model_name}")
        print("=" * 60)
        
        model = model_results['model']
        X_test = model_results['X_test']
        y_test = model_results['y_test']
        test_df = model_results['test_df']
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Classification report
        target_names = self.label_encoder.classes_
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        
        print(f"📋 Performance Summary:")
        print(f"   Accuracy: {report['accuracy']:.4f} ({report['accuracy']*100:.2f}%)")
        print(f"   Macro Avg Precision: {report['macro avg']['precision']:.4f}")
        print(f"   Macro Avg Recall: {report['macro avg']['recall']:.4f}")
        print(f"   Macro Avg F1-Score: {report['macro avg']['f1-score']:.4f}")
        
        # Top performing categories
        category_f1 = [(cat, report[cat]['f1-score']) for cat in target_names if cat in report]
        category_f1.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 Top 10 Performing Categories (F1-Score):")
        for cat, f1 in category_f1[:10]:
            print(f"   {cat}: {f1:.3f}")
        
        # Challenging categories
        print(f"\n⚠️ Most Challenging Categories (F1-Score):")
        for cat, f1 in category_f1[-5:]:
            support = report[cat]['support'] if cat in report else 0
            print(f"   {cat}: {f1:.3f} (support: {support})")
        
        # Sample predictions
        print(f"\n🔮 Sample Predictions:")
        sample_indices = np.random.choice(len(test_df), min(8, len(test_df)), replace=False)
        
        for idx in sample_indices:
            true_cat = test_df.iloc[idx]['Category L2']
            pred_cat = self.label_encoder.inverse_transform([y_pred[idx]])[0]
            desc = test_df.iloc[idx]['Item_Descripton'][:50]
            correct = "✅" if true_cat == pred_cat else "❌"
            
            print(f"   {correct} '{desc:<50}' → {pred_cat}")
            if true_cat != pred_cat:
                print(f"      (True: {true_cat})")
        
        return report
    
    def compare_models(self, baseline_results, enhanced_results):
        """Compare baseline vs enhanced model performance"""
        print(f"\n📈 MODEL COMPARISON")
        print("=" * 60)
        
        print(f"📊 Performance Comparison:")
        print(f"   Baseline Accuracy:  {baseline_results['accuracy']:.4f} ({baseline_results['accuracy']*100:.2f}%)")
        print(f"   Enhanced Accuracy:  {enhanced_results['accuracy']:.4f} ({enhanced_results['accuracy']*100:.2f}%)")
        
        accuracy_improvement = enhanced_results['accuracy'] - baseline_results['accuracy']
        print(f"   Improvement:        {accuracy_improvement:.4f} ({accuracy_improvement*100:.2f} percentage points)")
        
        print(f"\n📊 Scale Comparison:")
        print(f"   Baseline Categories: {baseline_results['categories']}")
        print(f"   Enhanced Categories: {enhanced_results['categories']}")
        print(f"   Category Expansion:  {enhanced_results['categories']/baseline_results['categories']:.1f}x")
        
        print(f"\n📊 Data Volume Comparison:")
        print(f"   Baseline Training:   {baseline_results['train_samples']:,} samples")
        print(f"   Enhanced Training:   {enhanced_results['train_samples']:,} samples")
        print(f"   Data Expansion:      {enhanced_results['train_samples']/baseline_results['train_samples']:.1f}x")
        
        print(f"\n⏱️ Training Time Comparison:")
        print(f"   Baseline Time:       {baseline_results['training_time']:.2f}s")
        print(f"   Enhanced Time:       {enhanced_results['training_time']:.2f}s")
        time_ratio = enhanced_results['training_time'] / baseline_results['training_time']
        print(f"   Time Ratio:          {time_ratio:.1f}x")
        
        # Performance per category
        baseline_perf_per_cat = baseline_results['accuracy'] / baseline_results['categories']
        enhanced_perf_per_cat = enhanced_results['accuracy'] / enhanced_results['categories']
        
        print(f"\n🎯 Efficiency Metrics:")
        print(f"   Baseline Performance/Category: {baseline_perf_per_cat:.6f}")
        print(f"   Enhanced Performance/Category: {enhanced_perf_per_cat:.6f}")
        
        if accuracy_improvement > 0:
            print(f"\n✅ ENHANCED MODEL OUTPERFORMS BASELINE!")
        elif accuracy_improvement < -0.02:  # More than 2% drop
            print(f"\n⚠️ Enhanced model shows significant performance drop")
        else:
            print(f"\n📊 Models show comparable performance")
    
    def save_enhanced_model(self, model_results, filename='enhanced_svm_model.pkl'):
        """Save the enhanced model"""
        output_path = f'/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.5/models/{filename}'
        
        print(f"\n💾 SAVING ENHANCED MODEL")
        print("=" * 50)
        
        model_package = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'accuracy': model_results['accuracy'],
            'cv_accuracy': model_results['cv_accuracy'],
            'categories': model_results['categories'],
            'training_samples': model_results['train_samples'],
            'feature_count': self.vectorizer.max_features,
            'model_type': 'Enhanced RBF SVM',
            'training_date': '2025-09-09'
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"✅ Saved enhanced model: {filename}")
        print(f"📁 Path: {output_path}")
        print(f"🎯 Model Details:")
        print(f"   Type: Enhanced RBF SVM")
        print(f"   Accuracy: {model_results['accuracy']:.4f}")
        print(f"   Categories: {model_results['categories']}")
        print(f"   Features: {self.vectorizer.max_features}")
        
        return output_path

def main():
    """Main training and evaluation function"""
    print("🚀 ENHANCED SVM TRAINING & EVALUATION")
    print("=" * 70)
    
    trainer = EnhancedSVMTrainer()
    
    # Load datasets
    original_df, combined_df = trainer.load_datasets()
    
    # Preprocess datasets
    original_clean = trainer.preprocess_data(original_df, is_combined=False)
    combined_clean = trainer.preprocess_data(combined_df, is_combined=True)
    
    # Train baseline model (original data only)
    baseline_results = trainer.train_baseline_model(original_clean)
    baseline_report = trainer.detailed_evaluation(baseline_results, "BASELINE MODEL")
    
    # Train enhanced model (combined data)
    enhanced_results = trainer.train_enhanced_model(combined_clean)
    enhanced_report = trainer.detailed_evaluation(enhanced_results, "ENHANCED MODEL")
    
    # Compare models
    trainer.compare_models(baseline_results, enhanced_results)
    
    # Save enhanced model
    model_path = trainer.save_enhanced_model(enhanced_results)
    
    print(f"\n" + "=" * 70)
    print(f"🎉 ENHANCED SVM TRAINING COMPLETE!")
    print(f"✅ Baseline Model: {baseline_results['accuracy']*100:.2f}% accuracy ({baseline_results['categories']} categories)")
    print(f"✅ Enhanced Model: {enhanced_results['accuracy']*100:.2f}% accuracy ({enhanced_results['categories']} categories)")
    improvement = enhanced_results['accuracy'] - baseline_results['accuracy']
    print(f"📈 Improvement: {improvement*100:.2f} percentage points")
    print(f"💾 Saved Model: {model_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
