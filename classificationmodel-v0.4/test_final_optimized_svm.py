#!/usr/bin/env python3
"""
Test the optimized production SVM with unigrams-only configuration
"""

import pandas as pd
from production_svm_model import ProductionSVMCategorizer

def test_optimized_unigram_svm():
    """Test the optimized SVM model with unigrams-only"""
    print("🚀 Testing Optimized Production SVM (Unigrams + C=0.5)")
    print("="*60)
    
    # Load data
    train_data = pd.read_excel('preprocessed_data/train_data.xlsx')
    test_data = pd.read_excel('preprocessed_data/test_data.xlsx')
    
    print(f"📊 Data loaded:")
    print(f"   Training: {len(train_data)} records")
    print(f"   Testing: {len(test_data)} records")
    
    # Train model (without hyperparameter tuning to use optimized defaults)
    model = ProductionSVMCategorizer()
    model.train(train_data, tune_hyperparameters=False)
    
    # Verify the configuration
    c_value = getattr(model.model, 'C', 'Unknown')
    ngram_range = getattr(model.vectorizer, 'ngram_range', 'Unknown')
    max_features = getattr(model.vectorizer, 'max_features', 'Unknown')
    
    print(f"\n✅ Model Configuration:")
    print(f"   C parameter: {c_value}")
    print(f"   N-gram range: {ngram_range}")
    print(f"   Max features: {max_features}")
    
    # Evaluate performance
    results = model.evaluate(test_data)
    print(f"\n✅ Performance Results:")
    print(f"   Test accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    
    # Compare with previous performance
    previous_accuracy = 0.6360  # Previous performance with bigrams
    improvement = results['accuracy'] - previous_accuracy
    improvement_pct = (improvement / previous_accuracy) * 100
    
    print(f"\n📊 Improvement Analysis:")
    print(f"   Previous (bigrams): {previous_accuracy:.4f} ({previous_accuracy*100:.2f}%)")
    print(f"   Current (unigrams): {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    if improvement > 0:
        print(f"   ✅ Improvement: +{improvement:.4f} ({improvement_pct:+.2f}%)")
    else:
        print(f"   ❌ Change: {improvement:.4f} ({improvement_pct:+.2f}%)")
    
    return model, results

def compare_with_extended_features():
    """Test with extended features (1000) for comparison"""
    print(f"\n{'-'*60}")
    print("🔍 Testing Extended Features (1000) for Comparison")
    print('-'*60)
    
    # Temporarily modify the model to use 1000 features
    train_data = pd.read_excel('preprocessed_data/train_data.xlsx')
    test_data = pd.read_excel('preprocessed_data/test_data.xlsx')
    
    # Create a custom model instance for testing
    from production_svm_model import ProductionSVMCategorizer
    
    model = ProductionSVMCategorizer()
    
    # Manually override the vectorizer configuration
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import SVC
    from sklearn.preprocessing import LabelEncoder
    import time
    
    print("📝 Preprocessing text data...")
    texts = train_data['Item_Descripton'].apply(model.preprocess_text)
    
    print("📊 Extracting TF-IDF features (1000 features)...")
    model.vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 1),  # Unigrams only
        stop_words='english',
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    
    X_train = model.vectorizer.fit_transform(texts)
    
    print("🏷️  Encoding labels...")
    model.label_encoder = LabelEncoder()
    y_train = model.label_encoder.fit_transform(train_data['Category L2'])
    
    print(f"✅ Feature matrix: {X_train.shape}")
    
    # Train SVM
    print("🎯 Training SVM with extended features...")
    start_time = time.time()
    model.model = SVC(
        kernel='linear',
        C=0.5,
        class_weight='balanced',
        random_state=42,
        probability=True
    )
    
    model.model.fit(X_train, y_train)
    training_time = time.time() - start_time
    model.is_trained = True
    
    print(f"✅ Training completed in {training_time:.2f}s")
    
    # Evaluate
    results = model.evaluate(test_data)
    print(f"✅ Test accuracy (1000 features): {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    
    return results['accuracy']

if __name__ == "__main__":
    try:
        # Test optimized model
        model, results = test_optimized_unigram_svm()
        
        # Test with extended features
        extended_accuracy = compare_with_extended_features()
        
        print(f"\n{'='*60}")
        print("🎯 FINAL COMPARISON")
        print('='*60)
        print(f"Standard (500 features): {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"Extended (1000 features): {extended_accuracy:.4f} ({extended_accuracy*100:.2f}%)")
        
        if extended_accuracy > results['accuracy']:
            diff = extended_accuracy - results['accuracy']
            print(f"✅ Extended features perform better by {diff:.4f} ({diff*100:.2f}%)")
        else:
            print(f"✅ Standard features are optimal")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
