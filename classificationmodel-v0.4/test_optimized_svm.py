#!/usr/bin/env python3
"""
Quick test to verify the optimized SVM model with C=0.5
"""

import pandas as pd
from production_svm_model import ProductionSVMCategorizer

def test_optimized_svm():
    """Test the optimized SVM model with C=0.5"""
    print("🚀 Testing Optimized Production SVM (C=0.5)")
    print("="*50)
    
    # Load data
    train_data = pd.read_excel('preprocessed_data/train_data.xlsx')
    test_data = pd.read_excel('preprocessed_data/test_data.xlsx')
    
    print(f"📊 Data loaded:")
    print(f"   Training: {len(train_data)} records")
    print(f"   Testing: {len(test_data)} records")
    
    # Train model (without hyperparameter tuning to use default C=0.5)
    model = ProductionSVMCategorizer()
    model.train(train_data, tune_hyperparameters=False)
    
    # Verify the C parameter
    c_value = getattr(model.model, 'C', 'Unknown')
    print(f"\n✅ Model C parameter: {c_value}")
    
    # Evaluate performance
    results = model.evaluate(test_data)
    print(f"✅ Test accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    
    return model

if __name__ == "__main__":
    model = test_optimized_svm()
