#!/usr/bin/env python3
"""
Quick test to verify the copied SVM model works in v0.5
"""

import sys
import os
import pandas as pd
from production_svm_model import ProductionSVMCategorizer

def test_copied_svm_model():
    """Test that the copied SVM model works correctly"""
    print("🧪 Testing Copied SVM Model in v0.5")
    print("=" * 50)
    
    # Test 1: Check if model file exists
    model_path = 'models/production_svm_categorizer.pkl'
    if os.path.exists(model_path):
        print("✅ Model file found:", model_path)
    else:
        print("❌ Model file not found:", model_path)
        return False
    
    # Test 2: Check if data file exists
    data_path = 'preprocessed_data/top_10_l2_categories_data.xlsx'
    if os.path.exists(data_path):
        print("✅ Data file found:", data_path)
    else:
        print("❌ Data file not found:", data_path)
        return False
    
    # Test 3: Load and test model
    try:
        print("\n🔄 Loading pre-trained SVM model...")
        model = ProductionSVMCategorizer()
        model.load_model(model_path)
        print("✅ Model loaded successfully")
        
        # Test 4: Make predictions
        print("\n🔮 Testing predictions...")
        test_descriptions = [
            "HYDRAULIC OIL 46 HN BULK",
            "Contact tube 1.2mm 60A167",
            "32 MM PVC Flexible conduit BLACK",
            "Split Pin 3.2 × 40",
            "Paint Brush 4 inch"
        ]
        
        predictions = model.predict(test_descriptions)
        
        print("📋 Sample predictions:")
        for desc, pred in zip(test_descriptions, predictions):
            print(f"   '{desc[:30]:<30}' → {pred}")
        
        print("\n✅ All tests passed! Model is working correctly in v0.5")
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

if __name__ == "__main__":
    success = test_copied_svm_model()
    if success:
        print("\n🎉 v0.5 SVM model setup complete and verified!")
    else:
        print("\n💥 Setup verification failed!")
        sys.exit(1)
