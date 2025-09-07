#!/usr/bin/env python3
"""
Test script for the refactored prediction system
"""

import pandas as pd
import os
from predict_categories import MultiLevelModelPredictor

def test_prediction_system():
    """Test the prediction system with different models"""

    # Test data
    test_data = pd.DataFrame({
        'Item_Descripton': [
            'Office supplies including pens and paper',
            'Software license for Microsoft Office',
            'Travel expenses for business trip',
            'Coffee and snacks for office meeting',
            'IT equipment maintenance service'
        ]
    })

    print("🧪 Testing Prediction System")
    print("=" * 50)

    # Test with spaCy model
    print("\n1. Testing with spaCy model...")
    try:
        predictor_spacy = MultiLevelModelPredictor(model_path='spend_categorization_model_spacy.pkl')
        predictions_spacy, confidences_spacy = predictor_spacy.predict_multi_level(test_data)
        print("✅ spaCy model predictions successful")
        print(f"   Predicted categories: {predictions_spacy}")

    except Exception as e:
        print(f"❌ spaCy model failed: {e}")

    # Test with NLTK model
    print("\n2. Testing with NLTK model...")
    try:
        predictor_nltk = MultiLevelModelPredictor(model_path='spend_categorization_model_nltk.pkl')
        predictions_nltk, confidences_nltk = predictor_nltk.predict_multi_level(test_data)
        print("✅ NLTK model predictions successful")
        print(f"   Predicted categories: {predictions_nltk}")

    except Exception as e:
        print(f"❌ NLTK model failed: {e}")

    # Test with basic model
    print("\n3. Testing with basic model...")
    try:
        predictor_basic = MultiLevelModelPredictor(model_path='spend_categorization_model_basic_parser_+_fast_training.pkl')
        predictions_basic, confidences_basic = predictor_basic.predict_multi_level(test_data)
        print("✅ Basic model predictions successful")
        print(f"   Predicted categories: {predictions_basic}")

    except Exception as e:
        print(f"❌ Basic model failed: {e}")

    print("\n" + "=" * 50)
    print("🧪 Test completed!")

if __name__ == "__main__":
    test_prediction_system()
