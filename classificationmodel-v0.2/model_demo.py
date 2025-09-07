#!/usr/bin/env python3
"""
Demo script to use the trained spend categorization model
"""

import pandas as pd
import pickle
import numpy as np
import re

def load_model(model_path='spend_categorization_model.pkl'):
    """Load the trained model"""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data

def preprocess_text(text):
    """Preprocess text for prediction"""
    import re
    if not isinstance(text, str) or text == '':
        return ''
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def predict_category(description, model_data):
    """Predict category for a single item description"""
    # Preprocess the text
    processed_text = preprocess_text(description)

    # Extract features using the trained vectorizer
    text_features = model_data['vectorizer'].transform([processed_text])

    # Convert to DataFrame with proper column names
    text_feature_names = [f'text_{i}' for i in range(text_features.shape[1])]
    text_df = pd.DataFrame(text_features.toarray(), columns=text_feature_names)

    # Add length-based features
    length_features = pd.DataFrame({
        'desc_length': [len(description)],
        'word_count': [len(description.split())],
        'unique_words': [len(set(processed_text.split()))],
        'has_numbers': [1 if re.search(r'\d', description) else 0],
        'has_symbols': [1 if re.search(r'[^\w\s]', description) else 0]
    })

    # Combine features
    X = pd.concat([text_df, length_features], axis=1)

    # Make prediction
    prediction_encoded = model_data['model'].predict(X)[0]
    prediction = model_data['label_encoders']['Category L2'].inverse_transform([prediction_encoded])[0]

    # Get confidence score
    if hasattr(model_data['model'], 'predict_proba'):
        probabilities = model_data['model'].predict_proba(X)[0]
        confidence = np.max(probabilities)
    else:
        confidence = 0.5

    return prediction, confidence

def main():
    """Demo the trained model with sample predictions"""
    print("="*60)
    print("SPEND CATEGORIZATION MODEL DEMO")
    print("="*60)

    # Load the trained model
    try:
        model_data = load_model()
        print("✅ Model loaded successfully!")
    except FileNotFoundError:
        print("❌ Model file not found. Please run train_classification_model.py first.")
        return

    # Sample item descriptions for testing
    test_items = [
        "Stainless steel pipe 2 inch diameter",
        "Electrical cable 10mm copper wire",
        "Industrial filter replacement",
        "Power supply unit 220V",
        "Hand tools set with wrenches",
        "Chemical solvent for cleaning",
        "Office chair ergonomic design",
        "Pneumatic valve control system",
        "Welding machine portable",
        "Safety gloves nitrile material"
    ]

    print("\n🔍 Making predictions on sample items:")
    print("-" * 60)

    for i, item in enumerate(test_items, 1):
        prediction, confidence = predict_category(item, model_data)
        print(f"{i:2d}. {item:<50} → {prediction} (Confidence: {confidence:.2f})")

    print("\n" + "="*60)
    print("DEMO COMPLETED!")
    print("="*60)
    print("\n📊 Model Details:")
    print(f"  • Target Level: {model_data['target_levels'][0]}")
    print(f"  • Model Type: {type(model_data['model']).__name__}")
    print(f"  • Number of Features: {len(model_data['feature_names'])}")
    print(f"  • Number of Categories: {len(model_data['label_encoders']['Category L2'].classes_)}")
    print(f"  • Categories: {', '.join(model_data['label_encoders']['Category L2'].classes_)}")

if __name__ == "__main__":
    main()
