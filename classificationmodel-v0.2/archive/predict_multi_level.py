#!/usr/bin/env python3
"""
Spend Platform Multi-Level Categorization Model Predictor
Load trained multi-level model and make predictions on new data for L2, L3, and L5
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import warnings
from typing import Dict, List, Optional, Any, Union

warnings.filterwarnings('ignore')

class MultiLevelModelPredictor:
    """Predictor class for the trained multi-level categorization model"""

    def __init__(self, model_path: str):
        self.model_path: str = model_path
        self.model: Optional[Any] = None
        self.load_model()

    def load_model(self):
        """Load the trained multi-level model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        print(f"Loading multi-level model from {self.model_path}...")
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)

        # Create a simple object to hold model components
        class MultiLevelModelContainer:
            def __init__(self, model_data):
                self.models = model_data['models']
                self.vectorizer = model_data['vectorizer']
                self.label_encoders = model_data['label_encoders']
                self.feature_names = model_data['feature_names']
                self.target_levels = model_data['target_levels']
                self.scaler = model_data['scaler']

        self.model = MultiLevelModelContainer(model_data)
        print("✅ Multi-level model loaded successfully")

    def preprocess_text(self, text):
        """Preprocess text data for feature extraction"""
        if not isinstance(text, str) or text == '':
            return ''

        # Convert to lowercase
        text = text.lower()

        # Remove special characters but keep spaces and alphanumeric
        import re
        text = re.sub(r'[^\w\s]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def extract_features(self, df):
        """Extract features from new data"""
        if self.model is None:
            raise ValueError("Model not loaded")

        print("Extracting features from new data...")

        df_processed = df.copy()
        df_processed = df_processed.reset_index(drop=True)

        # Ensure required columns exist
        if 'Item_Descripton' not in df_processed.columns:
            if 'Description' in df_processed.columns:
                df_processed['Item_Descripton'] = df_processed['Description']
            elif 'Descripton' in df_processed.columns:
                df_processed['Item_Descripton'] = df_processed['Descripton']
            else:
                raise ValueError("Data must contain 'Item_Descripton', 'Description', or 'Descripton' column")

        # Apply text preprocessing
        df_processed['processed_description'] = df_processed['Item_Descripton'].apply(self.preprocess_text)

        # Text features using TF-IDF
        text_features = self.model.vectorizer.transform(df_processed['processed_description'])

        # Convert to DataFrame
        text_feature_names = [f'text_{i}' for i in range(text_features.shape[1])]
        text_df = pd.DataFrame(text_features.toarray(), columns=text_feature_names, index=df_processed.index)

        # Length-based features
        length_features = pd.DataFrame({
            'desc_length': df_processed['Item_Descripton'].str.len(),
            'word_count': df_processed['Item_Descripton'].str.split().str.len(),
            'unique_words': df_processed['processed_description'].apply(lambda x: len(set(x.split()))),
            'has_numbers': df_processed['Item_Descripton'].str.contains(r'\d').astype(int),
            'has_symbols': df_processed['Item_Descripton'].str.contains(r'[^\w\s]').astype(int)
        }, index=df_processed.index)

        # Combine all features
        X = pd.concat([text_df, length_features], axis=1)

        print(f"✅ Extracted {X.shape[1]} features from {X.shape[0]} samples")
        return X

    def predict_multi_level(self, df):
        """Make predictions for all levels"""
        if self.model is None:
            raise ValueError("Model not loaded")

        print("Making multi-level predictions...")

        # Extract features once
        X = self.extract_features(df)

        # Make predictions for each level
        predictions = {}
        confidences = {}

        for level in self.model.target_levels:  # type: ignore
            print(f"Predicting {level}...")

            # Make predictions
            predictions_encoded = self.model.models[level].predict(X)  # type: ignore

            # Decode predictions
            predictions[level] = self.model.label_encoders[level].inverse_transform(predictions_encoded)  # type: ignore

            # Get prediction probabilities
            if hasattr(self.model.models[level], 'predict_proba'):  # type: ignore
                probabilities = self.model.models[level].predict_proba(X)  # type: ignore
                confidences[level] = np.max(probabilities, axis=1)
            else:
                confidences[level] = np.ones(len(predictions[level])) * 0.5  # Default confidence

        print(f"✅ Generated predictions for {len(df)} samples across {len(self.model.target_levels)} levels")  # type: ignore

        return predictions, confidences

    def predict_with_details(self, df):
        """Make predictions with detailed output for all levels"""
        predictions, confidences = self.predict_multi_level(df)

        # Create results dataframe
        results_df = df.copy()

        # Add predictions and confidences for each level
        for level in self.model.target_levels:  # type: ignore
            results_df[f'Predicted_{level.replace(" ", "_")}'] = predictions[level]
            results_df[f'Confidence_{level.replace(" ", "_")}'] = confidences[level]
            results_df[f'Confidence_Level_{level.replace(" ", "_")}'] = pd.cut(
                confidences[level],
                bins=[0, 0.5, 0.8, 1.0],
                labels=['Low', 'Medium', 'High']
            )

        return results_df

def predict_from_file(input_file, output_file='multi_level_predictions.xlsx'):
    """Predict categories from an Excel file using multi-level model"""
    print(f"Predicting categories from file: {input_file}")

    # Load model
    predictor = MultiLevelModelPredictor('multi_level_spend_categorization_model.pkl')

    # Load input data - handle files with headers in row 2
    df_raw = pd.read_excel(input_file, header=None)

    # Check if headers are in row 2 (common in some Excel files)
    if len(df_raw) > 2 and pd.notna(df_raw.iloc[2, 1]) and str(df_raw.iloc[2, 1]).strip() == 'Item Code':
        # Use row 2 as headers
        df = df_raw.iloc[2:].reset_index(drop=True)
        df.columns = df_raw.iloc[2].values
        print("Detected headers in row 2, using them as column names")
    else:
        # Use default reading
        df = pd.read_excel(input_file)

    print(f"Loaded {len(df)} records from {input_file}")
    print(f"Columns: {list(df.columns)}")

    # Make predictions
    results_df = predictor.predict_with_details(df)

    # Save results
    results_df.to_excel(output_file, index=False)
    print(f"✅ Multi-level predictions saved to {output_file}")

    return results_df

def predict_unclear_categories(input_file, output_file='unclear_multi_level_predictions.xlsx'):
    """Predict categories only for records where Category L1 is 'unclear' using multi-level model"""
    print(f"Predicting categories for unclear records from file: {input_file}")

    # Load model
    predictor = MultiLevelModelPredictor('multi_level_spend_categorization_model.pkl')

    # Load input data - handle files with headers in row 2
    df_raw = pd.read_excel(input_file, header=None)

    # Check if headers are in row 2 (common in some Excel files)
    if len(df_raw) > 2 and pd.notna(df_raw.iloc[2, 1]) and str(df_raw.iloc[2, 1]).strip() == 'Item Code':
        # Use row 2 as headers
        df = df_raw.iloc[2:].reset_index(drop=True)
        df.columns = df_raw.iloc[2].values
        print("Detected headers in row 2, using them as column names")
    else:
        # Use default reading
        df = pd.read_excel(input_file)

    print(f"Loaded {len(df)} records from {input_file}")

    # Filter for records where Category L1 is 'unclear' (case insensitive)
    unclear_mask = df['Category L1'].str.lower() == 'unclear'
    unclear_df = df[unclear_mask].copy()

    if len(unclear_df) == 0:
        print("❌ No records found with Category L1 = 'unclear'")
        return pd.DataFrame()

    print(f"Found {len(unclear_df)} records with Category L1 = 'unclear'")
    print(f"Columns: {list(unclear_df.columns)}")

    # Make predictions on unclear records
    results_df = predictor.predict_with_details(unclear_df)

    # Save results
    results_df.to_excel(output_file, index=False)
    print(f"✅ Multi-level predictions for unclear records saved to {output_file}")

    return results_df

def main():
    """Main function to demonstrate multi-level model predictions"""
    print("="*80)
    print("MULTI-LEVEL SPEND PLATFORM CATEGORIZATION MODEL PREDICTOR")
    print("="*80)

    # Model path
    model_path = 'multi_level_spend_categorization_model.pkl'

    # Initialize predictor
    try:
        predictor = MultiLevelModelPredictor(model_path)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure the multi-level model file exists in the current directory.")
        return

    # Create sample data for demonstration
    print("\nCreating sample data for demonstration...")

    sample_data = pd.DataFrame({
        'Item_Code': ['ITEM001', 'ITEM002', 'ITEM003', 'ITEM004', 'ITEM005'],
        'Item_Descripton': [
            'Industrial hydraulic oil filter replacement',
            'Office supplies paper and toner cartridges',
            'Heavy duty welding equipment and accessories',
            'IT server maintenance and software licenses',
            'Manufacturing plant safety equipment and PPE'
        ],
        'Supplier_Name': ['ABC Industrial', 'Office Depot', 'WeldTech', 'TechCorp', 'SafetyFirst']
    })

    print("Sample data:")
    for i, row in sample_data.iterrows():
        print(f"  {row['Item_Descripton'][:60]}...")

    # Make predictions
    print("\n" + "="*60)
    print("MAKING MULTI-LEVEL PREDICTIONS")
    print("="*60)

    results_df = predictor.predict_with_details(sample_data)

    # Display results
    print("\n" + "="*60)
    print("MULTI-LEVEL PREDICTION RESULTS")
    print("="*60)

    for i, row in results_df.iterrows():
        print(f"  Item: {row['Item_Descripton'][:50]}...")
        for level in predictor.model.target_levels:  # type: ignore
            pred_col = f'Predicted_{level.replace(" ", "_")}'
            conf_col = f'Confidence_{level.replace(" ", "_")}'
            conf_level_col = f'Confidence_Level_{level.replace(" ", "_")}'
            print(f"   {level}: {row[pred_col]} (Confidence: {row[conf_col]:.2f}, Level: {row[conf_level_col]})")

    # Summary statistics
    print("\n" + "="*60)
    print("MULTI-LEVEL PREDICTION SUMMARY")
    print("="*60)

    print(f"Total predictions: {len(results_df)}")
    print(f"Levels predicted: {predictor.model.target_levels}")  # type: ignore

    for level in predictor.model.target_levels:  # type: ignore
        pred_col = f'Predicted_{level.replace(" ", "_")}'
        conf_col = f'Confidence_{level.replace(" ", "_")}'
        unique_preds = results_df[pred_col].nunique()
        avg_conf = results_df[conf_col].mean()
        print(f"  {level}: {unique_preds} unique predictions, Avg. confidence: {avg_conf:.2f}")

    # Save results
    results_file = 'multi_level_prediction_results.xlsx'
    results_df.to_excel(results_file, index=False)
    print(f"\n✅ Results saved to {results_file}")

    print("\n" + "="*80)
    print("MULTI-LEVEL PREDICTION DEMONSTRATION COMPLETED!")
    print("="*80)

if __name__ == "__main__":
    main()
