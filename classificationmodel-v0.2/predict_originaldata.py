#!/usr/bin/env python3
"""
Predict Categories for Original Training Data
Use the spaCy parser model to predict categories for all rows in the original training data Excel file
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add the current directory to Python path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def predict_original_data():
    """Predict categories for all rows in the original training data"""

    print("="*80)
    print("SPEND PLATFORM - PREDICT ORIGINAL TRAINING DATA")
    print("="*80)

    # Model path
    model_path = "/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.2/pipeline_run_20250907_143517/spend_categorization_model_spacy_parser_+_optimized.pkl"

    # Input file path
    input_file = "/Users/sujoymukherjee/code/spendplatform/context/Original Catergorization Working Sheet_Labelled Data for Training.xlsx"

    # Output file path
    output_file = "/Users/sujoymukherjee/code/spendplatform/context/predicted_original_training_data.xlsx"

    print(f"Model path: {model_path}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")

    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found: {model_path}")
        return

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file not found: {input_file}")
        return

    # Import the predictor class
    try:
        from predict_categories import MultiLevelModelPredictor
    except ImportError as e:
        print(f"❌ Error importing predictor: {e}")
        print("Make sure you're running this from the classificationmodel-v0.2 directory")
        return

    # Initialize predictor with the specific model path
    print("\nLoading spaCy model...")
    try:
        predictor = MultiLevelModelPredictor(model_path=model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Load the original training data
    print("\nLoading original training data...")
    try:
        # Read with header in row 2 (index 2)
        df = pd.read_excel(input_file, header=2)
        print(f"✅ Loaded {len(df)} records")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Check if we have the required description column
    if 'Descripton' not in df.columns:
        print("❌ Error: 'Descripton' column not found in the data")
        print(f"Available columns: {list(df.columns)}")
        return

    # Filter out rows with missing descriptions
    original_count = len(df)
    df = df.dropna(subset=['Descripton'])
    filtered_count = len(df)

    if filtered_count < original_count:
        print(f"⚠️ Filtered out {original_count - filtered_count} rows with missing descriptions")
        print(f"Remaining records: {filtered_count}")

    if len(df) == 0:
        print("❌ Error: No valid records found with descriptions")
        return

    # Make predictions
    print("\n" + "="*60)
    print("MAKING PREDICTIONS")
    print("="*60)

    try:
        results_df = predictor.predict_with_details(df)
        print("✅ Predictions completed successfully")
    except Exception as e:
        print(f"❌ Error making predictions: {e}")
        return

    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)

    try:
        results_df.to_excel(output_file, index=False)
        print(f"✅ Results saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return

    # Display summary
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)

    print(f"Total records processed: {len(results_df)}")
    print(f"Model used: spaCy Parser + Optimized")

    if predictor.model is None:
        print("❌ Error: Model not properly loaded")
        return

    print(f"Target levels: {predictor.model.target_levels}")

    # Show prediction statistics for each level
    for level in predictor.model.target_levels:
        pred_col = f'Predicted_{level.replace(" ", "_")}'
        conf_col = f'Confidence_{level.replace(" ", "_")}'

        if pred_col in results_df.columns and conf_col in results_df.columns:
            unique_preds = results_df[pred_col].nunique()
            avg_conf = results_df[conf_col].mean()
            high_conf_pct = (results_df[conf_col] >= 0.8).sum() / len(results_df) * 100

            print(f"\n{level}:")
            print(f"  • Unique predictions: {unique_preds}")
            print(f"  • Average confidence: {avg_conf:.2f}")
            print(f"  • High confidence predictions: {high_conf_pct:.1f}%")

            # Show top 5 most common predictions
            top_predictions = results_df[pred_col].value_counts().head(5)
            print("  • Top 5 predictions:")
            for pred, count in top_predictions.items():
                pct = count / len(results_df) * 100
                print(f"                {pred}: {count} ({pct:.1f}%)")

    print("\n" + "="*80)
    print("PREDICTION PROCESS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Results saved to: {output_file}")

def main():
    """Main function"""
    predict_original_data()

if __name__ == "__main__":
    main()
