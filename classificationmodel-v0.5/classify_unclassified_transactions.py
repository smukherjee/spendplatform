#!/usr/bin/env python3
"""
Classify Unclassified Transactions using Enhanced SVM Model
Process Excel file and add L2 category predictions with confidence scores
"""

import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_production_model():
    """Load the production SVM model"""
    model_path = '/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.5/models/production_svm_enhanced.pkl'
    
    try:
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)
        print(f"✅ Loaded production model: {model_package.get('model_type', 'SVM')}")
        print(f"📊 Model accuracy: {model_package.get('accuracy', 0):.4f}")
        print(f"🏷️ Categories: {model_package.get('categories', 0)}")
        return model_package
    except FileNotFoundError:
        print(f"❌ Model file not found: {model_path}")
        return None

def examine_unclassified_file():
    """Examine the structure of the unclassified transactions file"""
    file_path = '/Users/sujoymukherjee/code/spendplatform/context/unclassified transactions.xlsx'
    
    print("🔍 EXAMINING UNCLASSIFIED TRANSACTIONS FILE")
    print("=" * 60)
    
    try:
        # Try to read the Excel file
        df = pd.read_excel(file_path)
        
        print(f"✅ File loaded successfully!")
        print(f"📊 File info:")
        print(f"   • Shape: {df.shape}")
        print(f"   • Columns: {list(df.columns)}")
        
        # Show first few rows
        print(f"\n📋 First 5 rows:")
        print(df.head())
        
        # Check for description-like columns
        description_columns = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['description', 'item', 'product', 'name', 'detail']):
                description_columns.append(col)
        
        print(f"\n🔤 Potential description columns: {description_columns}")
        
        # Check for existing category columns
        category_columns = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['category', 'class', 'type', 'l2', 'l3', 'l4']):
                category_columns.append(col)
        
        print(f"🏷️ Existing category columns: {category_columns}")
        
        # Check for missing values in key columns
        if description_columns:
            main_desc_col = description_columns[0]
            missing_descriptions = df[main_desc_col].isna().sum()
            print(f"\n⚠️ Missing descriptions in '{main_desc_col}': {missing_descriptions}")
        
        return df, description_columns, category_columns
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None, [], []

def predict_category(description, model_package):
    """Make prediction for a single description"""
    if pd.isna(description) or str(description).strip() == '':
        return {
            'category': 'Unknown',
            'confidence': 0.0,
            'status': 'Empty Description'
        }
    
    try:
        # Preprocess text
        clean_text = str(description).lower().strip()
        
        # Extract features
        features = model_package['vectorizer'].transform([clean_text])
        
        # Make prediction
        prediction = model_package['model'].predict(features)[0]
        probabilities = model_package['model'].predict_proba(features)[0]
        
        # Get category name and confidence
        category = model_package['label_encoder'].inverse_transform([prediction])[0]
        confidence = max(probabilities)
        
        # Determine prediction quality
        if confidence > 0.8:
            status = 'High Confidence'
        elif confidence > 0.6:
            status = 'Medium Confidence'
        else:
            status = 'Low Confidence'
        
        return {
            'category': category,
            'confidence': confidence,
            'status': status
        }
    
    except Exception as e:
        return {
            'category': 'Error',
            'confidence': 0.0,
            'status': f'Prediction Error: {str(e)}'
        }

def process_unclassified_transactions(df, description_column, model_package):
    """Process all transactions and add predictions"""
    print(f"\n🔄 PROCESSING TRANSACTIONS")
    print("=" * 60)
    
    print(f"📝 Using description column: '{description_column}'")
    print(f"📊 Total transactions to process: {len(df)}")
    
    # Initialize result columns
    df['Predicted_L2_Category'] = ''
    df['Prediction_Confidence'] = 0.0
    df['Prediction_Status'] = ''
    df['Processing_Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Process in batches for progress tracking
    batch_size = 100
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    processed_count = 0
    high_confidence_count = 0
    medium_confidence_count = 0
    low_confidence_count = 0
    error_count = 0
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(df))
        
        print(f"   Processing batch {batch_idx + 1}/{total_batches} ({start_idx+1}-{end_idx})...")
        
        for idx in range(start_idx, end_idx):
            description = df.iloc[idx][description_column]
            
            # Make prediction
            result = predict_category(description, model_package)
            
            # Update dataframe
            df.at[idx, 'Predicted_L2_Category'] = result['category']
            df.at[idx, 'Prediction_Confidence'] = result['confidence']
            df.at[idx, 'Prediction_Status'] = result['status']
            
            # Update counters
            processed_count += 1
            if result['status'] == 'High Confidence':
                high_confidence_count += 1
            elif result['status'] == 'Medium Confidence':
                medium_confidence_count += 1
            elif result['status'] == 'Low Confidence':
                low_confidence_count += 1
            else:
                error_count += 1
    
    print(f"\n✅ Processing complete!")
    print(f"📊 Summary:")
    print(f"   • Total processed: {processed_count}")
    print(f"   • High confidence (>0.8): {high_confidence_count} ({high_confidence_count/processed_count*100:.1f}%)")
    print(f"   • Medium confidence (0.6-0.8): {medium_confidence_count} ({medium_confidence_count/processed_count*100:.1f}%)")
    print(f"   • Low confidence (<0.6): {low_confidence_count} ({low_confidence_count/processed_count*100:.1f}%)")
    print(f"   • Errors: {error_count} ({error_count/processed_count*100:.1f}%)")
    
    return df, {
        'total': processed_count,
        'high_confidence': high_confidence_count,
        'medium_confidence': medium_confidence_count,
        'low_confidence': low_confidence_count,
        'errors': error_count
    }

def analyze_predictions(df):
    """Analyze the prediction results"""
    print(f"\n📊 PREDICTION ANALYSIS")
    print("=" * 60)
    
    # Category distribution
    category_counts = df['Predicted_L2_Category'].value_counts()
    print(f"🏷️ Predicted Categories Distribution:")
    for category, count in category_counts.head(10).items():
        percentage = count / len(df) * 100
        print(f"   {category}: {count} ({percentage:.1f}%)")
    
    if len(category_counts) > 10:
        print(f"   ... and {len(category_counts) - 10} more categories")
    
    # Confidence distribution
    print(f"\n🎯 Confidence Score Distribution:")
    confidence_stats = df['Prediction_Confidence'].describe()
    print(f"   Mean: {confidence_stats['mean']:.3f}")
    print(f"   Median: {confidence_stats['50%']:.3f}")
    print(f"   Std: {confidence_stats['std']:.3f}")
    print(f"   Min: {confidence_stats['min']:.3f}")
    print(f"   Max: {confidence_stats['max']:.3f}")
    
    # Status distribution
    print(f"\n📈 Prediction Status Distribution:")
    status_counts = df['Prediction_Status'].value_counts()
    for status, count in status_counts.items():
        percentage = count / len(df) * 100
        print(f"   {status}: {count} ({percentage:.1f}%)")
    
    # Sample predictions
    print(f"\n🔮 Sample Predictions:")
    
    # High confidence samples
    high_conf_samples = df[df['Prediction_Status'] == 'High Confidence'].head(3)
    if not high_conf_samples.empty:
        print(f"   🟢 High Confidence Examples:")
        for _, row in high_conf_samples.iterrows():
            desc_col = [col for col in df.columns if 'description' in col.lower() or 'item' in col.lower()][0]
            desc = str(row[desc_col])[:50] if desc_col in df.columns else "N/A"
            print(f"      '{desc}' → {row['Predicted_L2_Category']} (conf: {row['Prediction_Confidence']:.3f})")
    
    # Medium confidence samples
    med_conf_samples = df[df['Prediction_Status'] == 'Medium Confidence'].head(2)
    if not med_conf_samples.empty:
        print(f"   🟡 Medium Confidence Examples:")
        for _, row in med_conf_samples.iterrows():
            desc_col = [col for col in df.columns if 'description' in col.lower() or 'item' in col.lower()][0]
            desc = str(row[desc_col])[:50] if desc_col in df.columns else "N/A"
            print(f"      '{desc}' → {row['Predicted_L2_Category']} (conf: {row['Prediction_Confidence']:.3f})")
    
    # Low confidence samples
    low_conf_samples = df[df['Prediction_Status'] == 'Low Confidence'].head(2)
    if not low_conf_samples.empty:
        print(f"   🔴 Low Confidence Examples:")
        for _, row in low_conf_samples.iterrows():
            desc_col = [col for col in df.columns if 'description' in col.lower() or 'item' in col.lower()][0]
            desc = str(row[desc_col])[:50] if desc_col in df.columns else "N/A"
            print(f"      '{desc}' → {row['Predicted_L2_Category']} (conf: {row['Prediction_Confidence']:.3f})")

def save_classified_results(df, output_path):
    """Save the classified results to Excel"""
    print(f"\n💾 SAVING CLASSIFIED RESULTS")
    print("=" * 60)
    
    try:
        # Create a summary sheet
        summary_data = {
            'Metric': [
                'Total Transactions',
                'High Confidence Predictions (>0.8)',
                'Medium Confidence Predictions (0.6-0.8)',
                'Low Confidence Predictions (<0.6)',
                'Average Confidence',
                'Processing Date',
                'Model Used'
            ],
            'Value': [
                len(df),
                len(df[df['Prediction_Status'] == 'High Confidence']),
                len(df[df['Prediction_Status'] == 'Medium Confidence']),
                len(df[df['Prediction_Status'] == 'Low Confidence']),
                f"{df['Prediction_Confidence'].mean():.3f}",
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Enhanced SVM v2.0'
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        
        # Category summary
        category_summary = df['Predicted_L2_Category'].value_counts().reset_index()
        category_summary.columns = ['Category', 'Count']
        category_summary['Percentage'] = (category_summary['Count'] / len(df) * 100).round(2)
        
        # Save to Excel with multiple sheets
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Main data with predictions
            df.to_excel(writer, sheet_name='Classified_Transactions', index=False)
            
            # Summary sheet
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Category breakdown
            category_summary.to_excel(writer, sheet_name='Category_Breakdown', index=False)
        
        print(f"✅ Results saved to: {output_path}")
        print(f"📊 File contains {len(df)} classified transactions")
        print(f"📝 Sheets created:")
        print(f"   • Classified_Transactions - Main data with predictions")
        print(f"   • Summary - Processing summary and statistics")
        print(f"   • Category_Breakdown - Category distribution")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        return False

def main():
    """Main function to process unclassified transactions"""
    print("🚀 UNCLASSIFIED TRANSACTIONS PROCESSING")
    print("=" * 70)
    
    # Load production model
    model_package = load_production_model()
    if model_package is None:
        return
    
    # Examine input file
    df, description_columns, category_columns = examine_unclassified_file()
    if df is None:
        return
    
    # Determine which column to use for descriptions
    if not description_columns:
        print(f"❌ No description columns found. Available columns: {list(df.columns)}")
        return
    
    main_description_column = description_columns[0]
    
    # Process transactions
    classified_df, stats = process_unclassified_transactions(df, main_description_column, model_package)
    
    # Analyze predictions
    analyze_predictions(classified_df)
    
    # Save results
    output_path = '/Users/sujoymukherjee/code/spendplatform/context/classified_transactions_output.xlsx'
    success = save_classified_results(classified_df, output_path)
    
    if success:
        print(f"\n" + "=" * 70)
        print(f"🎉 PROCESSING COMPLETE!")
        print(f"✅ {stats['total']} transactions classified")
        print(f"🟢 {stats['high_confidence']} high confidence predictions")
        print(f"🟡 {stats['medium_confidence']} medium confidence predictions")
        print(f"🔴 {stats['low_confidence']} low confidence predictions")
        print(f"💾 Results saved: classified_transactions_output.xlsx")
        print("=" * 70)
    
    return classified_df, stats

if __name__ == "__main__":
    classified_df, processing_stats = main()
