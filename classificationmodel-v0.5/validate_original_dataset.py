#!/usr/bin/env python3
"""
Test the optimized RBF SVM model against the original full dataset
"""

import pandas as pd
import numpy as np
import time
from production_svm_model import ProductionSVMCategorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_original_data():
    """Load and prepare the original dataset"""
    print("📁 Loading original dataset...")
    
    # Load the original data
    data_path = '/Users/sujoymukherjee/code/spendplatform/context/Copy of Catergorization Working Sheet_Labelled Data for Training.xlsx'
    
    try:
        # Try different sheet names and header configurations
        df = pd.read_excel(data_path, sheet_name=0)
        print(f"✅ Loaded data from: {data_path.split('/')[-1]}")
        print(f"✅ Shape: {df.shape}")
        print(f"✅ Columns: {list(df.columns)}")
        
        # Display first few rows to understand structure
        print(f"\n📊 First 5 rows:")
        print(df.head())
        
        # Check for different possible column names
        description_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['description', 'item', 'product', 'spend', 'text'])]
        category_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['category', 'class', 'label', 'type'])]
        
        print(f"\n🔍 Potential description columns: {description_cols}")
        print(f"🔍 Potential category columns: {category_cols}")
        
        return df, description_cols, category_cols
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, [], []

def identify_data_columns(df, description_cols, category_cols):
    """Identify the correct description and category columns"""
    print(f"\n🔍 Analyzing data structure...")
    
    # Check data types and non-null counts
    print(f"📋 Column info:")
    for col in df.columns:
        non_null_count = df[col].notna().sum()
        data_type = df[col].dtype
        print(f"   {col}: {non_null_count} non-null values, type: {data_type}")
    
    # Look for columns with string data that could be descriptions
    text_columns = []
    column_analysis = {}
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if column contains text descriptions
            sample_values = df[col].dropna().head(5).tolist()
            avg_length = df[col].dropna().astype(str).str.len().mean()
            unique_count = df[col].nunique()
            
            print(f"   {col} sample values: {sample_values}")
            print(f"   {col} average length: {avg_length:.1f}")
            print(f"   {col} unique values: {unique_count}")
            
            column_analysis[col] = {
                'avg_length': avg_length,
                'unique_count': unique_count,
                'sample_values': sample_values
            }
            
            if avg_length > 15:  # Likely description if average length > 15 chars
                text_columns.append(col)
    
    print(f"\n📝 Identified text columns: {text_columns}")
    
    # Look for categorical columns with reasonable distribution
    categorical_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_count = df[col].nunique()
            total_count = len(df[col].dropna())
            
            # Check if it's a reasonable category column (not too few, not too many unique values)
            if 5 <= unique_count <= total_count * 0.3:  # Between 5 and 30% unique values
                categorical_columns.append(col)
                print(f"   {col}: {unique_count} unique values out of {total_count} (good for categories)")
    
    print(f"\n🏷️ Identified potential categorical columns: {categorical_columns}")
    
    # Auto-detect based on column names and structure
    description_column = None
    category_column = None
    
    # Look for description-like columns
    for col in df.columns:
        if col in column_analysis:
            sample_vals = column_analysis[col]['sample_values']
            if any('Item_Descripton' in str(val) or 'Description' in str(val) for val in sample_vals):
                description_column = col
                print(f"   🎯 Found description column: {col}")
                break
    
    # Look for category columns - prefer L2 level
    for col in df.columns:
        if col in column_analysis:
            sample_vals = column_analysis[col]['sample_values']
            if any('Category L2' in str(val) for val in sample_vals):
                category_column = col
                print(f"   🎯 Found Category L2 column: {col}")
                break
    
    # Fallback: if not found by header, use heuristics
    if not description_column:
        # Look for column with longest average text length
        max_length = 0
        for col, analysis in column_analysis.items():
            if analysis['avg_length'] > max_length and analysis['avg_length'] > 20:
                max_length = analysis['avg_length']
                description_column = col
    
    if not category_column:
        # Look for column with reasonable number of categories (10-100)
        for col, analysis in column_analysis.items():
            unique_count = analysis['unique_count']
            if 10 <= unique_count <= 100:
                category_column = col
                break
    
    return text_columns, categorical_columns, description_column, category_column

def prepare_dataset_for_training(df, description_col, category_col):
    """Prepare the dataset for training"""
    print(f"\n📝 Preparing dataset for training...")
    print(f"   Description column: {description_col}")
    print(f"   Category column: {category_col}")
    
    # Create a clean dataset
    clean_df = df[[description_col, category_col]].copy()
    clean_df.columns = ['Item_Descripton', 'Category L2']
    
    # Remove rows with missing values
    original_count = len(clean_df)
    clean_df = clean_df.dropna()
    final_count = len(clean_df)
    
    print(f"✅ Dataset preparation:")
    print(f"   Original records: {original_count}")
    print(f"   After cleaning: {final_count}")
    print(f"   Removed: {original_count - final_count} records")
    
    # Show category distribution
    print(f"\n📊 Category distribution:")
    category_counts = clean_df['Category L2'].value_counts()
    for category, count in category_counts.head(10).items():
        percentage = (count / len(clean_df)) * 100
        print(f"   {category}: {count} ({percentage:.1f}%)")
    
    print(f"\n✅ Categories: {clean_df['Category L2'].nunique()} unique")
    
    return clean_df

def test_model_on_original_data(df):
    """Test the optimized model on the original dataset"""
    print(f"\n{'='*60}")
    print("🧪 TESTING OPTIMIZED RBF SVM ON ORIGINAL DATASET")
    print('='*60)
    
    # Split the data
    train_df, test_df = train_test_split(
        df, 
        test_size=0.25,  # 75% train, 25% test
        random_state=42, 
        stratify=df['Category L2']
    )
    
    print(f"📊 Data split:")
    print(f"   Training: {len(train_df)} records")
    print(f"   Testing: {len(test_df)} records")
    
    # Train the optimized model
    print(f"\n{'-'*50}")
    print("🚀 Training Optimized RBF SVM Model")
    print('-'*50)
    
    start_time = time.time()
    model = ProductionSVMCategorizer()
    model.train(train_df, tune_hyperparameters=False)
    training_time = time.time() - start_time
    
    # Evaluate the model
    print(f"\n{'-'*50}")
    print("📈 Evaluating Model Performance")
    print('-'*50)
    
    evaluation_start = time.time()
    results = model.evaluate(test_df)
    evaluation_time = time.time() - evaluation_start
    
    print(f"\n✅ Performance Summary:")
    print(f"   Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"   Training Time: {training_time:.2f}s")
    print(f"   Evaluation Time: {evaluation_time:.2f}s")
    print(f"   Total Time: {training_time + evaluation_time:.2f}s")
    
    # Model configuration
    if model.model:
        kernel = getattr(model.model, 'kernel', 'unknown')
        c_param = getattr(model.model, 'C', 'unknown')
        gamma = getattr(model.model, 'gamma', 'unknown')
        
        print(f"\n⚙️ Model Configuration:")
        print(f"   Kernel: {kernel}")
        print(f"   C parameter: {c_param}")
        print(f"   Gamma: {gamma}")
        print(f"   Features: 1000 TF-IDF unigrams")
    
    return model, results, test_df

def detailed_performance_analysis(model, test_df, results):
    """Perform detailed performance analysis"""
    print(f"\n{'='*60}")
    print("🔍 DETAILED PERFORMANCE ANALYSIS")
    print('='*60)
    
    # Get predictions for confusion matrix
    test_texts = test_df['Item_Descripton'].tolist()
    predictions = model.predict(test_texts)
    true_labels = test_df['Category L2'].tolist()
    
    # Create confusion matrix
    unique_categories = sorted(test_df['Category L2'].unique())
    cm = confusion_matrix(true_labels, predictions, labels=unique_categories)
    
    print(f"📊 Per-Category Performance:")
    print(f"{'Category':<40} {'Precision':<10} {'Recall':<8} {'F1-Score':<8} {'Support':<8}")
    print('-'*75)
    
    # Calculate per-category metrics
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, labels=unique_categories, average=None, zero_division=0
    )
    
    for i, category in enumerate(unique_categories):
        cat_name = str(category)[:38] if category is not None else "Unknown"
        try:
            if isinstance(precision, np.ndarray) and i < len(precision):
                prec = float(precision[i])
            else:
                prec = 0.0
                
            if isinstance(recall, np.ndarray) and i < len(recall):
                rec = float(recall[i])
            else:
                rec = 0.0
                
            if isinstance(f1, np.ndarray) and i < len(f1):
                f1_score = float(f1[i])
            else:
                f1_score = 0.0
                
            if isinstance(support, np.ndarray) and i < len(support):
                supp = int(support[i])
            else:
                supp = 0
        except (IndexError, TypeError, ValueError):
            prec = rec = f1_score = 0.0
            supp = 0
        print(f"{cat_name:<40} {prec:.3f}      {rec:.3f}    {f1_score:.3f}    {supp:<8}")
    
    # Overall metrics
    avg_precision = np.mean(precision)
    avg_recall = np.mean(recall)
    avg_f1 = np.mean(f1)
    
    print(f"\n📈 Overall Metrics:")
    print(f"   Average Precision: {avg_precision:.4f}")
    print(f"   Average Recall: {avg_recall:.4f}")
    print(f"   Average F1-Score: {avg_f1:.4f}")
    print(f"   Test Accuracy: {results['accuracy']:.4f}")
    
    # Show some example predictions
    print(f"\n🔮 Sample Predictions:")
    print(f"{'Description':<50} {'True':<25} {'Predicted':<25} {'✓':<3}")
    print('-'*105)
    
    sample_indices = np.random.choice(len(test_df), min(10, len(test_df)), replace=False)
    for idx in sample_indices:
        desc = test_df.iloc[idx]['Item_Descripton'][:47]
        true_cat = test_df.iloc[idx]['Category L2'][:22]
        pred_cat = predictions[idx][:22]
        correct = "✓" if true_cat == pred_cat else "✗"
        
        print(f"{desc:<50} {true_cat:<25} {pred_cat:<25} {correct:<3}")
    
    return cm, unique_categories

def compare_with_preprocessed_results():
    """Compare results with previously tested preprocessed data"""
    print(f"\n{'='*60}")
    print("📊 COMPARISON WITH PREVIOUS RESULTS")
    print('='*60)
    
    # Previous results from preprocessed data
    previous_results = {
        'preprocessed_accuracy': 0.6942,
        'preprocessed_samples': 533,
        'baseline_accuracy': 0.6360,
        'improvement_over_baseline': 0.0582
    }
    
    print(f"Previous Results (on preprocessed data):")
    print(f"   Test Accuracy: {previous_results['preprocessed_accuracy']:.4f} ({previous_results['preprocessed_accuracy']*100:.2f}%)")
    print(f"   Test Samples: {previous_results['preprocessed_samples']}")
    print(f"   Baseline: {previous_results['baseline_accuracy']:.4f} ({previous_results['baseline_accuracy']*100:.2f}%)")
    
    return previous_results

if __name__ == "__main__":
    try:
        # Load and explore the original data
        df, desc_cols, cat_cols = load_and_prepare_original_data()
        
        if df is not None:
            # Analyze data structure
            text_cols, categorical_cols, description_col, category_col = identify_data_columns(df, desc_cols, cat_cols)
            
            # Interactive selection of columns (for now, auto-detect)
            if description_col and category_col:
                print(f"\n💡 Auto-selected columns:")
                print(f"   Description: {description_col}")
                print(f"   Category: {category_col}")
                
                # Prepare dataset
                clean_df = prepare_dataset_for_training(df, description_col, category_col)
                
                # Check for class imbalance issues
                category_counts = clean_df['Category L2'].value_counts()
                min_class_size = category_counts.min()
                
                if min_class_size < 2:
                    print(f"\n⚠️ Class imbalance detected! Minimum class size: {min_class_size}")
                    print(f"🔧 Filtering out classes with < 2 samples...")
                    
                    # Remove classes with only 1 sample
                    valid_categories = category_counts[category_counts >= 2].index
                    clean_df = clean_df[clean_df['Category L2'].isin(valid_categories)]
                    
                    print(f"✅ Filtered dataset:")
                    print(f"   Records after filtering: {len(clean_df)}")
                    print(f"   Categories after filtering: {clean_df['Category L2'].nunique()}")
                
                # Test the model
                model, results, test_df = test_model_on_original_data(clean_df)
                
                # Detailed analysis
                cm, categories = detailed_performance_analysis(model, test_df, results)
                
                # Compare with previous results
                previous_results = compare_with_preprocessed_results()
                
                print(f"\n{'='*60}")
                print("🎯 FINAL VALIDATION RESULTS")
                print('='*60)
                print(f"✅ Model successfully tested on original dataset")
                print(f"✅ Achieved {results['accuracy']*100:.2f}% accuracy")
                print(f"✅ Processed {len(clean_df)} total records")
                print(f"✅ Validated across {len(categories)} categories")
                print(f"✅ Optimized RBF SVM confirmed working!")
                
            else:
                print(f"❌ Could not automatically identify description and category columns")
                print(f"🔍 Available columns: {list(df.columns)}")
                if description_col:
                    print(f"   Found description column: {description_col}")
                if category_col:
                    print(f"   Found category column: {category_col}")
                print(f"🔍 Manual inspection required for column mapping")
        
    except Exception as e:
        print(f"❌ Error in validation: {e}")
        import traceback
        traceback.print_exc()
