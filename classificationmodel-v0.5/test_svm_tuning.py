#!/usr/bin/env python3
"""
Test script to demonstrate SVM C parameter tuning with grid search
"""

import pandas as pd
import numpy as np
from production_svm_model import ProductionSVMCategorizer
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the preprocessed data"""
    print("📁 Loading preprocessed data...")
    
    # Load the preprocessed data files
    train_data = pd.read_excel('preprocessed_data/train_data.xlsx')
    test_data = pd.read_excel('preprocessed_data/test_data.xlsx')
    
    # Combine for full dataset info
    df = pd.concat([train_data, test_data], ignore_index=True)
    
    print(f"✅ Loaded preprocessed data:")
    print(f"   Training: {len(train_data)} records")
    print(f"   Testing: {len(test_data)} records")
    print(f"   Total: {len(df)} records")
    print(f"✅ Columns: {list(df.columns)}")
    
    # Basic data info
    if 'Category L2' in df.columns:
        print(f"✅ Categories: {df['Category L2'].nunique()} unique")
        print(f"✅ Category distribution:")
        print(df['Category L2'].value_counts().head(10))
    
    return df

def test_baseline_vs_tuned():
    """Compare baseline SVM (C=1.0) vs grid search tuned SVM"""
    print("\n" + "="*60)
    print("🔬 SVM C Parameter Tuning Comparison")
    print("="*60)
    
    # Load preprocessed data
    print("📁 Loading preprocessed data...")
    train_df = pd.read_excel('preprocessed_data/train_data.xlsx')
    test_df = pd.read_excel('preprocessed_data/test_data.xlsx')
    
    print(f"\n📊 Data loaded:")
    print(f"   Training: {len(train_df)} records")
    print(f"   Testing: {len(test_df)} records")
    
    # Test 1: Baseline model (C=1.0, no tuning)
    print("\n" + "-"*40)
    print("🎯 BASELINE MODEL (C=1.0)")
    print("-"*40)
    
    baseline_model = ProductionSVMCategorizer()
    baseline_model.train(train_df, tune_hyperparameters=False)
    baseline_results = baseline_model.evaluate(test_df)
    baseline_accuracy = baseline_results['accuracy']
    
    # Test 2: Grid search tuned model
    print("\n" + "-"*40)
    print("🔍 GRID SEARCH TUNED MODEL")
    print("-"*40)
    
    tuned_model = ProductionSVMCategorizer()
    tuned_model.train(train_df, tune_hyperparameters=True)
    tuned_results = tuned_model.evaluate(test_df)
    tuned_accuracy = tuned_results['accuracy']
    
    # Compare results
    print("\n" + "="*60)
    print("📈 PERFORMANCE COMPARISON")
    print("="*60)
    
    print(f"Baseline Model (C=1.0):")
    print(f"   Accuracy: {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
    
    print(f"\nTuned Model (Grid Search):")
    print(f"   Accuracy: {tuned_accuracy:.4f} ({tuned_accuracy*100:.2f}%)")
    
    # Safely get C parameter
    best_c = getattr(tuned_model.model, 'C', 'Unknown') if tuned_model.model else 'Unknown'
    print(f"   Best C parameter: {best_c}")
    
    # Calculate improvement
    improvement = tuned_accuracy - baseline_accuracy
    improvement_pct = (improvement / baseline_accuracy) * 100
    
    print(f"\nImprovement:")
    if improvement > 0:
        print(f"   ✅ +{improvement:.4f} ({improvement_pct:+.2f}%)")
        print(f"   🎉 Grid search improved performance!")
    elif improvement < 0:
        print(f"   ❌ {improvement:.4f} ({improvement_pct:+.2f}%)")
        print(f"   📝 Baseline was better - consider other parameters")
    else:
        print(f"   ➖ No change in performance")
    
    return {
        'baseline_accuracy': baseline_accuracy,
        'tuned_accuracy': tuned_accuracy,
        'improvement': improvement,
        'best_c': getattr(tuned_model.model, 'C', 'Unknown') if tuned_model.model else 'Unknown'
    }

def demo_grid_search_details():
    """Show detailed grid search process"""
    print("\n" + "="*60)
    print("🔍 DETAILED GRID SEARCH ANALYSIS")
    print("="*60)
    
    # Load preprocessed data
    print("📁 Loading preprocessed data...")
    train_data = pd.read_excel('preprocessed_data/train_data.xlsx')
    
    # Use a smaller sample for quick demonstration
    sample_df = train_data.sample(n=min(1000, len(train_data)), random_state=42)
    print(f"\n📊 Using sample of {len(sample_df)} records for quick analysis")
    
    # Train with grid search
    model = ProductionSVMCategorizer()
    model.train(sample_df, tune_hyperparameters=True)
    
    print(f"\n✅ Grid search completed!")
    
    # Safely get C parameter
    best_c = getattr(model.model, 'C', 'Unknown') if model.model else 'Unknown'
    print(f"   Best C parameter: {best_c}")
    
    return model

if __name__ == "__main__":
    try:
        # Run the comparison test
        results = test_baseline_vs_tuned()
        
        print(f"\n🎯 Summary:")
        print(f"   Baseline accuracy: {results['baseline_accuracy']*100:.2f}%")
        print(f"   Tuned accuracy: {results['tuned_accuracy']*100:.2f}%")
        print(f"   Best C parameter: {results['best_c']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("🔍 Running detailed analysis instead...")
        demo_grid_search_details()
