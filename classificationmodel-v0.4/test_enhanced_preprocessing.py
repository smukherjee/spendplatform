#!/usr/bin/env python3
"""
Test enhanced preprocessing vs basic preprocessing performance
"""

import pandas as pd
import time
from production_svm_model import ProductionSVMCategorizer

def test_enhanced_preprocessing():
    """Test the model with enhanced preprocessing"""
    print("🚀 Testing Enhanced Preprocessing Performance")
    print("="*60)
    
    # Load data
    train_data = pd.read_excel('preprocessed_data/train_data.xlsx')
    test_data = pd.read_excel('preprocessed_data/test_data.xlsx')
    
    print(f"📊 Data loaded:")
    print(f"   Training: {len(train_data)} records")
    print(f"   Testing: {len(test_data)} records")
    
    # Test enhanced preprocessing model
    print(f"\n{'-'*50}")
    print("🔬 Enhanced Preprocessing Model")
    print('-'*50)
    
    start_time = time.time()
    
    # Train model
    model = ProductionSVMCategorizer()
    model.train(train_data, tune_hyperparameters=False)
    
    # Evaluate performance
    results = model.evaluate(test_data)
    
    total_time = time.time() - start_time
    
    print(f"\n✅ Enhanced Preprocessing Results:")
    print(f"   Test accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"   Total time: {total_time:.2f}s")
    
    # Show some preprocessing examples
    print(f"\n📝 Preprocessing Examples:")
    sample_texts = [
        "Heavy Duty Industrial Pump - Model XYZ-123",
        "Electrical Cable, 10ft, Black Color, Standard Type",
        "Office Chair - Ergonomic Design, Adjustable Height",
        "Chemical Lubricant - Multi-Purpose, 5L Container",
        "Safety Equipment: Hard Hat & Protective Gear"
    ]
    
    print("   Original → Processed:")
    for text in sample_texts:
        processed = model.preprocess_text(text)
        print(f"   '{text}' → '{processed}'")
    
    return model, results

def compare_preprocessing_methods():
    """Compare different preprocessing approaches"""
    print(f"\n{'='*60}")
    print("🔍 Preprocessing Method Comparison")
    print('='*60)
    
    # Sample texts for comparison
    sample_texts = [
        "Heavy-Duty Industrial Pump (Model: XYZ-123) - Brand New!",
        "Office Equipment: Chairs, Desks & Storage Units",
        "Chemical Products - Cleaning & Maintenance Supplies",
        "Electrical Components: Wires, Cables & Connectors",
        "Safety Gear: Hard Hats, Gloves & Protective Equipment"
    ]
    
    model = ProductionSVMCategorizer()
    
    print("Preprocessing Examples:")
    print("-" * 80)
    print(f"{'Original':<50} {'Enhanced Processing':<30}")
    print("-" * 80)
    
    for text in sample_texts:
        processed = model.preprocess_text(text)
        print(f"{text:<50} {processed:<30}")
    
    return True

if __name__ == "__main__":
    try:
        # Test enhanced preprocessing
        model, results = test_enhanced_preprocessing()
        
        # Compare preprocessing methods
        compare_preprocessing_methods()
        
        # Previous performance for comparison
        previous_accuracy = 0.6698  # Previous best performance
        improvement = results['accuracy'] - previous_accuracy
        improvement_pct = (improvement / previous_accuracy) * 100
        
        print(f"\n{'='*60}")
        print("📈 FINAL COMPARISON")
        print('='*60)
        print(f"Previous (basic preprocessing): {previous_accuracy:.4f} ({previous_accuracy*100:.2f}%)")
        print(f"Enhanced preprocessing:         {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        
        if improvement > 0:
            print(f"✅ Improvement: +{improvement:.4f} ({improvement_pct:+.2f}%)")
        elif improvement < 0:
            print(f"❌ Change: {improvement:.4f} ({improvement_pct:+.2f}%)")
        else:
            print(f"➖ No change in performance")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
