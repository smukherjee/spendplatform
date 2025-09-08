#!/usr/bin/env python3
"""
Test optimized domain-specific preprocessing vs basic preprocessing
"""

import pandas as pd
import time
from production_svm_model import ProductionSVMCategorizer

def test_optimized_preprocessing():
    """Test the model with optimized domain-specific preprocessing"""
    print("🚀 Testing Optimized Domain-Specific Preprocessing")
    print("="*65)
    
    # Load data
    train_data = pd.read_excel('preprocessed_data/train_data.xlsx')
    test_data = pd.read_excel('preprocessed_data/test_data.xlsx')
    
    print(f"📊 Data loaded:")
    print(f"   Training: {len(train_data)} records")
    print(f"   Testing: {len(test_data)} records")
    
    # Test optimized preprocessing model
    print(f"\n{'-'*55}")
    print("🔬 Optimized Domain-Specific Preprocessing Model")
    print('-'*55)
    
    start_time = time.time()
    
    # Train model
    model = ProductionSVMCategorizer()
    model.train(train_data, tune_hyperparameters=False)
    
    # Evaluate performance
    results = model.evaluate(test_data)
    
    total_time = time.time() - start_time
    
    print(f"\n✅ Optimized Preprocessing Results:")
    print(f"   Test accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"   Total time: {total_time:.2f}s")
    
    # Show preprocessing examples
    print(f"\n📝 Optimized Preprocessing Examples:")
    sample_texts = [
        "Heavy-Duty Industrial Pump (Model: XYZ-123) - Brand New!",
        "Office Equipment: Chairs, Desks & Storage Units", 
        "Chemical Products - Cleaning & Maintenance Supplies",
        "Electrical Components: Wires, Cables & Connectors",
        "Safety Equipment: Hard Hats, Gloves & Protective Gear",
        "Hydraulic Valve - Model HV-2000, 3/4 inch NPT",
        "Stainless Steel Pipe Fitting - 90° Elbow, 2-inch",
        "Industrial Bearing - SKF 6205-2RS, Sealed"
    ]
    
    print("   Original → Optimized Processing:")
    print("   " + "-" * 85)
    for text in sample_texts:
        processed = model.preprocess_text(text)
        print(f"   '{text}' → '{processed}'")
    
    return model, results

def compare_all_preprocessing_approaches():
    """Compare basic, enhanced, and optimized preprocessing"""
    print(f"\n{'='*65}")
    print("🔍 Complete Preprocessing Comparison")
    print('='*65)
    
    sample_texts = [
        "Heavy-Duty Industrial Pump (Model: XYZ-123)",
        "Electrical Cable, 10ft, Black Color",
        "Office Chair - Ergonomic Design", 
        "Chemical Lubricant - Multi-Purpose",
        "Safety Equipment: Hard Hat & Gloves"
    ]
    
    # Basic preprocessing (simple)
    def basic_preprocess(text):
        if pd.isna(text) or text == '':
            return ''
        return str(text).lower().strip()
    
    # Current optimized preprocessing
    model = ProductionSVMCategorizer()
    
    print("Preprocessing Method Comparison:")
    print("-" * 95)
    print(f"{'Original':<35} {'Basic':<20} {'Optimized':<35}")
    print("-" * 95)
    
    for text in sample_texts:
        basic = basic_preprocess(text)
        optimized = model.preprocess_text(text)
        print(f"{text:<35} {basic[:18]:<20} {optimized[:33]:<35}")
    
    return True

if __name__ == "__main__":
    try:
        # Test optimized preprocessing  
        model, results = test_optimized_preprocessing()
        
        # Compare all preprocessing methods
        compare_all_preprocessing_approaches()
        
        # Compare with previous performances
        basic_accuracy = 0.6698      # Previous best with basic preprocessing
        enhanced_accuracy = 0.6548   # Enhanced preprocessing result
        current_accuracy = results['accuracy']
        
        print(f"\n{'='*65}")
        print("📈 COMPREHENSIVE PERFORMANCE COMPARISON")
        print('='*65)
        print(f"Basic preprocessing:     {basic_accuracy:.4f} ({basic_accuracy*100:.2f}%)")
        print(f"Enhanced preprocessing:  {enhanced_accuracy:.4f} ({enhanced_accuracy*100:.2f}%)")
        print(f"Optimized preprocessing: {current_accuracy:.4f} ({current_accuracy*100:.2f}%)")
        
        # Compare optimized vs basic
        improvement_vs_basic = current_accuracy - basic_accuracy
        improvement_pct_basic = (improvement_vs_basic / basic_accuracy) * 100
        
        # Compare optimized vs enhanced  
        improvement_vs_enhanced = current_accuracy - enhanced_accuracy
        improvement_pct_enhanced = (improvement_vs_enhanced / enhanced_accuracy) * 100
        
        print(f"\n📊 Improvements:")
        if improvement_vs_basic > 0:
            print(f"   ✅ vs Basic: +{improvement_vs_basic:.4f} ({improvement_pct_basic:+.2f}%)")
        elif improvement_vs_basic < 0:
            print(f"   ❌ vs Basic: {improvement_vs_basic:.4f} ({improvement_pct_basic:+.2f}%)")
        else:
            print(f"   ➖ vs Basic: No change")
            
        if improvement_vs_enhanced > 0:
            print(f"   ✅ vs Enhanced: +{improvement_vs_enhanced:.4f} ({improvement_pct_enhanced:+.2f}%)")
        elif improvement_vs_enhanced < 0:
            print(f"   ❌ vs Enhanced: {improvement_vs_enhanced:.4f} ({improvement_pct_enhanced:+.2f}%)")
        else:
            print(f"   ➖ vs Enhanced: No change")
        
        # Determine best approach
        best_accuracy = max(basic_accuracy, enhanced_accuracy, current_accuracy)
        if best_accuracy == current_accuracy:
            print(f"\n🏆 WINNER: Optimized preprocessing achieves best performance!")
        elif best_accuracy == basic_accuracy:
            print(f"\n🏆 WINNER: Basic preprocessing remains the best approach!")
        else:
            print(f"\n🏆 WINNER: Enhanced preprocessing was the best approach!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
