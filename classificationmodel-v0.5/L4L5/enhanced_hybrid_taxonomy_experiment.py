#!/usr/bin/env python3
"""
Enhanced Hybrid Experiment with Proper Taxonomy Mapping
This version uses the proper taxonomy from Categorization File.xlsx
to ensure L2 and L4 categories are correctly derived from L5 predictions
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
warnings.filterwarnings('ignore')

def load_taxonomy_mappings():
    """Load proper taxonomy mappings from Categorization File.xlsx"""
    categorization_file = '/Users/sujoymukherjee/code/spendplatform/context/archive/Categorization File.xlsx'
    
    try:
        taxonomy_df = pd.read_excel(categorization_file, sheet_name='Taxonomy')
        
        # Create proper mappings based on taxonomy
        # Category 2 = L2, Category 4 = L4, Category 5 = L5
        l5_to_l4_mapping = dict(zip(taxonomy_df['Category 5'], taxonomy_df['Category 4']))
        l5_to_l2_mapping = dict(zip(taxonomy_df['Category 5'], taxonomy_df['Category 2']))
        l4_to_l2_mapping = dict(zip(taxonomy_df['Category 4'], taxonomy_df['Category 2']))
        
        print(f"📋 Loaded taxonomy mappings:")
        print(f"   • L5 → L4: {len(l5_to_l4_mapping)} mappings")
        print(f"   • L5 → L2: {len(l5_to_l2_mapping)} mappings") 
        print(f"   • L4 → L2: {len(l4_to_l2_mapping)} mappings")
        
        return l5_to_l4_mapping, l5_to_l2_mapping, l4_to_l2_mapping
        
    except Exception as e:
        print(f"❌ Error loading taxonomy: {e}")
        return {}, {}, {}

def load_models_and_data():
    """Load all trained models and test data"""
    models = {}
    
    # Load L2 model (use production SVM enhanced model)
    l2_model_path = '/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.5/models/production_svm_enhanced.pkl'
    with open(l2_model_path, 'rb') as f:
        l2_model_data = pickle.load(f)
        models['l2'] = l2_model_data
        models['l2_vectorizer'] = l2_model_data['vectorizer']
    
    # Load L4 model  
    with open('l4_simplified_model.pkl', 'rb') as f:
        models['l4'] = pickle.load(f)
        models['l4_vectorizer'] = models['l4']['vectorizer']
        
    # Load L5 model
    with open('l5_simplified_model.pkl', 'rb') as f:
        models['l5'] = pickle.load(f)
        models['l5_vectorizer'] = models['l5']['vectorizer']
    
    # Load test data (use unclassified transactions for real-world testing)
    test_data = pd.read_excel('/Users/sujoymukherjee/code/spendplatform/context/unclassified transactions.xlsx')
    
    return models, test_data

def enhanced_hybrid_classification_with_taxonomy(models, test_data, l5_to_l4_mapping, l5_to_l2_mapping, l4_to_l2_mapping, sample_size=1500):
    """
    Enhanced hybrid classification with proper taxonomy-based category adjustment
    """
    print(f"🧪 ENHANCED HYBRID EXPERIMENT WITH PROPER TAXONOMY")
    print("=" * 70)
    
    # Sample data
    if len(test_data) > sample_size:
        test_sample = test_data.sample(n=sample_size, random_state=42)
    else:
        test_sample = test_data.copy()
    
    print(f"📊 Analyzing {len(test_sample)} samples...")
    
    results = []
    medium_enhanced = []
    low_enhanced = []
    category_adjustments = []
    
    for idx, row in test_sample.iterrows():
        description = row['Description']  # Column name from unclassified transactions
        
        # Get L2 prediction with calibrated confidence
        l2_text = models['l2_vectorizer'].transform([description])
        l2_proba = models['l2']['model'].predict_proba(l2_text)[0]
        l2_pred_idx = np.argmax(l2_proba)
        l2_prediction = models['l2']['label_encoder'].inverse_transform([l2_pred_idx])[0]
        l2_confidence = float(np.max(l2_proba) * 0.8028)  # Calibrate based on model accuracy
        
        # Get L4 prediction with calibrated confidence
        l4_text = models['l4_vectorizer'].transform([description])
        if hasattr(models['l4']['model'], 'predict_proba'):
            l4_proba = models['l4']['model'].predict_proba(l4_text)[0]
            l4_pred_idx = np.argmax(l4_proba)
            l4_prediction = models['l4']['model'].classes_[l4_pred_idx]
            l4_confidence = float(np.max(l4_proba) * 0.5469)
        else:
            # Use decision function for SVM models without probability
            l4_decision = models['l4']['model'].decision_function(l4_text)[0]
            l4_pred_idx = np.argmax(l4_decision)
            l4_prediction = models['l4']['model'].classes_[l4_pred_idx]
            # Convert decision function to confidence (normalized)
            l4_confidence = float((np.max(l4_decision) + 1) / 2 * 0.5469)
        
        # Get L5 prediction with calibrated confidence
        l5_text = models['l5_vectorizer'].transform([description])
        if hasattr(models['l5']['model'], 'predict_proba'):
            l5_proba = models['l5']['model'].predict_proba(l5_text)[0]
            l5_pred_idx = np.argmax(l5_proba)
            l5_prediction = models['l5']['model'].classes_[l5_pred_idx]
            l5_confidence = float(np.max(l5_proba) * 0.5030)
        else:
            # Use decision function for SVM models without probability
            l5_decision = models['l5']['model'].decision_function(l5_text)[0]
            l5_pred_idx = np.argmax(l5_decision)
            l5_prediction = models['l5']['model'].classes_[l5_pred_idx]
            # Convert decision function to confidence (normalized)
            l5_confidence = float((np.max(l5_decision) + 1) / 2 * 0.5030)
        
        # Initialize result
        final_prediction = l2_prediction
        final_confidence = l2_confidence
        strategy = "high_confidence_l2"
        improvement = 0.0
        category_adjusted = False
        adjusted_l2_category = l2_prediction
        adjusted_l4_category = l4_prediction
        
        # Enhanced hybrid logic for medium and low confidence
        if l2_confidence < 0.75:  # Medium or low confidence
            
            # Try L4 enhancement
            l4_improvement = max(0, l4_confidence - l2_confidence)
            
            # Try L5 enhancement
            l5_improvement = max(0, l5_confidence - l2_confidence)
            
            # Choose best enhancement
            if l5_improvement > l4_improvement and l5_improvement > 0.05:  # L5 is better
                final_prediction = l5_prediction
                final_confidence = l5_confidence
                improvement = l5_improvement
                
                if l2_confidence < 0.50:
                    strategy = "low_confidence_replaced_by_l5"
                else:
                    strategy = "medium_confidence_enhanced_by_l5"
                
                # CRITICAL: Apply taxonomy-based category adjustment when L5 confidence ≥ 65%
                if l5_confidence >= 0.65:
                    category_adjusted = True
                    
                    # Use taxonomy to get proper L2 and L4 categories from L5
                    if l5_prediction in l5_to_l2_mapping:
                        adjusted_l2_category = l5_to_l2_mapping[l5_prediction]
                    else:
                        adjusted_l2_category = l2_prediction  # Keep original if no mapping
                        
                    if l5_prediction in l5_to_l4_mapping:
                        adjusted_l4_category = l5_to_l4_mapping[l5_prediction]  
                    else:
                        adjusted_l4_category = l4_prediction  # Keep original if no mapping
                        
            elif l4_improvement > 0.05:  # L4 is better than L2
                final_prediction = l4_prediction
                final_confidence = l4_confidence
                improvement = l4_improvement
                
                if l2_confidence < 0.50:
                    strategy = "low_confidence_enhanced_by_l4"
                else:
                    strategy = "medium_confidence_enhanced_by_l4"
                
                # For L4 enhancement, adjust L2 using L4→L2 mapping
                if l4_prediction in l4_to_l2_mapping:
                    adjusted_l2_category = l4_to_l2_mapping[l4_prediction]
        
        # Store result
        result = {
            'description': description,
            'l2_prediction': l2_prediction,
            'l2_confidence': l2_confidence,
            'l4_prediction': l4_prediction,
            'l4_confidence': l4_confidence,
            'l5_prediction': l5_prediction,
            'l5_confidence': l5_confidence,
            'final_prediction': final_prediction,
            'final_confidence': final_confidence,
            'strategy': strategy,
            'improvement': improvement,
            'category_adjusted': category_adjusted,
            'adjusted_l2_category': adjusted_l2_category,
            'adjusted_l4_category': adjusted_l4_category
        }
        results.append(result)
        
        # Track enhancements
        if improvement > 0:
            if l2_confidence >= 0.50 and l2_confidence < 0.75:
                medium_enhanced.append(result)
            elif l2_confidence < 0.50:
                low_enhanced.append(result)
        
        # Track category adjustments
        if category_adjusted:
            category_adjustments.append(result)
    
    # Convert to DataFrames
    results_df = pd.DataFrame(results)
    medium_enhanced_df = pd.DataFrame(medium_enhanced)
    low_enhanced_df = pd.DataFrame(low_enhanced)
    category_adjustments_df = pd.DataFrame(category_adjustments)
    
    # Calculate statistics
    total_enhanced = len(medium_enhanced_df) + len(low_enhanced_df)
    enhancement_rate = total_enhanced / len(results_df) * 100
    overall_improvement = results_df['final_confidence'].mean() - results_df['l2_confidence'].mean()
    
    print(f"\n📊 ENHANCED HYBRID RESULTS WITH PROPER TAXONOMY:")
    print(f"   • Total items analyzed: {len(results_df)}")
    print(f"   • Items enhanced: {total_enhanced} ({enhancement_rate:.1f}%)")
    print(f"   • Overall confidence improvement: +{overall_improvement:.4f} ({overall_improvement*100:.2f}%)")
    print(f"   • Items with category adjustments: {len(category_adjustments_df)}")
    
    # Medium confidence analysis
    medium_total = len(results_df[(results_df['l2_confidence'] >= 0.50) & (results_df['l2_confidence'] < 0.75)])
    medium_enhancement_rate = len(medium_enhanced_df) / medium_total * 100 if medium_total > 0 else 0
    print(f"\n🎯 Medium Confidence (50-75%):")
    print(f"   • Total medium confidence: {medium_total}")
    print(f"   • Enhanced: {len(medium_enhanced_df)} ({medium_enhancement_rate:.1f}%)")
    if len(medium_enhanced_df) > 0:
        print(f"   • Average improvement: +{medium_enhanced_df['improvement'].mean():.4f}")
    
    # Low confidence analysis  
    low_total = len(results_df[results_df['l2_confidence'] < 0.50])
    low_enhancement_rate = len(low_enhanced_df) / low_total * 100 if low_total > 0 else 0
    print(f"\n🎯 Low Confidence (<50%):")
    print(f"   • Total low confidence: {low_total}")
    print(f"   • Enhanced: {len(low_enhanced_df)} ({low_enhancement_rate:.1f}%)")
    if len(low_enhanced_df) > 0:
        print(f"   • Average improvement: +{low_enhanced_df['improvement'].mean():.4f}")
    
    # Strategy distribution
    print(f"\n📈 Strategy Distribution:")
    strategy_counts = results_df['strategy'].value_counts()
    for strategy, count in strategy_counts.items():
        percentage = count / len(results_df) * 100
        print(f"   • {strategy}: {count} ({percentage:.1f}%)")
    
    # Category adjustment analysis
    print(f"\n🔄 Category Adjustments (L5 confidence ≥ 65%):")
    print(f"   • Items with adjustments: {len(category_adjustments_df)}")
    
    if len(category_adjustments_df) > 0:
        l2_changes = (category_adjustments_df['l2_prediction'] != category_adjustments_df['adjusted_l2_category']).sum()
        l4_changes = (category_adjustments_df['l4_prediction'] != category_adjustments_df['adjusted_l4_category']).sum()
        
        print(f"   • L2 category changes: {l2_changes}")
        print(f"   • L4 category changes: {l4_changes}")
        
        if l2_changes > 0:
            print(f"\n🔄 Sample L2 Category Adjustments:")
            l2_sample_changes = category_adjustments_df[
                category_adjustments_df['l2_prediction'] != category_adjustments_df['adjusted_l2_category']
            ].head(5)
            
            for idx, row in l2_sample_changes.iterrows():
                print(f"   • {row['l2_prediction']} → {row['adjusted_l2_category']}")
                print(f"     Based on L5: {row['l5_prediction']} (conf: {row['l5_confidence']:.3f})")
    
    # Save results
    output_file = 'enhanced_hybrid_taxonomy_experiment_results.xlsx'
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Enhanced_Hybrid_Results', index=False)
        medium_enhanced_df.to_excel(writer, sheet_name='Medium_Confidence_Enhanced', index=False)
        low_enhanced_df.to_excel(writer, sheet_name='Low_Confidence_Enhanced', index=False)
        category_adjustments_df.to_excel(writer, sheet_name='Category_Adjustments', index=False)
    
    print(f"\n💾 Results saved to: {output_file}")
    print(f"📊 Sheets: Enhanced_Hybrid_Results, Medium_Confidence_Enhanced, Low_Confidence_Enhanced, Category_Adjustments")
    
    return results_df, medium_enhanced_df, low_enhanced_df, category_adjustments_df

def main():
    """Main execution function"""
    print("🧪 ENHANCED HYBRID EXPERIMENT WITH PROPER TAXONOMY MAPPING")
    print("=" * 70)
    
    # Load taxonomy mappings
    l5_to_l4_mapping, l5_to_l2_mapping, l4_to_l2_mapping = load_taxonomy_mappings()
    
    if not l5_to_l2_mapping:
        print("❌ Failed to load taxonomy mappings. Exiting.")
        return
    
    # Load models and data
    print("🔄 Loading models and test data...")
    models, test_data = load_models_and_data()
    
    # Run enhanced hybrid experiment with proper taxonomy
    results_df, medium_enhanced_df, low_enhanced_df, category_adjustments_df = enhanced_hybrid_classification_with_taxonomy(
        models, test_data, l5_to_l4_mapping, l5_to_l2_mapping, l4_to_l2_mapping
    )
    
    print(f"\n" + "=" * 70)
    print(f"🎉 ENHANCED HYBRID EXPERIMENT WITH PROPER TAXONOMY COMPLETE!")
    print(f"📋 Key Improvement: L2 and L4 categories now properly derived from L5 using taxonomy")
    print(f"🔄 Category adjustments follow proper hierarchical structure")
    print("=" * 70)

if __name__ == "__main__":
    main()
