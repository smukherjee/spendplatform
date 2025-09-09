#!/usr/bin/env python3
"""
Enhanced Hybrid Classification Experiment
Test L4/L5 enhancement for medium and low confidence L2 predictions
When L5 confidence > 65%, also adjust L2 and L4 category predictions
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

def load_models():
    """Load all trained models"""
    print("🔧 LOADING CLASSIFICATION MODELS")
    print("=" * 60)
    
    models = {}
    
    # Load L2 model
    try:
        l2_path = '/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.5/models/production_svm_enhanced.pkl'
        with open(l2_path, 'rb') as f:
            models['L2'] = pickle.load(f)
        print(f"✅ L2 Model loaded: {models['L2'].get('accuracy', 0):.4f} accuracy")
    except Exception as e:
        print(f"❌ Error loading L2 model: {e}")
        return None
    
    # Load L4 model
    try:
        l4_path = '/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.5/L4L5/l4_simplified_model.pkl'
        with open(l4_path, 'rb') as f:
            models['L4'] = pickle.load(f)
        print(f"✅ L4 Model loaded: {models['L4'].get('accuracy', 0):.4f} accuracy")
    except Exception as e:
        print(f"❌ Error loading L4 model: {e}")
        models['L4'] = None
    
    # Load L5 model
    try:
        l5_path = '/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.5/L4L5/l5_simplified_model.pkl'
        with open(l5_path, 'rb') as f:
            models['L5'] = pickle.load(f)
        print(f"✅ L5 Model loaded: {models['L5'].get('accuracy', 0):.4f} accuracy")
    except Exception as e:
        print(f"❌ Error loading L5 model: {e}")
        models['L5'] = None
    
    return models

def load_mapping_data():
    """Load L4/L5 mapping data for category adjustment"""
    try:
        file_path = '/Users/sujoymukherjee/code/spendplatform/context/Copy of Catergorization Working Sheet_Labelled Data for Training.xlsx'
        df = pd.read_excel(file_path, sheet_name='Working File_KGP', header=2)
        
        # Create mapping dictionaries
        l5_to_l4_mapping = {}
        l5_to_l2_mapping = {}
        l4_to_l2_mapping = {}
        
        for _, row in df.iterrows():
            if pd.notna(row['Category L5']) and pd.notna(row['Category L4']):
                l5_to_l4_mapping[row['Category L5']] = row['Category L4']
            
            if pd.notna(row['Category L5']) and pd.notna(row['Category L2']):
                l5_to_l2_mapping[row['Category L5']] = row['Category L2']
            
            if pd.notna(row['Category L4']) and pd.notna(row['Category L2']):
                l4_to_l2_mapping[row['Category L4']] = row['Category L2']
        
        print(f"📊 Mapping data loaded:")
        print(f"   • L5→L4 mappings: {len(l5_to_l4_mapping)}")
        print(f"   • L5→L2 mappings: {len(l5_to_l2_mapping)}")
        print(f"   • L4→L2 mappings: {len(l4_to_l2_mapping)}")
        
        return {
            'l5_to_l4': l5_to_l4_mapping,
            'l5_to_l2': l5_to_l2_mapping,
            'l4_to_l2': l4_to_l2_mapping
        }
        
    except Exception as e:
        print(f"❌ Error loading mapping data: {e}")
        return None

def get_calibrated_confidence(model, vectorizer, text, model_accuracy):
    """Calculate calibrated prediction confidence score"""
    try:
        text_vec = vectorizer.transform([text.lower()])
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(text_vec)[0]
            confidence = np.max(proba)
        elif hasattr(model, 'decision_function'):
            decision_scores = model.decision_function(text_vec)[0]
            if len(decision_scores) > 1:
                confidence = 1 / (1 + np.exp(-np.max(decision_scores)))
            else:
                confidence = 1 / (1 + np.exp(-abs(decision_scores)))
        else:
            confidence = 0.5
        
        # Calibrate confidence based on model accuracy
        calibrated_confidence = confidence * model_accuracy + (1 - model_accuracy) * 0.5
        calibrated_confidence = calibrated_confidence * 0.95  # Max 95% confidence
        
        return max(0.1, min(0.95, calibrated_confidence))
        
    except Exception as e:
        return 0.5

def classify_with_model(models, description, model_name):
    """Classify with specified model and return prediction with confidence"""
    if not models[model_name]:
        return None, 0.0
    
    model_data = models[model_name]
    model = model_data['model']
    vectorizer = model_data['vectorizer']
    model_accuracy = model_data.get('accuracy', 0.5)
    
    try:
        text_vec = vectorizer.transform([description.lower()])
        
        if model_name == 'L2':
            prediction_encoded = model.predict(text_vec)[0]
            prediction = model_data['label_encoder'].inverse_transform([prediction_encoded])[0]
        else:
            prediction = model.predict(text_vec)[0]
        
        confidence = get_calibrated_confidence(model, vectorizer, description, model_accuracy)
        
        return prediction, confidence
    except Exception as e:
        return None, 0.0

def enhanced_hybrid_classification(models, mappings, description):
    """
    Enhanced hybrid classification with category adjustment when L5 confidence > 65%
    """
    
    # Get predictions from all models
    l2_pred, l2_conf = classify_with_model(models, description, 'L2')
    l4_pred, l4_conf = classify_with_model(models, description, 'L4')
    l5_pred, l5_conf = classify_with_model(models, description, 'L5')
    
    result = {
        'description': description[:50] + '...' if len(description) > 50 else description,
        'l2_prediction': l2_pred,
        'l2_confidence': l2_conf,
        'l4_prediction': l4_pred,
        'l4_confidence': l4_conf,
        'l5_prediction': l5_pred,
        'l5_confidence': l5_conf,
        'final_prediction': l2_pred,
        'final_confidence': l2_conf,
        'adjusted_l2_category': None,
        'adjusted_l4_category': None,
        'strategy': 'L2_ONLY',
        'improvement': 0.0,
        'category_adjusted': False
    }
    
    # Define confidence thresholds
    high_threshold = 0.75
    medium_threshold = 0.50
    l5_adjustment_threshold = 0.65
    improvement_threshold = 0.05
    
    # Check for L5 high confidence category adjustment
    category_adjusted = False
    if l5_pred and l5_conf >= l5_adjustment_threshold and mappings:
        # Adjust L2 category based on L5 prediction
        if l5_pred in mappings['l5_to_l2']:
            adjusted_l2 = mappings['l5_to_l2'][l5_pred]
            result['adjusted_l2_category'] = adjusted_l2
            category_adjusted = True
        
        # Adjust L4 category based on L5 prediction
        if l5_pred in mappings['l5_to_l4']:
            adjusted_l4 = mappings['l5_to_l4'][l5_pred]
            result['adjusted_l4_category'] = adjusted_l4
            category_adjusted = True
        
        result['category_adjusted'] = category_adjusted
    
    # Classification logic based on L2 confidence levels
    if l2_conf >= high_threshold:
        # High confidence L2 - use as is, but still check for category adjustment
        if category_adjusted and l5_conf > l2_conf:
            result['strategy'] = 'L2_HIGH_WITH_L5_CATEGORY_ADJUSTMENT'
            # Optionally use L5 prediction if significantly better
            if l5_conf > l2_conf + improvement_threshold:
                result['final_prediction'] = l5_pred
                result['final_confidence'] = l5_conf
                result['improvement'] = l5_conf - l2_conf
        else:
            result['strategy'] = 'L2_HIGH_CONFIDENCE'
    
    elif l2_conf >= medium_threshold:
        # Medium confidence L2 - try enhancement
        best_pred = l2_pred
        best_conf = l2_conf
        best_model = 'L2'
        
        # Check L4 enhancement
        if l4_pred and l4_conf > l2_conf + improvement_threshold:
            best_pred = l4_pred
            best_conf = l4_conf
            best_model = 'L4'
        
        # Check L5 enhancement
        if l5_pred and l5_conf > best_conf + improvement_threshold:
            best_pred = l5_pred
            best_conf = l5_conf
            best_model = 'L5'
        
        if best_model != 'L2':
            result['final_prediction'] = best_pred
            result['final_confidence'] = best_conf
            result['strategy'] = f'L2_MEDIUM_ENHANCED_BY_{best_model}'
            result['improvement'] = best_conf - l2_conf
        else:
            result['strategy'] = 'L2_MEDIUM_KEPT'
            if category_adjusted:
                result['strategy'] += '_WITH_CATEGORY_ADJUSTMENT'
    
    else:
        # Low confidence L2 - use best available
        candidates = []
        
        if l2_pred:
            candidates.append((l2_pred, l2_conf, 'L2'))
        if l4_pred:
            candidates.append((l4_pred, l4_conf, 'L4'))
        if l5_pred:
            candidates.append((l5_pred, l5_conf, 'L5'))
        
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_pred, best_conf, best_model = candidates[0]
            
            result['final_prediction'] = best_pred
            result['final_confidence'] = best_conf
            result['strategy'] = f'L2_LOW_REPLACED_BY_{best_model}'
            result['improvement'] = best_conf - l2_conf
            
            if category_adjusted:
                result['strategy'] += '_WITH_CATEGORY_ADJUSTMENT'
    
    return result

def analyze_enhanced_hybrid_performance(models, mappings, test_data, sample_size=1000):
    """Analyze enhanced hybrid classification performance"""
    print(f"\n🧪 ENHANCED HYBRID CLASSIFICATION EXPERIMENT")
    print("=" * 60)
    
    # Sample data for testing
    if len(test_data) > sample_size:
        test_sample = test_data.sample(n=sample_size, random_state=42)
        print(f"📊 Testing on {sample_size} random samples")
    else:
        test_sample = test_data
        print(f"📊 Testing on all {len(test_sample)} samples")
    
    results = []
    
    print(f"🔄 Processing enhanced classifications...")
    for idx, row in test_sample.iterrows():
        description = row['Description']
        result = enhanced_hybrid_classification(models, mappings, description)
        results.append(result)
        
        if len(results) % 200 == 0:
            print(f"   Processed {len(results)}/{len(test_sample)} items...")
    
    results_df = pd.DataFrame(results)
    return results_df

def generate_enhanced_performance_report(results_df):
    """Generate comprehensive enhanced performance report"""
    print(f"\n📊 ENHANCED HYBRID CLASSIFICATION REPORT")
    print("=" * 70)
    
    total_items = len(results_df)
    
    # Strategy distribution
    strategy_counts = results_df['strategy'].value_counts()
    print(f"📈 Classification Strategy Distribution:")
    for strategy, count in strategy_counts.items():
        percentage = count / total_items * 100
        print(f"   • {strategy}: {count} items ({percentage:.1f}%)")
    
    # Category adjustment analysis
    category_adjusted_items = results_df[results_df['category_adjusted'] == True]
    l5_high_conf_items = results_df[results_df['l5_confidence'] >= 0.65]
    
    print(f"\n🔄 Category Adjustment Analysis:")
    print(f"   • Items with L5 confidence ≥ 65%: {len(l5_high_conf_items)}/{total_items} ({len(l5_high_conf_items)/total_items*100:.1f}%)")
    print(f"   • Items with category adjustments: {len(category_adjusted_items)}/{total_items} ({len(category_adjusted_items)/total_items*100:.1f}%)")
    
    if len(category_adjusted_items) > 0:
        l2_adjustments = category_adjusted_items['adjusted_l2_category'].notna().sum()
        l4_adjustments = category_adjusted_items['adjusted_l4_category'].notna().sum()
        print(f"   • L2 category adjustments: {l2_adjustments}")
        print(f"   • L4 category adjustments: {l4_adjustments}")
    
    # Confidence improvements by L2 confidence tier
    print(f"\n🎯 Confidence Enhancement by L2 Tier:")
    
    # Medium confidence analysis (50-75%)
    medium_conf_items = results_df[(results_df['l2_confidence'] >= 0.50) & (results_df['l2_confidence'] < 0.75)]
    medium_enhanced = medium_conf_items[medium_conf_items['improvement'] > 0.05]
    
    print(f"   Medium Confidence (50-75%):")
    print(f"     • Total items: {len(medium_conf_items)}")
    print(f"     • Enhanced items: {len(medium_enhanced)} ({len(medium_enhanced)/len(medium_conf_items)*100 if len(medium_conf_items) > 0 else 0:.1f}%)")
    if len(medium_enhanced) > 0:
        print(f"     • Average improvement: {medium_enhanced['improvement'].mean():.4f} ({medium_enhanced['improvement'].mean()*100:.2f}%)")
    
    # Low confidence analysis (<50%)
    low_conf_items = results_df[results_df['l2_confidence'] < 0.50]
    low_enhanced = low_conf_items[low_conf_items['improvement'] > 0.05]
    
    print(f"   Low Confidence (<50%):")
    print(f"     • Total items: {len(low_conf_items)}")
    print(f"     • Enhanced items: {len(low_enhanced)} ({len(low_enhanced)/len(low_conf_items)*100 if len(low_conf_items) > 0 else 0:.1f}%)")
    if len(low_enhanced) > 0:
        print(f"     • Average improvement: {low_enhanced['improvement'].mean():.4f} ({low_enhanced['improvement'].mean()*100:.2f}%)")
    
    # Overall improvement analysis
    improved_items = results_df[results_df['improvement'] > 0]
    significant_improvements = results_df[results_df['improvement'] > 0.05]
    
    print(f"\n💡 Overall Enhancement Results:")
    print(f"   • Items with any improvement: {len(improved_items)}/{total_items} ({len(improved_items)/total_items*100:.1f}%)")
    print(f"   • Items with significant improvement (>5%): {len(significant_improvements)}/{total_items} ({len(significant_improvements)/total_items*100:.1f}%)")
    
    if len(improved_items) > 0:
        avg_improvement = improved_items['improvement'].mean()
        print(f"   • Average improvement (enhanced items): {avg_improvement:.4f} ({avg_improvement*100:.2f}%)")
    
    # Overall confidence comparison
    l2_conf_avg = results_df['l2_confidence'].mean()
    final_conf_avg = results_df['final_confidence'].mean()
    overall_improvement = final_conf_avg - l2_conf_avg
    
    print(f"\n📊 Overall Confidence Comparison:")
    print(f"   • L2 Average Confidence: {l2_conf_avg:.4f} ({l2_conf_avg*100:.2f}%)")
    print(f"   • Enhanced Average Confidence: {final_conf_avg:.4f} ({final_conf_avg*100:.2f}%)")
    print(f"   • Overall Improvement: {overall_improvement:.4f} ({overall_improvement*100:.2f}%)")
    
    # Examples of high L5 confidence with category adjustments
    high_l5_with_adjustments = results_df[
        (results_df['l5_confidence'] >= 0.65) & 
        (results_df['category_adjusted'] == True)
    ]
    
    if len(high_l5_with_adjustments) > 0:
        print(f"\n🔄 High L5 Confidence Category Adjustment Examples:")
        for idx, row in high_l5_with_adjustments.head(5).iterrows():
            print(f"   • {row['description']}")
            print(f"     L2: {row['l2_prediction']} → L5: {row['l5_prediction']} (conf: {row['l5_confidence']:.3f})")
            if pd.notna(row['adjusted_l2_category']):
                print(f"     Adjusted L2: {row['adjusted_l2_category']}")
            if pd.notna(row['adjusted_l4_category']):
                print(f"     Adjusted L4: {row['adjusted_l4_category']}")
    
    return {
        'total_items': total_items,
        'medium_enhanced': len(medium_enhanced),
        'medium_total': len(medium_conf_items),
        'low_enhanced': len(low_enhanced),
        'low_total': len(low_conf_items),
        'category_adjustments': len(category_adjusted_items),
        'overall_improvement': overall_improvement
    }

def save_enhanced_results(results_df):
    """Save enhanced results to Excel file"""
    output_path = '/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.5/L4L5/enhanced_hybrid_experiment_results.xlsx'
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Main results
        results_df.to_excel(writer, sheet_name='Enhanced_Hybrid_Results', index=False)
        
        # Medium confidence enhancements
        medium_enhanced = results_df[
            (results_df['l2_confidence'] >= 0.50) & 
            (results_df['l2_confidence'] < 0.75) & 
            (results_df['improvement'] > 0.05)
        ]
        medium_enhanced.to_excel(writer, sheet_name='Medium_Confidence_Enhanced', index=False)
        
        # Low confidence enhancements
        low_enhanced = results_df[
            (results_df['l2_confidence'] < 0.50) & 
            (results_df['improvement'] > 0.05)
        ]
        low_enhanced.to_excel(writer, sheet_name='Low_Confidence_Enhanced', index=False)
        
        # Category adjustments
        category_adjusted = results_df[results_df['category_adjusted'] == True]
        category_adjusted.to_excel(writer, sheet_name='Category_Adjustments', index=False)
        
        # Summary
        summary = pd.DataFrame({
            'Metric': [
                'Total Items',
                'Medium Confidence Items',
                'Medium Confidence Enhanced',
                'Low Confidence Items', 
                'Low Confidence Enhanced',
                'Category Adjustments',
                'Overall Improvement'
            ],
            'Value': [
                len(results_df),
                len(results_df[(results_df['l2_confidence'] >= 0.50) & (results_df['l2_confidence'] < 0.75)]),
                len(medium_enhanced),
                len(results_df[results_df['l2_confidence'] < 0.50]),
                len(low_enhanced),
                (results_df['category_adjusted'] == True).sum(),
                results_df['final_confidence'].mean() - results_df['l2_confidence'].mean()
            ]
        })
        summary.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"💾 Enhanced results saved to: {output_path}")

def main():
    """Main enhanced analysis function"""
    print("🧪 ENHANCED HYBRID CLASSIFICATION EXPERIMENT")
    print("=" * 70)
    print("Testing L4/L5 enhancement for medium and low confidence L2 predictions")
    print("With L5 confidence > 65% category adjustment feature")
    print("=" * 70)
    
    # Load models
    models = load_models()
    if not models or not models['L2']:
        print("❌ Cannot proceed without L2 model")
        return
    
    # Load mapping data
    mappings = load_mapping_data()
    if not mappings:
        print("⚠️ Proceeding without category mappings")
    
    # Load test data
    try:
        file_path = '/Users/sujoymukherjee/code/spendplatform/context/unclassified transactions.xlsx'
        test_data = pd.read_excel(file_path)
        test_data = test_data[test_data['Description'].notna()].copy()
        print(f"📊 Test data loaded: {len(test_data)} unclassified transactions")
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return
    
    # Analyze enhanced hybrid performance
    results_df = analyze_enhanced_hybrid_performance(models, mappings, test_data, sample_size=1500)
    
    # Generate performance report
    performance_summary = generate_enhanced_performance_report(results_df)
    
    # Save results
    save_enhanced_results(results_df)
    
    print(f"\n" + "=" * 70)
    print(f"🎉 ENHANCED HYBRID EXPERIMENT COMPLETE!")
    print(f"📊 Key Results:")
    print(f"   • Medium confidence enhanced: {performance_summary['medium_enhanced']}/{performance_summary['medium_total']}")
    print(f"   • Low confidence enhanced: {performance_summary['low_enhanced']}/{performance_summary['low_total']}")
    print(f"   • Category adjustments: {performance_summary['category_adjustments']}")
    print(f"   • Overall improvement: {performance_summary['overall_improvement']*100:.2f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
