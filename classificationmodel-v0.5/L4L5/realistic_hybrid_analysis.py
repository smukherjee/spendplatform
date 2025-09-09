#!/usr/bin/env python3
"""
Refined Hybrid Classification with Calibrated Confidence Scores
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
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
                # Multi-class case
                confidence = 1 / (1 + np.exp(-np.max(decision_scores)))
            else:
                # Binary case
                confidence = 1 / (1 + np.exp(-abs(decision_scores)))
        else:
            confidence = 0.5
        
        # Calibrate confidence based on model accuracy
        # Scale confidence by model reliability
        calibrated_confidence = confidence * model_accuracy + (1 - model_accuracy) * 0.5
        
        # Apply conservative scaling to prevent overconfidence
        calibrated_confidence = calibrated_confidence * 0.95  # Max 95% confidence
        
        return max(0.1, min(0.95, calibrated_confidence))  # Clamp between 10% and 95%
        
    except Exception as e:
        return 0.5

def classify_with_calibrated_confidence(models, description, model_name):
    """Classify with calibrated confidence based on model accuracy"""
    if not models[model_name]:
        return None, 0.0
    
    model_data = models[model_name]
    model = model_data['model']
    vectorizer = model_data['vectorizer']
    model_accuracy = model_data.get('accuracy', 0.5)
    
    try:
        # Vectorize text
        text_vec = vectorizer.transform([description.lower()])
        
        # Get prediction
        if model_name == 'L2':
            prediction_encoded = model.predict(text_vec)[0]
            prediction = model_data['label_encoder'].inverse_transform([prediction_encoded])[0]
        else:
            prediction = model.predict(text_vec)[0]
        
        # Get calibrated confidence
        confidence = get_calibrated_confidence(model, vectorizer, description, model_accuracy)
        
        return prediction, confidence
    except Exception as e:
        return None, 0.0

def realistic_hybrid_classification(models, description):
    """Realistic hybrid classification with proper confidence calibration"""
    
    # Get predictions from all models
    l2_pred, l2_conf = classify_with_calibrated_confidence(models, description, 'L2')
    l4_pred, l4_conf = classify_with_calibrated_confidence(models, description, 'L4')
    l5_pred, l5_conf = classify_with_calibrated_confidence(models, description, 'L5')
    
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
        'strategy': 'L2_ONLY',
        'improvement': 0.0
    }
    
    # Decision logic based on confidence levels and thresholds
    confidence_threshold_high = 0.75
    confidence_threshold_medium = 0.55
    improvement_threshold = 0.05  # Require at least 5% improvement
    
    # High confidence L2 - use as is
    if l2_conf >= confidence_threshold_high:
        result['strategy'] = 'L2_HIGH_CONFIDENCE'
        return result
    
    # Medium confidence L2 - consider L4/L5 enhancement
    elif l2_conf >= confidence_threshold_medium:
        best_alternative = None
        best_conf = l2_conf
        best_model = 'L2'
        
        # Check L4 enhancement
        if l4_pred and l4_conf > l2_conf + improvement_threshold:
            best_alternative = l4_pred
            best_conf = l4_conf
            best_model = 'L4'
        
        # Check L5 enhancement (only if better than L4)
        if l5_pred and l5_conf > best_conf + improvement_threshold:
            best_alternative = l5_pred
            best_conf = l5_conf
            best_model = 'L5'
        
        if best_alternative:
            result['final_prediction'] = best_alternative
            result['final_confidence'] = best_conf
            result['strategy'] = f'L2_MEDIUM_ENHANCED_BY_{best_model}'
            result['improvement'] = best_conf - l2_conf
        else:
            result['strategy'] = 'L2_MEDIUM_KEPT'
    
    # Low confidence L2 - use best available
    else:
        candidates = [
            (l2_pred, l2_conf, 'L2'),
            (l4_pred, l4_conf, 'L4'),
            (l5_pred, l5_conf, 'L5')
        ]
        
        valid_candidates = [(pred, conf, model) for pred, conf, model in candidates if pred is not None]
        
        if valid_candidates:
            valid_candidates.sort(key=lambda x: x[1], reverse=True)
            best_pred, best_conf, best_model = valid_candidates[0]
            
            result['final_prediction'] = best_pred
            result['final_confidence'] = best_conf
            result['strategy'] = f'L2_LOW_REPLACED_BY_{best_model}'
            result['improvement'] = best_conf - l2_conf
    
    return result

def analyze_realistic_hybrid_performance(models, test_data, sample_size=1000):
    """Analyze realistic hybrid classification performance"""
    print(f"\n🔬 REALISTIC HYBRID CLASSIFICATION ANALYSIS")
    print("=" * 60)
    
    # Sample data for testing
    if len(test_data) > sample_size:
        test_sample = test_data.sample(n=sample_size, random_state=42)
        print(f"📊 Testing on {sample_size} random samples")
    else:
        test_sample = test_data
        print(f"📊 Testing on all {len(test_sample)} samples")
    
    results = []
    
    print(f"🔄 Processing classifications...")
    for idx, row in test_sample.iterrows():
        description = row['Description']
        result = realistic_hybrid_classification(models, description)
        results.append(result)
        
        if len(results) % 200 == 0:
            print(f"   Processed {len(results)}/{len(test_sample)} items...")
    
    results_df = pd.DataFrame(results)
    return results_df

def generate_realistic_performance_report(results_df):
    """Generate realistic performance report"""
    print(f"\n📊 REALISTIC HYBRID CLASSIFICATION REPORT")
    print("=" * 70)
    
    total_items = len(results_df)
    
    # Strategy distribution
    strategy_counts = results_df['strategy'].value_counts()
    print(f"📈 Classification Strategy Distribution:")
    for strategy, count in strategy_counts.items():
        percentage = count / total_items * 100
        print(f"   • {strategy}: {count} items ({percentage:.1f}%)")
    
    # Confidence improvements
    improved_items = results_df[results_df['improvement'] > 0]
    significant_improvements = results_df[results_df['improvement'] > 0.05]
    
    print(f"\n💡 Confidence Enhancement Results:")
    print(f"   • Items with any improvement: {len(improved_items)}/{total_items} ({len(improved_items)/total_items*100:.1f}%)")
    print(f"   • Items with significant improvement (>5%): {len(significant_improvements)}/{total_items} ({len(significant_improvements)/total_items*100:.1f}%)")
    
    if len(improved_items) > 0:
        avg_improvement = improved_items['improvement'].mean()
        print(f"   • Average improvement (enhanced items): {avg_improvement:.4f} ({avg_improvement*100:.2f}%)")
        print(f"   • Maximum improvement: {results_df['improvement'].max():.4f}")
    
    # Overall confidence comparison
    l2_conf_avg = results_df['l2_confidence'].mean()
    final_conf_avg = results_df['final_confidence'].mean()
    overall_improvement = final_conf_avg - l2_conf_avg
    
    print(f"\n📊 Overall Confidence Comparison:")
    print(f"   • L2 Average Confidence: {l2_conf_avg:.4f} ({l2_conf_avg*100:.2f}%)")
    print(f"   • Hybrid Average Confidence: {final_conf_avg:.4f} ({final_conf_avg*100:.2f}%)")
    print(f"   • Overall Improvement: {overall_improvement:.4f} ({overall_improvement*100:.2f}%)")
    
    # Confidence tier analysis
    print(f"\n🎯 Confidence Tier Analysis:")
    
    # Define more realistic tiers
    high_threshold = 0.75
    medium_low_threshold = 0.55
    
    # L2 tiers
    l2_high = (results_df['l2_confidence'] >= high_threshold).sum()
    l2_medium = ((results_df['l2_confidence'] >= medium_low_threshold) & (results_df['l2_confidence'] < high_threshold)).sum()
    l2_low = (results_df['l2_confidence'] < medium_low_threshold).sum()
    
    # Final tiers
    final_high = (results_df['final_confidence'] >= high_threshold).sum()
    final_medium = ((results_df['final_confidence'] >= medium_low_threshold) & (results_df['final_confidence'] < high_threshold)).sum()
    final_low = (results_df['final_confidence'] < medium_low_threshold).sum()
    
    print(f"   L2 Model Tiers:")
    print(f"     • High (≥75%): {l2_high} items ({l2_high/total_items*100:.1f}%)")
    print(f"     • Medium (55-75%): {l2_medium} items ({l2_medium/total_items*100:.1f}%)")
    print(f"     • Low (<55%): {l2_low} items ({l2_low/total_items*100:.1f}%)")
    
    print(f"   Hybrid Model Tiers:")
    print(f"     • High (≥75%): {final_high} items ({final_high/total_items*100:.1f}%) [+{final_high-l2_high}]")
    print(f"     • Medium (55-75%): {final_medium} items ({final_medium/total_items*100:.1f}%) [+{final_medium-l2_medium}]")
    print(f"     • Low (<55%): {final_low} items ({final_low/total_items*100:.1f}%) [+{final_low-l2_low}]")
    
    # Medium confidence enhancement specific analysis
    medium_conf_items = results_df[(results_df['l2_confidence'] >= medium_low_threshold) & (results_df['l2_confidence'] < high_threshold)]
    enhanced_medium = medium_conf_items[medium_conf_items['improvement'] > 0.05]
    
    print(f"\n🎯 Medium Confidence Enhancement Focus:")
    print(f"   • Medium confidence L2 items: {len(medium_conf_items)}")
    print(f"   • Significantly enhanced (>5%): {len(enhanced_medium)} ({len(enhanced_medium)/len(medium_conf_items)*100 if len(medium_conf_items) > 0 else 0:.1f}%)")
    
    if len(enhanced_medium) > 0:
        avg_medium_improvement = enhanced_medium['improvement'].mean()
        print(f"   • Average enhancement: {avg_medium_improvement:.4f} ({avg_medium_improvement*100:.2f}%)")
        
        # Enhanced by which model?
        l4_enhanced = enhanced_medium[enhanced_medium['strategy'].str.contains('L4')].shape[0]
        l5_enhanced = enhanced_medium[enhanced_medium['strategy'].str.contains('L5')].shape[0]
        
        print(f"   • Enhanced by L4: {l4_enhanced}")
        print(f"   • Enhanced by L5: {l5_enhanced}")
    
    return {
        'total_items': total_items,
        'improved_items': len(improved_items),
        'significant_improvements': len(significant_improvements),
        'overall_improvement': overall_improvement,
        'medium_enhanced': len(enhanced_medium) if len(medium_conf_items) > 0 else 0,
        'medium_total': len(medium_conf_items)
    }

def save_realistic_results(results_df):
    """Save realistic results to Excel file"""
    output_path = '/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.5/L4L5/realistic_hybrid_results.xlsx'
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Main results
        results_df.to_excel(writer, sheet_name='Realistic_Hybrid_Results', index=False)
        
        # Focus on medium confidence enhancements
        medium_enhanced = results_df[
            (results_df['l2_confidence'] >= 0.55) & 
            (results_df['l2_confidence'] < 0.75) & 
            (results_df['improvement'] > 0.05)
        ]
        medium_enhanced.to_excel(writer, sheet_name='Medium_Confidence_Enhanced', index=False)
        
        # Summary
        summary = pd.DataFrame({
            'Metric': [
                'Total Items',
                'L2 Avg Confidence',
                'Hybrid Avg Confidence', 
                'Overall Improvement',
                'Items with Any Enhancement',
                'Items with Significant Enhancement (>5%)',
                'Medium Confidence Items Enhanced'
            ],
            'Value': [
                len(results_df),
                results_df['l2_confidence'].mean(),
                results_df['final_confidence'].mean(),
                results_df['final_confidence'].mean() - results_df['l2_confidence'].mean(),
                (results_df['improvement'] > 0).sum(),
                (results_df['improvement'] > 0.05).sum(),
                len(medium_enhanced)
            ]
        })
        summary.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"💾 Results saved to: {output_path}")

def main():
    """Main realistic analysis function"""
    print("🔍 REALISTIC HYBRID L2 + L4/L5 CLASSIFICATION")
    print("=" * 70)
    
    # Load models
    models = load_models()
    if not models or not models['L2']:
        print("❌ Cannot proceed without L2 model")
        return
    
    # Load test data
    try:
        file_path = '/Users/sujoymukherjee/code/spendplatform/context/unclassified transactions.xlsx'
        test_data = pd.read_excel(file_path)
        test_data = test_data[test_data['Description'].notna()].copy()
        print(f"📊 Test data loaded: {len(test_data)} unclassified transactions")
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return
    
    # Analyze realistic hybrid performance
    results_df = analyze_realistic_hybrid_performance(models, test_data, sample_size=1000)
    
    # Generate performance report
    performance_summary = generate_realistic_performance_report(results_df)
    
    # Save results
    save_realistic_results(results_df)
    
    print(f"\n" + "=" * 70)
    print(f"🎉 REALISTIC HYBRID ANALYSIS COMPLETE!")
    print(f"📊 Key Realistic Results:")
    print(f"   • Overall confidence improvement: {performance_summary['overall_improvement']*100:.2f}%")
    print(f"   • Significantly enhanced items: {performance_summary['significant_improvements']}/{performance_summary['total_items']}")
    print(f"   • Medium confidence enhanced: {performance_summary['medium_enhanced']}/{performance_summary['medium_total']}")
    print("=" * 70)

if __name__ == "__main__":
    main()
