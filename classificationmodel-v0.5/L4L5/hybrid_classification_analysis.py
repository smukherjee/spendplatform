#!/usr/bin/env python3
"""
Hybrid Classification: L2 + L4/L5 Enhancement for Medium Confidence Predictions
Test if L4/L5 models can improve confidence for medium-confidence L2 predictions
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
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

def load_test_data():
    """Load unclassified transactions for testing"""
    try:
        file_path = '/Users/sujoymukherjee/code/spendplatform/context/unclassified transactions.xlsx'
        df = pd.read_excel(file_path)
        
        # Clean and prepare data
        df = df[df['Description'].notna()].copy()
        df['Description'] = df['Description'].astype(str).str.strip()
        
        print(f"📊 Test data loaded: {len(df)} unclassified transactions")
        return df
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return None

def get_prediction_confidence(model, vectorizer, text, prediction_proba=None):
    """Calculate prediction confidence score"""
    if hasattr(model, 'predict_proba'):
        if prediction_proba is None:
            text_vec = vectorizer.transform([text.lower()])
            proba = model.predict_proba(text_vec)[0]
        else:
            proba = prediction_proba
        confidence = np.max(proba)
        return confidence
    else:
        # For SVM without probability, use decision function
        text_vec = vectorizer.transform([text.lower()])
        if hasattr(model, 'decision_function'):
            decision_scores = model.decision_function(text_vec)[0]
            # Normalize decision scores to [0,1] range
            confidence = 1 / (1 + np.exp(-np.max(decision_scores)))
            return confidence
    
    return 0.5  # Default medium confidence

def classify_with_l2(models, description):
    """Classify using L2 model and return prediction with confidence"""
    l2_model = models['L2']['model']
    l2_vectorizer = models['L2']['vectorizer']
    l2_label_encoder = models['L2']['label_encoder']
    
    # Vectorize text
    text_vec = l2_vectorizer.transform([description.lower()])
    
    # Get prediction and probability
    prediction_encoded = l2_model.predict(text_vec)[0]
    prediction = l2_label_encoder.inverse_transform([prediction_encoded])[0]
    
    # Get confidence
    confidence = get_prediction_confidence(l2_model, l2_vectorizer, description)
    
    return prediction, confidence

def classify_with_l4(models, description):
    """Classify using L4 model and return prediction with confidence"""
    if not models['L4']:
        return None, 0.0
    
    l4_model = models['L4']['model']
    l4_vectorizer = models['L4']['vectorizer']
    
    try:
        # Vectorize text
        text_vec = l4_vectorizer.transform([description.lower()])
        
        # Get prediction
        prediction = l4_model.predict(text_vec)[0]
        
        # Get confidence
        confidence = get_prediction_confidence(l4_model, l4_vectorizer, description)
        
        return prediction, confidence
    except Exception as e:
        return None, 0.0

def classify_with_l5(models, description):
    """Classify using L5 model and return prediction with confidence"""
    if not models['L5']:
        return None, 0.0
    
    l5_model = models['L5']['model']
    l5_vectorizer = models['L5']['vectorizer']
    
    try:
        # Vectorize text
        text_vec = l5_vectorizer.transform([description.lower()])
        
        # Get prediction
        prediction = l5_model.predict(text_vec)[0]
        
        # Get confidence
        confidence = get_prediction_confidence(l5_model, l5_vectorizer, description)
        
        return prediction, confidence
    except Exception as e:
        return None, 0.0

def hybrid_classification(models, description, l2_confidence_threshold_low=0.6, l2_confidence_threshold_high=0.8):
    """
    Hybrid classification strategy:
    - High L2 confidence (>0.8): Use L2 prediction
    - Medium L2 confidence (0.6-0.8): Try L4/L5 enhancement
    - Low L2 confidence (<0.6): Use best of L2/L4/L5
    """
    
    # Get L2 prediction
    l2_pred, l2_conf = classify_with_l2(models, description)
    
    result = {
        'description': description[:50] + '...' if len(description) > 50 else description,
        'l2_prediction': l2_pred,
        'l2_confidence': l2_conf,
        'final_prediction': l2_pred,
        'final_confidence': l2_conf,
        'strategy': 'L2_ONLY',
        'improvement': 0.0
    }
    
    # High confidence L2 - use as is
    if l2_conf >= l2_confidence_threshold_high:
        result['strategy'] = 'L2_HIGH_CONFIDENCE'
        return result
    
    # Medium/Low confidence L2 - try L4/L5 enhancement
    l4_pred, l4_conf = classify_with_l4(models, description)
    l5_pred, l5_conf = classify_with_l5(models, description)
    
    result['l4_prediction'] = l4_pred
    result['l4_confidence'] = l4_conf
    result['l5_prediction'] = l5_pred
    result['l5_confidence'] = l5_conf
    
    # Find best confidence among all models
    candidates = [
        (l2_pred, l2_conf, 'L2'),
        (l4_pred, l4_conf, 'L4'),
        (l5_pred, l5_conf, 'L5')
    ]
    
    # Filter out None predictions
    valid_candidates = [(pred, conf, model) for pred, conf, model in candidates if pred is not None]
    
    if not valid_candidates:
        return result
    
    # Sort by confidence
    valid_candidates.sort(key=lambda x: x[1], reverse=True)
    best_pred, best_conf, best_model = valid_candidates[0]
    
    # Decision logic
    if l2_conf >= l2_confidence_threshold_low:
        # Medium confidence L2 - enhance if L4/L5 significantly better
        if best_conf > l2_conf + 0.1:  # At least 10% improvement
            result['final_prediction'] = best_pred
            result['final_confidence'] = best_conf
            result['strategy'] = f'L2_MEDIUM_ENHANCED_BY_{best_model}'
            result['improvement'] = best_conf - l2_conf
        else:
            result['strategy'] = 'L2_MEDIUM_KEPT'
    else:
        # Low confidence L2 - use best available
        result['final_prediction'] = best_pred
        result['final_confidence'] = best_conf
        result['strategy'] = f'L2_LOW_REPLACED_BY_{best_model}'
        result['improvement'] = best_conf - l2_conf
    
    return result

def analyze_hybrid_performance(models, test_data, sample_size=500):
    """Analyze hybrid classification performance"""
    print(f"\n🔬 HYBRID CLASSIFICATION ANALYSIS")
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
        result = hybrid_classification(models, description)
        results.append(result)
        
        if len(results) % 100 == 0:
            print(f"   Processed {len(results)}/{len(test_sample)} items...")
    
    results_df = pd.DataFrame(results)
    
    return results_df

def generate_performance_report(results_df):
    """Generate comprehensive performance report"""
    print(f"\n📊 HYBRID CLASSIFICATION PERFORMANCE REPORT")
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
    avg_improvement = improved_items['improvement'].mean() if len(improved_items) > 0 else 0
    
    print(f"\n💡 Confidence Enhancement Results:")
    print(f"   • Items with improved confidence: {len(improved_items)}/{total_items} ({len(improved_items)/total_items*100:.1f}%)")
    print(f"   • Average confidence improvement: {avg_improvement:.4f} ({avg_improvement*100:.2f}%)")
    print(f"   • Maximum improvement: {results_df['improvement'].max():.4f}")
    
    # Confidence distribution comparison
    l2_conf_avg = results_df['l2_confidence'].mean()
    final_conf_avg = results_df['final_confidence'].mean()
    overall_improvement = final_conf_avg - l2_conf_avg
    
    print(f"\n📊 Overall Confidence Comparison:")
    print(f"   • L2 Average Confidence: {l2_conf_avg:.4f} ({l2_conf_avg*100:.2f}%)")
    print(f"   • Hybrid Average Confidence: {final_conf_avg:.4f} ({final_conf_avg*100:.2f}%)")
    print(f"   • Overall Improvement: {overall_improvement:.4f} ({overall_improvement*100:.2f}%)")
    
    # Confidence tier analysis
    print(f"\n🎯 Confidence Tier Analysis:")
    
    # L2 tiers
    l2_high = (results_df['l2_confidence'] >= 0.8).sum()
    l2_medium = ((results_df['l2_confidence'] >= 0.6) & (results_df['l2_confidence'] < 0.8)).sum()
    l2_low = (results_df['l2_confidence'] < 0.6).sum()
    
    # Final tiers
    final_high = (results_df['final_confidence'] >= 0.8).sum()
    final_medium = ((results_df['final_confidence'] >= 0.6) & (results_df['final_confidence'] < 0.8)).sum()
    final_low = (results_df['final_confidence'] < 0.6).sum()
    
    print(f"   L2 Model Tiers:")
    print(f"     • High (≥80%): {l2_high} items ({l2_high/total_items*100:.1f}%)")
    print(f"     • Medium (60-80%): {l2_medium} items ({l2_medium/total_items*100:.1f}%)")
    print(f"     • Low (<60%): {l2_low} items ({l2_low/total_items*100:.1f}%)")
    
    print(f"   Hybrid Model Tiers:")
    print(f"     • High (≥80%): {final_high} items ({final_high/total_items*100:.1f}%) [+{final_high-l2_high}]")
    print(f"     • Medium (60-80%): {final_medium} items ({final_medium/total_items*100:.1f}%) [+{final_medium-l2_medium}]")
    print(f"     • Low (<60%): {final_low} items ({final_low/total_items*100:.1f}%) [+{final_low-l2_low}]")
    
    # Medium confidence enhancement analysis
    medium_conf_items = results_df[(results_df['l2_confidence'] >= 0.6) & (results_df['l2_confidence'] < 0.8)]
    enhanced_medium = medium_conf_items[medium_conf_items['improvement'] > 0]
    
    print(f"\n🎯 Medium Confidence Enhancement Analysis:")
    print(f"   • Medium confidence L2 items: {len(medium_conf_items)}")
    print(f"   • Enhanced by L4/L5: {len(enhanced_medium)} ({len(enhanced_medium)/len(medium_conf_items)*100:.1f}%)")
    
    if len(enhanced_medium) > 0:
        avg_medium_improvement = enhanced_medium['improvement'].mean()
        print(f"   • Average enhancement: {avg_medium_improvement:.4f} ({avg_medium_improvement*100:.2f}%)")
        
        # Show examples
        print(f"\n📋 Top Enhancement Examples:")
        top_improvements = enhanced_medium.nlargest(5, 'improvement')
        for idx, row in top_improvements.iterrows():
            print(f"   • {row['description']}")
            print(f"     L2: {row['l2_prediction']} ({row['l2_confidence']:.3f}) → "
                  f"Final: {row['final_prediction']} ({row['final_confidence']:.3f}) "
                  f"[+{row['improvement']:.3f}] via {row['strategy']}")
    
    return {
        'total_items': total_items,
        'improved_items': len(improved_items),
        'avg_improvement': avg_improvement,
        'overall_improvement': overall_improvement,
        'tier_improvements': {
            'high': final_high - l2_high,
            'medium': final_medium - l2_medium,
            'low': final_low - l2_low
        }
    }

def save_results(results_df):
    """Save results to Excel file"""
    output_path = '/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.5/L4L5/hybrid_classification_results.xlsx'
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Main results
        results_df.to_excel(writer, sheet_name='Hybrid_Results', index=False)
        
        # Summary statistics
        summary_stats = pd.DataFrame({
            'Metric': [
                'Total Items',
                'L2 Avg Confidence',
                'Hybrid Avg Confidence',
                'Overall Improvement',
                'Items Enhanced',
                'Enhancement Rate'
            ],
            'Value': [
                len(results_df),
                results_df['l2_confidence'].mean(),
                results_df['final_confidence'].mean(),
                results_df['final_confidence'].mean() - results_df['l2_confidence'].mean(),
                (results_df['improvement'] > 0).sum(),
                (results_df['improvement'] > 0).mean()
            ]
        })
        summary_stats.to_excel(writer, sheet_name='Summary', index=False)
        
        # Strategy breakdown
        strategy_breakdown = results_df['strategy'].value_counts().reset_index()
        strategy_breakdown.columns = ['Strategy', 'Count']
        strategy_breakdown['Percentage'] = strategy_breakdown['Count'] / len(results_df) * 100
        strategy_breakdown.to_excel(writer, sheet_name='Strategy_Breakdown', index=False)
    
    print(f"💾 Results saved to: {output_path}")

def main():
    """Main analysis function"""
    print("🔍 HYBRID L2 + L4/L5 CLASSIFICATION ANALYSIS")
    print("=" * 70)
    
    # Load models
    models = load_models()
    if not models or not models['L2']:
        print("❌ Cannot proceed without L2 model")
        return
    
    # Load test data
    test_data = load_test_data()
    if test_data is None:
        print("❌ Cannot proceed without test data")
        return
    
    # Analyze hybrid performance
    results_df = analyze_hybrid_performance(models, test_data)
    
    # Generate performance report
    performance_summary = generate_performance_report(results_df)
    
    # Save results
    save_results(results_df)
    
    print(f"\n" + "=" * 70)
    print(f"🎉 HYBRID CLASSIFICATION ANALYSIS COMPLETE!")
    print(f"📊 Key Results:")
    print(f"   • Overall confidence improvement: {performance_summary['overall_improvement']*100:.2f}%")
    print(f"   • Items enhanced: {performance_summary['improved_items']}/{performance_summary['total_items']}")
    print(f"   • High-confidence items gained: {performance_summary['tier_improvements']['high']}")
    print("=" * 70)

if __name__ == "__main__":
    main()
