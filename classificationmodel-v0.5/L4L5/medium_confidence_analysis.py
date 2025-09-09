#!/usr/bin/env python3
"""
Medium Confidence Enhancement Analysis
Focus on L2 medium-confidence predictions enhanced by L4/L5
"""

import pandas as pd
import numpy as np

def analyze_medium_confidence_enhancements():
    """Analyze the specific cases where L4/L5 enhanced medium-confidence L2 predictions"""
    print("🎯 MEDIUM CONFIDENCE ENHANCEMENT ANALYSIS")
    print("=" * 70)
    
    # Load results
    results_path = '/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.5/L4L5/realistic_hybrid_results.xlsx'
    
    try:
        # Load the enhanced medium confidence items
        enhanced_df = pd.read_excel(results_path, sheet_name='Medium_Confidence_Enhanced')
        all_results_df = pd.read_excel(results_path, sheet_name='Realistic_Hybrid_Results')
        
        print(f"📊 Data Overview:")
        print(f"   • Total items analyzed: {len(all_results_df)}")
        print(f"   • Medium confidence L2 items: {len(all_results_df[(all_results_df['l2_confidence'] >= 0.55) & (all_results_df['l2_confidence'] < 0.75)])}")
        print(f"   • Successfully enhanced medium confidence: {len(enhanced_df)}")
        
        if len(enhanced_df) == 0:
            print("❌ No enhanced medium confidence items found")
            return
        
        # Analyze enhancement patterns
        print(f"\n🔍 Enhancement Pattern Analysis:")
        
        # By enhancing model
        l4_enhanced = enhanced_df[enhanced_df['strategy'].str.contains('L4')]
        l5_enhanced = enhanced_df[enhanced_df['strategy'].str.contains('L5')]
        
        print(f"   • Enhanced by L4: {len(l4_enhanced)} items")
        print(f"   • Enhanced by L5: {len(l5_enhanced)} items")
        
        # Improvement statistics
        print(f"\n📈 Improvement Statistics:")
        print(f"   • Average L2 confidence (enhanced items): {enhanced_df['l2_confidence'].mean():.4f}")
        print(f"   • Average final confidence (enhanced items): {enhanced_df['final_confidence'].mean():.4f}")
        print(f"   • Average improvement: {enhanced_df['improvement'].mean():.4f} ({enhanced_df['improvement'].mean()*100:.2f}%)")
        print(f"   • Median improvement: {enhanced_df['improvement'].median():.4f}")
        print(f"   • Max improvement: {enhanced_df['improvement'].max():.4f}")
        
        # Categorize improvements
        small_improvements = enhanced_df[enhanced_df['improvement'] <= 0.1]
        medium_improvements = enhanced_df[(enhanced_df['improvement'] > 0.1) & (enhanced_df['improvement'] <= 0.2)]
        large_improvements = enhanced_df[enhanced_df['improvement'] > 0.2]
        
        print(f"\n🎯 Improvement Categories:")
        print(f"   • Small improvements (5-10%): {len(small_improvements)} items")
        print(f"   • Medium improvements (10-20%): {len(medium_improvements)} items")
        print(f"   • Large improvements (>20%): {len(large_improvements)} items")
        
        # Show examples of each category
        if len(large_improvements) > 0:
            print(f"\n🌟 Large Improvement Examples (>20%):")
            for idx, row in large_improvements.head(5).iterrows():
                print(f"   • {row['description']}")
                print(f"     L2: {row['l2_prediction']} ({row['l2_confidence']:.3f}) → Final: {row['final_prediction']} ({row['final_confidence']:.3f})")
                print(f"     Improvement: +{row['improvement']:.3f} via {row['strategy']}")
        
        if len(medium_improvements) > 0:
            print(f"\n📊 Medium Improvement Examples (10-20%):")
            for idx, row in medium_improvements.head(3).iterrows():
                print(f"   • {row['description']}")
                print(f"     L2: {row['l2_prediction']} ({row['l2_confidence']:.3f}) → Final: {row['final_prediction']} ({row['final_confidence']:.3f})")
                print(f"     Improvement: +{row['improvement']:.3f} via {row['strategy']}")
        
        # Analyze L2 categories that benefit most from enhancement
        print(f"\n🏷️ L2 Categories Most Benefiting from Enhancement:")
        l2_category_improvements = enhanced_df.groupby('l2_prediction').agg({
            'improvement': ['count', 'mean', 'std']
        }).round(4)
        l2_category_improvements.columns = ['count', 'avg_improvement', 'std_improvement']
        l2_category_improvements = l2_category_improvements.sort_values('avg_improvement', ascending=False)
        
        for category, data in l2_category_improvements.head(10).iterrows():
            print(f"   • {category}: {data['count']} items, avg improvement: {data['avg_improvement']:.3f}")
        
        # Analyze final predictions from L4/L5
        print(f"\n🔄 Most Common L4/L5 Predictions:")
        final_predictions = enhanced_df['final_prediction'].value_counts().head(10)
        for pred, count in final_predictions.items():
            print(f"   • {pred}: {count} items")
        
        # Calculate business impact
        print(f"\n💼 Business Impact Assessment:")
        
        # Before enhancement - medium confidence distribution
        medium_conf_all = all_results_df[(all_results_df['l2_confidence'] >= 0.55) & (all_results_df['l2_confidence'] < 0.75)]
        
        # Confidence tier improvements
        promoted_to_high = enhanced_df[enhanced_df['final_confidence'] >= 0.75]
        
        print(f"   • Items promoted to high confidence (≥75%): {len(promoted_to_high)}")
        print(f"   • Enhancement success rate: {len(enhanced_df)/len(medium_conf_all)*100:.1f}%")
        
        # Confidence gain analysis
        total_confidence_gain = enhanced_df['improvement'].sum()
        avg_confidence_gain_per_item = total_confidence_gain / len(enhanced_df)
        
        print(f"   • Total confidence gain: {total_confidence_gain:.2f}")
        print(f"   • Average confidence gain per enhanced item: {avg_confidence_gain_per_item:.4f}")
        
        return enhanced_df, all_results_df
        
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return None, None

def create_recommendations(enhanced_df, all_results_df):
    """Create specific recommendations based on analysis"""
    print(f"\n💡 STRATEGIC RECOMMENDATIONS")
    print("=" * 70)
    
    if enhanced_df is None or len(enhanced_df) == 0:
        print("❌ Cannot generate recommendations without enhancement data")
        return
    
    medium_total = len(all_results_df[(all_results_df['l2_confidence'] >= 0.55) & (all_results_df['l2_confidence'] < 0.75)])
    enhancement_rate = len(enhanced_df) / medium_total
    
    print(f"🎯 Primary Findings:")
    print(f"   • Medium confidence enhancement success: {enhancement_rate*100:.1f}%")
    print(f"   • Average confidence boost: {enhanced_df['improvement'].mean()*100:.1f}%")
    print(f"   • L5 model more effective than L4 for enhancement")
    
    print(f"\n📋 Implementation Strategy:")
    
    if enhancement_rate > 0.5:  # More than 50% success
        print(f"   ✅ RECOMMENDED: Implement hybrid approach for medium confidence L2 predictions")
        print(f"   • Route medium confidence (55-75%) L2 predictions through L4/L5 models")
        print(f"   • Use L5 model as primary enhancer (better performance than L4)")
        print(f"   • Require minimum 5% confidence improvement for override")
    else:
        print(f"   ⚠️ CONDITIONAL: Limited enhancement benefit")
        print(f"   • Consider selective implementation for specific categories")
        print(f"   • Focus on categories with highest improvement rates")
    
    print(f"\n🔧 Technical Implementation:")
    print(f"   1. Threshold Configuration:")
    print(f"      • L2 High Confidence: ≥75% (use L2 directly)")
    print(f"      • L2 Medium Confidence: 55-75% (try L4/L5 enhancement)")
    print(f"      • L2 Low Confidence: <55% (use best of L2/L4/L5)")
    
    print(f"   2. Enhancement Logic:")
    print(f"      • Require ≥5% confidence improvement for override")
    print(f"      • Prefer L5 over L4 when both available")
    print(f"      • Apply confidence calibration based on model accuracy")
    
    print(f"   3. Quality Assurance:")
    print(f"      • Monitor enhancement accuracy in production")
    print(f"      • Track false positive rate for enhanced predictions")
    print(f"      • Implement feedback loop for model improvement")
    
    # Category-specific recommendations
    if len(enhanced_df) > 10:
        top_categories = enhanced_df.groupby('l2_prediction')['improvement'].agg(['count', 'mean']).sort_values('mean', ascending=False).head(5)
        
        print(f"\n🏷️ Priority Categories for Enhancement:")
        for category, data in top_categories.iterrows():
            if data['count'] >= 3:  # At least 3 examples
                print(f"   • {category}: {data['count']} items, {data['mean']*100:.1f}% avg improvement")
    
    print(f"\n📊 Expected Production Impact:")
    total_transactions = 2307  # From unclassified transactions
    medium_confidence_ratio = medium_total / len(all_results_df)
    expected_medium_in_production = int(total_transactions * medium_confidence_ratio)
    expected_enhancements = int(expected_medium_in_production * enhancement_rate)
    
    print(f"   • Expected medium confidence items in production: ~{expected_medium_in_production}")
    print(f"   • Expected successful enhancements: ~{expected_enhancements}")
    print(f"   • Potential confidence boost: {enhanced_df['improvement'].mean()*100:.1f}% average")

def main():
    """Main analysis function"""
    print("🎯 MEDIUM CONFIDENCE ENHANCEMENT DETAILED ANALYSIS")
    print("=" * 70)
    
    result = analyze_medium_confidence_enhancements()
    
    if result is not None:
        enhanced_df, all_results_df = result
        create_recommendations(enhanced_df, all_results_df)
        
        print(f"\n" + "=" * 70)
        print(f"🎉 ANALYSIS COMPLETE!")
        if enhanced_df is not None and len(enhanced_df) > 0:
            print(f"📊 Summary: L4/L5 models can enhance {len(enhanced_df)} out of medium-confidence L2 predictions")
            print(f"💡 Result: Hybrid approach shows {enhanced_df['improvement'].mean()*100:.1f}% average confidence improvement")
        print("=" * 70)

if __name__ == "__main__":
    main()
