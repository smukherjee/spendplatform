#!/usr/bin/env python3
"""
Enhanced Hybrid Experiment Results Analysis
Detailed analysis of the medium/low confidence enhancement experiment
"""

import pandas as pd
import numpy as np

def analyze_experiment_results():
    """Analyze the enhanced hybrid experiment results"""
    print("📊 ENHANCED HYBRID EXPERIMENT ANALYSIS")
    print("=" * 70)
    
    try:
        # Load results
        results_path = '/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.5/L4L5/enhanced_hybrid_experiment_results.xlsx'
        
        # Load all sheets
        all_results = pd.read_excel(results_path, sheet_name='Enhanced_Hybrid_Results')
        medium_enhanced = pd.read_excel(results_path, sheet_name='Medium_Confidence_Enhanced')
        low_enhanced = pd.read_excel(results_path, sheet_name='Low_Confidence_Enhanced')
        category_adjusted = pd.read_excel(results_path, sheet_name='Category_Adjustments')
        
        print(f"📋 Data Overview:")
        print(f"   • Total analyzed: {len(all_results)} items")
        print(f"   • Medium confidence enhanced: {len(medium_enhanced)} items")
        print(f"   • Low confidence enhanced: {len(low_enhanced)} items")
        print(f"   • Category adjustments: {len(category_adjusted)} items")
        
        # Detailed medium confidence analysis
        print(f"\n🎯 MEDIUM CONFIDENCE ANALYSIS (50-75%)")
        print("=" * 50)
        
        medium_conf_items = all_results[
            (all_results['l2_confidence'] >= 0.50) & 
            (all_results['l2_confidence'] < 0.75)
        ]
        
        print(f"📊 Medium Confidence Statistics:")
        print(f"   • Total medium confidence items: {len(medium_conf_items)}")
        print(f"   • Successfully enhanced: {len(medium_enhanced)}")
        print(f"   • Enhancement rate: {len(medium_enhanced)/len(medium_conf_items)*100:.1f}%")
        print(f"   • Average L2 confidence: {medium_conf_items['l2_confidence'].mean():.4f}")
        print(f"   • Average final confidence: {medium_enhanced['final_confidence'].mean():.4f}")
        print(f"   • Average improvement: {medium_enhanced['improvement'].mean():.4f} ({medium_enhanced['improvement'].mean()*100:.2f}%)")
        
        # Medium confidence enhancement by model
        medium_l4_enhanced = medium_enhanced[medium_enhanced['strategy'].str.contains('L4')]
        medium_l5_enhanced = medium_enhanced[medium_enhanced['strategy'].str.contains('L5')]
        
        print(f"\n📈 Medium Confidence Enhancement by Model:")
        print(f"   • Enhanced by L4: {len(medium_l4_enhanced)} items")
        print(f"   • Enhanced by L5: {len(medium_l5_enhanced)} items")
        print(f"   • L5 effectiveness: {len(medium_l5_enhanced)/len(medium_enhanced)*100:.1f}%")
        
        # Detailed low confidence analysis
        print(f"\n🎯 LOW CONFIDENCE ANALYSIS (<50%)")
        print("=" * 50)
        
        low_conf_items = all_results[all_results['l2_confidence'] < 0.50]
        
        print(f"📊 Low Confidence Statistics:")
        print(f"   • Total low confidence items: {len(low_conf_items)}")
        print(f"   • Successfully enhanced: {len(low_enhanced)}")
        print(f"   • Enhancement rate: {len(low_enhanced)/len(low_conf_items)*100:.1f}%")
        print(f"   • Average L2 confidence: {low_conf_items['l2_confidence'].mean():.4f}")
        print(f"   • Average final confidence: {low_enhanced['final_confidence'].mean():.4f}")
        print(f"   • Average improvement: {low_enhanced['improvement'].mean():.4f} ({low_enhanced['improvement'].mean()*100:.2f}%)")
        
        # Low confidence enhancement by model
        low_l4_enhanced = low_enhanced[low_enhanced['strategy'].str.contains('L4')]
        low_l5_enhanced = low_enhanced[low_enhanced['strategy'].str.contains('L5')]
        
        print(f"\n📈 Low Confidence Enhancement by Model:")
        print(f"   • Enhanced by L4: {len(low_l4_enhanced)} items")
        print(f"   • Enhanced by L5: {len(low_l5_enhanced)} items")
        print(f"   • L5 effectiveness: {len(low_l5_enhanced)/len(low_enhanced)*100:.1f}%")
        
        # Category adjustment analysis
        print(f"\n🔄 CATEGORY ADJUSTMENT ANALYSIS")
        print("=" * 50)
        
        print(f"📊 L5 High Confidence Category Adjustments:")
        print(f"   • Items with L5 confidence ≥ 65%: {(all_results['l5_confidence'] >= 0.65).sum()}")
        print(f"   • L2 category adjustments: {category_adjusted['adjusted_l2_category'].notna().sum()}")
        print(f"   • L4 category adjustments: {category_adjusted['adjusted_l4_category'].notna().sum()}")
        
        # Analyze most common adjustments
        l2_adjustment_changes = []
        l4_adjustment_changes = []
        
        for idx, row in category_adjusted.iterrows():
            if pd.notna(row['adjusted_l2_category']) and row['l2_prediction'] != row['adjusted_l2_category']:
                l2_adjustment_changes.append(f"{row['l2_prediction']} → {row['adjusted_l2_category']}")
            
            if pd.notna(row['adjusted_l4_category']) and row['l4_prediction'] != row['adjusted_l4_category']:
                l4_adjustment_changes.append(f"{row['l4_prediction']} → {row['adjusted_l4_category']}")
        
        print(f"\n🔄 Most Common L2 Category Adjustments:")
        l2_changes_series = pd.Series(l2_adjustment_changes)
        if len(l2_changes_series) > 0:
            for change, count in l2_changes_series.value_counts().head(10).items():
                print(f"   • {change}: {count} times")
        else:
            print("   • No L2 category changes detected")
        
        print(f"\n🔄 Most Common L4 Category Adjustments:")
        l4_changes_series = pd.Series(l4_adjustment_changes)
        if len(l4_changes_series) > 0:
            for change, count in l4_changes_series.value_counts().head(10).items():
                print(f"   • {change}: {count} times")
        else:
            print("   • No L4 category changes detected")
        
        # Confidence tier promotion analysis
        print(f"\n📈 CONFIDENCE TIER PROMOTION ANALYSIS")
        print("=" * 50)
        
        # Before enhancement tiers
        l2_high = (all_results['l2_confidence'] >= 0.75).sum()
        l2_medium = ((all_results['l2_confidence'] >= 0.50) & (all_results['l2_confidence'] < 0.75)).sum()
        l2_low = (all_results['l2_confidence'] < 0.50).sum()
        
        # After enhancement tiers
        final_high = (all_results['final_confidence'] >= 0.75).sum()
        final_medium = ((all_results['final_confidence'] >= 0.50) & (all_results['final_confidence'] < 0.75)).sum()
        final_low = (all_results['final_confidence'] < 0.50).sum()
        
        print(f"📊 Confidence Tier Changes:")
        print(f"   Before Enhancement:")
        print(f"     • High (≥75%): {l2_high} items ({l2_high/len(all_results)*100:.1f}%)")
        print(f"     • Medium (50-75%): {l2_medium} items ({l2_medium/len(all_results)*100:.1f}%)")
        print(f"     • Low (<50%): {l2_low} items ({l2_low/len(all_results)*100:.1f}%)")
        
        print(f"   After Enhancement:")
        print(f"     • High (≥75%): {final_high} items ({final_high/len(all_results)*100:.1f}%) [+{final_high-l2_high}]")
        print(f"     • Medium (50-75%): {final_medium} items ({final_medium/len(all_results)*100:.1f}%) [+{final_medium-l2_medium}]")
        print(f"     • Low (<50%): {final_low} items ({final_low/len(all_results)*100:.1f}%) [+{final_low-l2_low}]")
        
        # Success examples
        print(f"\n🌟 SUCCESS EXAMPLES")
        print("=" * 50)
        
        # High improvement examples
        high_improvements = all_results[all_results['improvement'] > 0.20].sort_values('improvement', ascending=False)
        
        print(f"📈 Top 5 Highest Improvements:")
        for idx, row in high_improvements.head(5).iterrows():
            print(f"   • {row['description']}")
            print(f"     L2: {row['l2_prediction']} ({row['l2_confidence']:.3f}) → Final: {row['final_prediction']} ({row['final_confidence']:.3f})")
            print(f"     Improvement: +{row['improvement']:.3f} ({row['improvement']*100:.1f}%) via {row['strategy']}")
            if row['category_adjusted']:
                print(f"     L2 adjusted: {row['adjusted_l2_category']}, L4 adjusted: {row['adjusted_l4_category']}")
        
        return {
            'total_items': len(all_results),
            'medium_enhanced': len(medium_enhanced),
            'medium_total': len(medium_conf_items),
            'low_enhanced': len(low_enhanced),
            'low_total': len(low_conf_items),
            'category_adjustments': len(category_adjusted),
            'overall_improvement': all_results['final_confidence'].mean() - all_results['l2_confidence'].mean(),
            'tier_promotions': final_high - l2_high
        }
        
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        return None

def generate_business_recommendations(analysis_results):
    """Generate business recommendations based on analysis"""
    print(f"\n💡 BUSINESS RECOMMENDATIONS")
    print("=" * 70)
    
    if analysis_results is None:
        print("❌ Cannot generate recommendations without analysis results")
        return
    
    medium_enhancement_rate = analysis_results['medium_enhanced'] / analysis_results['medium_total']
    low_enhancement_rate = analysis_results['low_enhanced'] / analysis_results['low_total']
    
    print(f"🎯 Key Findings:")
    print(f"   • Medium confidence enhancement: {medium_enhancement_rate*100:.1f}% success rate")
    print(f"   • Low confidence enhancement: {low_enhancement_rate*100:.1f}% success rate")
    print(f"   • Overall confidence improvement: {analysis_results['overall_improvement']*100:.2f}%")
    print(f"   • High-confidence tier promotions: +{analysis_results['tier_promotions']} items")
    
    print(f"\n📋 Strategic Recommendations:")
    
    if medium_enhancement_rate > 0.6 and low_enhancement_rate > 0.6:
        print(f"   ✅ HIGHLY RECOMMENDED: Deploy hybrid enhancement system")
        print(f"   • Implement L4/L5 enhancement for both medium and low confidence L2 predictions")
        print(f"   • Use L5 model as primary enhancer (superior performance)")
        print(f"   • Enable L5 category adjustment feature for confidence > 65%")
        
        print(f"\n🔧 Implementation Framework:")
        print(f"   1. Confidence Routing Logic:")
        print(f"      • High L2 confidence (≥75%): Use L2 directly")
        print(f"      • Medium L2 confidence (50-75%): Try L4/L5 enhancement")
        print(f"      • Low L2 confidence (<50%): Use best of L2/L4/L5")
        
        print(f"   2. Enhancement Criteria:")
        print(f"      • Require minimum 5% confidence improvement")
        print(f"      • Prefer L5 over L4 when both available")
        print(f"      • Apply category adjustment when L5 confidence ≥ 65%")
        
        print(f"   3. Category Adjustment Rules:")
        print(f"      • Use L5→L2 mapping for L2 category adjustment")
        print(f"      • Use L5→L4 mapping for L4 category refinement")
        print(f"      • Maintain audit trail of all adjustments")
        
    else:
        print(f"   ⚠️ CONDITIONAL RECOMMENDATION: Selective implementation")
        print(f"   • Focus on specific categories with high enhancement rates")
        print(f"   • Implement pilot program for validation")
    
    print(f"\n📊 Expected Production Impact:")
    
    # Estimate production impact based on 2,307 unclassified transactions
    total_production = 2307
    expected_medium = int(total_production * (analysis_results['medium_total'] / analysis_results['total_items']))
    expected_low = int(total_production * (analysis_results['low_total'] / analysis_results['total_items']))
    
    expected_medium_enhanced = int(expected_medium * medium_enhancement_rate)
    expected_low_enhanced = int(expected_low * low_enhancement_rate)
    
    print(f"   • Expected medium confidence items: ~{expected_medium}")
    print(f"   • Expected medium enhancements: ~{expected_medium_enhanced}")
    print(f"   • Expected low confidence items: ~{expected_low}")
    print(f"   • Expected low enhancements: ~{expected_low_enhanced}")
    print(f"   • Total expected enhancements: ~{expected_medium_enhanced + expected_low_enhanced}")
    
    print(f"\n🎯 Quality Assurance Measures:")
    print(f"   • Monitor enhancement accuracy vs ground truth")
    print(f"   • Track category adjustment effectiveness")
    print(f"   • Implement confidence calibration validation")
    print(f"   • Establish feedback loop for continuous improvement")

def main():
    """Main analysis function"""
    print("📊 ENHANCED HYBRID EXPERIMENT DETAILED ANALYSIS")
    print("=" * 70)
    
    analysis_results = analyze_experiment_results()
    
    if analysis_results:
        generate_business_recommendations(analysis_results)
        
        print(f"\n" + "=" * 70)
        print(f"🎉 EXPERIMENT ANALYSIS COMPLETE!")
        print(f"📊 Summary: Hybrid L4/L5 enhancement shows significant promise")
        print(f"💡 Result: {analysis_results['overall_improvement']*100:.2f}% overall confidence improvement")
        print("=" * 70)

if __name__ == "__main__":
    main()
