#!/usr/bin/env python3
"""
Analyze and Validate Classified Transactions Results
Provide insights and recommendations for the classified data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def load_classified_results():
    """Load the classified results and analyze them"""
    print("📊 CLASSIFIED TRANSACTIONS ANALYSIS")
    print("=" * 70)
    
    file_path = '/Users/sujoymukherjee/code/spendplatform/context/classified_transactions_output.xlsx'
    
    try:
        # Load all sheets
        classified_df = pd.read_excel(file_path, sheet_name='Classified_Transactions')
        summary_df = pd.read_excel(file_path, sheet_name='Summary')
        category_breakdown_df = pd.read_excel(file_path, sheet_name='Category_Breakdown')
        
        print(f"✅ Results loaded successfully!")
        print(f"📊 Dataset info:")
        print(f"   • Total transactions: {len(classified_df):,}")
        print(f"   • Columns: {len(classified_df.columns)}")
        print(f"   • Predicted categories: {classified_df['Predicted_L2_Category'].nunique()}")
        
        return classified_df, summary_df, category_breakdown_df
        
    except Exception as e:
        print(f"❌ Error loading results: {e}")
        return None, None, None

def analyze_confidence_distribution(df):
    """Analyze confidence score distribution"""
    print(f"\n🎯 CONFIDENCE ANALYSIS")
    print("=" * 50)
    
    confidence_scores = df['Prediction_Confidence']
    
    # Statistical summary
    print(f"📊 Confidence Statistics:")
    print(f"   • Mean: {confidence_scores.mean():.3f}")
    print(f"   • Median: {confidence_scores.median():.3f}")
    print(f"   • Standard Deviation: {confidence_scores.std():.3f}")
    print(f"   • Min: {confidence_scores.min():.3f}")
    print(f"   • Max: {confidence_scores.max():.3f}")
    
    # Confidence bins
    high_conf = (confidence_scores > 0.8).sum()
    medium_conf = ((confidence_scores > 0.6) & (confidence_scores <= 0.8)).sum()
    low_conf = (confidence_scores <= 0.6).sum()
    
    print(f"\n📈 Confidence Distribution:")
    print(f"   🟢 High (>0.8): {high_conf:,} ({high_conf/len(df)*100:.1f}%)")
    print(f"   🟡 Medium (0.6-0.8): {medium_conf:,} ({medium_conf/len(df)*100:.1f}%)")
    print(f"   🔴 Low (≤0.6): {low_conf:,} ({low_conf/len(df)*100:.1f}%)")
    
    # Category-wise confidence
    print(f"\n🏷️ Average Confidence by Category (Top 10):")
    category_confidence = df.groupby('Predicted_L2_Category')['Prediction_Confidence'].agg(['mean', 'count']).sort_values('mean', ascending=False)
    
    for category, stats in category_confidence.head(10).iterrows():
        print(f"   {category}: {stats['mean']:.3f} ({stats['count']} items)")

def analyze_category_predictions(df):
    """Analyze category prediction patterns"""
    print(f"\n🏷️ CATEGORY PREDICTION ANALYSIS")
    print("=" * 50)
    
    # Category distribution
    category_counts = df['Predicted_L2_Category'].value_counts()
    
    print(f"📊 Category Distribution (All {len(category_counts)} categories):")
    for i, (category, count) in enumerate(category_counts.items(), 1):
        percentage = count / len(df) * 100
        print(f"   {i:2d}. {category}: {count:,} ({percentage:.1f}%)")
    
    # Most confident predictions per category
    print(f"\n🎯 Most Confident Predictions by Category:")
    for category in category_counts.head(5).index:
        cat_data = df[df['Predicted_L2_Category'] == category]
        best_prediction = cat_data.loc[cat_data['Prediction_Confidence'].idxmax()]
        print(f"\n   🏆 {category}:")
        print(f"      Description: '{best_prediction['Description'][:60]}...'")
        print(f"      Confidence: {best_prediction['Prediction_Confidence']:.3f}")
        print(f"      Status: {best_prediction['Prediction_Status']}")

def identify_review_candidates(df):
    """Identify transactions that need manual review"""
    print(f"\n⚠️ REVIEW CANDIDATES ANALYSIS")
    print("=" * 50)
    
    # Low confidence predictions
    low_confidence = df[df['Prediction_Confidence'] < 0.4]
    print(f"🔴 Very Low Confidence (<0.4): {len(low_confidence):,} transactions")
    
    if len(low_confidence) > 0:
        print(f"   Top 5 candidates for review:")
        for i, (_, row) in enumerate(low_confidence.head().iterrows(), 1):
            print(f"   {i}. '{row['Description'][:50]}...' → {row['Predicted_L2_Category']} (conf: {row['Prediction_Confidence']:.3f})")
    
    # Generic descriptions
    generic_keywords = ['generic', 'replacement', 'part', 'component', 'item', 'product', 'material']
    generic_mask = df['Description'].str.lower().str.contains('|'.join(generic_keywords), na=False)
    generic_items = df[generic_mask]
    
    print(f"\n🔍 Generic Descriptions: {len(generic_items):,} transactions")
    if len(generic_items) > 0:
        print(f"   Sample generic descriptions:")
        for i, (_, row) in enumerate(generic_items.head(3).iterrows(), 1):
            print(f"   {i}. '{row['Description'][:50]}...' → {row['Predicted_L2_Category']} (conf: {row['Prediction_Confidence']:.3f})")
    
    # Short descriptions
    short_descriptions = df[df['Description'].str.len() < 20]
    print(f"\n📝 Short Descriptions (<20 chars): {len(short_descriptions):,} transactions")
    if len(short_descriptions) > 0:
        print(f"   Sample short descriptions:")
        for i, (_, row) in enumerate(short_descriptions.head(3).iterrows(), 1):
            print(f"   {i}. '{row['Description']}' → {row['Predicted_L2_Category']} (conf: {row['Prediction_Confidence']:.3f})")

def compare_existing_categories(df):
    """Compare predictions with existing categories if available"""
    print(f"\n🔄 EXISTING VS PREDICTED CATEGORIES")
    print("=" * 50)
    
    if 'Category L2' in df.columns:
        # Check how many already have L2 categories
        existing_l2 = df['Category L2'].notna().sum()
        print(f"📊 Existing L2 Categories: {existing_l2:,} ({existing_l2/len(df)*100:.1f}%)")
        
        if existing_l2 > 0:
            # Compare predictions with existing categories
            comparison_df = df[df['Category L2'].notna()].copy()
            matches = comparison_df['Category L2'] == comparison_df['Predicted_L2_Category']
            match_count = matches.sum()
            
            print(f"✅ Exact Matches: {match_count:,} ({match_count/len(comparison_df)*100:.1f}%)")
            
            # Show some examples of matches and mismatches
            if match_count > 0:
                print(f"\n🎯 Sample Exact Matches:")
                matches_df = comparison_df[matches].head(3)
                for i, (_, row) in enumerate(matches_df.iterrows(), 1):
                    print(f"   {i}. '{row['Description'][:40]}...' → {row['Category L2']} (conf: {row['Prediction_Confidence']:.3f})")
            
            mismatches = comparison_df[~matches]
            if len(mismatches) > 0:
                print(f"\n❌ Sample Mismatches:")
                for i, (_, row) in enumerate(mismatches.head(3).iterrows(), 1):
                    print(f"   {i}. '{row['Description'][:40]}...'")
                    print(f"      Existing: {row['Category L2']}")
                    print(f"      Predicted: {row['Predicted_L2_Category']} (conf: {row['Prediction_Confidence']:.3f})")
    else:
        print("ℹ️ No existing L2 categories found for comparison")

def generate_quality_report(df):
    """Generate a quality assessment report"""
    print(f"\n📋 QUALITY ASSESSMENT REPORT")
    print("=" * 50)
    
    total_transactions = len(df)
    
    # Quality metrics
    high_quality = (df['Prediction_Confidence'] > 0.8).sum()
    medium_quality = ((df['Prediction_Confidence'] > 0.6) & (df['Prediction_Confidence'] <= 0.8)).sum()
    needs_review = (df['Prediction_Confidence'] <= 0.6).sum()
    
    print(f"🎯 Quality Metrics:")
    print(f"   ✅ High Quality (>0.8): {high_quality:,} ({high_quality/total_transactions*100:.1f}%)")
    print(f"   🟡 Medium Quality (0.6-0.8): {medium_quality:,} ({medium_quality/total_transactions*100:.1f}%)")
    print(f"   ⚠️ Needs Review (≤0.6): {needs_review:,} ({needs_review/total_transactions*100:.1f}%)")
    
    # Deployment readiness
    ready_for_production = high_quality + medium_quality
    print(f"\n🚀 Deployment Readiness:")
    print(f"   • Ready for production: {ready_for_production:,} ({ready_for_production/total_transactions*100:.1f}%)")
    print(f"   • Requires manual review: {needs_review:,} ({needs_review/total_transactions*100:.1f}%)")
    
    # Category coverage
    categories_used = df['Predicted_L2_Category'].nunique()
    print(f"\n📊 Category Coverage:")
    print(f"   • Categories utilized: {categories_used}/17 model categories")
    print(f"   • Average items per category: {total_transactions/categories_used:.1f}")
    
    # Model performance assessment
    avg_confidence = df['Prediction_Confidence'].mean()
    if avg_confidence > 0.7:
        performance_rating = "Excellent"
        performance_emoji = "🏆"
    elif avg_confidence > 0.6:
        performance_rating = "Good"
        performance_emoji = "✅"
    elif avg_confidence > 0.5:
        performance_rating = "Fair"
        performance_emoji = "🟡"
    else:
        performance_rating = "Needs Improvement"
        performance_emoji = "⚠️"
    
    print(f"\n{performance_emoji} Overall Performance Rating: {performance_rating}")
    print(f"   • Average confidence: {avg_confidence:.3f}")
    print(f"   • Model utilization: {categories_used}/17 categories")

def create_recommendations(df):
    """Create actionable recommendations"""
    print(f"\n💡 RECOMMENDATIONS")
    print("=" * 50)
    
    low_confidence_count = (df['Prediction_Confidence'] <= 0.6).sum()
    high_confidence_count = (df['Prediction_Confidence'] > 0.8).sum()
    
    print(f"🎯 Immediate Actions:")
    
    if low_confidence_count > len(df) * 0.3:  # More than 30% low confidence
        print(f"   1. 🔍 PRIORITY: Review {low_confidence_count:,} low-confidence predictions")
        print(f"      • Focus on descriptions with confidence < 0.4")
        print(f"      • Verify generic or unclear item descriptions")
        print(f"      • Consider creating standard naming conventions")
    
    if high_confidence_count > len(df) * 0.2:  # Good number of high confidence
        print(f"   2. ✅ DEPLOY: {high_confidence_count:,} high-confidence predictions ready for production")
        print(f"      • These can be automatically categorized")
        print(f"      • Monitor for any edge cases")
    
    print(f"\n🚀 Process Improvements:")
    print(f"   1. Data Quality:")
    print(f"      • Standardize item descriptions")
    print(f"      • Include technical specifications")
    print(f"      • Add manufacturer information where possible")
    
    print(f"   2. Model Enhancement:")
    print(f"      • Retrain with manual corrections")
    print(f"      • Add domain-specific features")
    print(f"      • Consider hierarchical classification")
    
    print(f"   3. Workflow Integration:")
    print(f"      • Auto-approve high confidence (>0.8)")
    print(f"      • Queue medium confidence for review")
    print(f"      • Flag low confidence for manual processing")

def main():
    """Main analysis function"""
    print("🔍 CLASSIFIED TRANSACTIONS VALIDATION")
    print("=" * 70)
    
    # Load results
    classified_df, summary_df, category_breakdown_df = load_classified_results()
    if classified_df is None:
        return
    
    # Run all analyses
    analyze_confidence_distribution(classified_df)
    analyze_category_predictions(classified_df)
    identify_review_candidates(classified_df)
    compare_existing_categories(classified_df)
    generate_quality_report(classified_df)
    create_recommendations(classified_df)
    
    print(f"\n" + "=" * 70)
    print(f"🎉 ANALYSIS COMPLETE!")
    print(f"✅ {len(classified_df):,} transactions analyzed")
    print(f"📊 Results saved in: classified_transactions_output.xlsx")
    print(f"🚀 Ready for production integration with confidence thresholds")
    print("=" * 70)

if __name__ == "__main__":
    main()
