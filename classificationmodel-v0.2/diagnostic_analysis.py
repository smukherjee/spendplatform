#!/usr/bin/env python3
"""
Enhanced Features Diagnostic Analysis
Diagnose why enhanced features are performing worse than existing approach
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

warnings.filterwarnings('ignore')

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config import ModelConfig
from text_parsers import NLTKTextParser

def analyze_text_processing_differences():
    """Analyze differences in text processing between approaches"""
    print("🔍 ANALYZING TEXT PROCESSING DIFFERENCES")
    print("=" * 60)
    
    # Load sample data
    data_path = project_root / "preprocessed_data" / "train_data.xlsx"
    df = pd.read_excel(data_path).head(100)  # Small sample for analysis
    
    print(f"Analyzing {len(df)} sample descriptions...")
    
    # Original descriptions
    original_descriptions = df['Item_Descripton'].fillna('').astype(str)
    
    # Process with NLTK
    model_config = ModelConfig()
    text_processor = model_config.get_text_parser()
    
    processed_descriptions = []
    for desc in original_descriptions:
        processed = text_processor.preprocess_text(str(desc))
        processed_descriptions.append(processed)
    
    # Analyze processing results
    print("\n📊 TEXT PROCESSING ANALYSIS:")
    
    original_lengths = [len(desc) for desc in original_descriptions]
    processed_lengths = [len(desc) for desc in processed_descriptions]
    
    print(f"Average original length: {np.mean(original_lengths):.1f} chars")
    print(f"Average processed length: {np.mean(processed_lengths):.1f} chars")
    print(f"Length reduction: {((np.mean(original_lengths) - np.mean(processed_lengths)) / np.mean(original_lengths)) * 100:.1f}%")
    
    # Show examples
    print("\n📋 PROCESSING EXAMPLES:")
    for i in range(5):
        print(f"\nExample {i+1}:")
        print(f"Original:  {original_descriptions.iloc[i][:100]}...")
        print(f"Processed: {processed_descriptions[i][:100]}...")
    
    return original_descriptions, processed_descriptions

def analyze_tfidf_differences():
    """Analyze TF-IDF parameter differences"""
    print("\n🔍 ANALYZING TF-IDF PARAMETER DIFFERENCES")
    print("=" * 60)
    
    # Load sample data
    data_path = project_root / "preprocessed_data" / "train_data.xlsx"
    df = pd.read_excel(data_path).head(500)  # Larger sample for TF-IDF
    
    # Get processed descriptions
    model_config = ModelConfig()
    text_processor = model_config.get_text_parser()
    
    processed_descriptions = []
    for desc in df['Item_Descripton'].fillna(''):
        processed = text_processor.preprocess_text(str(desc))
        processed_descriptions.append(processed)
    
    # Existing approach parameters
    existing_vectorizer = TfidfVectorizer(
        max_features=model_config.tfidf_max_features,  # 1500
        ngram_range=model_config.tfidf_ngram_range,    # (1, 3)
        min_df=model_config.tfidf_min_df,              # 2
        max_df=model_config.tfidf_max_df               # 0.95
    )
    
    # Enhanced v2 parameters
    enhanced_vectorizer = TfidfVectorizer(
        max_features=1300,  # Reduced from 2000
        ngram_range=(1, 3),  # Reduced from (1, 4)
        min_df=4,  # Increased from 2
        max_df=0.90,  # Reduced from 0.92
        sublinear_tf=True,
        use_idf=True,
        smooth_idf=True
    )
    
    # Fit both vectorizers
    existing_matrix = existing_vectorizer.fit_transform(processed_descriptions)
    enhanced_matrix = enhanced_vectorizer.fit_transform(processed_descriptions)
    
    print("\n📊 TF-IDF COMPARISON:")
    print(f"Existing approach:")
    print(f"  Features: {existing_matrix.shape[1]}")
    print(f"  Parameters: max_features={model_config.tfidf_max_features}, ngram_range={model_config.tfidf_ngram_range}, min_df={model_config.tfidf_min_df}, max_df={model_config.tfidf_max_df}")
    
    print(f"\nEnhanced approach:")
    print(f"  Features: {enhanced_matrix.shape[1]}")
    print(f"  Parameters: max_features=1300, ngram_range=(1,3), min_df=4, max_df=0.90")
    
    # Analyze vocabulary overlap
    existing_vocab = set(existing_vectorizer.get_feature_names_out())
    enhanced_vocab = set(enhanced_vectorizer.get_feature_names_out())
    
    overlap = existing_vocab.intersection(enhanced_vocab)
    existing_only = existing_vocab - enhanced_vocab
    enhanced_only = enhanced_vocab - existing_vocab
    
    print(f"\n📋 VOCABULARY ANALYSIS:")
    print(f"Existing vocabulary size: {len(existing_vocab)}")
    print(f"Enhanced vocabulary size: {len(enhanced_vocab)}")
    print(f"Overlapping terms: {len(overlap)} ({len(overlap)/len(existing_vocab)*100:.1f}%)")
    print(f"Existing only: {len(existing_only)}")
    print(f"Enhanced only: {len(enhanced_only)}")
    
    # Show different terms
    print(f"\n📝 TERMS ONLY IN EXISTING (first 20):")
    print(list(existing_only)[:20])
    
    print(f"\n📝 TERMS ONLY IN ENHANCED (first 20):")
    print(list(enhanced_only)[:20])
    
    return existing_vectorizer, enhanced_vectorizer

def analyze_feature_importance_patterns():
    """Analyze feature importance patterns"""
    print("\n🔍 ANALYZING FEATURE IMPORTANCE PATTERNS")
    print("=" * 60)
    
    # Key findings from test results
    print("\n📊 KEY OBSERVATIONS FROM TEST RESULTS:")
    print("1. Top features in optimized v2 are mostly structural (alpha_ratio, upper_ratio, etc.)")
    print("2. Domain-specific features appear but with lower importance")
    print("3. Character n-grams features appear in top 20")
    print("4. TF-IDF features have lower individual importance")
    
    print("\n💡 POTENTIAL ISSUES:")
    print("1. Feature scaling: Different feature types might need scaling")
    print("2. Feature correlation: Many features might be correlated")
    print("3. Overfitting: Too many features for the sample size")
    print("4. Feature quality: New features might not be as discriminative")
    
    print("\n🎯 RECOMMENDED FIXES:")
    print("1. Try feature scaling/normalization")
    print("2. Use correlation analysis to remove redundant features")
    print("3. Increase sample size or reduce features further")
    print("4. Analyze per-category feature effectiveness")

def create_improved_config():
    """Create an improved configuration based on analysis"""
    print("\n🔧 CREATING IMPROVED CONFIGURATION")
    print("=" * 60)
    
    improved_config = {
        "approach": "conservative_enhancement",
        "tfidf_config": {
            "max_features": 1500,  # Same as existing
            "ngram_range": (1, 3),  # Same as existing
            "min_df": 2,  # Same as existing
            "max_df": 0.95,  # Same as existing
            "sublinear_tf": True,
            "use_idf": True,
            "smooth_idf": True
        },
        "char_ngram_config": {
            "max_features": 100,  # Much smaller
            "ngram_range": (3, 4),
            "min_df": 10,  # Much higher threshold
            "max_df": 0.80
        },
        "feature_selection": {
            "method": "SelectKBest",
            "k": 800,  # Conservative selection
            "score_func": "mutual_info_classif"  # Try different scoring
        },
        "feature_scaling": True,  # Add feature scaling
        "remove_correlated": True,  # Remove highly correlated features
        "correlation_threshold": 0.95
    }
    
    print("📋 IMPROVED CONFIGURATION:")
    for section, params in improved_config.items():
        print(f"\n{section}:")
        if isinstance(params, dict):
            for param, value in params.items():
                print(f"  {param}: {value}")
        else:
            print(f"  {params}")
    
    return improved_config

def performance_analysis_summary():
    """Provide comprehensive performance analysis summary"""
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE PERFORMANCE ANALYSIS SUMMARY")
    print("=" * 60)
    
    print("\n🎯 CURRENT RESULTS:")
    print("• Existing (1073 features): 62.67% accuracy")
    print("• Enhanced v1 (1698 features): 57.33% accuracy (-8.51%)")
    print("• Optimized v2 (607 features): 55.33% accuracy (-11.70%)")
    
    print("\n🔍 ROOT CAUSE ANALYSIS:")
    print("1. FEATURE DILUTION:")
    print("   • Adding more features without proper validation")
    print("   • Many new features may not be discriminative")
    print("   • Character n-grams might be capturing noise")
    
    print("\n2. PARAMETER MISMATCH:")
    print("   • Enhanced parameters differ from well-tuned existing ones")
    print("   • min_df=4 (vs 2) might be filtering useful rare terms")
    print("   • max_df=0.90 (vs 0.95) might be removing common but useful terms")
    
    print("\n3. FEATURE SELECTION ISSUES:")
    print("   • Chi-squared might not be optimal for this task")
    print("   • Feature selection after combination might prefer structural features")
    print("   • No feature scaling before selection")
    
    print("\n4. DOMAIN FEATURE QUALITY:")
    print("   • Keywords might be too generic")
    print("   • Ratios and counts might be redundant")
    print("   • No validation of keyword effectiveness per category")
    
    print("\n💡 NEXT STEPS PRIORITY ORDER:")
    print("1. HIGH PRIORITY:")
    print("   • Revert to existing TF-IDF parameters as baseline")
    print("   • Add only proven domain features incrementally")
    print("   • Use proper feature scaling and selection")
    
    print("\n2. MEDIUM PRIORITY:")
    print("   • Validate domain keywords per category")
    print("   • Test different feature selection methods")
    print("   • Analyze category-specific feature importance")
    
    print("\n3. LOW PRIORITY:")
    print("   • Experiment with character n-grams carefully")
    print("   • Add more sophisticated domain features")
    print("   • Try ensemble methods")
    
    print("\n🎯 SUCCESS CRITERIA:")
    print("• Step 1: Match existing performance (62.67%)")
    print("• Step 2: Incremental improvements (+2-3%)")
    print("• Step 3: Target 70%+ accuracy")

def main():
    """Main diagnostic function"""
    print("🔬 ENHANCED FEATURES DIAGNOSTIC ANALYSIS")
    print("=" * 60)
    
    try:
        # Analyze text processing
        original_desc, processed_desc = analyze_text_processing_differences()
        
        # Analyze TF-IDF differences
        existing_vec, enhanced_vec = analyze_tfidf_differences()
        
        # Analyze feature importance patterns
        analyze_feature_importance_patterns()
        
        # Create improved configuration
        improved_config = create_improved_config()
        
        # Performance analysis summary
        performance_analysis_summary()
        
        print(f"\n✅ DIAGNOSTIC ANALYSIS COMPLETE")
        print(f"📋 Ready to implement conservative improvements")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
