#!/usr/bin/env python3
"""
Analysis and Optimization of Enhanced Features
Based on the test results, optimize the feature set to improve performance
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def analyze_feature_performance():
    """Analyze the feature performance results"""
    print("🔍 ENHANCED FEATURES PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    print("\n📊 KEY FINDINGS:")
    print("• Enhanced features: 1698 features (+58% vs existing 1073)")
    print("• Performance: -8.51% decrease in accuracy")
    print("• Existing accuracy: 62.67%")
    print("• Enhanced accuracy: 57.33%")
    
    print("\n🎯 TOP PERFORMING FEATURE GROUPS:")
    print("1. Text Features: 76.77% total importance (TF-IDF + char n-grams)")
    print("2. Structural Features: 16.38% total importance (length, ratios, patterns)")
    print("3. Domain Features: 10.40% total importance (category-specific keywords)")
    print("4. Semantic Features: 1.35% total importance (technical specifications)")
    print("5. Statistical Features: 0.68% total importance (word frequency stats)")
    
    print("\n🔬 ANALYSIS:")
    print("• Text features (TF-IDF + char n-grams) dominate importance")
    print("• Enhanced TF-IDF (2000 features) vs existing (1073) may be overfitting")
    print("• Character n-grams (500 features) might be adding noise")
    print("• Domain-specific features show promise but need refinement")
    print("• Structural features have high average importance per feature")
    
    print("\n💡 OPTIMIZATION RECOMMENDATIONS:")
    
    recommendations = [
        {
            "area": "TF-IDF Optimization",
            "issues": ["Increased from 1500 to 2000 features", "May be overfitting", "Lower min_df might include noise"],
            "solutions": ["Reduce max_features to 1200-1500", "Increase min_df to 3-5", "Optimize ngram_range"],
            "priority": "HIGH"
        },
        {
            "area": "Character N-grams",
            "issues": ["500 additional features", "May capture irrelevant patterns", "High dimensionality"],
            "solutions": ["Reduce to 200-300 features", "Increase min_df to 5", "Focus on 3-4 char n-grams"],
            "priority": "HIGH"
        },
        {
            "area": "Domain Features",
            "issues": ["Keywords may be too generic", "Ratio features might be redundant"],
            "solutions": ["Refine keyword lists per category", "Use feature selection", "Add L3/L4 specific terms"],
            "priority": "MEDIUM"
        },
        {
            "area": "Feature Selection",
            "issues": ["Too many low-importance features", "Potential overfitting"],
            "solutions": ["Apply feature selection (SelectKBest)", "Use recursive feature elimination", "Focus on top 1000-1200 features"],
            "priority": "HIGH"
        },
        {
            "area": "Structural Features",
            "issues": ["High individual importance", "Well-performing group"],
            "solutions": ["Keep most structural features", "Add more length/pattern features", "Optimize thresholds"],
            "priority": "LOW"
        }
    ]
    
    print("\n📋 DETAILED RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['area']} ({rec['priority']} Priority)")
        print("   Issues:")
        for issue in rec['issues']:
            print(f"   • {issue}")
        print("   Solutions:")
        for solution in rec['solutions']:
            print(f"   • {solution}")
    
    print("\n🎯 IMMEDIATE ACTION PLAN:")
    print("1. Create optimized feature extractor with reduced TF-IDF features")
    print("2. Apply feature selection to identify most important features")
    print("3. Refine domain keywords based on actual category patterns")
    print("4. Test incremental improvements to measure impact")
    
    return recommendations

def create_optimization_strategy():
    """Create a strategy for optimizing the enhanced features"""
    
    print("\n" + "=" * 60)
    print("🚀 FEATURE OPTIMIZATION STRATEGY")
    print("=" * 60)
    
    strategy = {
        "phase_1": {
            "name": "TF-IDF Optimization",
            "goal": "Reduce overfitting from text features",
            "changes": [
                "Reduce max_features from 2000 to 1300",
                "Increase min_df from 2 to 4",
                "Reduce char n-grams from 500 to 250",
                "Keep ngram_range (1,4) but test (1,3)"
            ],
            "expected_impact": "+3-5% accuracy improvement"
        },
        "phase_2": {
            "name": "Feature Selection",
            "goal": "Keep only most predictive features",
            "changes": [
                "Apply SelectKBest with k=1000",
                "Use chi-squared or mutual information",
                "Remove features with importance < 0.001"
            ],
            "expected_impact": "+2-4% accuracy improvement"
        },
        "phase_3": {
            "name": "Domain Keywords Refinement",
            "goal": "Improve category-specific detection",
            "changes": [
                "Analyze top misclassified categories",
                "Add L3/L4 specific keywords",
                "Create category-specific stop words",
                "Weight keywords by category frequency"
            ],
            "expected_impact": "+1-3% accuracy improvement"
        },
        "phase_4": {
            "name": "Advanced Features",
            "goal": "Add sophisticated domain features",
            "changes": [
                "Add word embeddings similarity features",
                "Create category co-occurrence features",
                "Add sequence/pattern matching",
                "Include part number detection"
            ],
            "expected_impact": "+2-5% accuracy improvement"
        }
    }
    
    for phase, details in strategy.items():
        print(f"\n📋 {details['name']}:")
        print(f"   Goal: {details['goal']}")
        print(f"   Expected Impact: {details['expected_impact']}")
        print("   Changes:")
        for change in details['changes']:
            print(f"   • {change}")
    
    print(f"\n🎯 TOTAL EXPECTED IMPROVEMENT: +8-17% accuracy")
    print(f"   Target: Achieve 70-75% L2 categorization accuracy")
    print(f"   Current: 57.33% (enhanced) vs 62.67% (existing)")
    
    return strategy

def create_optimized_config():
    """Create configuration for optimized enhanced features"""
    
    config = {
        "tfidf_config": {
            "max_features": 1300,  # Reduced from 2000
            "ngram_range": (1, 3),  # Reduced from (1, 4)
            "min_df": 4,  # Increased from 2
            "max_df": 0.90,  # Reduced from 0.92
            "sublinear_tf": True,
            "use_idf": True,
            "smooth_idf": True
        },
        "char_ngram_config": {
            "max_features": 250,  # Reduced from 500
            "ngram_range": (3, 4),  # More focused on 3-4 char patterns
            "min_df": 5,  # Increased from 3
            "max_df": 0.85
        },
        "feature_selection": {
            "method": "SelectKBest",
            "k": 1000,  # Total features to keep
            "score_func": "chi2"  # or "mutual_info_classif"
        },
        "domain_keywords": {
            "max_keywords_per_category": 15,  # Limit to most relevant
            "min_keyword_frequency": 5,  # Only frequent keywords
            "use_l3_keywords": True,  # Add L3-specific terms
            "weight_by_category_size": True
        }
    }
    
    print("\n📁 OPTIMIZED CONFIGURATION:")
    for section, params in config.items():
        print(f"\n{section}:")
        for param, value in params.items():
            print(f"  {param}: {value}")
    
    return config

def next_steps():
    """Outline the next steps for implementation"""
    
    print("\n" + "=" * 60)
    print("📋 IMPLEMENTATION ROADMAP")
    print("=" * 60)
    
    steps = [
        {
            "step": 1,
            "title": "Create Optimized Enhanced Features v2",
            "tasks": [
                "Implement reduced TF-IDF configuration",
                "Add feature selection capabilities",
                "Refine domain keyword extraction",
                "Add performance monitoring"
            ],
            "files": ["enhanced_features_v2.py"],
            "timeline": "Immediate"
        },
        {
            "step": 2,
            "title": "Test Optimized Features",
            "tasks": [
                "Run comparative tests with v1 and existing",
                "Measure accuracy improvements",
                "Analyze feature importance changes",
                "Validate on different data splits"
            ],
            "files": ["test_optimized_features.py"],
            "timeline": "After step 1"
        },
        {
            "step": 3,
            "title": "Integrate with Training Pipeline",
            "tasks": [
                "Update train_classification_model.py",
                "Add feature selection to pipeline",
                "Update model saving/loading",
                "Test end-to-end workflow"
            ],
            "files": ["train_classification_model.py", "predict_categories.py"],
            "timeline": "After validation"
        },
        {
            "step": 4,
            "title": "Production Deployment",
            "tasks": [
                "Update production prediction script",
                "Validate on full dataset",
                "Monitor performance improvements",
                "Document changes and results"
            ],
            "files": ["Production deployment"],
            "timeline": "After integration"
        }
    ]
    
    for step in steps:
        print(f"\n{step['step']}. {step['title']} ({step['timeline']})")
        print("   Tasks:")
        for task in step['tasks']:
            print(f"   • {task}")
        print(f"   Files: {', '.join(step['files'])}")
    
    print(f"\n🎯 SUCCESS METRICS:")
    print(f"• L2 Accuracy: Target 70%+ (current best: 62.67%)")
    print(f"• Feature Count: ~1000-1200 (vs current 1698)")
    print(f"• Training Time: Maintain or improve")
    print(f"• Memory Usage: Reduce due to fewer features")

def main():
    """Main analysis function"""
    
    # Run analysis
    recommendations = analyze_feature_performance()
    strategy = create_optimization_strategy()
    config = create_optimized_config()
    next_steps()
    
    print(f"\n" + "=" * 60)
    print(f"✅ ANALYSIS COMPLETE")
    print(f"📊 Ready to implement optimized enhanced features v2")
    print(f"🎯 Goal: Improve L2 accuracy from 57.33% to 70%+")
    print(f"=" * 60)

if __name__ == "__main__":
    main()
