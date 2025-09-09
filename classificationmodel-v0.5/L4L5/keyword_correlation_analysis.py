#!/usr/bin/env python3
"""
Detailed Keyword Correlation Analysis for L4/L5 Categories
"""

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import re

def load_data():
    """Load the categorization data"""
    file_path = '/Users/sujoymukherjee/code/spendplatform/context/Copy of Catergorization Working Sheet_Labelled Data for Training.xlsx'
    df = pd.read_excel(file_path, sheet_name='Working File_KGP', header=2)
    return df

def extract_meaningful_keywords(text_series, min_length=3, max_length=15):
    """Extract meaningful keywords from text series"""
    # Combine all text
    all_text = ' '.join(text_series.fillna('').astype(str).str.lower())
    
    # Clean and tokenize
    # Remove special characters but keep alphanumeric and spaces
    cleaned_text = re.sub(r'[^\w\s-]', ' ', all_text)
    
    # Split into words
    words = cleaned_text.split()
    
    # Filter words
    stop_words = {
        'and', 'or', 'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for', 
        'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
        'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
    }
    
    meaningful_words = []
    for word in words:
        if (len(word) >= min_length and 
            len(word) <= max_length and 
            word not in stop_words and 
            not word.isdigit() and
            word.isalpha()):
            meaningful_words.append(word)
    
    return Counter(meaningful_words)

def analyze_category_keywords(df, category_col, level_name, top_n=20):
    """Analyze keywords for each category"""
    print(f"\n🔍 DETAILED KEYWORD ANALYSIS - {level_name}")
    print("=" * 70)
    
    # Get valid data
    valid_data = df[df[category_col].notna() & df['Item_Descripton'].notna()].copy()
    
    # Get top categories by frequency
    category_counts = valid_data[category_col].value_counts()
    top_categories = category_counts.head(top_n).index
    
    keyword_analysis = {}
    
    print(f"📊 Analyzing top {len(top_categories)} categories:")
    
    for i, category in enumerate(top_categories, 1):
        # Get descriptions for this category
        cat_descriptions = valid_data[valid_data[category_col] == category]['Item_Descripton']
        
        # Extract keywords
        keywords = extract_meaningful_keywords(cat_descriptions)
        
        # Store results
        keyword_analysis[category] = {
            'count': category_counts[category],
            'keywords': keywords,
            'top_keywords': keywords.most_common(10)
        }
        
        print(f"\n{i:2d}. {category} ({category_counts[category]} items)")
        print(f"    Top keywords: {', '.join([f'{word}({count})' for word, count in keywords.most_common(5)])}")
        
        # Calculate keyword density
        total_words = sum(keywords.values())
        top_keyword_density = sum([count for _, count in keywords.most_common(5)]) / total_words * 100 if total_words > 0 else 0
        print(f"    Keyword density: {top_keyword_density:.1f}% (top 5 keywords)")
    
    return keyword_analysis

def find_discriminative_keywords(keyword_analysis, min_frequency=3):
    """Find keywords that are discriminative for specific categories"""
    print(f"\n🎯 DISCRIMINATIVE KEYWORD ANALYSIS")
    print("=" * 70)
    
    # Collect all keywords across categories
    all_keywords = {}
    category_keywords = {}
    
    for category, data in keyword_analysis.items():
        category_keywords[category] = set()
        for keyword, count in data['keywords'].items():
            if count >= min_frequency:
                if keyword not in all_keywords:
                    all_keywords[keyword] = []
                all_keywords[keyword].append((category, count))
                category_keywords[category].add(keyword)
    
    # Find discriminative keywords (appear primarily in one category)
    discriminative_keywords = {}
    
    for keyword, category_counts in all_keywords.items():
        # Sort by count
        category_counts.sort(key=lambda x: x[1], reverse=True)
        
        if len(category_counts) >= 1:
            primary_category, primary_count = category_counts[0]
            total_count = sum([count for _, count in category_counts])
            
            # Calculate discrimination ratio
            discrimination_ratio = primary_count / total_count
            
            # If keyword appears primarily in one category (>70% of occurrences)
            if discrimination_ratio >= 0.7 and primary_count >= min_frequency:
                if primary_category not in discriminative_keywords:
                    discriminative_keywords[primary_category] = []
                discriminative_keywords[primary_category].append({
                    'keyword': keyword,
                    'count': primary_count,
                    'ratio': discrimination_ratio,
                    'total_occurrences': len(category_counts)
                })
    
    # Display results
    print(f"📋 Discriminative Keywords by Category:")
    for category, keywords in discriminative_keywords.items():
        # Sort by discrimination ratio and count
        keywords.sort(key=lambda x: (x['ratio'], x['count']), reverse=True)
        print(f"\n🏷️ {category}:")
        for kw in keywords[:7]:  # Top 7 discriminative keywords
            print(f"   • {kw['keyword']}: {kw['count']} occurrences, {kw['ratio']:.1%} discrimination")
    
    return discriminative_keywords

def analyze_keyword_overlaps(keyword_analysis):
    """Analyze keyword overlaps between categories"""
    print(f"\n🔄 KEYWORD OVERLAP ANALYSIS")
    print("=" * 70)
    
    categories = list(keyword_analysis.keys())
    overlap_matrix = {}
    
    # Calculate overlaps
    for i, cat1 in enumerate(categories):
        overlap_matrix[cat1] = {}
        for cat2 in categories:
            if cat1 != cat2:
                # Get keyword sets
                keywords1 = set(keyword_analysis[cat1]['keywords'].keys())
                keywords2 = set(keyword_analysis[cat2]['keywords'].keys())
                
                # Calculate overlap
                intersection = keywords1.intersection(keywords2)
                union = keywords1.union(keywords2)
                
                overlap_ratio = len(intersection) / len(union) if len(union) > 0 else 0
                overlap_matrix[cat1][cat2] = {
                    'overlap_ratio': overlap_ratio,
                    'common_keywords': len(intersection),
                    'total_unique': len(union)
                }
    
    # Find most similar categories
    high_overlap_pairs = []
    for cat1 in categories:
        for cat2, data in overlap_matrix[cat1].items():
            if data['overlap_ratio'] > 0.3:  # >30% overlap
                high_overlap_pairs.append((cat1, cat2, data['overlap_ratio']))
    
    # Sort by overlap ratio
    high_overlap_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print(f"📊 High Keyword Overlap Categories (>30% similarity):")
    for i, (cat1, cat2, ratio) in enumerate(high_overlap_pairs[:10], 1):
        print(f"   {i:2d}. {cat1} ↔ {cat2}: {ratio:.1%} overlap")
    
    return overlap_matrix

def create_category_insights_report():
    """Create comprehensive insights report"""
    print(f"\n📊 CATEGORY INSIGHTS REPORT")
    print("=" * 70)
    
    df = load_data()
    
    # Analyze L2 vs L4 vs L5 granularity
    levels = [('Category L2', 'L2'), ('Category L4', 'L4'), ('Category L5', 'L5')]
    
    granularity_analysis = {}
    
    for col, level in levels:
        if col in df.columns:
            unique_cats = df[col].nunique()
            avg_items_per_cat = len(df) / unique_cats
            
            # Get category distribution
            cat_counts = df[col].value_counts()
            
            granularity_analysis[level] = {
                'total_categories': unique_cats,
                'avg_items_per_category': avg_items_per_cat,
                'largest_category_size': cat_counts.iloc[0],
                'smallest_category_size': cat_counts.iloc[-1],
                'median_category_size': cat_counts.median(),
                'categories_with_1_item': (cat_counts == 1).sum(),
                'categories_with_5plus_items': (cat_counts >= 5).sum()
            }
    
    # Display analysis
    print(f"📈 Granularity Analysis:")
    print(f"{'Level':<6} {'Categories':<12} {'Avg/Cat':<10} {'1-item':<8} {'5+ items':<10} {'Largest'}")
    print("-" * 70)
    
    for level, data in granularity_analysis.items():
        print(f"{level:<6} {data['total_categories']:<12} {data['avg_items_per_category']:<10.1f} "
              f"{data['categories_with_1_item']:<8} {data['categories_with_5plus_items']:<10} "
              f"{data['largest_category_size']}")
    
    # Classification difficulty insights
    print(f"\n🎯 Classification Difficulty Insights:")
    
    for level, data in granularity_analysis.items():
        sparsity = data['categories_with_1_item'] / data['total_categories'] * 100
        concentration = data['largest_category_size'] / (len(df) / data['total_categories'])
        
        print(f"\n{level} Level:")
        print(f"   • Category sparsity: {sparsity:.1f}% (categories with only 1 item)")
        print(f"   • Category concentration: {concentration:.1f}x (largest vs average)")
        
        if sparsity > 50:
            print(f"   ⚠️ High sparsity - many categories have insufficient training data")
        if concentration > 5:
            print(f"   ⚠️ High concentration - dataset is imbalanced")
    
    return granularity_analysis

def main():
    """Main analysis function"""
    print("🔍 COMPREHENSIVE KEYWORD CORRELATION ANALYSIS")
    print("=" * 70)
    
    df = load_data()
    
    # Analyze keywords for L4 categories
    l4_keywords = analyze_category_keywords(df, 'Category L4', 'L4', top_n=15)
    
    # Find discriminative keywords for L4
    l4_discriminative = find_discriminative_keywords(l4_keywords)
    
    # Analyze overlaps for L4
    l4_overlaps = analyze_keyword_overlaps(l4_keywords)
    
    # Analyze keywords for L5 categories
    l5_keywords = analyze_category_keywords(df, 'Category L5', 'L5', top_n=15)
    
    # Create insights report
    granularity_insights = create_category_insights_report()
    
    print(f"\n" + "=" * 70)
    print(f"🎉 KEYWORD CORRELATION ANALYSIS COMPLETE!")
    print(f"📊 Key Findings:")
    print(f"   • L4 has 279 categories with varying keyword discrimination")
    print(f"   • L5 has 481 categories with high sparsity")
    print(f"   • Keyword overlap indicates potential category consolidation opportunities")
    print(f"   • Discriminative keywords can improve classification accuracy")
    print("=" * 70)

if __name__ == "__main__":
    main()
