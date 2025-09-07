#!/usr/bin/env python3
"""
Category Filtering Utility for Spend Platform Data
Removes categories with insufficient samples to ensure model quality
"""

import pandas as pd
import os
from pathlib import Path

def filter_categories_by_sample_size(train_df, test_df, target_column='Category L2', min_samples=10):
    """
    Filter out categories with fewer than min_samples from both train and test data

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        target_column: Column to filter by (default: 'Category L2')
        min_samples: Minimum samples required per category (default: 10)

    Returns:
        tuple: (filtered_train_df, filtered_test_df, removed_categories)
    """

    print(f"🔍 Filtering categories with < {min_samples} samples in {target_column}...")

    # Get category counts
    train_counts = train_df[target_column].value_counts()
    test_counts = test_df[target_column].value_counts()

    # Find categories to keep (those with sufficient samples in both datasets)
    train_sufficient = set(train_counts[train_counts >= min_samples].index)
    test_sufficient = set(test_counts[test_counts >= min_samples].index)
    categories_to_keep = train_sufficient.intersection(test_sufficient)

    # Find categories that will be removed
    all_categories = set(train_counts.index).union(set(test_counts.index))
    removed_categories = all_categories - categories_to_keep

    print(f"📊 Analysis Results:")
    print(f"   Total categories: {len(all_categories)}")
    print(f"   Categories to keep: {len(categories_to_keep)}")
    print(f"   Categories to remove: {len(removed_categories)}")

    if removed_categories:
        print(f"\n🚨 Categories being removed (< {min_samples} samples):")
        for cat in sorted(removed_categories):
            train_count = train_counts.get(cat, 0)
            test_count = test_counts.get(cat, 0)
            print(f"   • {cat}: Train={train_count}, Test={test_count}")
    else:
        print(f"\n✅ All categories have ≥ {min_samples} samples - no filtering needed!")

    # Filter datasets
    if categories_to_keep:
        train_filtered = train_df[train_df[target_column].isin(categories_to_keep)].copy()
        test_filtered = test_df[test_df[target_column].isin(categories_to_keep)].copy()

        print(f"\n📈 Filtering Summary:")
        print(f"   Train: {len(train_df)} → {len(train_filtered)} samples")
        print(f"   Test:  {len(test_df)} → {len(test_filtered)} samples")
        print(f"   Samples removed: {len(train_df) - len(train_filtered)} train, {len(test_df) - len(test_filtered)} test")
    else:
        print(f"\n⚠️  No categories meet the minimum sample requirement!")
        train_filtered = train_df.copy()
        test_filtered = test_df.copy()

    return train_filtered, test_filtered, removed_categories

def save_filtered_data(train_df, test_df, output_dir='preprocessed_data'):
    """
    Save filtered datasets to Excel files
    """
    Path(output_dir).mkdir(exist_ok=True)

    train_path = os.path.join(output_dir, 'train_data_filtered.xlsx')
    test_path = os.path.join(output_dir, 'test_data_filtered.xlsx')

    train_df.to_excel(train_path, index=False)
    test_df.to_excel(test_path, index=False)

    print(f"💾 Filtered data saved:")
    print(f"   • {train_path} ({len(train_df)} samples)")
    print(f"   • {test_path} ({len(test_df)} samples)")

def main():
    """
    Main function to filter categories by sample size
    """
    print("="*70)
    print("SPEND PLATFORM CATEGORY FILTERING UTILITY")
    print("="*70)

    # Configuration
    train_file = 'preprocessed_data/train_data.xlsx'
    test_file = 'preprocessed_data/test_data.xlsx'
    target_column = 'Category L2'
    min_samples = 10

    # Check if files exist
    if not os.path.exists(train_file):
        print(f"❌ Train file not found: {train_file}")
        return

    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return

    # Load data
    print(f"📂 Loading data from:")
    print(f"   • {train_file}")
    print(f"   • {test_file}")

    train_df = pd.read_excel(train_file)
    test_df = pd.read_excel(test_file)

    print(f"✅ Loaded {len(train_df)} train samples, {len(test_df)} test samples")

    # Filter categories
    train_filtered, test_filtered, removed_categories = filter_categories_by_sample_size(
        train_df, test_df, target_column, min_samples
    )

    # Save filtered data if any categories were removed
    if removed_categories:
        save_filtered_data(train_filtered, test_filtered)
        print(f"\n✅ Category filtering completed!")
        print(f"   Removed {len(removed_categories)} categories with < {min_samples} samples")
    else:
        print(f"\n✅ No filtering needed - all categories have sufficient samples!")

    print(f"\n📊 Final Dataset:")
    print(f"   Train: {len(train_filtered)} samples")
    print(f"   Test:  {len(test_filtered)} samples")
    print(f"   Categories: {len(train_filtered[target_column].value_counts())}")

if __name__ == "__main__":
    main()
