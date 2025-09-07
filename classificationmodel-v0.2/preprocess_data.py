#!/usr/bin/env python3
"""
Preprocessing Script for Spend Platform Categorization Data
Filters training data to include only top 10 categories from Category L2
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import the data analyzer
from data_analysis import CategorizationDataAnalyzer

class DataPreprocessor:
    """Preprocessor for filtering categorization training data"""

    def __init__(self, excel_file_path, output_dir='preprocessed_data'):
        self.excel_file_path = excel_file_path
        self.output_dir = output_dir
        self.df = None
        self.top_categories = None

        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(exist_ok=True)

    def load_and_analyze_data(self):
        """Load data and perform analysis to identify top categories"""
        print("🔍 Loading and analyzing data...")

        # Use the data analyzer to load and analyze data
        analyzer = CategorizationDataAnalyzer(self.excel_file_path)
        self.df = analyzer.load_data()

        # Get top 10 categories from Category L2
        if 'Category L2' in self.df.columns:
            l2_counts = self.df['Category L2'].value_counts()
            self.top_categories = l2_counts.head(10).index.tolist()

            print(f"\n📊 Top 10 Category L2 categories identified:")
            for i, (category, count) in enumerate(l2_counts.head(10).items(), 1):
                percentage = (count / len(self.df)) * 100
                print(f"{i:2d}. {category:<35} {count:5d} ({percentage:5.1f}%)")

            return True
        else:
            print("❌ Category L2 column not found in data")
            return False

    def filter_data_by_top_categories(self):
        """Filter the dataset to include only records from top 10 L2 categories"""
        if self.df is None or self.top_categories is None:
            print("❌ Data not loaded or top categories not identified")
            return None

        print(f"\n🔄 Filtering data to top 10 Category L2 categories...")

        # Filter data to include only top categories
        filtered_df = self.df[self.df['Category L2'].isin(self.top_categories)].copy()

        print(f"✅ Original dataset: {len(self.df)} records")
        print(f"✅ Filtered dataset: {len(filtered_df)} records")
        print(".1f")

        return filtered_df

    def filter_rare_categories(self, filtered_df, min_l3_count=10):
        """Filter rare L3 categories and set corresponding L4 to 'Others'"""
        if filtered_df is None:
            return None

        print(f"\n🔄 Filtering rare L3 categories (count < {min_l3_count})...")

        # Get L3 category counts
        l3_counts = filtered_df['Category L3'].value_counts()

        # Identify rare categories (less than min_l3_count)
        rare_categories = l3_counts[l3_counts < min_l3_count].index.tolist()
        keep_categories = l3_counts[l3_counts >= min_l3_count].index.tolist()

        print(f"📊 L3 categories before filtering: {len(l3_counts)}")
        print(f"📊 Rare L3 categories (count < {min_l3_count}): {len(rare_categories)}")
        print(f"📊 Categories to keep: {len(keep_categories)}")

        if rare_categories:
            print(f"🔄 Rare categories being grouped as 'Others':")
            for cat in rare_categories[:5]:  # Show first 5
                count = l3_counts[cat]
                print(f"  • {cat} ({count} records)")
            if len(rare_categories) > 5:
                print(f"  • ... and {len(rare_categories) - 5} more")

        # Create a copy to avoid modifying the original
        processed_df = filtered_df.copy()

        # Replace rare L3 categories with 'Others'
        processed_df['Category L3'] = processed_df['Category L3'].apply(
            lambda x: 'Others' if x in rare_categories else x
        )

        # For rows where L3 is 'Others', also set L4 to 'Others'
        others_mask = processed_df['Category L3'] == 'Others'
        processed_df.loc[others_mask, 'Category L4'] = 'Others'

        # Also set L5 to 'Others' for consistency
        processed_df.loc[others_mask, 'Category L5'] = 'Others'

        # Show statistics after filtering
        final_l3_counts = processed_df['Category L3'].value_counts()
        print(f"\n✅ L3 categories after filtering: {len(final_l3_counts)}")
        print("📊 Top L3 categories after filtering:")
        for cat, count in final_l3_counts.head(5).items():
            percentage = (count / len(processed_df)) * 100
            print(f"  {cat:<35} {count:5d} ({percentage:5.1f}%)")

        others_count = final_l3_counts.get('Others', 0)
        if others_count > 0:
            others_percentage = (others_count / len(processed_df)) * 100
            print(f"  {'Others':<35} {others_count:5d} ({others_percentage:5.1f}%)")

        return processed_df

    def save_filtered_data(self, filtered_df, output_filename='top_10_l2_categories_data.xlsx'):
        """Save the filtered data to Excel file"""
        if filtered_df is None:
            print("❌ No filtered data to save")
            return False

        output_path = os.path.join(self.output_dir, output_filename)

        try:
            filtered_df.to_excel(output_path, index=False)
            print(f"✅ Filtered data saved to: {output_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving data: {e}")
            return False

    def generate_category_summary(self, filtered_df):
        """Generate a summary of the filtered categories"""
        if filtered_df is None:
            return None

        print(f"\n📋 Category Summary for Filtered Data:")

        summary = {}

        # Category L2 distribution in filtered data
        l2_dist = filtered_df['Category L2'].value_counts()
        summary['l2_distribution'] = l2_dist.to_dict()

        print(f"🏷️  Category L2 Distribution:")
        for category, count in l2_dist.items():
            percentage = (count / len(filtered_df)) * 100
            print(f"  {category:<35} {count:5d} ({percentage:5.1f}%)")

        # Other category levels distribution
        for level in ['Category L3', 'Category L4', 'Category L5']:
            if level in filtered_df.columns:
                level_dist = filtered_df[level].value_counts()
                summary[f'{level.lower()}_distribution'] = level_dist.to_dict()
                print(f"\n🏷️  {level} Distribution (Top 5):")
                for category, count in level_dist.head(5).items():
                    percentage = (count / len(filtered_df)) * 100
                    print(f"  {category:<35} {count:5d} ({percentage:5.1f}%)")

        return summary

    def create_train_test_split(self, filtered_df, test_size=0.2, random_state=42):
        """Create train/test split from filtered data"""
        if filtered_df is None:
            return None, None

        print(f"\n✂️  Creating train/test split (test_size={test_size})...")

        # Stratified split by Category L2 to maintain category distribution
        from sklearn.model_selection import train_test_split

        try:
            train_df, test_df = train_test_split(
                filtered_df,
                test_size=test_size,
                random_state=random_state,
                stratify=filtered_df['Category L2']
            )

            print(f"✅ Training set: {len(train_df)} records")
            print(f"✅ Test set: {len(test_df)} records")

            # Apply rare category filtering to each split separately
            print(f"\n🔄 Applying rare category filtering to train and test sets...")
            train_df = self.filter_rare_categories(train_df, min_l3_count=10)
            test_df = self.filter_rare_categories(test_df, min_l3_count=10)

            if train_df is None or test_df is None:
                print("❌ Error applying rare category filtering to train/test splits")
                return None, None

            # Save train/test splits
            train_path = os.path.join(self.output_dir, 'train_data.xlsx')
            test_path = os.path.join(self.output_dir, 'test_data.xlsx')

            train_df.to_excel(train_path, index=False)
            test_df.to_excel(test_path, index=False)

            print(f"✅ Training data saved to: {train_path}")
            print(f"✅ Test data saved to: {test_path}")

            return train_df, test_df

        except Exception as e:
            print(f"⚠️  Could not create stratified split: {e}")
            print("Creating random split instead...")

            # Fallback to random split
            train_df, test_df = train_test_split(
                filtered_df,
                test_size=test_size,
                random_state=random_state
            )

            print(f"✅ Training set: {len(train_df)} records")
            print(f"✅ Test set: {len(test_df)} records")

            # Apply rare category filtering to each split separately
            train_df = self.filter_rare_categories(train_df, min_l3_count=10)
            test_df = self.filter_rare_categories(test_df, min_l3_count=10)

            if train_df is None or test_df is None:
                print("❌ Error applying rare category filtering to train/test splits")
                return None, None

            return train_df, test_df

    def run_preprocessing_pipeline(self):
        """Run the complete preprocessing pipeline"""
        print("="*80)
        print("SPEND PLATFORM DATA PREPROCESSING PIPELINE")
        print("="*80)

        # Step 1: Load and analyze data
        if not self.load_and_analyze_data():
            return False

        # Step 2: Filter data by top categories
        filtered_df = self.filter_data_by_top_categories()
        if filtered_df is None:
            return False

        # Step 3: Save intermediate filtered data (before rare category filtering)
        intermediate_filename = 'top_10_l2_categories_data.xlsx'
        if not self.save_filtered_data(filtered_df, intermediate_filename):
            return False

        # Step 4: Generate category summary for intermediate data
        self.generate_category_summary(filtered_df)

        # Step 5: Create train/test split (rare category filtering applied within this step)
        train_df, test_df = self.create_train_test_split(filtered_df)
        if train_df is None or test_df is None:
            return False

        print("\n" + "="*80)
        print("PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("📁 Files Generated:")
        print(f"  • Filtered data (before rare filtering): {self.output_dir}/{intermediate_filename}")
        print(f"  • Training data: {self.output_dir}/train_data.xlsx")
        print(f"  • Test data: {self.output_dir}/test_data.xlsx")
        print("\n📊 Summary:")
        print(f"  • Original records: {len(self.df) if self.df is not None else 0}")
        print(f"  • Filtered records (top L2): {len(filtered_df)}")
        print(f"  • Final processed records: {len(train_df) + len(test_df)}")
        print(f"  • Top categories: {len(self.top_categories) if self.top_categories else 0}")
        data_reduction = ((len(self.df) - len(filtered_df)) / len(self.df)) * 100 if self.df is not None else 0
        print(".1f")
        return True

def main():
    """Main function to run the preprocessing pipeline"""
    excel_file = '/Users/sujoymukherjee/code/spendplatform/context/Copy of Catergorization Working Sheet_Labelled Data for Training.xlsx'

    preprocessor = DataPreprocessor(excel_file)
    success = preprocessor.run_preprocessing_pipeline()

    if not success:
        print("❌ Preprocessing failed!")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
