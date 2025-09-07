#!/usr/bin/env python3
"""
Categorization File Generator
Creates comprehensive categorization files from training data for spend analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys

class CategorizationFileGenerator:
    """Generate categorization files from training data."""

    def __init__(self, input_file=None, output_dir=None):
        self.input_file = input_file or '/Users/sujoymukherjee/code/spendplatform/context/Copy of Catergorization Working Sheet_Labelled Data for Training.xlsx'
        self.output_dir = output_dir or '/Users/sujoymukherjee/code/spendplatform/context'
        self.training_data = None

    def load_training_data(self):
        """Load and preprocess the training data."""
        print(f"Loading training data from: {self.input_file}")

        try:
            # Load the training data
            training_df = pd.read_excel(self.input_file)
            print(f"Raw data shape: {training_df.shape}")

            # Use second row as headers
            training_df.columns = training_df.iloc[1]
            training_df = training_df.iloc[2:].reset_index(drop=True)

            print(f"Processed data shape: {training_df.shape}")
            print(f"Columns: {list(training_df.columns)}")

            self.training_data = training_df
            return True

        except Exception as e:
            print(f"Error loading training data: {e}")
            return False

    def create_comprehensive_categorization(self):
        """Create a comprehensive categorization file with all available data."""
        if self.training_data is None:
            print("Training data not loaded!")
            return None

        print("\nCreating comprehensive categorization file...")

        # Select all available categorization columns
        required_columns = ['Item Code', 'Item_Descripton', 'Category L1', 'Category L2',
                          'Category L3', 'Category L4', 'Category L5', 'UDS comment']
        available_columns = [col for col in required_columns if col in self.training_data.columns]

        categorization_df = self.training_data[available_columns].copy()

        # Clean the data
        print("Cleaning data...")
        initial_count = len(categorization_df)

        # Remove rows with missing essential fields
        categorization_df = categorization_df.dropna(subset=['Item_Descripton', 'Category L1'])
        categorization_df = categorization_df[categorization_df['Category L1'].str.strip() != '']
        categorization_df = categorization_df[categorization_df['Item_Descripton'].str.strip() != '']

        # Remove unclear categories
        if 'Unclear' in categorization_df['Category L1'].values:
            categorization_df = categorization_df[categorization_df['Category L1'] != 'Unclear']

        cleaned_count = len(categorization_df)
        print(f"Removed {initial_count - cleaned_count} invalid rows")

        # Add analysis columns
        print("Adding analysis columns...")

        # Text analysis
        categorization_df['description_length'] = categorization_df['Item_Descripton'].str.len()
        categorization_df['word_count'] = categorization_df['Item_Descripton'].str.split().str.len()

        # Category hierarchy completeness
        hierarchy_levels = ['Category L1', 'Category L2', 'Category L3', 'Category L4', 'Category L5']
        available_hierarchy = [col for col in hierarchy_levels if col in categorization_df.columns]
        categorization_df['hierarchy_completeness'] = categorization_df[available_hierarchy].notna().sum(axis=1)

        # Create category path
        def create_category_path(row):
            path_parts = []
            for level in available_hierarchy:
                if level in row.index and pd.notna(row[level]):
                    path_parts.append(str(row[level]).strip())
            return ' > '.join(path_parts)

        categorization_df['category_path'] = categorization_df.apply(create_category_path, axis=1)

        # Save comprehensive file
        output_path = Path(self.output_dir) / 'comprehensive_categorization.xlsx'
        categorization_df.to_excel(output_path, index=False)
        print(f"Comprehensive categorization saved to: {output_path}")

        return categorization_df

    def create_simple_categorization(self):
        """Create a simplified categorization file for easy reference."""
        if self.training_data is None:
            print("Training data not loaded!")
            return None

        print("\nCreating simple categorization file...")

        # Create simplified mapping
        simple_columns = ['Item Code', 'Item_Descripton', 'Category L1', 'Category L5']
        available_simple = [col for col in simple_columns if col in self.training_data.columns]

        simple_df = self.training_data[available_simple].copy()

        # Clean the data
        simple_df = simple_df.dropna(subset=['Item_Descripton', 'Category L1'])
        simple_df = simple_df[simple_df['Category L1'].str.strip() != '']
        simple_df = simple_df[simple_df['Item_Descripton'].str.strip() != '']

        # Remove unclear categories
        if 'Unclear' in simple_df['Category L1'].values:
            simple_df = simple_df[simple_df['Category L1'] != 'Unclear']

        # Add category path for simple version
        if 'Category L5' in simple_df.columns:
            simple_df['category_path'] = simple_df['Category L1'] + ' > ' + simple_df['Category L5']

        # Save simple file
        output_path = Path(self.output_dir) / 'simple_categorization.xlsx'
        simple_df.to_excel(output_path, index=False)
        print(f"Simple categorization saved to: {output_path}")

        return simple_df

    def create_category_reference(self):
        """Create a category reference file with unique categories and their frequencies."""
        if self.training_data is None:
            print("Training data not loaded!")
            return None

        print("\nCreating category reference file...")

        # Category L1 reference
        l1_ref = self.training_data['Category L1'].value_counts().reset_index()
        l1_ref.columns = ['Category L1', 'Frequency']
        l1_ref = l1_ref[l1_ref['Category L1'] != 'Unclear']

        # Category L5 reference
        l5_ref = self.training_data['Category L5'].value_counts().reset_index()
        l5_ref.columns = ['Category L5', 'Frequency']
        l5_ref = l5_ref[l5_ref['Category L5'] != 'Unclear']

        # Create reference file
        with pd.ExcelWriter(Path(self.output_dir) / 'category_reference.xlsx') as writer:
            l1_ref.to_excel(writer, sheet_name='Category_L1', index=False)
            l5_ref.to_excel(writer, sheet_name='Category_L5', index=False)

        print(f"Category reference saved to: {Path(self.output_dir) / 'category_reference.xlsx'}")

        return l1_ref, l5_ref

    def generate_all_files(self):
        """Generate all categorization files."""
        print("Starting categorization file generation...")

        if not self.load_training_data():
            return False

        # Generate all files
        self.create_comprehensive_categorization()
        self.create_simple_categorization()
        self.create_category_reference()

        print("\n" + "="*60)
        print("CATEGORIZATION FILE GENERATION COMPLETE")
        print("="*60)
        print("Generated files:")
        print("1. comprehensive_categorization.xlsx - Full categorization with analysis")
        print("2. simple_categorization.xlsx - Simplified mapping for easy reference")
        print("3. category_reference.xlsx - Category frequencies and reference data")

        return True

def main():
    """Main function to run the categorization file generator."""
    parser = argparse.ArgumentParser(description='Generate categorization files from training data')
    parser.add_argument('--input', '-i', help='Input Excel file path')
    parser.add_argument('--output', '-o', help='Output directory path')

    args = parser.parse_args()

    generator = CategorizationFileGenerator(
        input_file=args.input,
        output_dir=args.output
    )

    success = generator.generate_all_files()

    if success:
        print("\nCategorization files generated successfully!")
    else:
        print("\nFailed to generate categorization files!")
        sys.exit(1)

if __name__ == "__main__":
    main()
