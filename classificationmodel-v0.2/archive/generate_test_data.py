#!/usr/bin/env python3
"""
Generate Test Data for Spend Platform Categorization
Creates testdata.xlsx with 1000 records (20% existing, 80% new)
"""

import pandas as pd
import numpy as np
import random
from pathlib import Path

def load_training_data():
    """Load and process training data"""
    excel_file = '/Users/sujoymukherjee/code/spendplatform/context/Copy of Catergorization Working Sheet_Labelled Data for Training.xlsx'

    # Read with proper header
    df_raw = pd.read_excel(excel_file, header=None)
    header_row = 2
    df = pd.read_excel(excel_file, header=header_row)
    df.columns = df.columns.str.strip()

    # Remove unnamed column
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    return df

def extract_category_words(df):
    """Extract words from category levels L2-L5"""
    category_words = set()

    for level in ['Category L2', 'Category L3', 'Category L4', 'Category L5']:
        if level in df.columns:
            categories = df[level].dropna().unique()
            for category in categories:
                # Split category into words and clean
                words = str(category).lower().split()
                for word in words:
                    # Clean word (remove special chars, keep alphanumeric)
                    clean_word = ''.join(c for c in word if c.isalnum())
                    if len(clean_word) > 2:  # Only words longer than 2 chars
                        category_words.add(clean_word)

    return list(category_words)

def generate_new_description(category_words, index):
    """Generate a new item description containing at least one category word"""
    # Base templates
    templates = [
        "Industrial {} equipment and supplies",
        "{} system components and accessories",
        "Professional {} tools and machinery",
        "{} maintenance and replacement parts",
        "High quality {} materials and consumables",
        "{} processing and manufacturing supplies",
        "Advanced {} technology and solutions",
        "{} safety and protection equipment",
        "Precision {} instruments and devices",
        "{} construction and building materials"
    ]

    # Select random category word
    category_word = random.choice(category_words)

    # Select random template
    template = random.choice(templates)

    # Generate description
    description = template.format(category_word)

    # Add some variation with numbers and additional words
    variations = [
        f"{description} - Model {random.randint(100, 999)}",
        f"{description} Size {random.choice(['Small', 'Medium', 'Large'])}",
        f"{description} Quantity {random.randint(1, 100)}",
        f"{description} Grade {random.choice(['A', 'B', 'C'])}",
        f"{description} Type {random.randint(1, 10)}"
    ]

    # 30% chance to add variation
    if random.random() < 0.3:
        description = random.choice(variations)

    return description

def generate_test_data():
    """Generate the complete test dataset"""
    print("Loading training data...")
    df = load_training_data()

    print(f"Training data loaded: {len(df)} records")

    # Extract category words
    print("Extracting category words...")
    category_words = extract_category_words(df)
    print(f"Found {len(category_words)} unique category words")

    # Sample existing data (20% = 200 records)
    print("Generating existing data samples...")
    existing_data = df[['Item Code', 'Item_Descripton']].dropna().sample(
        n=200, random_state=42
    ).copy()
    existing_data['Data_Type'] = 'Existing'

    # Generate new data (80% = 800 records)
    print("Generating new test data...")
    new_data = []

    for i in range(800):
        item_code = f"TEST{i+1:04d}"  # TEST0001, TEST0002, etc.
        description = generate_new_description(category_words, i)
        new_data.append({
            'Item Code': item_code,
            'Item_Descripton': description,
            'Data_Type': 'New'
        })

    new_df = pd.DataFrame(new_data)

    # Combine all data
    print("Combining datasets...")
    test_data = pd.concat([existing_data, new_df], ignore_index=True)

    # Shuffle the data
    test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Generated {len(test_data)} test records")
    print(f"Existing data: {len(existing_data)} records")
    print(f"New data: {len(new_df)} records")

    return test_data

def validate_test_data(test_data, category_words):
    """Validate that new data contains category words"""
    print("\nValidating test data...")

    new_records = test_data[test_data['Data_Type'] == 'New']
    valid_count = 0

    for desc in new_records['Item_Descripton']:
        desc_words = set(str(desc).lower().split())
        # Clean words for comparison
        desc_words_clean = set()
        for word in desc_words:
            clean_word = ''.join(c for c in word if c.isalnum())
            if len(clean_word) > 2:
                desc_words_clean.add(clean_word)

        # Check if any category word is in the description
        if desc_words_clean.intersection(set(category_words)):
            valid_count += 1

    validation_rate = valid_count / len(new_records) * 100
    print(f"Validation rate: {validation_rate:.1f}%")

    return validation_rate >= 95  # Should be at least 95% valid

def main():
    """Main function to generate test data"""
    print("="*60)
    print("SPEND PLATFORM TEST DATA GENERATOR")
    print("="*60)

    # Generate test data
    test_data = generate_test_data()

    # Load training data again to get category words for validation
    df = load_training_data()
    category_words = extract_category_words(df)

    # Validate data
    is_valid = validate_test_data(test_data, category_words)

    if not is_valid:
        print("⚠️  Warning: Some new records may not contain category words")
    else:
        print("✅ All new records contain at least one category word")

    # Show sample of generated data
    print("\n" + "="*40)
    print("SAMPLE GENERATED DATA")
    print("="*40)

    sample = test_data.sample(10, random_state=42)
    for i, row in sample.iterrows():
        data_type = row['Data_Type']
        item_code = row['Item Code']
        desc = row['Item_Descripton'][:60] + "..." if len(row['Item_Descripton']) > 60 else row['Item_Descripton']
        print(f"  [{data_type}] {item_code} - {desc}")

    # Data distribution
    print("\n" + "="*40)
    print("DATA DISTRIBUTION")
    print("="*40)

    data_dist = test_data['Data_Type'].value_counts()
    for data_type, count in data_dist.items():
        percentage = (count / len(test_data)) * 100
        print(f"   {data_type}: {count} records ({percentage:.1f}%)")

    # Save to Excel
    output_file = 'testdata.xlsx'
    test_data.to_excel(output_file, index=False)

    print("\n✅ Test data saved to:")
    print(f"   {output_file}")
    print(f"   Total records: {len(test_data)}")
    print(f"   File size: {Path(output_file).stat().st_size / 1024:.1f} KB")

    print("\n" + "="*60)
    print("TEST DATA GENERATION COMPLETED!")
    print("="*60)

if __name__ == "__main__":
    main()
