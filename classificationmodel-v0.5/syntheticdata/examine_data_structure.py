#!/usr/bin/env python3
"""
Examine existing data structures to understand patterns for synthetic data generation
"""

import pandas as pd
import numpy as np

def examine_existing_data():
    """Examine the existing training data structure"""
    print("📊 EXAMINING EXISTING DATA STRUCTURE")
    print("=" * 60)
    
    # Load the training data
    train_data_path = '/Users/sujoymukherjee/code/spendplatform/classificationmodel-v0.5/preprocessed_data/top_10_l2_categories_data.xlsx'
    
    try:
        df = pd.read_excel(train_data_path)
        print(f"✅ Loaded training data: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        
        print(f"\n📊 Data Overview:")
        print(df.head(10))
        
        print(f"\n🏷️ Category Distribution:")
        category_counts = df['Category L2'].value_counts()
        for category, count in category_counts.items():
            print(f"   {category}: {count} records")
        
        print(f"\n📝 Sample Item Descriptions by Category:")
        for category in category_counts.index[:5]:  # Top 5 categories
            print(f"\n{category}:")
            samples = df[df['Category L2'] == category]['Item_Descripton'].head(5).tolist()
            for i, sample in enumerate(samples, 1):
                print(f"   {i}. {sample}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return None

def examine_categorization_file():
    """Examine the categorization file for hierarchy"""
    print(f"\n📁 EXAMINING CATEGORIZATION FILE")
    print("=" * 60)
    
    cat_file_path = '/Users/sujoymukherjee/code/spendplatform/context/archive/Categorization File.xlsx'
    
    try:
        # Try different sheet names and configurations
        df = pd.read_excel(cat_file_path, sheet_name=0)
        print(f"✅ Loaded categorization file: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        
        print(f"\n📊 First 10 rows:")
        print(df.head(10))
        
        # Look for category hierarchy columns
        category_cols = [col for col in df.columns if 'category' in col.lower() or 'l2' in col.lower() or 'l3' in col.lower() or 'l4' in col.lower()]
        print(f"\n🏷️ Category-related columns: {category_cols}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading categorization file: {e}")
        return None

if __name__ == "__main__":
    train_df = examine_existing_data()
    cat_df = examine_categorization_file()
