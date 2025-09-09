#!/usr/bin/env python3
"""
Examine categorization file more thoroughly to find the hierarchy
"""

import pandas as pd

def examine_categorization_file_detailed():
    """Examine the categorization file in detail"""
    print("📁 DETAILED CATEGORIZATION FILE ANALYSIS")
    print("=" * 60)
    
    cat_file_path = '/Users/sujoymukherjee/code/spendplatform/context/archive/Categorization File.xlsx'
    
    try:
        # Get all sheet names
        xl_file = pd.ExcelFile(cat_file_path)
        print(f"📋 Available sheets: {xl_file.sheet_names}")
        
        # Examine each sheet
        for sheet_name in xl_file.sheet_names:
            print(f"\n📊 Sheet: {sheet_name}")
            print("-" * 40)
            
            df = pd.read_excel(cat_file_path, sheet_name=sheet_name)
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Look for category hierarchy
            category_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['category', 'l1', 'l2', 'l3', 'l4', 'l5'])]
            if category_cols:
                print(f"Category columns found: {category_cols}")
                
                # Show unique values in category columns
                for col in category_cols:
                    unique_vals = df[col].dropna().unique()
                    print(f"  {col}: {len(unique_vals)} unique values")
                    if len(unique_vals) <= 20:
                        print(f"    Values: {list(unique_vals)}")
                    else:
                        print(f"    Sample values: {list(unique_vals[:10])}...")
            
            print(f"\nFirst 5 rows:")
            print(df.head())
            
    except Exception as e:
        print(f"❌ Error examining categorization file: {e}")

def examine_attachment():
    """Check if there's additional structure from attachment"""
    print(f"\n📎 CHECKING ATTACHMENT CONTEXT")
    print("=" * 60)
    
    # The attachment shows "Item_Description" which suggests the structure
    print("Attachment shows: Item_Description")
    print("This suggests the data should have item descriptions with categories")

if __name__ == "__main__":
    examine_categorization_file_detailed()
    examine_attachment()
