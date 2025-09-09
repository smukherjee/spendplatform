#!/usr/bin/env python3
"""
Examine Excel file structure for L4/L5 analysis
"""

import pandas as pd
import numpy as np

def examine_excel_structure():
    """Examine the structure of the categorization Excel file"""
    print("🔍 EXAMINING EXCEL FILE STRUCTURE")
    print("=" * 60)
    
    file_path = '/Users/sujoymukherjee/code/spendplatform/context/Copy of Catergorization Working Sheet_Labelled Data for Training.xlsx'
    
    try:
        # First, check what sheets are available
        xl_file = pd.ExcelFile(file_path)
        print(f"📄 Available sheets: {xl_file.sheet_names}")
        
        # Load the first sheet with different options
        for sheet_name in xl_file.sheet_names[:2]:  # Check first 2 sheets
            print(f"\n📊 Analyzing sheet: '{sheet_name}'")
            
            # Try different header rows
            for header_row in [0, 1, 2]:
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row)
                    print(f"\n   Header row {header_row}:")
                    print(f"   • Shape: {df.shape}")
                    print(f"   • Columns: {list(df.columns)}")
                    
                    # Look for L4/L5 patterns
                    l4_cols = [col for col in df.columns if 'l4' in str(col).lower()]
                    l5_cols = [col for col in df.columns if 'l5' in str(col).lower()]
                    
                    if l4_cols or l5_cols:
                        print(f"   ✅ Found L4 columns: {l4_cols}")
                        print(f"   ✅ Found L5 columns: {l5_cols}")
                        
                        # Show some sample data
                        print(f"   📋 Sample data:")
                        sample_cols = ['Item_Descripton'] + l4_cols + l5_cols
                        available_cols = [col for col in sample_cols if col in df.columns]
                        if available_cols:
                            print(df[available_cols].head(3))
                        
                        return df, sheet_name, header_row, l4_cols, l5_cols
                    
                except Exception as e:
                    print(f"   ❌ Error with header {header_row}: {e}")
        
        # If no L4/L5 found, show raw structure
        print(f"\n🔍 Raw file structure (first sheet, no header):")
        df_raw = pd.read_excel(file_path, header=None)
        print(f"Shape: {df_raw.shape}")
        print(f"First 10 rows:")
        print(df_raw.head(10))
        
        return None, None, None, [], []
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None, None, None, [], []

def find_item_description_column(df):
    """Find the item description column"""
    desc_candidates = ['Item_Descripton', 'Description', 'Item Description', 'Item_Description']
    
    for col in df.columns:
        if any(candidate.lower() in str(col).lower() for candidate in desc_candidates):
            return col
    
    # Look in the data itself for description-like content
    for col in df.columns:
        sample_values = df[col].dropna().head(20)
        if len(sample_values) > 0:
            # Check if values look like item descriptions
            avg_length = sample_values.astype(str).str.len().mean()
            if avg_length > 10:  # Descriptions are usually longer
                print(f"   🔍 Potential description column: {col} (avg length: {avg_length:.1f})")
                return col
    
    return None

def main():
    print("🔍 EXCEL STRUCTURE EXAMINATION")
    print("=" * 70)
    
    result = examine_excel_structure()
    
    if result[0] is not None:
        df, sheet_name, header_row, l4_cols, l5_cols = result
        
        # Find item description column
        desc_col = find_item_description_column(df)
        print(f"\n🎯 Best configuration found:")
        print(f"   • Sheet: {sheet_name}")
        print(f"   • Header row: {header_row}")
        print(f"   • Description column: {desc_col}")
        print(f"   • L4 columns: {l4_cols}")
        print(f"   • L5 columns: {l5_cols}")
        
        # Show data quality
        if desc_col and l4_cols:
            l4_col = l4_cols[0]
            valid_data = df[df[desc_col].notna() & df[l4_col].notna()]
            print(f"\n📊 Data quality:")
            print(f"   • Total records: {len(df)}")
            print(f"   • Records with description: {df[desc_col].notna().sum()}")
            print(f"   • Records with L4: {df[l4_col].notna().sum()}")
            print(f"   • Usable records: {len(valid_data)}")
            
            if len(valid_data) > 0:
                print(f"\n📋 Sample usable data:")
                print(valid_data[[desc_col, l4_col]].head(5))
    else:
        print(f"\n❌ Could not find proper L4/L5 structure in the file")

if __name__ == "__main__":
    main()
