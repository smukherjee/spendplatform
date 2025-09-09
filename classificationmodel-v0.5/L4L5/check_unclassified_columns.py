#!/usr/bin/env python3
"""
Check column names in unclassified transactions file
"""

import pandas as pd

def examine_unclassified_file():
    """Examine the unclassified transactions file structure"""
    file_path = '/Users/sujoymukherjee/code/spendplatform/context/unclassified transactions.xlsx'
    
    try:
        df = pd.read_excel(file_path)
        print(f"📊 File structure:")
        print(f"   • Shape: {df.shape}")
        print(f"   • Columns: {list(df.columns)}")
        print(f"\n📋 First 3 rows:")
        print(df.head(3))
        
        # Look for description-like columns
        desc_candidates = [col for col in df.columns if 'desc' in col.lower() or 'item' in col.lower()]
        print(f"\n🔍 Potential description columns: {desc_candidates}")
        
        return df.columns.tolist()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

if __name__ == "__main__":
    examine_unclassified_file()
