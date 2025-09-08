#!/usr/bin/env python3
"""
Inspect Excel file structure
"""

import pandas as pd

def inspect_excel_files():
    """Inspect the structure of available Excel files"""
    files_to_check = [
        '/Users/sujoymukherjee/code/spendplatform/context/Copy of Catergorization Working Sheet_Labelled Data for Training.xlsx',
        '/Users/sujoymukherjee/code/spendplatform/context/Original Catergorization Working Sheet_Labelled Data for Training.xlsx',
        '/Users/sujoymukherjee/code/spendplatform/context/predicted_original_training_data.xlsx'
    ]
    
    for file_path in files_to_check:
        try:
            print(f"\n{'='*60}")
            print(f"📁 File: {file_path.split('/')[-1]}")
            print('='*60)
            
            # Read with different options
            df = pd.read_excel(file_path, header=0)
            print(f"✅ Shape: {df.shape}")
            print(f"✅ Columns: {list(df.columns)}")
            
            # Show first few rows
            print(f"\n📊 First 3 rows:")
            print(df.head(3))
            
            # Check for data types
            print(f"\n🔍 Data types:")
            print(df.dtypes)
            
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")

if __name__ == "__main__":
    inspect_excel_files()
