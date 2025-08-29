"""
Utility functions for the Spend Platform application
"""

import pandas as pd
import sqlite3
import hashlib
from datetime import datetime
import re
from config import Config

def hash_password(password):
    """Hash a password for storing in database"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    """Verify a password against its hash"""
    return hash_password(password) == hashed

def get_db_connection():
    """Get database connection"""
    return sqlite3.connect(Config.DATABASE_PATH)

def normalize_supplier_name(supplier_name):
    """Normalize supplier name for better matching"""
    if pd.isna(supplier_name):
        return ""
    
    # Convert to uppercase and remove extra spaces
    normalized = str(supplier_name).strip().upper()
    
    # Remove common company suffixes
    suffixes = ['LTD', 'LIMITED', 'INC', 'CORP', 'CORPORATION', 'LLC', 'PVT', 'PRIVATE']
    for suffix in suffixes:
        normalized = re.sub(f'\\b{suffix}\\b', '', normalized)
    
    # Remove special characters except alphanumeric and spaces
    normalized = re.sub(r'[^A-Z0-9\s]', '', normalized)
    
    # Remove extra spaces
    normalized = ' '.join(normalized.split())
    
    return normalized

def detect_duplicates(df, columns=['supplier_name', 'item_invoice_value', 'invoice_date']):
    """Detect potential duplicate transactions"""
    duplicates = df.duplicated(subset=columns, keep=False)
    return df[duplicates]

def validate_currency(currency):
    """Validate currency code"""
    valid_currencies = ['USD', 'EUR', 'GBP', 'INR', 'JPY', 'CAD', 'AUD', 'CHF', 'CNY']
    return str(currency).upper() in valid_currencies if pd.notna(currency) else False

def format_currency(amount, currency='USD'):
    """Format amount as currency"""
    if pd.isna(amount):
        return "N/A"
    
    try:
        amount = float(amount)
        if currency == 'USD':
            return f"${amount:,.2f}"
        else:
            return f"{currency} {amount:,.2f}"
    except (ValueError, TypeError):
        return "Invalid Amount"

def get_date_range_filter(df, date_column, days=30):
    """Get recent data within specified days"""
    if date_column not in df.columns:
        return df
    
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    cutoff_date = datetime.now() - pd.Timedelta(days=days)
    
    return df[df[date_column] >= cutoff_date]

def calculate_spend_metrics(df):
    """Calculate key spend metrics"""
    if df.empty or 'item_invoice_value' not in df.columns:
        return {}
    
    return {
        'total_spend': df['item_invoice_value'].sum(),
        'average_transaction': df['item_invoice_value'].mean(),
        'median_transaction': df['item_invoice_value'].median(),
        'min_transaction': df['item_invoice_value'].min(),
        'max_transaction': df['item_invoice_value'].max(),
        'transaction_count': len(df),
        'unique_suppliers': df['supplier_name'].nunique() if 'supplier_name' in df.columns else 0
    }

def export_to_excel(data_dict, filename):
    """Export multiple dataframes to Excel with different sheets"""
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        for sheet_name, df in data_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Get workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]
            
            # Add formatting
            header_format = workbook.add_format({ # type: ignore
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BC',
                'border': 1
            })
            
            # Apply header formatting
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # Auto-adjust column widths
            for i, col in enumerate(df.columns):
                # Get the maximum length of the column
                max_length = max(
                    df[col].astype(str).map(len).max(),  # len of largest item
                    len(str(col))  # len of column name/header
                ) + 2  # adding a little extra space
                worksheet.set_column(i, i, min(max_length, 50))  # cap at 50

def clean_data(df):
    """Clean and standardize data"""
    df_clean = df.copy()
    
    # Strip whitespace from string columns
    string_columns = df_clean.select_dtypes(include=['object']).columns
    df_clean[string_columns] = df_clean[string_columns].apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
    
    # Normalize supplier names
    if 'supplier_name' in df_clean.columns:
        df_clean['normalized_supplier'] = df_clean['supplier_name'].apply(normalize_supplier_name)
    
    # Convert date columns
    date_columns = ['invoice_date', 'po_creation_date']
    for col in date_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
    
    # Convert numeric columns
    numeric_columns = ['item_invoice_value', 'invoice_amount', 'po_price', 'unit_price']
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    return df_clean

def generate_data_quality_report(df):
    """Generate data quality report"""
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': {},
        'data_types': {},
        'duplicate_rows': len(df[df.duplicated()]),
        'negative_values': {},
        'invalid_dates': {}
    }
    
    # Check missing values
    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            report['missing_values'][col] = {
                'count': int(missing_count),
                'percentage': round((missing_count / len(df)) * 100, 2)
            }
    
    # Check data types
    for col in df.columns:
        report['data_types'][col] = str(df[col].dtype)
    
    # Check for negative values in numeric columns
    numeric_columns = df.select_dtypes(include=['number']).columns
    for col in numeric_columns:
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            report['negative_values'][col] = int(negative_count)
    
    # Check for invalid dates
    date_columns = ['invoice_date', 'po_creation_date']
    for col in date_columns:
        if col in df.columns:
            df_temp = df.copy()
            df_temp[col] = pd.to_datetime(df_temp[col], errors='coerce')
            invalid_count = df_temp[col].isna().sum() - df[col].isna().sum()
            if invalid_count > 0:
                report['invalid_dates'][col] = int(invalid_count)
    
    return report

def get_user_permissions(role):
    """Get user permissions based on role"""
    return Config.USER_ROLES.get(role, Config.USER_ROLES['Data Analyst'])

def log_user_action(user_id, action_type, module_affected, details=""):
    """Log user actions for audit trail"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO user_actions_audit (user_id, action_type, module_affected, details)
            VALUES (?, ?, ?, ?)
        ''', (user_id, action_type, module_affected, details))
        conn.commit()
    except Exception as e:
        print(f"Error logging user action: {e}")
    finally:
        conn.close()

def get_spend_summary_by_period(df, period='M'):
    """Get spend summary grouped by time period"""
    if 'invoice_date' not in df.columns or 'item_invoice_value' not in df.columns:
        return pd.DataFrame()
    
    df['invoice_date'] = pd.to_datetime(df['invoice_date'], errors='coerce')
    df_valid = df.dropna(subset=['invoice_date', 'item_invoice_value'])
    
    if df_valid.empty:
        return pd.DataFrame()
    
    # Group by period
    period_mapping = {
        'D': 'Daily',
        'W': 'Weekly', 
        'M': 'Monthly',
        'Q': 'Quarterly',
        'Y': 'Yearly'
    }
    
    grouped = df_valid.groupby(pd.Grouper(key='invoice_date', freq=period)).agg({
        'item_invoice_value': ['sum', 'mean', 'count'],
        'supplier_name': 'nunique'
    }).round(2)
    
    # Flatten column names
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns]
    grouped = grouped.reset_index()
    
    return grouped
