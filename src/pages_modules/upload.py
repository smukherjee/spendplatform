"""Upload page module for data upload and validation."""
import streamlit as st
import pandas as pd
import io
from typing import Optional
from src.utils.db_simple import get_db_connection
from src.config import config
from src.utils.debug import debug_logger, show_error_block, safe_execute


def render_page() -> None:
    """Render the data upload page."""
    try:
        debug_logger.info("Rendering upload page")
        st.title("📤 Data Upload")
        st.markdown("### Upload and validate spend data files")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an Excel file",
            type=['xlsx', 'xls'],
            help="Upload Excel files with spend transaction data"
        )
        
        if uploaded_file is not None:
            debug_logger.debug("File uploaded", {"filename": uploaded_file.name, "size": uploaded_file.size})
            
            success, df, error = safe_execute(
                pd.read_excel, uploaded_file,
                error_title="Failed to Read Excel File",
                show_ui_error=True
            )
            
            if not success or df is None:
                debug_logger.error("Failed to read uploaded file")
                return
                
            debug_logger.info("File read successfully", {"rows": len(df), "columns": len(df.columns)})
            st.success(f"✅ File uploaded successfully! {len(df)} rows loaded.")
            
            # Preview data
            with st.expander("📋 Data Preview"):
                st.dataframe(df.head(10))
            
            # Validation
            debug_logger.debug("Starting data validation")
            success, validation_results, error = safe_execute(
                validate_data, df,
                error_title="Failed to Validate Data",
                show_ui_error=True
            )
            
            if success and validation_results:
                render_validation_results(validation_results)
                
                # Process data if valid
                if validation_results['is_valid']:
                    if st.button("🚀 Process Data"):
                        debug_logger.info("Processing data", {"rows": len(df)})
                        safe_execute(
                            process_data, df,
                            error_title="Failed to Process Data",
                            show_ui_error=True
                        )
                        
    except Exception as e:
        debug_logger.exception("Error rendering upload page", e)
        show_error_block("Upload Page Error", e)


def validate_data(df: pd.DataFrame) -> dict:
    """Validate uploaded data and return validation results."""
    debug_logger.debug("Validating uploaded data", {"rows": len(df), "columns": len(df.columns)})
    
    results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'summary': {}
    }
    
    # Required columns check
    required_columns = [
        'supplier_name', 'item_invoice_value', 'invoice_date'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        results['errors'].append(f"Missing required columns: {', '.join(missing_columns)}")
        results['is_valid'] = False
    
    # Data quality checks
    if 'item_invoice_value' in df.columns:
        negative_values = df[df['item_invoice_value'] < 0]
        if not negative_values.empty:
            results['warnings'].append(f"Found {len(negative_values)} negative amounts")
    
    if 'supplier_name' in df.columns:
        missing_suppliers = df[df['supplier_name'].isna()]
        if not missing_suppliers.empty:
            results['warnings'].append(f"Found {len(missing_suppliers)} records with missing supplier names")
    
    # Summary statistics
    results['summary'] = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum()
    }
    
    return results


def render_validation_results(results: dict) -> None:
    """Render validation results."""
    st.subheader("📋 Validation Results")
    
    # Summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", results['summary']['total_rows'])
    with col2:
        st.metric("Total Columns", results['summary']['total_columns'])
    with col3:
        st.metric("Missing Values", results['summary']['missing_values'])
    
    # Errors
    if results['errors']:
        st.error("❌ Validation Errors:")
        for error in results['errors']:
            st.error(f"• {error}")
    
    # Warnings
    if results['warnings']:
        st.warning("⚠️ Data Quality Warnings:")
        for warning in results['warnings']:
            st.warning(f"• {warning}")
    
    if results['is_valid'] and not results['warnings']:
        st.success("✅ All validations passed!")


def process_data(df: pd.DataFrame) -> None:
    """Process and save validated data to database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Insert data into spend_transactions table
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT INTO spend_transactions (
                    bu_code, bu_name, region, po_no, po_item_no,
                    invoice_date, supplier_no, supplier_name,
                    item_invoice_value, currency_type, material_code,
                    material_item_name, tower_practice, category, subcategory
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row.get('BU_CODE'),
                row.get('BU _NAME'),
                row.get('Region'),
                row.get('PO_NO'),
                row.get('PO_ITEM_NO'),
                row.get('Invoice Date'),
                row.get('SUPPLIER_NO'),
                row.get('SUPPLIER_NAME', row.get('supplier_name')),
                row.get('Item Invoice Value', row.get('item_invoice_value')),
                row.get('CURRENCY_TYPE'),
                row.get('Material Code'),
                row.get('Material Item Name'),
                row.get('Tower (Practice)'),
                row.get('Category'),
                row.get('Subcategory')
            ))
        
        conn.commit()
