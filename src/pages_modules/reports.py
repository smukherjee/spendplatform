"""Reports page module for analytics and reporting."""
import streamlit as st
import pandas as pd
import plotly.express as px
import io
from src.utils.db_simple import get_db_connection
from src.config import config
from src.utils.debug import debug_logger, show_error_block, safe_execute


def render_page() -> None:
    """Render the reports and analytics page."""
    try:
        debug_logger.info("Rendering reports page")
        st.title("📈 Reports & Analytics")
        st.markdown("### Interactive filtering and report generation")
        
        # Load data
        debug_logger.debug("Loading transaction data for reports")
        success, df, error = safe_execute(
            load_transaction_data,
            error_title="Failed to Load Report Data",
            show_ui_error=True
        )
        
        if not success or df is None or df.empty:
            debug_logger.warning("No data available for reporting")
            st.warning("No data available for reporting.")
            return
        
        debug_logger.debug("Report data loaded", {"rows": len(df), "columns": len(df.columns)})
        
        # Filters
        debug_logger.debug("Rendering filters")
        safe_execute(render_filters, df, error_title="Failed to Render Filters", show_ui_error=True)
        
        # Apply filters from session state
        debug_logger.debug("Applying filters")
        success, filtered_df, error = safe_execute(
            apply_filters, df,
            error_title="Failed to Apply Filters",
            show_ui_error=True
        )
        
        if not success or filtered_df is None:
            return
        
        debug_logger.debug("Filters applied", {"filtered_rows": len(filtered_df)})
        
        # Reports
        col1, col2 = st.columns(2)
        
        with col1:
            safe_execute(render_spend_summary, filtered_df, error_title="Failed to Render Spend Summary", show_ui_error=True)
            safe_execute(render_supplier_analysis, filtered_df, error_title="Failed to Render Supplier Analysis", show_ui_error=True)
        
        with col2:
            safe_execute(render_category_analysis, filtered_df, error_title="Failed to Render Category Analysis", show_ui_error=True)
            safe_execute(render_regional_analysis, filtered_df, error_title="Failed to Render Regional Analysis", show_ui_error=True)
        
        # Export options
        safe_execute(render_export_options, filtered_df, error_title="Failed to Render Export Options", show_ui_error=True)
        
    except Exception as e:
        debug_logger.exception("Error rendering reports page", e)
        show_error_block("Reports Page Error", e)


def render_filters(df: pd.DataFrame) -> None:
    """Render filter controls."""
    st.subheader("🔍 Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Region filter
        regions = df['region'].unique().tolist() if 'region' in df.columns else []
        selected_regions = st.multiselect(
            "Region",
            options=regions,
            default=regions,
            key="region_filter"
        )
    
    with col2:
        # Supplier filter
        suppliers = df['supplier_name'].unique().tolist() if 'supplier_name' in df.columns else []
        selected_suppliers = st.multiselect(
            "Supplier",
            options=suppliers[:20],  # Limit to first 20 for performance
            key="supplier_filter"
        )
    
    with col3:
        # Category filter
        categories = df['category'].unique().tolist() if 'category' in df.columns else []
        selected_categories = st.multiselect(
            "Category",
            options=categories,
            default=categories,
            key="category_filter"
        )


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply filters based on session state."""
    filtered_df = df.copy()
    
    # Apply region filter
    if 'region_filter' in st.session_state and st.session_state.region_filter:
        if 'region' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['region'].isin(st.session_state.region_filter)]
    
    # Apply supplier filter
    if 'supplier_filter' in st.session_state and st.session_state.supplier_filter:
        if 'supplier_name' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['supplier_name'].isin(st.session_state.supplier_filter)]
    
    # Apply category filter
    if 'category_filter' in st.session_state and st.session_state.category_filter:
        if 'category' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['category'].isin(st.session_state.category_filter)]
    
    return filtered_df


def render_spend_summary(df: pd.DataFrame) -> None:
    """Render spend summary metrics."""
    st.subheader("💰 Spend Summary")
    
    col1, col2 = st.columns(2)
    with col1:
        total_spend = df['item_invoice_value'].sum() if 'item_invoice_value' in df.columns else 0
        st.metric("Total Spend", f"${total_spend:,.2f}")
    
    with col2:
        avg_transaction = df['item_invoice_value'].mean() if 'item_invoice_value' in df.columns else 0
        st.metric("Avg Transaction", f"${avg_transaction:,.2f}")


def render_supplier_analysis(df: pd.DataFrame) -> None:
    """Render supplier analysis chart."""
    if 'supplier_name' in df.columns and 'item_invoice_value' in df.columns:
        st.subheader("🏢 Top Suppliers")
        
        supplier_spend = df.groupby('supplier_name')['item_invoice_value'].sum().nlargest(10).reset_index()
        
        fig = px.bar(
            supplier_spend,
            x='item_invoice_value',
            y='supplier_name',
            orientation='h',
            title="Top 10 Suppliers by Spend"
        )
        st.plotly_chart(fig, width='stretch')


def render_category_analysis(df: pd.DataFrame) -> None:
    """Render category analysis chart."""
    if 'category' in df.columns and 'item_invoice_value' in df.columns:
        st.subheader("📂 Spend by Category")
        
        category_spend = df.groupby('category')['item_invoice_value'].sum().reset_index()
        
        fig = px.pie(
            category_spend,
            values='item_invoice_value',
            names='category',
            title="Spend Distribution by Category"
        )
        st.plotly_chart(fig, width='stretch')


def render_regional_analysis(df: pd.DataFrame) -> None:
    """Render regional analysis chart."""
    if 'region' in df.columns and 'item_invoice_value' in df.columns:
        st.subheader("🌍 Regional Analysis")
        
        regional_spend = df.groupby('region')['item_invoice_value'].sum().reset_index()
        
        fig = px.bar(
            regional_spend,
            x='region',
            y='item_invoice_value',
            title="Spend by Region"
        )
        st.plotly_chart(fig, width='stretch')


def render_export_options(df: pd.DataFrame) -> None:
    """Render data export options."""
    st.subheader("📁 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export to CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"spend_report_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📈 Export to Excel"):
            # Create Excel file in memory
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Spend Data', index=False)
            
            st.download_button(
                label="Download Excel",
                data=output.getvalue(),
                file_name=f"spend_report_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


def load_transaction_data() -> pd.DataFrame:
    """Load transaction data for reporting."""
    with get_db_connection() as conn:
        query = """
            SELECT 
                st.*,
                v.vendor_name,
                c.category_name
            FROM spend_transactions st
            LEFT JOIN vendors v ON st.supplier_name = v.vendor_name
            LEFT JOIN categories c ON st.category_id = c.category_id
            ORDER BY st.invoice_date DESC
        """
        return pd.read_sql_query(query, conn)
