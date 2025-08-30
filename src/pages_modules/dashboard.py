"""Dashboard page module for displaying analytics and KPIs."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from src.utils.db_simple import get_db_connection
from src.config import config
from src.utils.debug import debug_logger, show_error_block, safe_execute


def render_page() -> None:
    """Render the dashboard page with analytics and KPIs."""
    try:
        debug_logger.info("Rendering dashboard page")
        st.title("📊 Dashboard")
        st.markdown("### Real-time spend analytics and key performance indicators")
        
        # Load data
        debug_logger.debug("Loading dashboard data")
        success, df, error = safe_execute(
            load_dashboard_data,
            error_title="Failed to Load Dashboard Data",
            show_ui_error=True
        )
        
        if not success or df is None:
            debug_logger.error("Dashboard data loading failed")
            return
        
        if df.empty:
            debug_logger.debug("No data available for dashboard")
            st.warning("No data available. Please upload spend data first.")
            return
        
        debug_logger.debug("Dashboard data loaded", {"rows": len(df)})
        
        # KPI Cards
        debug_logger.debug("Rendering KPI cards")
        success, _, error = safe_execute(
            render_kpi_cards, df,
            error_title="Failed to Render KPIs",
            show_ui_error=True
        )
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            debug_logger.debug("Rendering left column charts")
            safe_execute(render_spend_by_region, df, error_title="Failed to Render Region Chart", show_ui_error=True)
            safe_execute(render_top_suppliers, df, error_title="Failed to Render Top Suppliers", show_ui_error=True)
        
        with col2:
            debug_logger.debug("Rendering right column charts")
            safe_execute(render_spend_trend, df, error_title="Failed to Render Spend Trend", show_ui_error=True)
            safe_execute(render_category_breakdown, df, error_title="Failed to Render Category Breakdown", show_ui_error=True)
        
        # Recent transactions
        debug_logger.debug("Rendering recent transactions")
        safe_execute(render_recent_transactions, df, error_title="Failed to Render Recent Transactions", show_ui_error=True)
        
    except Exception as e:
        debug_logger.exception("Error rendering dashboard page", e)
        show_error_block("Dashboard Page Error", e)


def load_dashboard_data() -> pd.DataFrame:
    """Load transaction data for dashboard."""
    debug_logger.debug("Loading dashboard data from database")
    try:
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
                LIMIT 1000
            """
            df = pd.read_sql_query(query, conn)
            debug_logger.debug("Dashboard data loaded successfully", {"rows": len(df)})
            return df
    except Exception as e:
        debug_logger.exception("Error loading dashboard data", e)
        raise Exception(f"Database error loading dashboard data: {str(e)}")


def render_kpi_cards(df: pd.DataFrame) -> None:
    """Render KPI metric cards."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_spend = df['item_invoice_value'].sum()
        st.metric("Total Spend", f"${total_spend:,.2f}")
    
    with col2:
        transaction_count = len(df)
        st.metric("Transactions", f"{transaction_count:,}")
    
    with col3:
        supplier_count = df['supplier_name'].nunique()
        st.metric("Suppliers", f"{supplier_count:,}")
    
    with col4:
        # Error count (placeholder - would need actual error data)
        st.metric("Active Errors", "0")


def render_spend_by_region(df: pd.DataFrame) -> None:
    """Render spend by region pie chart."""
    if 'region' in df.columns:
        region_spend = df.groupby('region')['item_invoice_value'].sum().reset_index()
        fig = px.pie(
            region_spend, 
            values='item_invoice_value', 
            names='region',
            title="Spend by Region"
        )
        st.plotly_chart(fig, use_container_width=True)


def render_top_suppliers(df: pd.DataFrame) -> None:
    """Render top suppliers bar chart."""
    top_suppliers = df.groupby('supplier_name')['item_invoice_value'].sum().nlargest(10).reset_index()
    fig = px.bar(
        top_suppliers,
        x='item_invoice_value',
        y='supplier_name',
        orientation='h',
        title="Top 10 Suppliers"
    )
    st.plotly_chart(fig, use_container_width=True)


def render_spend_trend(df: pd.DataFrame) -> None:
    """Render spend trend over time."""
    if 'invoice_date' in df.columns:
        df['invoice_date'] = pd.to_datetime(df['invoice_date'])
        monthly_spend = df.groupby(df['invoice_date'].dt.to_period('M'))['item_invoice_value'].sum().reset_index()
        monthly_spend['invoice_date'] = monthly_spend['invoice_date'].astype(str)
        
        fig = px.line(
            monthly_spend,
            x='invoice_date',
            y='item_invoice_value',
            title="Monthly Spend Trend"
        )
        st.plotly_chart(fig, use_container_width=True)


def render_category_breakdown(df: pd.DataFrame) -> None:
    """Render category breakdown chart."""
    if 'category_name' in df.columns:
        category_spend = df.groupby('category_name')['item_invoice_value'].sum().reset_index()
        fig = px.bar(
            category_spend,
            x='category_name',
            y='item_invoice_value',
            title="Spend by Category"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)


def render_recent_transactions(df: pd.DataFrame) -> None:
    """Render recent transactions table."""
    st.subheader("Recent Transactions")
    
    # Show only recent transactions
    recent_df = df.head(config.ui.recent_items_limit)
    
    # Select key columns for display
    display_columns = [
        'invoice_date', 'supplier_name', 'item_invoice_value', 
        'currency_type', 'region', 'category_name'
    ]
    
    # Filter to available columns
    available_columns = [col for col in display_columns if col in recent_df.columns]
    
    if available_columns:
        st.dataframe(
            recent_df[available_columns],
            use_container_width=True,
            hide_index=True
        )
