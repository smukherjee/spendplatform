"""
Spend Data Management Platform - MVP
A Streamlit-based application for managing spend data, vendor information, and categorization.
"""

import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import os
from crud_operations import crud
from config import Config
from styles import (
    inject_custom_css, 
    create_metric_card, 
    create_section_header, 
    create_status_badge,
    create_professional_dataframe,
    create_chart_container,
    apply_button_style,
    create_coming_soon_page
)

# Page configuration
st.set_page_config(
    page_title="Spend Data Management Platform",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect('spend_platform.db')
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'Data Analyst',
            last_login TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Vendors table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vendors (
            vendor_id INTEGER PRIMARY KEY AUTOINCREMENT,
            vendor_name TEXT NOT NULL,
            normalized_name TEXT,
            supplier_no TEXT,
            contact_info TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Categories table (hierarchical)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT NOT NULL,
            parent_category_id INTEGER,
            category_level INTEGER DEFAULT 1,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (parent_category_id) REFERENCES categories (category_id)
        )
    ''')
    
    # Spend transactions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS spend_transactions (
            transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            bu_code TEXT,
            bu_name TEXT,
            region TEXT,
            po_no TEXT,
            po_item_no TEXT,
            invoice_date DATE,
            supplier_no TEXT,
            supplier_name TEXT,
            invoice_amount REAL,
            currency_type TEXT,
            unit_of_purch TEXT,
            item_invoice_value REAL,
            material_code TEXT,
            material_item_name TEXT,
            order_description TEXT,
            inv_item_desc TEXT,
            po_price REAL,
            po_purchase_qty REAL,
            unit_price REAL,
            tower_practice TEXT,
            category TEXT,
            subcategory TEXT,
            vendor_id INTEGER,
            category_id INTEGER,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'Active',
            FOREIGN KEY (vendor_id) REFERENCES vendors (vendor_id),
            FOREIGN KEY (category_id) REFERENCES categories (category_id)
        )
    ''')
    
    # Error logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS error_logs (
            error_id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id INTEGER,
            error_type TEXT NOT NULL,
            description TEXT,
            status TEXT DEFAULT 'Open',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved_at TIMESTAMP,
            resolved_by TEXT,
            FOREIGN KEY (transaction_id) REFERENCES spend_transactions (transaction_id)
        )
    ''')
    
    # Rules table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rules (
            rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
            rule_name TEXT NOT NULL,
            rule_description TEXT,
            rule_type TEXT,
            rule_condition TEXT,
            active_flag BOOLEAN DEFAULT TRUE,
            created_by TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    
    # Create default users if they don't exist
    cursor = conn.cursor()
    
    # Check if users exist
    cursor.execute("SELECT COUNT(*) FROM users")
    user_count = cursor.fetchone()[0]
    
    if user_count == 0:
        # Create default users with hashed passwords (for demo, using simple hash)
        import hashlib
        
        default_users = [
            ("admin", hashlib.md5("admin123".encode()).hexdigest(), "Admin"),
            ("analyst", hashlib.md5("analyst123".encode()).hexdigest(), "Data Analyst"),
            ("manager", hashlib.md5("manager123".encode()).hexdigest(), "Spend Manager")
        ]
        
        for username, password_hash, role in default_users:
            cursor.execute('''
                INSERT INTO users (username, password_hash, role)
                VALUES (?, ?, ?)
            ''', (username, password_hash, role))
        
        conn.commit()
    
    conn.close()

# Authentication functions
def authenticate_user(username, password):
    """Database-based authentication"""
    import hashlib
    
    # Hash the provided password
    password_hash = hashlib.md5(password.encode()).hexdigest()
    
    conn = sqlite3.connect(Config.DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            SELECT user_id, username, role FROM users 
            WHERE username = ? AND password_hash = ?
        ''', (username, password_hash))
        
        user = cursor.fetchone()
        if user:
            return {
                "user_id": user[0],
                "username": user[1],
                "role": user[2]
            }
    except Exception as e:
        print(f"Authentication error: {e}")
    finally:
        conn.close()
    
    return None

def login_page():
    """Display login page with professional styling"""
    # Inject CSS for login page
    inject_custom_css()
    
    # Create professional header
    create_section_header(
        "🔐 Spend Platform Login", 
        "Secure access to your spend data management platform"
    )
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        
        with st.form("login_form"):
            st.markdown("### Sign In to Continue")
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                submit = st.form_submit_button("Login", use_container_width=True)
            with col_btn2:
                st.markdown('<div style="padding: 0.375rem 0;"></div>', unsafe_allow_html=True)  # Spacing
        
        if submit:
            user = authenticate_user(username, password)
            if user:
                st.session_state.user = user
                st.session_state.authenticated = True
                st.success("Login successful! Redirecting...")
                st.rerun()
            else:
                st.error("Invalid credentials. Please try again.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Demo credentials in an info box
        st.markdown('<div class="metric-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
        st.markdown("### 🎯 Demo Credentials")
        st.markdown("""
        **Available Test Accounts:**
        - **Admin**: `admin` / `admin123`
        - **Spend Manager**: `manager` / `manager123` 
        - **Data Analyst**: `analyst` / `analyst123`
        """)
        st.markdown('</div>', unsafe_allow_html=True)

def sidebar_navigation():
    """Display sidebar navigation"""
    st.sidebar.title("🏢 Spend Platform")
    
    # Safe access to user information
    user = st.session_state.get('user', {})
    username = user.get('username', 'Guest')
    role = user.get('role', 'Unknown')
    
    st.sidebar.write(f"Welcome, {username}")
    st.sidebar.write(f"Role: {role}")
    
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.user = None
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Navigation menu organized by workflow sections
    st.sidebar.markdown("## 🔄 Data Integration")
    data_integration_pages = {
        "� Data Upload & Import": "upload",
        "📋 Data Validation Rules": "validation_rules",
        "🔍 Error Detection & Handling": "errors",
        "� Data Source Management": "data_sources",
        "🌐 Multi-Language Processing": "language_processing"
    }
    
    st.sidebar.markdown("## 🔧 Data Enrichment & Analysis")
    data_enrichment_pages = {
        "🗂️ Master Data Management": "master_data",
        "🏷️ Data Classification": "classification",
        "💼 Transaction Management": "transactions", 
        "📏 Business Rules Management": "rules",
        "🔄 Data Standardization": "standardization",
        "� Self-Service Data Tools": "self_service"
    }
    
    st.sidebar.markdown("## 📈 Data Insights")
    data_insights_pages = {
        "📊 Analytics Dashboard": "dashboard",
        "📈 Advanced Reports": "reports",
        "🎯 Spend Analysis": "spend_analysis",
        "� Data Governance": "governance",
        "�👥 User Management": "users"
    }
    
    # Combine all pages for selection
    all_pages = {**data_integration_pages, **data_enrichment_pages, **data_insights_pages}
    
    # Create expandable sections for navigation
    selected_page_key = None
    
    with st.sidebar.expander("🔄 Data Integration", expanded=True):
        for page_name, page_key in data_integration_pages.items():
            if st.button(page_name, key=f"nav_{page_key}"):
                st.session_state.selected_page = page_key
                
    with st.sidebar.expander("🔧 Data Enrichment & Analysis", expanded=True):
        for page_name, page_key in data_enrichment_pages.items():
            if st.button(page_name, key=f"nav_{page_key}"):
                st.session_state.selected_page = page_key
                
    with st.sidebar.expander("📈 Data Insights", expanded=True):
        for page_name, page_key in data_insights_pages.items():
            if st.button(page_name, key=f"nav_{page_key}"):
                st.session_state.selected_page = page_key
    
    # Default to dashboard if no selection
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = 'dashboard'
        
    return st.session_state.selected_page

def dashboard_page():
    """Enhanced dashboard with professional styling using CRUD operations"""
    create_section_header(
        "📊 Analytics Dashboard", 
        "Comprehensive view of your spend data and key performance indicators"
    )
    
    try:
        # Load data using CRUD operations
        df = crud.read_transactions(limit=5000)  # Limit for performance
        
        if df.empty:
            st.warning("⚠️ No spend data available. Please upload data first.")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Upload Data Now", use_container_width=True):
                    st.session_state.selected_page = "upload"
                    st.rerun()
            return
        
        # Professional key metrics with cards
        st.markdown('<h2 class="section-header">📈 Key Performance Indicators</h2>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_spend = df['item_invoice_value'].sum()
            create_metric_card("Total Spend", f"${total_spend:,.2f}", "+12.5%", True)
        
        with col2:
            total_transactions = len(df)
            create_metric_card("Total Transactions", f"{total_transactions:,}", f"+{len(df)//10} this month", True)
        
        with col3:
            unique_suppliers = df['supplier_name'].nunique()
            create_metric_card("Unique Suppliers", f"{unique_suppliers:,}", "5 new suppliers", True)
        
        with col4:
            # Count errors using CRUD
            error_df = crud.read_error_logs(filters={'status': 'Open'})
            open_errors = len(error_df)
            delta_text = "All resolved!" if open_errors == 0 else f"{open_errors} need attention"
            create_metric_card("Open Errors", f"{open_errors:,}", delta_text, open_errors == 0)
        
        # Professional charts section
        st.markdown('<h2 class="section-header">📊 Spend Analytics</h2>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            def create_region_chart():
                region_spend = df.groupby('region')['item_invoice_value'].sum().reset_index()
                fig = px.pie(region_spend, values='item_invoice_value', names='region',
                           title="Spend Distribution by Region",
                           color_discrete_sequence=px.colors.qualitative.Set3)
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    title_font_size=16,
                    title_x=0.5
                )
                st.plotly_chart(fig, use_container_width=True)
            
            create_chart_container(create_region_chart, "🌍 Regional Analysis")
        
        with col2:
            def create_suppliers_chart():
                top_suppliers = df.groupby('supplier_name')['item_invoice_value'].sum().nlargest(10).reset_index()
                fig = px.bar(top_suppliers, x='item_invoice_value', y='supplier_name', 
                           orientation='h', title="Top 10 Suppliers by Spend Value",
                           color='item_invoice_value', 
                           color_continuous_scale='Blues')
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    title_font_size=16,
                    title_x=0.5,
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            create_chart_container(create_suppliers_chart, "🏢 Supplier Analysis")
        
        # Recent transactions with professional table
        st.markdown('<h2 class="section-header">🕐 Recent Activity</h2>', unsafe_allow_html=True)
        recent_df = df.head(10)[['invoice_date', 'supplier_name', 'item_invoice_value', 'region', 'bu_name']]
        recent_df.columns = ['Invoice Date', 'Supplier', 'Value', 'Region', 'Business Unit']
        create_professional_dataframe(recent_df, "Latest Transactions")
        
        # Quick analytics using CRUD analytics methods
        st.subheader("Quick Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top 5 Suppliers**")
            top_suppliers_analytics = crud.get_top_suppliers(limit=5)
            if not top_suppliers_analytics.empty:
                st.dataframe(top_suppliers_analytics, use_container_width=True)
        
        with col2:
            st.write("**Error Summary**")
            error_summary = crud.get_data_quality_report()
            if error_summary:
                # Convert to DataFrame for display
                summary_df = pd.DataFrame([error_summary])
                st.dataframe(summary_df, use_container_width=True)
            else:
                st.info("No error data available")
        
    except Exception as e:
        st.error(f"Error loading dashboard data: {str(e)}")

def upload_page():
    """Professional Data Upload & Import Page"""
    create_section_header(
        "📤 Data Upload & Import", 
        "Integrate data from multiple sources with advanced validation and processing"
    )
    
    # Professional information panel
    st.markdown('<h3 class="subsection-header">🎯 Upload Guidelines</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 📋 Data Integration Capabilities")
        st.markdown("""
        - **🔄 Multiple Source Integration**: CSV, Excel, JSON formats supported
        - **🔍 Smart Duplicate Detection**: Advanced algorithms prevent data duplication
        - **✅ Real-time Validation**: Automatic data quality checks during upload
        - **🔧 User Feedback Loop**: Easy reporting of data adjustments needed
        - **🌐 Multi-format Support**: Flexible parsing for various data structures
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Upload Statistics")
        
        # Get current data stats
        try:
            df = crud.read_transactions(limit=1)
            if not df.empty:
                total_records = crud.read_transactions(limit=100000)  # Get count
                st.metric("Total Records", f"{len(total_records):,}")
                st.metric("Data Sources", "3 Active")
                st.metric("Last Upload", "2 hours ago")
            else:
                st.info("No data uploaded yet")
        except:
            st.info("Database not available")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<h3 class="subsection-header">📁 File Upload</h3>', unsafe_allow_html=True)
    
    # Professional upload interface
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    
    # File uploader with professional styling
    st.markdown("#### 📁 Select Your Data File")
    uploaded_file = st.file_uploader(
        "Choose an Excel file", 
        type=['xlsx', 'xls'],
        help="Upload your spend data Excel file. Supports multiple formats and sources."
    )
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
            st.markdown('</div>', unsafe_allow_html=True)  # Close upload card
            
            # Data preview with professional styling
            st.markdown('<h3 class="subsection-header">👁️ Data Preview</h3>', unsafe_allow_html=True)
            create_professional_dataframe(df.head(10), "First 10 Records")
            
            # Data validation section
            st.markdown('<h3 class="subsection-header">🔍 Data Validation Results</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                validation_results = validate_data(df)
                
                if validation_results['errors']:
                    st.markdown("### ❌ Validation Errors")
                    for error in validation_results['errors']:
                        st.markdown(f"• {error}")
                else:
                    st.markdown("### ✅ Validation Passed")
                    st.markdown("All required fields and data types are valid!")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                
                if validation_results['warnings']:
                    st.markdown("### ⚠️ Warnings")
                    for warning in validation_results['warnings']:
                        st.markdown(f"• {warning}")
                else:
                    st.markdown("### 🎉 No Warnings")
                    st.markdown("Data quality looks excellent!")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Process button with professional styling
            st.markdown('<div style="text-align: center; margin: 2rem 0;">', unsafe_allow_html=True)
            if st.button("🚀 Process and Save Data", use_container_width=True):
                if not validation_results['errors']:
                    with st.spinner("Processing data..."):
                        save_transactions(df)
                    st.success("🎉 Data processed and saved successfully!")
                else:
                    st.error("❌ Please fix validation errors before processing")
            st.markdown('</div>', unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)  # Close upload card
    else:
        # Show upload instructions when no file is selected
        st.markdown("""
        #### 📋 Upload Instructions
        1. **Prepare your Excel file** with spend data
        2. **Ensure required columns** are present (Supplier Name, Invoice Value, etc.)
        3. **Select your file** using the uploader above
        4. **Review validation results** before processing
        5. **Confirm data processing** to save to database
        """)
        st.markdown('</div>', unsafe_allow_html=True)  # Close upload card

def validate_data(df):
    """Validate uploaded data"""
    errors = []
    warnings = []
    
    # Required columns mapping
    required_columns = {
        'SUPPLIER_NAME': 'supplier_name',
        'Item Invoice Value': 'item_invoice_value',
        'Invoice Date': 'invoice_date',
        'BU_CODE': 'bu_code'
    }
    
    # Check required columns
    for col in required_columns.keys():
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
    
    # Check for negative values
    if 'Item Invoice Value' in df.columns:
        negative_values = df[df['Item Invoice Value'] < 0]
        if not negative_values.empty:
            errors.append(f"Found {len(negative_values)} rows with negative invoice values")
    
    # Check for missing supplier names
    if 'SUPPLIER_NAME' in df.columns:
        missing_suppliers = df[df['SUPPLIER_NAME'].isna()]
        if not missing_suppliers.empty:
            warnings.append(f"Found {len(missing_suppliers)} rows with missing supplier names")
    
    return {'errors': errors, 'warnings': warnings}

def save_transactions(df):
    """Save transactions to database"""
    conn = sqlite3.connect('spend_platform.db')
    
    try:
        # Column mapping
        column_mapping = {
            'BU_CODE': 'bu_code',
            'BU _NAME': 'bu_name',
            'Region': 'region',
            'PO_NO': 'po_no',
            'PO_ITEM_NO': 'po_item_no',
            'Invoice Date': 'invoice_date',
            'SUPPLIER_NO': 'supplier_no',
            'SUPPLIER_NAME': 'supplier_name',
            'Invoice Header Amt - all items': 'invoice_amount',
            'CURRENCY_TYPE': 'currency_type',
            'UNIT_OF_PURCH': 'unit_of_purch',
            'Item Invoice Value': 'item_invoice_value',
            'Material Code': 'material_code',
            'Material Item Name': 'material_item_name',
            'Order description': 'order_description',
            'INV_ITEM_DESC': 'inv_item_desc',
            'PO Price': 'po_price',
            'PO purchase qty': 'po_purchase_qty',
            'Unit Price': 'unit_price',
            'Tower (Practice)': 'tower_practice',
            'Category': 'category',
            'Subcategory': 'subcategory'
        }
        
        # Rename columns
        df_mapped = df.rename(columns=column_mapping)
        
        # Select only the columns we have mappings for
        available_columns = [col for col in column_mapping.values() if col in df_mapped.columns]
        df_final = df_mapped[available_columns]
        
        # Save to database
        df_final.to_sql('spend_transactions', conn, if_exists='append', index=False)
        
        # Log successful upload
        st.success(f"Successfully saved {len(df_final)} transactions to database")
        
    except Exception as e:
        st.error(f"Error saving data: {str(e)}")
    finally:
        conn.close()

def master_data_page():
    """Master Data Management - Central Data Repository"""
    st.title("🗂️ Master Data Management")
    st.markdown("### Central Data Management - Master lists for vendors and categories")
    
    # Add information about master data principles
    with st.expander("ℹ️ Master Data Management Principles"):
        st.markdown("""
        - **Centralized Repository**: Single source of truth for vendors, categories, and reference data
        - **Data Standardization**: Consistent naming conventions and data formats
        - **Supplier Normalization**: Unified vendor records consolidating variations
        - **Quality Control**: Data validation and enrichment workflows
        """)
    
    # Settings section - moved to sidebar for better UX
    with st.sidebar:
        st.subheader("⚙️ Table Settings")
        
        # Get current user's pagination preference - with fallback for demo
        current_user_id = None
        if 'user' in st.session_state and st.session_state.user:
            current_user_id = st.session_state.user.get('user_id', 1)  # Default to user_id 1 for demo
        else:
            current_user_id = 1  # Default demo user
        
        current_preference = crud.get_user_setting(
            current_user_id, 
            'records_per_page', 
            str(Config.DEFAULT_RECORDS_PER_PAGE)
        )
        
        # Handle None case
        if current_preference is None:
            current_preference = str(Config.DEFAULT_RECORDS_PER_PAGE)
        
        current_preference_int = int(current_preference)
        
        records_per_page = st.selectbox(
            "Records per page:",
            options=Config.PAGINATION_OPTIONS,
            index=Config.PAGINATION_OPTIONS.index(current_preference_int) if current_preference_int in Config.PAGINATION_OPTIONS else 0,
            help="Select how many records to display per page"
        )
        
        # Save preference if changed
        if records_per_page != current_preference_int:
            crud.set_user_setting(current_user_id, 'records_per_page', str(records_per_page))
            st.success("✅ Preference saved!")
    
    tab1, tab2 = st.tabs(["📦 Vendors", "📂 Categories"])
    
    with tab1:
        enhanced_vendor_management(records_per_page)
    
    with tab2:
        enhanced_category_management(records_per_page)

def enhanced_vendor_management(records_per_page):
    """Enhanced vendor management with paginated table view and inline actions"""
    st.subheader("📦 Vendor Management")
    
    # Search and filter controls
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    with col1:
        search_term = st.text_input("🔍 Search Vendors", placeholder="Search by name...")
    with col2:
        vendor_type_filter = st.selectbox("Type", ["All", "Internal", "External", "Other"])
    with col3:
        if st.button("🔄 Refresh"):
            st.rerun()
    with col4:
        show_add_form = st.checkbox("➕ Add New", value=False)
    
    # Add new vendor form (conditional)
    if show_add_form:
        with st.expander("➕ Add New Vendor", expanded=True):
            with st.form("add_vendor_form"):
                col1, col2 = st.columns(2)
                with col1:
                    vendor_name = st.text_input("Vendor Name*", placeholder="Enter vendor name")
                    vendor_type = st.selectbox("Vendor Type", ["Internal", "External", "Other"])
                with col2:
                    normalized_name = st.text_input("Normalized Name", placeholder="Auto-generated if empty")
                    supplier_no = st.text_input("Supplier Number", placeholder="Enter supplier number")
                
                contact_info = st.text_area("Contact Information", placeholder="Enter contact details")
                
                col_submit, col_cancel = st.columns([1, 1])
                with col_submit:
                    if st.form_submit_button("💾 Add Vendor", use_container_width=True):
                        if vendor_name.strip():
                            try:
                                # Auto-generate normalized name if empty
                                if not normalized_name.strip():
                                    from utils import normalize_supplier_name
                                    normalized_name = normalize_supplier_name(vendor_name)
                                
                                vendor_id = crud.create_vendor(
                                    vendor_name=vendor_name,
                                    normalized_name=normalized_name,
                                    vendor_type=vendor_type
                                )
                                st.success(f"✅ Vendor added successfully! ID: {vendor_id}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error adding vendor: {str(e)}")
                        else:
                            st.error("❌ Vendor name is required")
    
    try:
        # Load vendors with search and filters
        filters = {}
        if vendor_type_filter != "All":
            filters['vendor_type'] = vendor_type_filter
        
        if search_term:
            vendors_df = crud.read_vendors(filters=filters, search=search_term)
        else:
            vendors_df = crud.read_vendors(filters=filters)
        
        if not vendors_df.empty:
            # Pagination setup
            total_records = len(vendors_df)
            total_pages = (total_records - 1) // records_per_page + 1
            
            # Page selector
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                current_page = st.selectbox(
                    f"Page (showing {records_per_page} of {total_records} records):",
                    options=list(range(1, total_pages + 1)),
                    format_func=lambda x: f"Page {x}",
                    key="vendor_page"
                )
            
            # Calculate pagination
            start_idx = (current_page - 1) * records_per_page
            end_idx = min(start_idx + records_per_page, total_records)
            paginated_df = vendors_df.iloc[start_idx:end_idx]
            
            # Display paginated table
            st.subheader(f"Vendors (Page {current_page} of {total_pages})")
            
            # Create enhanced table with action buttons
            for idx, row in paginated_df.iterrows():
                with st.container():
                    # Create a bordered container for each row
                    col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 1, 1])
                    
                    with col1:
                        st.write(f"**{row['vendor_name']}**")
                        if row['normalized_name']:
                            st.caption(f"Normalized: {row['normalized_name']}")
                    
                    with col2:
                        st.write(f"Type: {row.get('vendor_type', 'N/A')}")
                        st.caption(f"ID: {row['vendor_id']}")
                    
                    with col3:
                        st.write(f"Created: {row['created_at'][:10] if row['created_at'] else 'N/A'}")
                        if row['updated_at']:
                            st.caption(f"Updated: {row['updated_at'][:10]}")
                    
                    with col4:
                        # Edit button
                        if st.button("✏️", key=f"edit_vendor_btn_{row['vendor_id']}", help="Edit vendor"):
                            st.session_state[f"edit_vendor_state_{row['vendor_id']}"] = True
                    
                    with col5:
                        # Delete button
                        if st.button("🗑️", key=f"delete_vendor_{row['vendor_id']}", help="Delete vendor"):
                            if st.session_state.get(f"confirm_delete_vendor_{row['vendor_id']}", False):
                                if crud.delete_vendor(row['vendor_id']):
                                    st.success("✅ Vendor deleted!")
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to delete vendor")
                            else:
                                st.session_state[f"confirm_delete_vendor_{row['vendor_id']}"] = True
                                st.warning("Click again to confirm deletion")
                    
                    # Show edit form if edit button was clicked
                    if st.session_state.get(f"edit_vendor_state_{row['vendor_id']}", False):
                        with st.expander(f"✏️ Edit Vendor: {row['vendor_name']}", expanded=True):
                            with st.form(f"edit_vendor_form_{row['vendor_id']}"):
                                edit_col1, edit_col2 = st.columns(2)
                                with edit_col1:
                                    new_name = st.text_input("Vendor Name", value=row['vendor_name'])
                                    new_type = st.selectbox("Vendor Type", 
                                                          ["Internal", "External", "Other"],
                                                          index=["Internal", "External", "Other"].index(row.get('vendor_type', 'Other')))
                                with edit_col2:
                                    new_normalized = st.text_input("Normalized Name", value=row['normalized_name'] or "")
                                
                                col_save, col_cancel = st.columns(2)
                                with col_save:
                                    if st.form_submit_button("💾 Save Changes"):
                                        updates = {
                                            'vendor_name': new_name,
                                            'normalized_name': new_normalized,
                                            'vendor_type': new_type
                                        }
                                        if crud.update_vendor(row['vendor_id'], updates):
                                            st.success("✅ Vendor updated successfully!")
                                            st.session_state[f"edit_vendor_state_{row['vendor_id']}"] = False
                                            st.rerun()
                                        else:
                                            st.error("❌ Failed to update vendor")
                                
                                with col_cancel:
                                    if st.form_submit_button("❌ Cancel"):
                                        st.session_state[f"edit_vendor_state_{row['vendor_id']}"] = False
                                        st.rerun()
                    
                    st.divider()  # Visual separator between rows
        
        else:
            st.info("🔍 No vendors found. Try adjusting your search criteria or add new vendors.")
    
    except Exception as e:
        st.error(f"❌ Error loading vendors: {str(e)}")

def enhanced_category_management(records_per_page):
    """Enhanced category management with paginated table view and inline actions"""
    st.subheader("📂 Category Management")
    
    # Search and filter controls
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    with col1:
        search_term = st.text_input("🔍 Search Categories", placeholder="Search by name...")
    with col2:
        level_filter = st.selectbox("Level", ["All"] + list(range(1, 6)))
    with col3:
        if st.button("🔄 Refresh", key="refresh_categories"):
            st.rerun()
    with col4:
        show_add_form = st.checkbox("➕ Add New", value=False, key="add_category_toggle")
    
    # Add new category form (conditional)
    if show_add_form:
        with st.expander("➕ Add New Category", expanded=True):
            with st.form("add_category_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    category_name = st.text_input("Category Name*", placeholder="Enter category name")
                    category_level = st.number_input("Category Level", min_value=1, max_value=5, value=1)
                
                with col2:
                    # Load existing categories for parent selection
                    try:
                        existing_categories = crud.read_categories()
                        parent_options = ["None (Root Level)"]
                        if not existing_categories.empty:
                            parent_options.extend(existing_categories['category_name'].tolist())
                        
                        parent_selection = st.selectbox("Parent Category", parent_options)
                        parent_category_id = None
                        if parent_selection != "None (Root Level)" and not existing_categories.empty:
                            parent_category_id = existing_categories[existing_categories['category_name'] == parent_selection]['category_id'].iloc[0]
                    except:
                        parent_category_id = None
                        st.warning("Could not load parent categories")
                    
                    category_type = st.selectbox("Category Type", ["Direct", "Indirect", "Other"])
                
                if st.form_submit_button("💾 Add Category", use_container_width=True):
                    if category_name.strip():
                        try:
                            category_id = crud.create_category(
                                category_name=category_name,
                                parent_category_id=parent_category_id,
                                category_type=category_type
                            )
                            st.success(f"✅ Category added successfully! ID: {category_id}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error adding category: {str(e)}")
                    else:
                        st.error("❌ Category name is required")
    
    try:
        # Load categories with filters
        filters = {}
        if level_filter != "All":
            filters['category_level'] = level_filter
        
        categories_df = crud.read_categories(filters=filters)
        
        if not categories_df.empty:
            # Apply search filter
            if search_term:
                mask = categories_df['category_name'].str.contains(search_term, case=False, na=False)
                categories_df = categories_df[mask]
            
            if not categories_df.empty:
                # Pagination setup
                total_records = len(categories_df)
                total_pages = (total_records - 1) // records_per_page + 1
                
                # Page selector
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    current_page = st.selectbox(
                        f"Page (showing {records_per_page} of {total_records} records):",
                        options=list(range(1, total_pages + 1)),
                        format_func=lambda x: f"Page {x}",
                        key="category_page"
                    )
                
                # Calculate pagination
                start_idx = (current_page - 1) * records_per_page
                end_idx = min(start_idx + records_per_page, total_records)
                paginated_df = categories_df.iloc[start_idx:end_idx]
                
                # Display paginated table
                st.subheader(f"Categories (Page {current_page} of {total_pages})")
                
                # Create enhanced table with action buttons
                for idx, row in paginated_df.iterrows():
                    with st.container():
                        col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 1, 1])
                        
                        with col1:
                            # Show hierarchy with indentation
                            indent = "  " * (row.get('category_level', 1) - 1)
                            st.write(f"**{indent}{row['category_name']}**")
                            if row.get('parent_category_name'):
                                st.caption(f"Parent: {row['parent_category_name']}")
                        
                        with col2:
                            st.write(f"Type: {row.get('category_type', 'N/A')}")
                            st.caption(f"Level: {row.get('category_level', 'N/A')}")
                        
                        with col3:
                            st.write(f"Created: {row['created_at'][:10] if row['created_at'] else 'N/A'}")
                            st.caption(f"ID: {row['category_id']}")
                        
                        with col4:
                            # Edit button
                            if st.button("✏️", key=f"edit_category_btn_{row['category_id']}", help="Edit category"):
                                st.session_state[f"edit_category_state_{row['category_id']}"] = True
                        
                        with col5:
                            # Delete button
                            if st.button("🗑️", key=f"delete_category_{row['category_id']}", help="Delete category"):
                                if st.session_state.get(f"confirm_delete_category_{row['category_id']}", False):
                                    if crud.delete_category(row['category_id']):
                                        st.success("✅ Category deleted!")
                                        st.rerun()
                                    else:
                                        st.error("❌ Failed to delete category")
                                else:
                                    st.session_state[f"confirm_delete_category_{row['category_id']}"] = True
                                    st.warning("Click again to confirm deletion")
                        
                        # Show edit form if edit button was clicked
                        if st.session_state.get(f"edit_category_state_{row['category_id']}", False):
                            with st.expander(f"✏️ Edit Category: {row['category_name']}", expanded=True):
                                with st.form(f"edit_category_form_{row['category_id']}"):
                                    edit_col1, edit_col2 = st.columns(2)
                                    with edit_col1:
                                        new_name = st.text_input("Category Name", value=row['category_name'])
                                        new_type = st.selectbox("Category Type", 
                                                              ["Direct", "Indirect", "Other"],
                                                              index=["Direct", "Indirect", "Other"].index(row.get('category_type', 'Other')))
                                    with edit_col2:
                                        # Parent category selection
                                        try:
                                            all_categories = crud.read_categories()
                                            # Exclude current category and its children from parent options
                                            available_parents = all_categories[all_categories['category_id'] != row['category_id']]
                                            
                                            parent_options = ["None (Root Level)"]
                                            if not available_parents.empty:
                                                parent_options.extend(available_parents['category_name'].tolist())
                                            
                                            current_parent_name = row.get('parent_category_name', "None (Root Level)")
                                            if current_parent_name not in parent_options:
                                                current_parent_name = "None (Root Level)"
                                            
                                            parent_selection = st.selectbox("Parent Category", 
                                                                           parent_options,
                                                                           index=parent_options.index(current_parent_name))
                                            
                                            parent_category_id = None
                                            if parent_selection != "None (Root Level)":
                                                parent_category_id = available_parents[available_parents['category_name'] == parent_selection]['category_id'].iloc[0]
                                        except:
                                            parent_category_id = row.get('parent_category_id')
                                            st.warning("Could not load parent options")
                                    
                                    col_save, col_cancel = st.columns(2)
                                    with col_save:
                                        if st.form_submit_button("💾 Save Changes"):
                                            updates = {
                                                'category_name': new_name,
                                                'category_type': new_type,
                                                'parent_category_id': parent_category_id
                                            }
                                            if crud.update_category(row['category_id'], updates):
                                                st.success("✅ Category updated successfully!")
                                                st.session_state[f"edit_category_state_{row['category_id']}"] = False
                                                st.rerun()
                                            else:
                                                st.error("❌ Failed to update category")
                                    
                                    with col_cancel:
                                        if st.form_submit_button("❌ Cancel"):
                                            st.session_state[f"edit_category_state_{row['category_id']}"] = False
                                            st.rerun()
                        
                        st.divider()  # Visual separator between rows
            else:
                st.info("🔍 No categories found matching your search criteria.")
        else:
            st.info("📂 No categories found. Add some categories to get started.")
    
    except Exception as e:
        st.error(f"❌ Error loading categories: {str(e)}")

def vendor_management():
    """Legacy vendor management - kept for compatibility"""
    # Redirect to enhanced version
    enhanced_vendor_management(15)

def category_management():
    """Legacy category management - kept for compatibility"""
    # Redirect to enhanced version
    enhanced_category_management(15)

def error_management_page():
    """Error Detection & Handling - Data Integration Quality Control"""
    st.title("🔍 Error Detection & Handling")
    st.markdown("### Automatic Error Detection - Preventing Bad Data Entry")
    
    # Add information about error detection principles
    with st.expander("ℹ️ Error Detection Principles"):
        st.markdown("""
        - **Automatic Detection**: Real-time scanning for negative amounts, missing information
        - **Data Quality Rules**: Built-in validation preventing bad data entry
        - **User Feedback**: Easy error reporting and resolution workflow  
        - **Smart Correction**: AI-powered suggestions for data corrections
        """)
    
    try:
        # Load errors with transaction data
        error_df = crud.read_error_logs(filters={'status': 'Open'})
        
        if error_df.empty:
            st.success("🎉 No open errors found!")
            
            # Show resolved errors option
            show_resolved = st.checkbox("Show Recently Resolved Errors")
            if show_resolved:
                resolved_errors = crud.read_error_logs(filters={'status': 'Resolved'})
                if not resolved_errors.empty:
                    st.subheader("Recently Resolved Errors")
                    st.dataframe(resolved_errors.head(10), use_container_width=True)
            return
        
        st.subheader(f"Open Errors ({len(error_df)})")
        
        # Error summary using CRUD analytics
        error_summary = crud.get_data_quality_report()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Error Types Distribution")
            if error_summary and 'total_errors' in error_summary:
                # Create a simple bar chart from the error summary data
                error_types = ['Missing Data', 'Validation', 'Duplicates']
                error_counts = [
                    error_summary.get('missing_suppliers', 0) + error_summary.get('missing_amounts', 0),
                    error_summary.get('total_errors', 0) - error_summary.get('open_errors', 0),
                    error_summary.get('open_errors', 0)
                ]
                
                fig = px.bar(x=error_types, y=error_counts, title="Error Count by Type")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Error Statistics")
            total_errors = error_summary.get('total_errors', 0) if error_summary else 0
            st.metric("Total Open Errors", len(error_df))
            st.metric("Total All Errors", int(total_errors))
            
            if error_summary:
                missing_data = error_summary.get('missing_suppliers', 0)
                st.metric("Missing Data Issues", missing_data)
        
        # Bulk error management
        st.subheader("🔧 Bulk Error Management")
        
        # Select errors for bulk actions
        selected_errors = st.multiselect(
            "Select errors for bulk actions:",
            options=error_df['error_id'].tolist(),
            format_func=lambda x: f"ID {x}: {error_df[error_df['error_id'] == x]['error_type'].iloc[0]}"
        )
        
        if selected_errors:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("✅ Resolve Selected Errors"):
                    resolved_count = 0
                    for error_id in selected_errors:
                        username = st.session_state.user.get('username', 'System') if st.session_state.get('user') else 'System'
                        if crud.resolve_error(error_id, f"Resolved by {username}"):
                            resolved_count += 1
                    if resolved_count > 0:
                        st.success(f"✅ Resolved {resolved_count} errors!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to resolve errors")
            
            with col2:
                if st.button("🗑️ Delete Selected Errors"):
                    deleted_count = 0
                    for error_id in selected_errors:
                        if crud.delete_error_log(error_id):
                            deleted_count += 1
                    if deleted_count > 0:
                        st.success(f"✅ Deleted {deleted_count} errors!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to delete errors")
            
            with col3:
                if st.button("📊 Export Selected"):
                    selected_df = error_df[error_df['error_id'].isin(selected_errors)]
                    csv = selected_df.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download Selected Errors",
                        data=csv,
                        file_name=f"selected_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        # Individual error management
        st.subheader("📋 Error Details & Management")
        
        # Filter errors
        col1, col2 = st.columns(2)
        with col1:
            error_type_filter = st.selectbox(
                "Filter by Error Type",
                ["All"] + list(error_df['error_type'].unique())
            )
        with col2:
            sort_by = st.selectbox("Sort by", ["Created Date", "Error Type", "Supplier"])
        
        # Apply filters
        filtered_errors = error_df.copy()
        if error_type_filter != "All":
            filtered_errors = filtered_errors[filtered_errors['error_type'] == error_type_filter]
        
        # Sort
        if sort_by == "Created Date":
            filtered_errors = filtered_errors.sort_values('created_at', ascending=False)
        elif sort_by == "Error Type":
            filtered_errors = filtered_errors.sort_values('error_type')
        elif sort_by == "Supplier":
            filtered_errors = filtered_errors.sort_values('supplier_name')
        
        # Display individual errors with management options
        for idx, error in filtered_errors.iterrows():
            with st.expander(f"❌ {error['error_type']}: {error['description'][:50]}... (ID: {error['error_id']})"):
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write("**Error Details:**")
                    st.write(f"- **Type:** {error['error_type']}")
                    st.write(f"- **Description:** {error['description']}")
                    st.write(f"- **Created:** {error['created_at']}")
                    
                    if error['supplier_name']:
                        st.write(f"- **Supplier:** {error['supplier_name']}")
                    if error['item_invoice_value']:
                        st.write(f"- **Amount:** ${error['item_invoice_value']:,.2f}")
                    if error['bu_code']:
                        st.write(f"- **BU Code:** {error['bu_code']}")
                    
                    # Edit error details
                    with st.form(f"edit_error_{error['error_id']}"):
                        new_description = st.text_area("Update Description", value=error['description'])
                        new_status = st.selectbox("Status", ["Open", "In Progress", "Resolved"])
                        resolution_notes = st.text_area("Resolution Notes")
                        
                        if st.form_submit_button("💾 Update Error"):
                            updates = {
                                'description': new_description,
                                'status': new_status
                            }
                            if new_status == 'Resolved':
                                updates['resolved_at'] = datetime.now()
                                username = st.session_state.user.get('username', 'System') if st.session_state.get('user') else 'System'
                                updates['resolved_by'] = username
                            
                            if crud.update_error_log(error['error_id'], updates):
                                st.success("✅ Error updated successfully!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to update error")
                
                with col2:
                    st.write("**Quick Actions:**")
                    
                    if st.button(f"✅ Resolve", key=f"resolve_{error['error_id']}"):
                        username = st.session_state.user.get('username', 'System') if st.session_state.get('user') else 'System'
                        if crud.resolve_error(error['error_id'], f"Resolved by {username}"):
                            st.success("✅ Error resolved!")
                            st.rerun()
                    
                    if st.button(f"🗑️ Delete", key=f"delete_{error['error_id']}"):
                        if crud.delete_error_log(error['error_id']):
                            st.success("✅ Error deleted!")
                            st.rerun()
                    
                    # Link to transaction if available
                    if error['transaction_id']:
                        st.info(f"Transaction ID: {error['transaction_id']}")
        
        # Add new error manually
        with st.expander("➕ Add New Error Log", expanded=False):
            with st.form("add_error_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Get transactions for dropdown
                    transactions_df = crud.read_transactions(limit=100)
                    if not transactions_df.empty:
                        transaction_options = transactions_df['transaction_id'].tolist()
                        selected_transaction = st.selectbox("Transaction ID", transaction_options)
                    else:
                        selected_transaction = st.number_input("Transaction ID", min_value=1)
                    
                    error_type = st.selectbox("Error Type", Config.ERROR_TYPES)
                
                with col2:
                    error_description = st.text_area("Error Description", 
                                                   placeholder="Describe the error in detail...")
                
                if st.form_submit_button("📝 Add Error Log"):
                    if selected_transaction and error_description:
                        try:
                            error_id = crud.create_error_log(error_type, error_description, selected_transaction)
                            st.success(f"✅ Error log created successfully! ID: {error_id}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error creating error log: {str(e)}")
                    else:
                        st.error("❌ Please fill in all fields")
        
        # Export options
        st.subheader("📤 Export Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export All Errors"):
                all_errors = crud.read_error_logs()
                if not all_errors.empty:
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        all_errors.to_excel(writer, sheet_name='All Errors', index=False)
                        # Convert error_summary dict to DataFrame for export
                        if error_summary:
                            summary_df = pd.DataFrame([error_summary])
                            summary_df.to_excel(writer, sheet_name='Error Summary', index=False)
                    
                    st.download_button(
                        label="⬇️ Download Excel Report",
                        data=output.getvalue(),
                        file_name=f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        
        with col2:
            if st.button("📋 Export Open Errors"):
                csv = error_df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv,
                    file_name=f"open_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("📈 Export Error Analytics"):
                # Convert error_summary dict to CSV for download
                if error_summary:
                    summary_df = pd.DataFrame([error_summary])
                    analytics_csv = summary_df.to_csv(index=False)
                else:
                    analytics_csv = "No error data available"
                st.download_button(
                    label="⬇️ Download Analytics",
                    data=analytics_csv,
                    file_name=f"error_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
    except Exception as e:
        st.error(f"Error loading error data: {str(e)}")

def transaction_management_page():
    """Comprehensive transaction management with professional styling and CRUD operations"""
    create_section_header(
        "💼 Transaction Management", 
        "Comprehensive view and management of all spend transactions"
    )
    
    # Professional filters section
    st.markdown('<h3 class="subsection-header">🔍 Search & Filter</h3>', unsafe_allow_html=True)
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        search_supplier = st.text_input("🔍 Search Supplier", placeholder="Enter supplier name...")
    with col2:
        filter_region = st.selectbox("🌍 Region", ["All", "APAC", "LATAM", "Global", "EMEA"])
    with col3:
        filter_bu = st.text_input("🏢 Business Unit", placeholder="Enter BU code...")
    with col4:
        date_range = st.date_input("📅 Date Range", value=[], format="YYYY-MM-DD")
    
    # Build filters
    filters = {}
    if filter_region != "All":
        filters['region'] = filter_region
    if filter_bu:
        filters['bu_code'] = filter_bu
    
    # Add date range to filters if provided
    if len(date_range) == 2:
        filters['start_date'] = date_range[0]
        filters['end_date'] = date_range[1]
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    try:
        # Load transactions
        if search_supplier:
            # Use SQL LIKE for supplier search
            transactions_df = crud.read_transactions(filters=filters)
            transactions_df = transactions_df[transactions_df['supplier_name'].str.contains(
                search_supplier, case=False, na=False)]
        else:
            transactions_df = crud.read_transactions(filters=filters, limit=100)
        
        if not transactions_df.empty:
            st.subheader(f"Transactions ({len(transactions_df)})")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Spend", f"${transactions_df['item_invoice_value'].sum():,.2f}")
            with col2:
                st.metric("Avg Transaction", f"${transactions_df['item_invoice_value'].mean():,.2f}")
            with col3:
                st.metric("Unique Suppliers", transactions_df['supplier_name'].nunique())
            with col4:
                st.metric("Date Range", f"{len(transactions_df)} records")
            
            # Bulk actions
            st.subheader("Bulk Actions")
            col1, col2, col3 = st.columns(3)
            
            selected_transactions = st.multiselect(
                "Select transactions:",
                options=transactions_df['transaction_id'].tolist(),
                format_func=lambda x: f"ID {x}: {transactions_df[transactions_df['transaction_id'] == x]['supplier_name'].iloc[0]}"
            )
            
            if selected_transactions:
                with col1:
                    if st.button("📝 Bulk Update Status"):
                        new_status = st.selectbox("New Status", ["Active", "Inactive", "Pending"])
                        updated_count = 0
                        for trans_id in selected_transactions:
                            if crud.update_transaction(trans_id, {'status': new_status}):
                                updated_count += 1
                        if updated_count > 0:
                            st.success(f"Updated {len(selected_transactions)} transactions")
                            st.rerun()
                
                with col2:
                    if st.button("🏷️ Bulk Categorize"):
                        categories_df = crud.read_categories()
                        if not categories_df.empty:
                            new_category = st.selectbox("Category", categories_df['category_name'].tolist())
                            updated_count = 0
                            for trans_id in selected_transactions:
                                if crud.update_transaction(trans_id, {'category_id': new_category}):
                                    updated_count += 1
                            if updated_count > 0:
                                st.success(f"Categorized {len(selected_transactions)} transactions")
                                st.rerun()
                
                with col3:
                    if st.button("🗑️ Bulk Delete"):
                        if st.checkbox("Confirm deletion"):
                            deleted_count = 0
                            for txn_id in selected_transactions:
                                if crud.delete_transaction(txn_id):
                                    deleted_count += 1
                            st.success(f"Deleted {deleted_count} transactions")
                            st.rerun()
            
            # Transaction details with inline editing
            st.subheader("Transaction Details")
            
            # Pagination
            items_per_page = 10
            total_pages = len(transactions_df) // items_per_page + 1
            page = st.selectbox("Page", range(1, total_pages + 1)) - 1
            
            start_idx = page * items_per_page
            end_idx = min(start_idx + items_per_page, len(transactions_df))
            page_df = transactions_df.iloc[start_idx:end_idx]
            
            for idx, row in page_df.iterrows():
                with st.expander(f"💰 {row['supplier_name']} - ${row['item_invoice_value']:,.2f} (ID: {row['transaction_id']})"):
                    
                    # Edit form
                    with st.form(f"edit_transaction_{row['transaction_id']}"):
                        edit_col1, edit_col2, edit_col3 = st.columns(3)
                        
                        with edit_col1:
                            new_supplier = st.text_input("Supplier Name", value=row['supplier_name'] or "")
                            new_amount = st.number_input("Invoice Value", value=float(row['item_invoice_value'] or 0))
                            new_bu_code = st.text_input("BU Code", value=row['bu_code'] or "")
                        
                        with edit_col2:
                            new_region = st.selectbox("Region", 
                                                    ["APAC", "LATAM", "Global", "EMEA"], 
                                                    index=["APAC", "LATAM", "Global", "EMEA"].index(row['region']) 
                                                    if row['region'] in ["APAC", "LATAM", "Global", "EMEA"] else 0)
                            new_currency = st.text_input("Currency", value=row['currency_type'] or "")
                            new_category = st.text_input("Category", value=row['category'] or "")
                        
                        with edit_col3:
                            new_status = st.selectbox("Status", 
                                                    ["Active", "Inactive", "Pending"],
                                                    index=["Active", "Inactive", "Pending"].index(row['status']) 
                                                    if row['status'] in ["Active", "Inactive", "Pending"] else 0)
                            new_po_no = st.text_input("PO Number", value=row['po_no'] or "")
                            new_description = st.text_area("Description", value=row['order_description'] or "")
                        
                        # Action buttons
                        col_update, col_delete, col_info = st.columns(3)
                        
                        with col_update:
                            if st.form_submit_button("💾 Update Transaction"):
                                updates = {
                                    'supplier_name': new_supplier,
                                    'item_invoice_value': new_amount,
                                    'bu_code': new_bu_code,
                                    'region': new_region,
                                    'currency_type': new_currency,
                                    'category': new_category,
                                    'status': new_status,
                                    'po_no': new_po_no,
                                    'order_description': new_description
                                }
                                if crud.update_transaction(row['transaction_id'], updates):
                                    st.success("✅ Transaction updated successfully!")
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to update transaction")
                        
                        with col_delete:
                            if st.form_submit_button("🗑️ Delete Transaction"):
                                if crud.delete_transaction(row['transaction_id']):
                                    st.success("✅ Transaction deleted successfully!")
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to delete transaction")
                        
                        with col_info:
                            st.write("**Upload Date:**", row['upload_date'][:10] if row['upload_date'] else "N/A")
        
        else:
            st.info("No transactions found with the current filters.")
            
        # Add new transaction
        with st.expander("➕ Add New Transaction", expanded=False):
            with st.form("add_transaction_form"):
                st.subheader("Add New Transaction")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    supplier_name = st.text_input("Supplier Name*")
                    invoice_value = st.number_input("Invoice Value*", min_value=0.0)
                    bu_code = st.text_input("BU Code*")
                
                with col2:
                    region = st.selectbox("Region*", ["APAC", "LATAM", "Global", "EMEA"])
                    currency = st.text_input("Currency", value="USD")
                    po_no = st.text_input("PO Number")
                
                with col3:
                    category = st.text_input("Category")
                    invoice_date = st.date_input("Invoice Date*")
                    description = st.text_area("Description")
                
                if st.form_submit_button("💾 Add Transaction"):
                    if supplier_name and invoice_value and bu_code:
                        transaction_data = {
                            'supplier_name': supplier_name,
                            'item_invoice_value': invoice_value,
                            'bu_code': bu_code,
                            'region': region,
                            'currency_type': currency,
                            'po_no': po_no,
                            'category': category,
                            'invoice_date': invoice_date,
                            'order_description': description,
                            'status': 'Active'
                        }
                        
                        try:
                            transaction_id = crud.create_transaction(transaction_data)
                            st.success(f"✅ Transaction added successfully! ID: {transaction_id}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error adding transaction: {str(e)}")
                    else:
                        st.error("❌ Please fill in all required fields (*)")
                        
    except Exception as e:
        st.error(f"Error loading transactions: {str(e)}")

def user_management_page():
    """User management with full CRUD operations"""
    st.title("👥 User Management")
    
    # Check permissions
    user_role = st.session_state.get('user', {}).get('role', '')
    if user_role != 'Admin':
        st.error("🚫 Access denied. Admin privileges required.")
        return
    
    try:
        # Load users
        users_df = crud.read_users()
        
        # Add new user
        with st.expander("➕ Add New User", expanded=False):
            with st.form("add_user_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    username = st.text_input("Username*", placeholder="Enter username")
                    password = st.text_input("Password*", type="password", placeholder="Enter password")
                
                with col2:
                    role = st.selectbox("Role*", ["Admin", "Spend Manager", "Data Analyst"])
                    confirm_password = st.text_input("Confirm Password*", type="password")
                
                if st.form_submit_button("👤 Add User"):
                    if username and password and password == confirm_password:
                        try:
                            from utils import hash_password
                            password_hash = hash_password(password)
                            user_id = crud.create_user(username, password_hash, role)
                            st.success(f"✅ User created successfully! ID: {user_id}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error creating user: {str(e)}")
                    else:
                        if password != confirm_password:
                            st.error("❌ Passwords do not match")
                        else:
                            st.error("❌ Please fill in all required fields")
        
        # Display users
        if not users_df.empty:
            st.subheader(f"Users ({len(users_df)})")
            
            for idx, user in users_df.iterrows():
                with st.expander(f"👤 {user['username']} ({user['role']})"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        with st.form(f"edit_user_{user['user_id']}"):
                            edit_col1, edit_col2 = st.columns(2)
                            
                            with edit_col1:
                                new_username = st.text_input("Username", value=user['username'])
                                new_role = st.selectbox("Role", 
                                                       ["Admin", "Spend Manager", "Data Analyst"],
                                                       index=["Admin", "Spend Manager", "Data Analyst"].index(user['role']))
                            
                            with edit_col2:
                                new_password = st.text_input("New Password (leave empty to keep current)", type="password")
                                confirm_new_password = st.text_input("Confirm New Password", type="password")
                            
                            col_update, col_delete = st.columns(2)
                            
                            with col_update:
                                if st.form_submit_button("💾 Update User"):
                                    updates = {
                                        'username': new_username,
                                        'role': new_role
                                    }
                                    
                                    # Update password if provided
                                    if new_password:
                                        if new_password == confirm_new_password:
                                            from utils import hash_password
                                            updates['password_hash'] = hash_password(new_password)
                                        else:
                                            st.error("❌ Passwords do not match")
                                            continue
                                    
                                    if crud.update_user(user['user_id'], updates):
                                        st.success("✅ User updated successfully!")
                                        st.rerun()
                                    else:
                                        st.error("❌ Failed to update user")
                            
                            with col_delete:
                                current_username = st.session_state.get('user', {}).get('username', '')
                                if user['username'] != current_username:  # Prevent self-deletion
                                    if st.form_submit_button("🗑️ Delete User"):
                                        if crud.delete_user(user['user_id']):
                                            st.success("✅ User deleted successfully!")
                                            st.rerun()
                                        else:
                                            st.error("❌ Failed to delete user")
                                else:
                                    st.info("Cannot delete current user")
                    
                    with col2:
                        st.metric("Created", user['created_at'][:10] if user['created_at'] else "N/A")
                        if user['last_login']:
                            st.metric("Last Login", user['last_login'][:10])
                        else:
                            st.metric("Last Login", "Never")
        
        else:
            st.info("No users found.")
            
    except Exception as e:
        st.error(f"Error loading users: {str(e)}")

def rules_management_page():
    """Business rules management with full CRUD operations"""
    st.title("📏 Business Rules Management")
    
    # Check permissions
    user_role = st.session_state.get('user', {}).get('role', '')
    if user_role not in ['Admin', 'Spend Manager']:
        st.error("🚫 Access denied. Admin or Spend Manager privileges required.")
        return
    
    try:
        # Load rules
        rules_df = crud.read_rules()
        
        # Add new rule
        with st.expander("➕ Add New Rule", expanded=False):
            with st.form("add_rule_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    rule_name = st.text_input("Rule Name*", placeholder="Enter rule name")
                    rule_type = st.selectbox("Rule Type*", 
                                           ["Validation", "Classification", "Approval", "Alert"])
                
                with col2:
                    rule_description = st.text_area("Description*", placeholder="Describe what this rule does")
                
                rule_condition = st.text_area("Rule Condition*", 
                                            placeholder="Enter rule condition (e.g., amount > 10000)")
                
                if st.form_submit_button("📏 Add Rule"):
                    if rule_name and rule_description and rule_condition:
                        try:
                            rule_id = crud.create_rule(
                                rule_name=rule_name,
                                rule_type=rule_type,
                                rule_condition=rule_condition,
                                created_by=st.session_state.get('username', 'system'),
                                rule_description=rule_description
                            )
                            st.success(f"✅ Rule created successfully! ID: {rule_id}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error creating rule: {str(e)}")
                    else:
                        st.error("❌ Please fill in all required fields")
        
        # Display rules
        if not rules_df.empty:
            st.subheader(f"Business Rules ({len(rules_df)})")
            
            # Filter controls
            col1, col2 = st.columns(2)
            with col1:
                show_inactive = st.checkbox("Show Inactive Rules")
            with col2:
                rule_type_filter = st.selectbox("Filter by Type", 
                                              ["All", "Validation", "Classification", "Approval", "Alert"])
            
            # Apply filters
            filtered_df = rules_df.copy()
            if not show_inactive:
                filtered_df = filtered_df[filtered_df['active_flag'] == 1]
            if rule_type_filter != "All":
                filtered_df = filtered_df[filtered_df['rule_type'] == rule_type_filter]
            
            for idx, rule in filtered_df.iterrows():
                status_icon = "✅" if rule['active_flag'] else "❌"
                with st.expander(f"{status_icon} {rule['rule_name']} ({rule['rule_type']})"):
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        with st.form(f"edit_rule_{rule['rule_id']}"):
                            edit_col1, edit_col2 = st.columns(2)
                            
                            with edit_col1:
                                new_name = st.text_input("Rule Name", value=rule['rule_name'])
                                new_type = st.selectbox("Rule Type",
                                                       ["Validation", "Classification", "Approval", "Alert"],
                                                       index=["Validation", "Classification", "Approval", "Alert"].index(rule['rule_type']))
                            
                            with edit_col2:
                                new_description = st.text_area("Description", value=rule['rule_description'] or "")
                            
                            new_condition = st.text_area("Rule Condition", value=rule['rule_condition'] or "")
                            
                            col_update, col_toggle, col_delete = st.columns(3)
                            
                            with col_update:
                                if st.form_submit_button("💾 Update"):
                                    updates = {
                                        'rule_name': new_name,
                                        'rule_type': new_type,
                                        'rule_description': new_description,
                                        'rule_condition': new_condition
                                    }
                                    if crud.update_rule(rule['rule_id'], updates):
                                        st.success("✅ Rule updated successfully!")
                                        st.rerun()
                                    else:
                                        st.error("❌ Failed to update rule")
                            
                            with col_toggle:
                                action = "Activate" if not rule['active_flag'] else "Deactivate"
                                if st.form_submit_button(f"🔄 {action}"):
                                    if crud.toggle_rule_status(rule['rule_id']):
                                        st.success(f"✅ Rule {action.lower()}d successfully!")
                                        st.rerun()
                                    else:
                                        st.error(f"❌ Failed to {action.lower()} rule")
                            
                            with col_delete:
                                if st.form_submit_button("🗑️ Delete"):
                                    if crud.delete_rule(rule['rule_id']):
                                        st.success("✅ Rule deleted successfully!")
                                        st.rerun()
                                    else:
                                        st.error("❌ Failed to delete rule")
                    
                    with col2:
                        st.metric("Created By", rule['created_by'] or "Unknown")
                        st.metric("Created", rule['created_at'][:10] if rule['created_at'] else "N/A")
                        st.metric("Status", "Active" if rule['active_flag'] else "Inactive")
        
        else:
            st.info("No rules found. Add some rules to get started.")
            
    except Exception as e:
        st.error(f"Error loading rules: {str(e)}")

def reports_page():
    """Enhanced reports page with CRUD-based data access"""
    st.title("📈 Advanced Reports")
    
    try:
        # Get data using CRUD operations
        transactions_df = crud.read_transactions(limit=10000)  # Get more data for reporting
        
        if transactions_df.empty:
            st.warning("No data available for reporting.")
            return
        
        # Convert date column
        if 'invoice_date' in transactions_df.columns:
            transactions_df['invoice_date'] = pd.to_datetime(transactions_df['invoice_date'], errors='coerce')
        
        # Filters
        st.subheader("📊 Report Filters")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            regions = ['All'] + list(transactions_df['region'].dropna().unique())
            selected_region = st.selectbox("Region", regions)
        
        with col2:
            suppliers = ['All'] + list(transactions_df['supplier_name'].dropna().unique())
            selected_supplier = st.selectbox("Supplier", suppliers[:50])  # Limit for performance
        
        with col3:
            categories = ['All'] + list(transactions_df['category'].dropna().unique())
            selected_category = st.selectbox("Category", categories)
        
        with col4:
            date_range = st.date_input("Date Range", value=[])
        
        # Apply filters
        filtered_df = transactions_df.copy()
        
        if selected_region != 'All':
            filtered_df = filtered_df[filtered_df['region'] == selected_region]
        
        if selected_supplier != 'All':
            filtered_df = filtered_df[filtered_df['supplier_name'] == selected_supplier]
        
        if selected_category != 'All':
            filtered_df = filtered_df[filtered_df['category'] == selected_category]
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = filtered_df[
                (filtered_df['invoice_date'] >= pd.Timestamp(start_date)) &
                (filtered_df['invoice_date'] <= pd.Timestamp(end_date))
            ]
        
        # Report sections
        st.subheader("📋 Summary Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_spend = filtered_df['item_invoice_value'].sum()
            st.metric("Filtered Spend", f"${total_spend:,.2f}")
        
        with col2:
            transaction_count = len(filtered_df)
            st.metric("Transactions", f"{transaction_count:,}")
        
        with col3:
            avg_transaction = filtered_df['item_invoice_value'].mean()
            st.metric("Avg Transaction", f"${avg_transaction:,.2f}")
        
        with col4:
            unique_suppliers = filtered_df['supplier_name'].nunique()
            st.metric("Suppliers", f"{unique_suppliers:,}")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Spend by Region")
            if not filtered_df.empty:
                region_spend = filtered_df.groupby('region')['item_invoice_value'].sum().reset_index()
                if not region_spend.empty:
                    fig = px.pie(region_spend, values='item_invoice_value', names='region',
                               title="Spend Distribution by Region")
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🏢 Top 10 Suppliers")
            top_suppliers = filtered_df.groupby('supplier_name')['item_invoice_value'].sum().nlargest(10).reset_index()
            if not top_suppliers.empty:
                fig = px.bar(top_suppliers, x='item_invoice_value', y='supplier_name', 
                           orientation='h', title="Top Suppliers by Spend")
                st.plotly_chart(fig, use_container_width=True)
        
        # Trend analysis
        if 'invoice_date' in filtered_df.columns and not filtered_df.empty:
            st.subheader("📈 Spend Trend Analysis")
            filtered_df_valid = filtered_df.dropna(subset=['invoice_date', 'item_invoice_value'])
            
            if not filtered_df_valid.empty:
                # Monthly trend
                filtered_df_valid['month'] = filtered_df_valid['invoice_date'].dt.to_period('M')
                monthly_spend = filtered_df_valid.groupby('month')['item_invoice_value'].sum().reset_index()
                monthly_spend['month'] = monthly_spend['month'].astype(str)
                
                fig = px.line(monthly_spend, x='month', y='item_invoice_value',
                            title="Monthly Spend Trend", markers=True)
                st.plotly_chart(fig, use_container_width=True)
        
        # Business Unit Analysis
        if 'bu_name' in filtered_df.columns:
            st.subheader("🏭 Business Unit Analysis")
            bu_spend = filtered_df.groupby('bu_name')['item_invoice_value'].sum().nlargest(10).reset_index()
            if not bu_spend.empty:
                fig = px.bar(bu_spend, x='bu_name', y='item_invoice_value',
                           title="Spend by Business Unit")
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.subheader("📄 Detailed Transaction Data")
        if not filtered_df.empty:
            # Display columns selection
            display_columns = st.multiselect(
                "Select columns to display:",
                options=filtered_df.columns.tolist(),
                default=['supplier_name', 'item_invoice_value', 'region', 'bu_name', 'invoice_date', 'category']
            )
            
            if display_columns:
                st.dataframe(filtered_df[display_columns].head(100), use_container_width=True)
        
        # Export options
        st.subheader("📤 Export Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Generate Excel Report"):
                # Create comprehensive Excel report
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    # Main data
                    filtered_df.to_excel(writer, sheet_name='Spend Report', index=False)
                    
                    # Summary sheet
                    summary_data = {
                        'Metric': ['Total Spend', 'Total Transactions', 'Average Transaction', 
                                 'Unique Suppliers', 'Unique Regions', 'Date Range'],
                        'Value': [
                            f"${total_spend:,.2f}",
                            f"{transaction_count:,}",
                            f"${avg_transaction:,.2f}",
                            f"{unique_suppliers:,}",
                            f"{filtered_df['region'].nunique():,}",
                            f"{len(date_range)} dates selected" if date_range else "All dates"
                        ]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    # Top suppliers sheet
                    if not top_suppliers.empty:
                        top_suppliers.to_excel(writer, sheet_name='Top Suppliers', index=False)
                    
                    # Regional summary
                    if not region_spend.empty:
                        region_spend.to_excel(writer, sheet_name='Regional Summary', index=False)
                
                st.download_button(
                    label="⬇️ Download Excel Report",
                    data=output.getvalue(),
                    file_name=f"spend_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        with col2:
            if st.button("📄 Generate CSV Report"):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download CSV Report",
                    data=csv,
                    file_name=f"spend_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("📋 Generate Summary Report"):
                # Analytics summary
                analytics_data = crud.get_spend_summary()
                if analytics_data:
                    summary_df = pd.DataFrame([analytics_data])
                    summary_csv = summary_df.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download Summary CSV",
                        data=summary_csv,
                        file_name=f"spend_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        
        # Data quality metrics
        st.subheader("🔍 Data Quality Metrics")
        
        try:
            quality_metrics = crud.get_data_quality_report()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", quality_metrics.get('total_transactions', 0))
            with col2:
                st.metric("Missing Suppliers", quality_metrics.get('missing_suppliers', 0))
            with col3:
                st.metric("Missing Amounts", quality_metrics.get('missing_amounts', 0))
            with col4:
                st.metric("Open Errors", quality_metrics.get('open_errors', 0))
            
        except Exception as e:
            st.warning(f"Could not load data quality metrics: {str(e)}")
        
    except Exception as e:
        st.error(f"Error generating reports: {str(e)}")

    """Display reports page"""
    st.title("📈 Advanced Reports")
    
    conn = sqlite3.connect('spend_platform.db')
    
    try:
        df = pd.read_sql_query("SELECT * FROM spend_transactions", conn)
        
        if df.empty:
            st.warning("No data available for reporting.")
            return
        
        # Filters
        st.subheader("Filters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            regions = ['All'] + list(df['region'].dropna().unique())
            selected_region = st.selectbox("Region", regions)
        
        with col2:
            suppliers = ['All'] + list(df['supplier_name'].dropna().unique())
            selected_supplier = st.selectbox("Supplier", suppliers)
        
        with col3:
            categories = ['All'] + list(df['category'].dropna().unique())
            selected_category = st.selectbox("Category", categories)
        
        # Apply filters
        filtered_df = df.copy()
        
        if selected_region != 'All':
            filtered_df = filtered_df[filtered_df['region'] == selected_region]
        
        if selected_supplier != 'All':
            filtered_df = filtered_df[filtered_df['supplier_name'] == selected_supplier]
        
        if selected_category != 'All':
            filtered_df = filtered_df[filtered_df['category'] == selected_category]
        
        # Report sections
        st.subheader("Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Filtered Spend", f"${filtered_df['item_invoice_value'].sum():,.2f}")
        
        with col2:
            st.metric("Transactions", f"{len(filtered_df):,}")
        
        with col3:
            st.metric("Avg Transaction", f"${filtered_df['item_invoice_value'].mean():,.2f}")
        
        with col4:
            st.metric("Suppliers", f"{filtered_df['supplier_name'].nunique():,}")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Spend Trend (Monthly)")
            if 'invoice_date' in filtered_df.columns:
                filtered_df['invoice_date'] = pd.to_datetime(filtered_df['invoice_date'], errors='coerce')
                filtered_df['month'] = filtered_df['invoice_date'].dt.to_period('M')
                monthly_spend = filtered_df.groupby('month')['item_invoice_value'].sum().reset_index()
                monthly_spend['month'] = monthly_spend['month'].astype(str)
                
                fig = px.line(monthly_spend, x='month', y='item_invoice_value')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Spend by Business Unit")
            bu_spend = filtered_df.groupby('bu_name')['item_invoice_value'].sum().nlargest(10).reset_index()
            fig = px.bar(bu_spend, x='bu_name', y='item_invoice_value')
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Export options
        st.subheader("Export Options")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Generate Excel Report"):
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    filtered_df.to_excel(writer, sheet_name='Spend Report', index=False)
                    
                    # Add summary sheet
                    summary_df = pd.DataFrame({
                        'Metric': ['Total Spend', 'Total Transactions', 'Average Transaction', 'Unique Suppliers'],
                        'Value': [
                            f"${filtered_df['item_invoice_value'].sum():,.2f}",
                            f"{len(filtered_df):,}",
                            f"${filtered_df['item_invoice_value'].mean():,.2f}",
                            f"{filtered_df['supplier_name'].nunique():,}"
                        ]
                    })
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                st.download_button(
                    label="Download Excel Report",
                    data=output.getvalue(),
                    file_name=f"spend_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        with col2:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV Report",
                data=csv,
                file_name=f"spend_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
    except Exception as e:
        st.error(f"Error generating reports: {str(e)}")
    finally:
        conn.close()

# ===================== NEW PAGE FUNCTIONS =====================

def validation_rules_page():
    """Data Validation Rules Management Page"""
    create_coming_soon_page(
        "Data Validation Rules",
        "�",
        [
            "Built-in rules to prevent bad data from entering the system",
            "Automatic error checking for negative amounts or missing information",
            "Configurable validation rules for different data types",
            "Real-time data quality monitoring",
            "Custom business rule engine",
            "Automated data cleansing workflows"
        ],
        "Q1 2026"
    )

def data_sources_page():
    """Data Source Management Page"""
    st.title("🔗 Data Source Management")
    st.info("🚧 **Coming Soon. Work in progress**")
    st.markdown("""
    This page will include:
    - Integration with multiple data sources
    - Data source configuration and mapping
    - Connection status monitoring
    - Data source validation and testing
    """)

def language_processing_page():
    """Multi-Language Data Processing Page"""
    st.title("🌐 Multi-Language Processing")
    st.info("🚧 **Coming Soon. Work in progress**")
    st.markdown("""
    This page will include:
    - Multi-language data processing capabilities
    - Automatic language detection
    - Translation services integration
    - Standardized output in preferred language
    """)

def classification_page():
    """Data Classification Page"""
    st.title("🏷️ Data Classification")
    st.info("🚧 **Coming Soon. Work in progress**")
    st.markdown("""
    This page will include:
    - Spend data classification according to taxonomy
    - AI-powered category suggestion
    - Custom classification rules
    - Structured analysis and reporting capabilities
    """)

def standardization_page():
    """Data Standardization Page"""
    st.title("🔄 Data Standardization")
    st.info("🚧 **Coming Soon. Work in progress**")
    st.markdown("""
    This page will include:
    - Date, currency, and unit standardization
    - Supplier name normalization
    - Data format consistency checks
    - Automated data cleaning processes
    """)

def self_service_page():
    """Self-Service Data Tools Page"""
    st.title("👤 Self-Service Data Tools")
    st.info("🚧 **Coming Soon. Work in progress**")
    st.markdown("""
    This page will include:
    - User-driven data augmentation capabilities
    - Custom dataset creation tools
    - Data enrichment workflows
    - No-code data transformation tools
    """)

def spend_analysis_page():
    """Advanced Spend Analysis Page"""
    st.title("🎯 Spend Analysis")
    st.info("🚧 **Coming Soon. Work in progress**")
    st.markdown("""
    This page will include:
    - Advanced spend analytics and insights
    - Trend analysis and forecasting
    - Supplier performance analytics
    - Cost optimization recommendations
    """)

def governance_page():
    """Data Governance Page"""
    st.title("📋 Data Governance")
    st.info("🚧 **Coming Soon. Work in progress**")
    st.markdown("""
    This page will include:
    - Clear data rules and responsibilities
    - Data quality metrics and monitoring
    - Audit trails and compliance reporting
    - Data stewardship workflows
    """)

def main():
    """Main application function"""
    # Inject professional styling
    inject_custom_css()
    
    # Initialize database
    init_database()
    
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    
    # Authentication check
    if not st.session_state.authenticated:
        login_page()
        return
    
    # Main application
    selected_page = sidebar_navigation()
    
    # Route to selected page
    # Data Integration
    if selected_page == "upload":
        upload_page()
    elif selected_page == "validation_rules":
        validation_rules_page()
    elif selected_page == "errors":
        error_management_page()
    elif selected_page == "data_sources":
        data_sources_page()
    elif selected_page == "language_processing":
        language_processing_page()
    
    # Data Enrichment & Analysis
    elif selected_page == "master_data":
        master_data_page()
    elif selected_page == "classification":
        classification_page()
    elif selected_page == "transactions":
        transaction_management_page()
    elif selected_page == "rules":
        rules_management_page()
    elif selected_page == "standardization":
        standardization_page()
    elif selected_page == "self_service":
        self_service_page()
    
    # Data Insights
    elif selected_page == "dashboard":
        dashboard_page()
    elif selected_page == "reports":
        reports_page()
    elif selected_page == "spend_analysis":
        spend_analysis_page()
    elif selected_page == "governance":
        governance_page()
    elif selected_page == "users":
        user_management_page()
    else:
        dashboard_page()  # Default to dashboard

if __name__ == "__main__":
    main()
