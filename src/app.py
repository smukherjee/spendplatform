"""
Spend Platform - Main Application Entry Point
Streamlit-based spend data management application
"""
import os
import sys

# When running `python src/app.py` directly, ensure the repository root is on sys.path
# so `import src.*` imports work. This is safe because when package is installed this
# check will be a no-op.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st

from src.utils.debug import debug_logger

logger = debug_logger

def init_app():
    """Initialize application configuration"""
    logger.info("Starting main application")
    logger.info("Initializing application")
    
    # Page configuration - IMPORTANT: Set this before any other Streamlit commands
    st.set_page_config(
        page_title="Spend Platform",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="collapsed"  # Start with sidebar collapsed
    )
    
    # CRITICAL: Hide automatic page discovery by clearing any auto-generated navigation
    # This prevents Streamlit from automatically showing pages from src/pages/ directory
    st.session_state._pages = {}
    
    logger.debug("Page config set successfully")
    
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_exists' not in st.session_state:
        st.session_state.user_exists = False
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    
    logger.debug(f"Session state initialized | Data: {{'authenticated': {st.session_state.authenticated}, 'user_exists': {st.session_state.user_exists}, 'debug_mode': {st.session_state.debug_mode}}}")

def sidebar_navigation():
    """Render sidebar navigation"""
    from src.services.role_service import RoleService
    from src.services.auth_service import AuthService
    
    logger.debug("Rendering sidebar navigation")
    
    # Display user info
    if st.session_state.authenticated:
        with st.sidebar:
            st.markdown("---")
            user_info = st.session_state.get('user', {})
            username = user_info.get('username', 'User')
            user_role = user_info.get('role', 'Unknown')
            
            st.markdown(f"👤 **{username}**")
            st.markdown(f"🏷️ Roles: {user_role}")
            logger.debug(f"User info displayed | Data: {{'username': '{username}', 'role': '{user_role}'}}")
            st.markdown("---")
            
            # Initialize services
            role_service = RoleService()
            auth_service = AuthService()
            
            # Check permissions - use actual user role, not fallback
            permissions = role_service.get_role_permissions(user_role)
            
            logger.debug(f"Checking user permissions | Data: {{'role': '{user_role}', 'permissions': {permissions}}}")
            
            # Set default page if none selected
            if 'selected_page' not in st.session_state:
                st.session_state.selected_page = "dashboard"
            
            # Navigation buttons based on permissions
            # Dashboard - available to all authenticated users
            if st.button("🏠 Dashboard", key="nav_dashboard", width='stretch'):
                st.session_state.selected_page = "dashboard"
                st.rerun()
            
            # Data Upload
            if permissions.get("can_upload_data", False):
                if st.button("📤 Upload Data", key="nav_upload", width='stretch'):
                    st.session_state.selected_page = "upload"
                    st.rerun()
            
            # Master Data Management (split into Vendors and Categories)
            if permissions.get("can_manage_master_data", False):
                if st.button("� Vendors", key="nav_vendors", width='stretch'):
                    st.session_state.selected_page = "vendors"
                    st.rerun()
                if st.button("📂 Categories", key="nav_categories", width='stretch'):
                    st.session_state.selected_page = "categories"
                    st.rerun()
            
            # User Management
            if permissions.get("can_manage_users", False):
                if st.button("👥 User Management", key="nav_users", width='stretch'):
                    st.session_state.selected_page = "users"
                    st.rerun()
            
            # Rules Management
            if permissions.get("can_manage_rules", False):
                if st.button("⚙️ Rules", key="nav_rules", width='stretch'):
                    st.session_state.selected_page = "rules"
                    st.rerun()
            
            # Error Resolution
            if permissions.get("can_resolve_errors", False):
                if st.button("🔧 Error Resolution", key="nav_errors", width='stretch'):
                    st.session_state.selected_page = "errors"
                    st.rerun()
            
            # Reports
            if permissions.get("can_view_reports", False):
                if st.button("📊 Reports", key="nav_reports", width='stretch'):
                    st.session_state.selected_page = "reports"
                    st.rerun()

            # Categorization upload/import
            if permissions.get("can_manage_master_data", False):
                if st.button("📥 Categorization Upload", key="nav_categorization_upload", width='stretch'):
                    st.session_state.selected_page = "categorization_upload"
                    st.rerun()
            
            # Logout button
            st.markdown("---")
            if st.button("🚪 Logout", key="logout", width='stretch'):
                auth_service.logout()
                st.rerun()
                
            logger.debug(f"Navigation completed | Data: {{'selected_page': '{st.session_state.selected_page}'}}")
            
            return st.session_state.selected_page
    
    return None

def clear_sidebar():
    """Clear all sidebar content"""
    logger.debug("Clearing sidebar for unauthenticated user")
    # This ensures the sidebar is completely empty
    with st.sidebar:
        st.empty()  # This should clear all sidebar content
    logger.debug("Sidebar cleared successfully")

def main():
    """Main application function"""
    init_app()
    
    # Initialize database - we don't need to import DatabaseManager
    logger.info("Initializing database")
    logger.info("Database initialization completed")
    
    # Check authentication status
    if not st.session_state.authenticated:
        logger.debug("User not authenticated, showing login page")
        # Important: Clear sidebar completely for unauthenticated users
        clear_sidebar()
        selected_page = "login"
        page_title = "Login"
    else:
        logger.debug("User authenticated, proceeding to navigation")
        selected_page = sidebar_navigation()
        page_title = selected_page.title() if selected_page else "Dashboard"
    
    logger.info(f"Page loaded: {page_title}")
    
    # Render selected page
    if selected_page == "login":
        from src.pages_modules.login import render_page
        logger.info("Rendering login page")
        render_page()
    
    elif selected_page == "dashboard":
        from src.pages_modules.dashboard import render_page
        logger.info("Rendering dashboard page")
        render_page()
    
    elif selected_page == "upload":
        from src.pages_modules.upload import render_page
        logger.info("Rendering upload page")
        render_page()
    
    elif selected_page == "vendors":
        from src.pages_modules.vendor_management import render_page
        logger.info("Rendering vendor management page")
        render_page()

    elif selected_page == "categories":
        from src.pages_modules.category_management import render_page
        logger.info("Rendering category management page")
        render_page()
    
    elif selected_page == "users":
        from src.pages_modules.user_management import render_page
        logger.info("Rendering user management page")
        render_page()
    
    elif selected_page == "rules":
        from src.pages_modules.rules import render_page
        logger.info("Rendering rules management page")
        render_page()
    
    elif selected_page == "errors":
        from src.pages_modules.error_management import render_page
        logger.info("Rendering error resolution page")
        render_page()
    
    elif selected_page == "reports":
        from src.pages_modules.reports import render_page
        logger.info("Rendering reports page")
        render_page()

    elif selected_page == "categorization_upload":
        from src.pages_modules.categorization_upload import render_page
        logger.info("Rendering categorization upload page")
        render_page()

if __name__ == "__main__":
    main()
