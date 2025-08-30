"""
Spend Platform - Main Application Entry Point
Streamlit-based spend data management application
"""
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
            if st.button("🏠 Dashboard", key="nav_dashboard", use_container_width=True):
                st.session_state.selected_page = "dashboard"
                st.rerun()
            
            # Data Upload
            if permissions.get("can_upload_data", False):
                if st.button("📤 Upload Data", key="nav_upload", use_container_width=True):
                    st.session_state.selected_page = "upload"
                    st.rerun()
            
            # Master Data Management
            if permissions.get("can_manage_master_data", False):
                if st.button("📋 Master Data", key="nav_master_data", use_container_width=True):
                    st.session_state.selected_page = "master_data"
                    st.rerun()
                    logger.debug("Master data page selected")
            
            # User Management
            if permissions.get("can_manage_users", False):
                if st.button("👥 User Management", key="nav_users", use_container_width=True):
                    st.session_state.selected_page = "users"
                    st.rerun()
            
            # Rules Management
            if permissions.get("can_manage_rules", False):
                if st.button("⚙️ Rules", key="nav_rules", use_container_width=True):
                    st.session_state.selected_page = "rules"
                    st.rerun()
            
            # Error Resolution
            if permissions.get("can_resolve_errors", False):
                if st.button("🔧 Error Resolution", key="nav_errors", use_container_width=True):
                    st.session_state.selected_page = "errors"
                    st.rerun()
            
            # Reports
            if permissions.get("can_view_reports", False):
                if st.button("📊 Reports", key="nav_reports", use_container_width=True):
                    st.session_state.selected_page = "reports"
                    st.rerun()
            
            # Logout button
            st.markdown("---")
            if st.button("🚪 Logout", key="logout", use_container_width=True):
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
    
    elif selected_page == "master_data":
        from src.pages_modules.master_data import render_page
        logger.info("Rendering master data management page")
        render_page()
    
    elif selected_page == "users":
        # TODO: Implement users page
        # st.info("User management page coming soon!")
        from src.pages_modules.user_management import render_page
        logger.info("Rendering user management page")
        render_page()
    
    elif selected_page == "rules":
        # TODO: Implement rules page
        st.info("Rules management page coming soon!")
        # from src.pages_modules.rules import render_rules_page
        # logger.info("Rendering rules management page")
        # render_rules_page()
    
    elif selected_page == "errors":
        # TODO: Implement errors page
        # st.info("Error resolution page coming soon!")
        from src.pages_modules.error_management import render_page
        logger.info("Rendering error resolution page")
        render_page()
    
    elif selected_page == "reports":
        # TODO: Implement reports page
        # st.info("Reports page coming soon!")
        from src.pages_modules.reports import render_page
        logger.info("Rendering reports page")
        render_page()

if __name__ == "__main__":
    main()
