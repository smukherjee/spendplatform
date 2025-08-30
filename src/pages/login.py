"""Login page module for user authentication."""
import streamlit as st
from src.services.auth_service import AuthService
from src.exceptions.base import AuthenticationError
from src.utils.debug import debug_logger, show_error_block, safe_execute


def render_page() -> None:
    """Render the login page."""
    try:
        debug_logger.info("Rendering login page")
        st.title("🔐 Login")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.form_submit_button("Login"):
                debug_logger.debug("Login form submitted", {"username": username})
                
                try:
                    auth_service = AuthService()
                    debug_logger.debug("AuthService initialized")
                    
                    user_info = auth_service.authenticate(username, password)
                    debug_logger.info("Authentication successful", {"username": username, "role": user_info.get('role')})
                    
                    # Set session state
                    st.session_state.authenticated = True
                    st.session_state.user = user_info
                    debug_logger.debug("Session state updated for authenticated user")
                    
                    st.success("✅ Login successful!")
                    st.rerun()
                    
                except AuthenticationError as e:
                    debug_logger.warning("Authentication failed", extra_data={"username": username, "error": str(e)})
                    st.error("❌ Invalid username or password")
                except Exception as e:
                    debug_logger.error("Unexpected error during authentication", e, {"username": username})
                    show_error_block("Login Error", e)
        
        # Demo credentials help
        with st.expander("ℹ️ Demo Credentials"):
            st.markdown("""
            Use these credentials for testing:
            
            - **Admin**: `admin` / `admin123`
            - **Spend Manager**: `manager` / `manager123`
            - **Data Analyst**: `analyst` / `analyst123`
            """)
            
    except Exception as e:
        debug_logger.error("Error rendering login page", e)
        show_error_block("Login Page Error", e)
