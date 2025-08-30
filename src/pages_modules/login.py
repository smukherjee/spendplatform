"""Login page module for user authentication."""
import streamlit as st
from src.services.auth_service import AuthService
from src.exceptions.base import AuthenticationError
from src.utils.debug import debug_logger, show_error_block, safe_execute


def render_page() -> None:
    """Render the login page."""
    try:
        debug_logger.info("Rendering login page")
        
        # Clear sidebar completely
        st.sidebar.empty()
        with st.sidebar:
            st.markdown("### 🔒 Please Log In")
            # st.markdown("Sidebar cleared for unauthenticated users")
        
        st.title("🔐 Login")
        # st.markdown("**This is the NEW refactored application - PORT 8502**")
        # st.success("✅ SUCCESS: Sidebar navigation is hidden before authentication!")
        
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
                    # Ensure the selected page defaults to dashboard for the new user
                    st.session_state.selected_page = "dashboard"
                    st.session_state.user = user_info
                    debug_logger.debug("Session state updated for authenticated user")
                    
                    st.success("✅ Login successful!")
                    st.rerun()
                    
                except AuthenticationError as e:
                    debug_logger.warning("Authentication failed", extra_data={"username": username, "error": str(e)})
                    st.error("❌ Invalid username or password")
                except Exception as e:
                    debug_logger.exception("Unexpected error during authentication", e, {"username": username})
                    show_error_block("Login Error", e)
        
        # Demo credentials help
        with st.expander("ℹ️ Demo Credentials"):
                st.markdown("""
                The demo password policy has been updated for this environment.

                Use the pattern: `username1234` (for example, `admin` -> `admin1234`).

                - **Admin**: `admin` / `admin1234`
                - **Spend Manager**: `manager` / `manager1234`
                - **Data Analyst**: `analyst` / `analyst1234`
                """)
            
    except Exception as e:
        debug_logger.exception("Error rendering login page", e)
        show_error_block("Login Page Error", e)
