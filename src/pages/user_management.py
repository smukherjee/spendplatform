"""User management page module."""
import streamlit as st
import pandas as pd
from src.utils.db_simple import get_db_connection
from src.services.auth_service import AuthService
from src.exceptions.base import ValidationError
from src.config import config


def render_page() -> None:
    """Render the user management page."""
    st.title("👥 User Management")
    st.markdown("### Manage user accounts and permissions")
    
    # Check permissions
    user_role = st.session_state.user.get('role', '')
    if user_role != 'Admin':
        st.error("❌ Access denied. Admin role required.")
        return
    
    # Load users
    users_df = load_users()
    
    # Add new user form
    with st.expander("➕ Add New User"):
        with st.form("add_user"):
            col1, col2 = st.columns(2)
            
            with col1:
                username = st.text_input("Username")
                role = st.selectbox("Role", ["Admin", "Spend Manager", "Data Analyst"])
            
            with col2:
                password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
            
            if st.form_submit_button("Create User"):
                if username and password and confirm_password:
                    if password == confirm_password:
                        try:
                            auth_service = AuthService()
                            auth_service.create_user(username, password, role)
                            st.success("✅ User created successfully!")
                            st.rerun()
                        except ValidationError as e:
                            st.error(f"❌ {e.message}")
                    else:
                        st.error("❌ Passwords do not match")
                else:
                    st.error("❌ Please fill in all required fields")
    
    # Display existing users
    if not users_df.empty:
        st.subheader(f"Existing Users ({len(users_df)})")
        
        for idx, user in users_df.iterrows():
            with st.expander(f"👤 {user['username']} ({user['role']})"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    with st.form(f"edit_user_{user['user_id']}"):
                        edit_col1, edit_col2 = st.columns(2)
                        
                        with edit_col1:
                            new_username = st.text_input("Username", value=user['username'])
                            new_role = st.selectbox(
                                "Role",
                                ["Admin", "Spend Manager", "Data Analyst"],
                                index=["Admin", "Spend Manager", "Data Analyst"].index(user['role'])
                            )
                        
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
                                        updates['password'] = new_password
                                    else:
                                        st.error("❌ Passwords do not match")
                                        continue
                                
                                try:
                                    auth_service = AuthService()
                                    auth_service.update_user(user['user_id'], updates)
                                    st.success("✅ User updated successfully!")
                                    st.rerun()
                                except ValidationError as e:
                                    st.error(f"❌ {e.message}")
                        
                        with col_delete:
                            current_username = st.session_state.get('user', {}).get('username', '')
                            if user['username'] != current_username:  # Prevent self-deletion
                                if st.form_submit_button("🗑️ Delete User"):
                                    if delete_user(user['user_id']):
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


def load_users() -> pd.DataFrame:
    """Load user data from database."""
    with get_db_connection() as conn:
        query = "SELECT * FROM users ORDER BY username"
        return pd.read_sql_query(query, conn)


def delete_user(user_id: int) -> bool:
    """Delete a user from the database."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            conn.commit()
            return cursor.rowcount > 0
    except Exception:
        return False
