"""Vendor management page module."""
import streamlit as st
import pandas as pd
from typing import Optional
from src.utils.db_simple import get_db_connection
from src.utils.debug import debug_logger, show_error_block, safe_execute


def render_page() -> None:
    """Render the vendor management page."""
    try:
        debug_logger.debug("Starting vendor management rendering")
        st.title("📦 Vendor Management")

        # Add new vendor form
        st.subheader("➕ Add New Vendor")
        with st.form("add_vendor"):
            vendor_name = st.text_input("Vendor Name")
            vendor_code = st.text_input("Vendor Code")
            contact_email = st.text_input("Contact Email")

            submit_button = st.form_submit_button("Add Vendor")
            if submit_button:
                if vendor_name and vendor_name.strip():
                    success, result, error = safe_execute(
                        lambda **kwargs: add_vendor(
                            vendor_name.strip(),
                            vendor_code.strip() if vendor_code else None,
                            contact_email.strip() if contact_email else None
                        )
                    )

                    if success and result:
                        st.success(f"✅ Vendor '{vendor_name}' added successfully!")
                    else:
                        error_msg = str(error) if error else "Failed to add vendor"
                        st.error(f"❌ Error adding vendor: {error_msg}")
                else:
                    st.error("❌ Please enter a vendor name")

        # Display existing vendors
        success, vendors_df, error = safe_execute(
            load_vendors,
            error_title="Failed to Load Vendors",
            show_ui_error=True
        )

        if success and vendors_df is not None and not vendors_df.empty:
            st.subheader(f"Existing Vendors ({len(vendors_df)})")
            st.dataframe(vendors_df, use_container_width=True)
        elif success:
            st.info("No vendors found. Add some vendors to get started.")

    except Exception as e:
        debug_logger.exception("Error in vendor management rendering", e)
        show_error_block("Vendor Management Error", e)


def load_vendors() -> pd.DataFrame:
    """Load vendors from database."""
    try:
        with get_db_connection() as conn:
            df = pd.read_sql_query(
                """
                SELECT vendor_id, vendor_name, supplier_no, contact_info, created_at, updated_at
                FROM vendors
                ORDER BY created_at DESC
                """,
                conn,
            )
            return df
    except Exception as e:
        debug_logger.exception("Error loading vendors", e)
        raise Exception(f"Database error loading vendors: {str(e)}")


def add_vendor(name: str, code: Optional[str] = None, email: Optional[str] = None) -> int:
    """Add a new vendor to the database."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT vendor_id FROM vendors WHERE normalized_name = ?", (name.lower().strip(),))
            existing = cursor.fetchone()
            if existing:
                raise Exception(f"Vendor '{name}' already exists")

            cursor.execute(
                """
                INSERT INTO vendors (vendor_name, supplier_no, contact_info, normalized_name, created_at)
                VALUES (?, ?, ?, ?, datetime('now'))
                """,
                (name, code, email, name.lower().strip()),
            )
            vendor_id = cursor.lastrowid
            conn.commit()
            return int(vendor_id) if vendor_id is not None else 0
    except Exception as e:
        debug_logger.exception("Error adding vendor to database", e, {"name": name, "code": code})
        if "UNIQUE constraint failed" in str(e):
            raise Exception(f"Vendor '{name}' already exists")
        else:
            raise Exception(f"Database error: {str(e)}")
