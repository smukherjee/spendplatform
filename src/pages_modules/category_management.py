"""Category management page module."""
import streamlit as st
import pandas as pd
from typing import Optional
from src.utils.db_simple import get_db_connection
from src.utils.debug import debug_logger, show_error_block, safe_execute


def render_page() -> None:
    """Render category management page."""
    try:
        debug_logger.debug("Starting category management rendering")
        st.title("📂 Category Management")

        with st.expander("➕ Add New Category"):
            with st.form("add_category"):
                category_name = st.text_input("Category Name")
                description = st.text_area("Description", height=80)

                success, parent_categories, error = safe_execute(
                    load_categories,
                    error_title="Failed to Load Categories",
                    show_ui_error=False
                )

                parent_option_tuples = [(None, "None")]
                if success and parent_categories is not None and not parent_categories.empty:
                    parent_option_tuples += list(zip(parent_categories['category_id'].tolist(), parent_categories['category_name'].tolist()))

                selected_parent = st.selectbox("Parent Category", parent_option_tuples, format_func=lambda x: x[1])
                parent_id = None
                try:
                    parent_id = selected_parent[0] if selected_parent is not None else None
                except Exception:
                    parent_id = None

                if st.form_submit_button("Add Category"):
                    if category_name:
                        success, result, error = safe_execute(
                            add_category,
                            category_name, parent_id, description,
                            error_title="Failed to Add Category",
                            show_ui_error=True
                        )

                        if success:
                            st.success("✅ Category added successfully!")
                            rerun_fn = getattr(st, 'experimental_rerun', None)
                            if callable(rerun_fn):
                                rerun_fn()
                            else:
                                try:
                                    st.rerun()
                                except Exception:
                                    pass
                    else:
                        st.error("❌ Category name is required")

        # Display existing categories
        success, categories_df, error = safe_execute(
            load_categories,
            error_title="Failed to Load Categories",
            show_ui_error=True
        )

        if success and categories_df is not None and not categories_df.empty:
            st.subheader(f"Existing Categories ({len(categories_df)})")
            st.dataframe(categories_df, use_container_width=True)
        elif success:
            st.info("No categories found. Add some categories to get started.")

    except Exception as e:
        debug_logger.exception("Error in category management rendering", e)
        show_error_block("Category Management Error", e)


def load_categories() -> pd.DataFrame:
    try:
        with get_db_connection() as conn:
            query = "SELECT * FROM categories ORDER BY category_name"
            df = pd.read_sql_query(query, conn)
            return df
    except Exception as e:
        debug_logger.exception("Error loading categories", e)
        raise Exception(f"Database error loading categories: {str(e)}")


def add_category(name: str, parent_id: Optional[int] = None, description: Optional[str] = None) -> int:
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Build insert dynamically in case older DBs don't have description
            try:
                cursor.execute("PRAGMA table_info(categories)")
                cols = [r[1] for r in cursor.fetchall()]
                if 'description' in cols:
                    cursor.execute(
                        "INSERT INTO categories (category_name, parent_category_id, description, created_at) VALUES (?, ?, ?, datetime('now'))",
                        (name, parent_id, description),
                    )
                else:
                    cursor.execute(
                        "INSERT INTO categories (category_name, parent_category_id, created_at) VALUES (?, ?, datetime('now'))",
                        (name, parent_id),
                    )

            except Exception:
                # Fallback to basic insert
                cursor.execute(
                    "INSERT INTO categories (category_name, parent_category_id, created_at) VALUES (?, ?, datetime('now'))",
                    (name, parent_id),
                )

            conn.commit()
            cid = cursor.lastrowid
            return int(cid) if cid is not None else 0
    except Exception as e:
        debug_logger.exception("Error adding category", e, {"name": name, "parent_id": parent_id})
        if "UNIQUE constraint failed" in str(e):
            raise Exception(f"Category '{name}' already exists")
        else:
            raise Exception(f"Database error: {str(e)}")
