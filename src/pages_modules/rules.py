"""Rules management page module."""
import streamlit as st
import pandas as pd
from src.utils.db_simple import get_db_connection
from src.utils.debug import debug_logger, show_error_block


def render_page() -> None:
    """Render the rules management page."""
    try:
        debug_logger.info("Rendering rules management page")
        st.title("⚙️ Rules Management")
        st.markdown("### Create, edit, and manage business rules used for validation and categorization")

        # Permission check: allow Admin and Spend Manager
        user_role = st.session_state.get('user', {}).get('role', '')
        if user_role not in ("Admin", "Spend Manager"):
            st.error("❌ Access denied. Admin or Spend Manager role required.")
            return

        # Tabs: List / Add
        tab_list, tab_add = st.tabs(["📜 Existing Rules", "➕ Add Rule"])

        with tab_add:
            render_add_rule()

        with tab_list:
            render_rule_list()

    except Exception as e:
        debug_logger.exception("Error rendering rules page", e)
        show_error_block("Rules Page Error", e)


def render_add_rule():
    """Render Add Rule form."""
    st.subheader("➕ Create a New Rule")
    with st.form("add_rule_form"):
        col1, col2 = st.columns([3, 1])
        with col1:
            name = st.text_input("Rule Name")
            rule_type = st.selectbox("Rule Type", ["Validation", "Categorization", "Transformation", "Other"])
            condition = st.text_area("Rule Condition / Expression", height=120)
            description = st.text_area("Description", height=80)
        with col2:
            active = st.checkbox("Active", value=True)
        if st.form_submit_button("Create Rule"):
            if not name:
                st.error("Please provide a rule name")
                return

            created_by = st.session_state.get('user', {}).get('username', 'system')
            try:
                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO rules (rule_name, rule_description, rule_type, rule_condition, active_flag, created_by) VALUES (?, ?, ?, ?, ?, ?)",
                        (name or '', description or '', rule_type or '', condition or '', 1 if active else 0, created_by)
                    )
                    conn.commit()
                st.success("✅ Rule created successfully")
                rerun_fn = getattr(st, 'experimental_rerun', None)
                if callable(rerun_fn):
                    rerun_fn()
            except Exception as e:
                debug_logger.exception("Failed to create rule", e)
                st.error(f"Failed to create rule: {e}")


def render_rule_list():
    """Display and edit existing rules."""
    st.subheader("📜 Existing Rules")
    try:
        df = load_rules()
    except Exception as e:
        debug_logger.exception("Failed to load rules", e)
        st.error(f"Failed to load rules: {e}")
        return

    if df.empty:
        st.info("No rules found. Add a rule using the tab on the right.")
        return

    # Render each rule in an expander with an edit form
    for _, row in df.iterrows():
        # Guard each row render so one bad row doesn't break the whole page
        try:
            rule_id = int(row['rule_id'])
            with st.expander(f"{row.get('rule_name', '')}  —  {row.get('rule_type', '')} (ID: {rule_id})"):
                with st.form(f"edit_rule_{rule_id}"):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        name = st.text_input("Rule Name", value=row.get('rule_name', '') )
                        # Make the index lookup safe in case the stored value is unexpected
                        rule_options = ["Validation", "Categorization", "Transformation", "Other"]
                        try:
                            default_type = row.get('rule_type') or 'Validation'
                            type_index = rule_options.index(default_type) if default_type in rule_options else 0
                        except Exception:
                            type_index = 0
                        rule_type = st.selectbox("Rule Type", rule_options, index=type_index)
                        condition = st.text_area("Condition / Expression", value=row.get('rule_condition', '') or '', height=120)
                        description = st.text_area("Description", value=row.get('rule_description', '') or '', height=80)
                    with col2:
                        active = st.checkbox("Active", value=bool(row.get('active_flag', 1)))

                    col_update, col_delete = st.columns([1, 1])
                    with col_update:
                        submitted_update = st.form_submit_button("💾 Update Rule")
                        if submitted_update:
                            try:
                                update_rule(
                                    rule_id,
                                    (name or ''),
                                    (description or ''),
                                    (rule_type or ''),
                                    (condition or ''),
                                    1 if active else 0
                                )
                                st.success("✅ Rule updated")
                                rerun_fn = getattr(st, 'experimental_rerun', None)
                                if callable(rerun_fn):
                                    rerun_fn()
                            except Exception as e:
                                debug_logger.exception("Failed to update rule", e)
                                st.error(f"Failed to update rule: {e}")

                    with col_delete:
                        submitted_delete = st.form_submit_button("🗑️ Delete Rule")
                        if submitted_delete:
                            if delete_rule(rule_id):
                                st.success("✅ Rule deleted")
                                rerun_fn = getattr(st, 'experimental_rerun', None)
                                if callable(rerun_fn):
                                    rerun_fn()
                            else:
                                st.error("Failed to delete rule")
        except Exception as e:
            debug_logger.exception("Failed rendering rule row", e)
            st.error(f"Failed to render rule ID {row.get('rule_id')}: {e}")


def load_rules() -> pd.DataFrame:
    """Load rules from the database ordered by creation time."""
    with get_db_connection() as conn:
        query = "SELECT * FROM rules ORDER BY created_at DESC"
        return pd.read_sql_query(query, conn)


def update_rule(rule_id: int, name: str, description: str, rule_type: str, condition: str, active_flag: int) -> None:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE rules SET rule_name = ?, rule_description = ?, rule_type = ?, rule_condition = ?, active_flag = ? WHERE rule_id = ?",
            (name, description, rule_type, condition, active_flag, rule_id)
        )
        conn.commit()


def delete_rule(rule_id: int) -> bool:
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM rules WHERE rule_id = ?", (rule_id,))
            conn.commit()
            return cursor.rowcount > 0
    except Exception as e:
        debug_logger.exception("Failed to delete rule", e)
        return False
