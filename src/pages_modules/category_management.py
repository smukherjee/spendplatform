"""Category management page module."""
import streamlit as st
import pandas as pd
from typing import Optional
from src.utils.db_simple import get_db_connection
from src.utils.debug import debug_logger, show_error_block, safe_execute
from src.config import config
import shutil
import time


def render_page() -> None:
    """Render category management page."""
    try:
        debug_logger.debug("Starting category management rendering")
        st.title("📂 Category Management")

    # The legacy 'Add New Category' expander was removed; use the editable table below to add categories

        # Display existing categories. Provide toggle to show soft-deleted rows.
        # Load non-deleted by default
        success, categories_df, error = safe_execute(
            load_categories, False,
            error_title="Failed to Load Categories",
            show_ui_error=True
        )

        # allow user to show soft-deleted rows
        try:
            show_deleted = st.checkbox("Show soft-deleted rows", value=False)
        except Exception:
            show_deleted = False

        if show_deleted:
            # reload with deleted included
            success, categories_df, error = safe_execute(
                load_categories, True,
                error_title="Failed to Load Categories",
                show_ui_error=True
            )

            # When showing deleted rows, expose hard-delete controls
            try:
                soft_options = []
                soft_map = {}
                if success and categories_df is not None and 'is_deleted' in categories_df.columns:
                    for _, r in categories_df.iterrows():
                        try:
                            isdel = r.get('is_deleted')
                            if pd.isna(isdel):
                                isdel = 0
                            try:
                                isdel_int = int(isdel)
                            except Exception:
                                isdel_int = 1 if str(isdel).lower() in ('true', '1', 'yes') else 0
                        except Exception:
                            isdel_int = 0

                        if isdel_int == 1:
                            cid = r.get('category_id')
                            name = r.get('category_name') or ''
                            label = f"{cid} — {name}"
                            soft_options.append(label)
                            soft_map[label] = cid

                if soft_options:
                    st.markdown("**Hard-delete soft-deleted categories**")
                    to_remove = st.multiselect("Select soft-deleted categories to permanently delete", options=soft_options, key='hard_delete_select')
                    if to_remove:
                        if st.button("Permanently delete selected"):
                            # require confirmation checkbox to avoid accidents
                            if not st.checkbox("Confirm permanent delete of selected items"):
                                st.info("Check confirmation to permanently delete selected items")
                            else:
                                db_path = config.database.path
                                timestamp = int(time.time())
                                bak = f"{db_path}.hard_delete_backup.{timestamp}.bak"
                                try:
                                    shutil.copyfile(db_path, bak)
                                except Exception as e:
                                    show_error_block("Failed to create DB backup", e)
                                    bak = None

                                try:
                                    ids = [soft_map[s] for s in to_remove]
                                    # coerce to ints
                                    ids_int = []
                                    for v in ids:
                                        try:
                                            ids_int.append(int(v))
                                        except Exception:
                                            continue

                                    if ids_int:
                                        placeholders = ','.join(['?'] * len(ids_int))
                                        with get_db_connection() as conn:
                                            cur = conn.cursor()
                                            try:
                                                conn.execute('BEGIN')
                                            except Exception:
                                                pass
                                            cur.execute(f"DELETE FROM categories WHERE category_id IN ({placeholders})", tuple(ids_int))
                                            conn.commit()

                                    st.success(f"Permanently deleted {len(ids_int)} categories")
                                    rerun_fn = getattr(st, 'experimental_rerun', None)
                                    if callable(rerun_fn):
                                        try:
                                            rerun_fn()
                                        except Exception:
                                            pass
                                except Exception as e:
                                    try:
                                        conn.rollback()
                                    except Exception:
                                        pass
                                    if bak:
                                        try:
                                            shutil.copyfile(bak, db_path)
                                            show_error_block("Failed to apply hard delete; DB restored from backup", e)
                                        except Exception as re:
                                            show_error_block("Failed to apply hard delete and failed to restore backup", re)
                                    else:
                                        show_error_block("Failed to apply hard delete", e)

                    # Danger: delete all soft-deleted rows
                    if st.button("Permanently delete ALL soft-deleted rows"):
                        if st.checkbox("Confirm permanent delete of ALL soft-deleted rows (irreversible)"):
                            db_path = config.database.path
                            timestamp = int(time.time())
                            bak = f"{db_path}.hard_delete_backup.{timestamp}.bak"
                            try:
                                shutil.copyfile(db_path, bak)
                            except Exception as e:
                                show_error_block("Failed to create DB backup", e)
                                bak = None

                            try:
                                with get_db_connection() as conn:
                                    cur = conn.cursor()
                                    try:
                                        conn.execute('BEGIN')
                                    except Exception:
                                        pass
                                    cur.execute("DELETE FROM categories WHERE COALESCE(is_deleted,0) = 1")
                                    conn.commit()
                                st.success("Permanently deleted all soft-deleted categories")
                                rerun_fn = getattr(st, 'experimental_rerun', None)
                                if callable(rerun_fn):
                                    try:
                                        rerun_fn()
                                    except Exception:
                                        pass
                            except Exception as e:
                                try:
                                    conn.rollback()
                                except Exception:
                                    pass
                                if bak:
                                    try:
                                        shutil.copyfile(bak, db_path)
                                        show_error_block("Failed to delete all; DB restored from backup", e)
                                    except Exception as re:
                                        show_error_block("Failed to delete all and failed to restore backup", re)
                                else:
                                    show_error_block("Failed to delete all soft-deleted rows", e)
                else:
                    st.info("No soft-deleted categories found")
            except Exception:
                # non-fatal: continue without hard-delete UI
                pass

        if success and categories_df is not None and not categories_df.empty:
            st.subheader(f"Existing Categories ({len(categories_df)})")
            # Normalize dataframe for Streamlit/pyarrow display to avoid dtype conversion warnings
            try:
                display_df = categories_df.copy()
                # Ensure numeric id columns use pandas nullable Int64 dtype
                if 'category_id' in display_df.columns:
                    display_df['category_id'] = pd.to_numeric(display_df['category_id'], errors='coerce').astype('Int64')
                if 'parent_category_id' in display_df.columns:
                    display_df['parent_category_id'] = pd.to_numeric(display_df['parent_category_id'], errors='coerce').astype('Int64')
                # Convert timestamps to string to avoid timezone/pyarrow issues
                if 'created_at' in display_df.columns:
                    try:
                        display_df['created_at'] = display_df['created_at'].astype(str)
                    except Exception:
                        pass
            except Exception:
                # Fallback to original dataframe if normalization fails
                display_df = categories_df

            # Build hierarchy path options for parent selection (used by quick-add)
            path_options = [(None, "None")]
            try:
                # defensive conversions
                tmp = categories_df.copy()
                if 'category_id' in tmp.columns:
                    tmp['category_id'] = pd.to_numeric(tmp['category_id'], errors='coerce').astype('Int64')
                if 'parent_category_id' in tmp.columns:
                    tmp['parent_category_id'] = pd.to_numeric(tmp['parent_category_id'], errors='coerce').astype('Int64')

                id_to_parent = {}
                id_to_name = {}

                def to_int_safe(v):
                    if pd.isna(v):
                        return None
                    try:
                        return int(v)
                    except Exception:
                        return None

                for _, r in tmp.iterrows():
                    cid = to_int_safe(r.get('category_id'))
                    if cid is None:
                        continue
                    pid = to_int_safe(r.get('parent_category_id'))
                    name = r.get('category_name') or ''
                    id_to_parent[cid] = pid
                    id_to_name[cid] = str(name)

                path_cache = {}

                def build_path(cid):
                    if cid in path_cache:
                        return path_cache[cid]
                    parts = []
                    cols_present = list(display_df.columns) if 'display_df' in locals() else []
                    # Manual add button removed; rely on data_editor's dynamic add/delete toolbar
                    left, right = st.columns([1, 2])
                    with right:
                        # right column intentionally left blank
                        pass
            except Exception:
                # non-fatal: if building path options fails, continue without hierarchical options
                pass
            # Rely on Streamlit's data_editor dynamic add/delete toolbar; no manual add button
            try:
                cols_present = list(display_df.columns)
            except Exception:
                cols_present = []

            # Use Streamlit's editable data editor so users can update rows inline.
            try:
                # Use editable data editor; attempt to make category_id read-only via column_config if supported
                try:
                    column_config = {}
                    try:
                        if 'category_id' in display_df.columns:
                            # many Streamlit versions support Column(label=..., disabled=True)
                            column_config['category_id'] = st.column_config.Column(label='ID', disabled=True)
                    except Exception:
                        # ColumnConfig not supported or signature differs; ignore
                        column_config = {}

                    # prefer any session override (e.g. new row added)
                    df_to_edit = st.session_state.get('categories_data_editor_df', display_df)
                    edited = st.data_editor(
                        df_to_edit,
                        use_container_width=True,
                        num_rows="dynamic",
                        column_config=(column_config if column_config else None),
                        key='categories_data_editor'
                    )

                    # If the user added rows via the editor, move new (null id) rows to the top and rerun
                    try:
                        if isinstance(edited, pd.DataFrame) and 'category_id' in edited.columns:
                            new_mask = edited['category_id'].isna()
                            if new_mask.any():
                                # Persist reordered editor state and force a rerun so the UI shows new rows at top
                                reordered = pd.concat([edited[new_mask], edited[~new_mask]], ignore_index=True, sort=False)
                                st.session_state['categories_data_editor'] = reordered.to_dict('records')
                                st.session_state['categories_data_editor_df'] = reordered
                                rerun_fn = getattr(st, 'experimental_rerun', None)
                                if callable(rerun_fn):
                                    try:
                                        rerun_fn()
                                    except Exception:
                                        pass
                    except Exception:
                        pass
                except Exception:
                    edited = st.data_editor(display_df, use_container_width=True, num_rows="dynamic", key='categories_data_editor')

                # Per-row validation messages: identify rows with problems and show them inline below the editor
                validation_msgs = []
                try:
                    for idx, row in edited.reset_index().iterrows():
                        row_id = row.get('category_id') if 'category_id' in row.index else None
                        name = row.get('category_name')
                        if name is None or str(name).strip() == '':
                            validation_msgs.append((idx, row_id, 'category_name is required'))
                except Exception:
                    validation_msgs = []

                if validation_msgs:
                    with st.expander(f"Validation issues ({len(validation_msgs)})"):
                        for idx, rid, msg in validation_msgs:
                            st.warning(f"Row #{idx} (id={rid}): {msg}")

                # Save changes button
                if st.button("Save changes to categories"):
                    # Determine changes by category_id
                    orig = display_df.set_index('category_id') if 'category_id' in display_df.columns else display_df
                    new = edited.set_index('category_id') if 'category_id' in edited.columns else edited

                    # helper to coerce parent ids
                    def to_parent_id(v):
                        try:
                            if pd.isna(v):
                                return None
                            return int(v)
                        except Exception:
                            return None

                    # Validation: require category_name non-empty for all edited rows
                    invalid_rows = []
                    for _, row in new.reset_index().iterrows():
                        name = row.get('category_name')
                        if name is None or str(name).strip() == '':
                            invalid_rows.append(row.get('category_id'))

                    if invalid_rows:
                        st.error(f"Validation failed: {len(invalid_rows)} row(s) have empty category_name. Please fill them before saving.")
                    else:
                        # Compute deletes: ids present in orig but missing in new
                        try:
                            orig_ids = set()
                            if 'category_id' in display_df.columns:
                                for v in display_df['category_id']:
                                    if pd.isna(v):
                                        continue
                                    try:
                                        orig_ids.add(int(v))
                                    except Exception:
                                        continue

                            new_ids = set()
                            if 'category_id' in edited.columns:
                                for v in edited['category_id']:
                                    if pd.isna(v):
                                        continue
                                    try:
                                        new_ids.add(int(v))
                                    except Exception:
                                        continue

                            to_delete = sorted(list(orig_ids - new_ids))
                        except Exception:
                            to_delete = []

                        # If there are deletions, require confirmation - we'll soft-delete instead of hard delete
                        if to_delete:
                            st.warning(f"This operation will SOFT-DELETE {len(to_delete)} existing category(ies). This can be undone by restoring a backup.")
                            if not st.button("Confirm soft-delete and apply changes"):
                                st.info("Click 'Confirm soft-delete and apply changes' to proceed.")
                                # Stop here so user can confirm
                                pass
                            else:
                                # proceed to apply changes
                                apply_now = True
                                # fallthrough to apply
                        else:
                            apply_now = True

                        if 'apply_now' in locals() and apply_now:
                            # Create DB backup
                            db_path = config.database.path
                            timestamp = int(time.time())
                            bak = f"{db_path}.edit_backup.{timestamp}.bak"
                            try:
                                shutil.copyfile(db_path, bak)
                            except Exception as e:
                                show_error_block("Failed to create DB backup", e)
                                bak = None

                            applied = 0
                            inserted = 0
                            try:
                                with get_db_connection() as conn:
                                    cur = conn.cursor()
                                    # determine if description column exists
                                    cur.execute("PRAGMA table_info(categories)")
                                    cols = [r[1] for r in cur.fetchall()]
                                    has_description = 'description' in cols
                                    # ensure soft-delete column exists
                                    if 'is_deleted' not in cols:
                                        try:
                                            cur.execute("ALTER TABLE categories ADD COLUMN is_deleted INTEGER DEFAULT 0")
                                            # refresh cols list
                                            cur.execute("PRAGMA table_info(categories)")
                                            cols = [r[1] for r in cur.fetchall()]
                                        except Exception:
                                            # if ALTER fails, continue; we'll fall back to hard delete
                                            pass

                                    # Begin transaction
                                    try:
                                        conn.execute('BEGIN')
                                    except Exception:
                                        pass

                                    # Perform deletions (soft-delete when possible)
                                    for cid in to_delete:
                                        try:
                                            if 'is_deleted' in cols:
                                                cur.execute("UPDATE categories SET is_deleted = 1 WHERE category_id = ?", (cid,))
                                            else:
                                                cur.execute("DELETE FROM categories WHERE category_id = ?", (cid,))
                                        except Exception:
                                            raise

                                    # Process inserts/updates
                                    for _, row in edited.reset_index().iterrows():
                                        cid = row.get('category_id') if 'category_id' in row.index or 'category_id' in edited.columns else None
                                        name = row.get('category_name')
                                        pid = to_parent_id(row.get('parent_category_id')) if 'parent_category_id' in row.index else None
                                        desc = row.get('description') if 'description' in row.index else None

                                        if pd.isna(cid) or cid is None:
                                            # insert
                                            # ensure inserted row is not marked deleted
                                            if has_description and 'is_deleted' in cols:
                                                cur.execute(
                                                    "INSERT INTO categories (category_name, parent_category_id, description, is_deleted, created_at) VALUES (?, ?, ?, 0, datetime('now'))",
                                                    (name, pid, desc),
                                                )
                                            elif has_description:
                                                cur.execute(
                                                    "INSERT INTO categories (category_name, parent_category_id, description, created_at) VALUES (?, ?, ?, datetime('now'))",
                                                    (name, pid, desc),
                                                )
                                            else:
                                                if 'is_deleted' in cols:
                                                    cur.execute(
                                                        "INSERT INTO categories (category_name, parent_category_id, is_deleted, created_at) VALUES (?, ?, 0, datetime('now'))",
                                                        (name, pid),
                                                    )
                                                else:
                                                    cur.execute(
                                                        "INSERT INTO categories (category_name, parent_category_id, created_at) VALUES (?, ?, datetime('now'))",
                                                        (name, pid),
                                                    )
                                            inserted += 1
                                        else:
                                            try:
                                                cid_int = int(cid)
                                            except Exception:
                                                continue
                                            # update
                                            cur.execute(
                                                "UPDATE categories SET category_name = ?, parent_category_id = ?, description = ? WHERE category_id = ?",
                                                (name, pid, desc, cid_int),
                                            )
                                            applied += 1

                                    conn.commit()

                                st.success(f"Applied {applied} updates, inserted {inserted} new categories")
                                # Rerun to refresh UI
                                rerun_fn = getattr(st, 'experimental_rerun', None)
                                if callable(rerun_fn):
                                    try:
                                        rerun_fn()
                                    except Exception:
                                        pass
                                else:
                                    try:
                                        st.rerun()
                                    except Exception:
                                        pass

                            except Exception as e:
                                # Rollback and attempt to restore backup
                                try:
                                    conn.rollback()
                                except Exception:
                                    pass
                                if bak:
                                    try:
                                        shutil.copyfile(bak, db_path)
                                        show_error_block("Failed to apply changes; DB restored from backup", e)
                                    except Exception as re:
                                        show_error_block("Failed to apply changes and failed to restore backup", re)
                                else:
                                    show_error_block("Failed to apply changes", e)

            except Exception:
                st.dataframe(display_df, use_container_width=True)
        elif success:
            st.info("No categories found. Add some categories to get started.")

    except Exception as e:
        debug_logger.exception("Error in category management rendering", e)
        show_error_block("Category Management Error", e)


def load_categories(show_deleted: bool = False) -> pd.DataFrame:
    """Load categories from DB.

    If show_deleted is False and the categories table has an `is_deleted` column,
    only return rows where is_deleted is 0 (or NULL treated as 0).
    """
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(categories)")
            cols = [r[1] for r in cur.fetchall()]
            has_is_deleted = 'is_deleted' in cols

            base = "SELECT * FROM categories"
            if not show_deleted and has_is_deleted:
                query = base + " WHERE COALESCE(is_deleted, 0) = 0 ORDER BY category_name"
            else:
                query = base + " ORDER BY category_name"

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
