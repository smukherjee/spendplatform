"""Categorization upload and import page.

Allows uploading a categorization CSV/XLSX matching the context/Categorization File.csv
and importing hierarchical categories into the database.
"""
import streamlit as st
import pandas as pd
import os
from typing import Optional, Tuple, List
from src.utils.db_simple import get_db_connection
from src.utils.debug import debug_logger, show_error_block
from src.utils.display import normalize_df_for_display


def parse_categorization_df(df: pd.DataFrame) -> List[List[str]]:
    """Return list of category paths (list of levels) for each row."""
    paths = []
    # Expect columns like 'Category 1','Category 2', ...
    level_cols = [c for c in df.columns if c.lower().startswith('category')]
    for _, row in df.iterrows():
        path = []
        for col in level_cols:
            val = row.get(col)
            if pd.isna(val) or str(val).strip() == '':
                continue
            path.append(str(val).strip())
        if path:
            paths.append(path)
    return paths


def build_unique_paths(paths: List[List[str]]) -> List[List[str]]:
    """Return unique paths preserving order."""
    seen = set()
    unique = []
    for p in paths:
        key = '||'.join(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def import_paths_to_db(paths: List[List[str]]) -> Tuple[int, int]:
    """Import hierarchical paths into categories table.

    Returns (created_count, total_processed)
    """
    created = 0
    processed = 0
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Cache existing categories by (parent_id, name) -> id
            cache = {}
            cursor.execute("SELECT category_id, category_name, parent_category_id FROM categories")
            for r in cursor.fetchall():
                cid, cname, pid = r
                cache[(pid if pid is not None else None, cname)] = cid

            for path in paths:
                parent_id = None
                processed += 1
                for level_name in path:
                    key = (parent_id, level_name)
                    if key in cache:
                        parent_id = cache[key]
                        continue
                    # create
                    cursor.execute(
                        "INSERT INTO categories (category_name, parent_category_id, created_at) VALUES (?, ?, datetime('now'))",
                        (level_name, parent_id),
                    )
                    new_id = cursor.lastrowid
                    conn.commit()
                    cache[key] = new_id
                    parent_id = new_id
                    created += 1

        return created, processed
    except Exception as e:
        debug_logger.exception("Failed importing categorization paths", e)
        raise


def render_page() -> None:
    st.title("📥 Categorization Upload")
    st.markdown("Upload a categorization CSV/XLSX (format: Category 1..Category N). You can also load the sample file included in the project context.")

    use_sample = st.checkbox("Use sample categorization file from repo context", value=False)
    uploaded_file = None
    if use_sample:
        sample_path = os.path.join('context', 'Categorization File.csv')
        if os.path.exists(sample_path):
            try:
                df = pd.read_csv(sample_path)
                st.success(f"Loaded sample file: {sample_path}")
            except Exception as e:
                show_error_block("Failed to load sample file", e)
                return
        else:
            st.error("Sample file not found in context/")
            df = None
    else:
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xls', 'xlsx'])
        df = None
        if uploaded_file is not None:
            try:
                if uploaded_file.name.lower().endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.success(f"Loaded file: {uploaded_file.name}")
            except Exception as e:
                show_error_block("Failed to parse uploaded file", e)
                return

    if df is None:
        st.info("Choose a sample or upload a file to proceed.")
        return

    st.subheader("Preview (first 10 rows)")
    try:
        st.dataframe(normalize_df_for_display(df.head(10)), width='stretch')
    except Exception:
        st.dataframe(df.head(10), width='stretch')

    paths = parse_categorization_df(df)
    unique_paths = build_unique_paths(paths)

    st.subheader("Parsed category paths (first 50)")
    for p in unique_paths[:50]:
        st.text(' > '.join(p))

    if st.button("Import parsed categories into DB"):
        try:
            created, processed = import_paths_to_db(unique_paths)
            st.success(f"Imported categories: created {created}, processed {processed} paths")
        except Exception as e:
            show_error_block("Failed to import categories", e)
