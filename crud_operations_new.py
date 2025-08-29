"""
CRUD operations for all database tables in the Spend Platform
"""

import sqlite3
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
from config import Config

class SpendPlatformCRUD:
    """Comprehensive CRUD operations for all database tables"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or Config.DATABASE_PATH
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    # ===================== USER SETTINGS CRUD =====================
    
    def get_user_setting(self, user_id: int, setting_name: str, default_value: str = None) -> str:
        """Get a user setting value"""
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT setting_value FROM user_settings 
                WHERE user_id = ? AND setting_name = ?
            ''', (user_id, setting_name))
            result = cursor.fetchone()
            return result[0] if result else default_value
        finally:
            conn.close()
    
    def set_user_setting(self, user_id: int, setting_name: str, setting_value: str) -> bool:
        """Set or update a user setting"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            # Check if setting exists
            cursor.execute('''
                SELECT setting_id FROM user_settings 
                WHERE user_id = ? AND setting_name = ?
            ''', (user_id, setting_name))
            
            if cursor.fetchone():
                # Update existing setting
                cursor.execute('''
                    UPDATE user_settings 
                    SET setting_value = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ? AND setting_name = ?
                ''', (setting_value, user_id, setting_name))
            else:
                # Insert new setting
                cursor.execute('''
                    INSERT INTO user_settings (user_id, setting_name, setting_value)
                    VALUES (?, ?, ?)
                ''', (user_id, setting_name, setting_value))
            
            conn.commit()
            return True
        except Exception as e:
            print(f"Error setting user preference: {e}")
            return False
        finally:
            conn.close()
    
    def get_user_settings(self, user_id: int) -> Dict:
        """Get all settings for a user"""
        conn = self.get_connection()
        try:
            df = pd.read_sql_query('''
                SELECT setting_name, setting_value FROM user_settings 
                WHERE user_id = ?
            ''', conn, params=(user_id,))
            
            return dict(zip(df['setting_name'], df['setting_value'])) if not df.empty else {}
        finally:
            conn.close()
    
    # ===================== USERS CRUD =====================
    
    def create_user(self, username: str, password_hash: str, role: str) -> int:
        """Create a new user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO users (username, password_hash, role)
                VALUES (?, ?, ?)
            ''', (username, password_hash, role))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()
    
    def read_user(self, user_id: int) -> Dict:
        """Read a single user by ID"""
        conn = self.get_connection()
        try:
            df = pd.read_sql_query('''
                SELECT * FROM users WHERE user_id = ?
            ''', conn, params=(user_id,))
            
            return df.iloc[0].to_dict() if not df.empty else {}
        finally:
            conn.close()
    
    def read_users(self, filters: Dict = None) -> pd.DataFrame:
        """Read multiple users with optional filters"""
        conn = self.get_connection()
        try:
            where_clause = "1=1"
            params = []
            
            if filters:
                conditions = []
                for key, value in filters.items():
                    if value is not None:
                        conditions.append(f"{key} = ?")
                        params.append(value)
                if conditions:
                    where_clause = " AND ".join(conditions)
            
            df = pd.read_sql_query(f'''
                SELECT * FROM users WHERE {where_clause}
                ORDER BY created_at DESC
            ''', conn, params=params)
            
            return df
        finally:
            conn.close()
    
    def update_user(self, user_id: int, updates: Dict) -> bool:
        """Update a user"""
        if not updates:
            return False
        
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            set_clauses = []
            params = []
            
            for key, value in updates.items():
                if key != 'user_id':  # Prevent updating the ID
                    set_clauses.append(f"{key} = ?")
                    params.append(value)
            
            if not set_clauses:
                return False
            
            params.append(user_id)
            cursor.execute(f'''
                UPDATE users 
                SET {", ".join(set_clauses)}
                WHERE user_id = ?
            ''', params)
            
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def delete_user(self, user_id: int) -> bool:
        """Delete a user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    # ===================== VENDORS CRUD =====================
    
    def create_vendor(self, vendor_name: str, normalized_name: Optional[str] = None,
                     vendor_type: Optional[str] = None) -> int:
        """Create a new vendor"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO vendors (vendor_name, normalized_name, vendor_type)
                VALUES (?, ?, ?)
            ''', (vendor_name, normalized_name, vendor_type))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()
    
    def read_vendor(self, vendor_id: int) -> Dict:
        """Read a single vendor by ID"""
        conn = self.get_connection()
        try:
            df = pd.read_sql_query('''
                SELECT * FROM vendors WHERE vendor_id = ?
            ''', conn, params=(vendor_id,))
            
            return df.iloc[0].to_dict() if not df.empty else {}
        finally:
            conn.close()
    
    def read_vendors(self, filters: Dict = None, search: str = None) -> pd.DataFrame:
        """Read multiple vendors with optional filters and search"""
        conn = self.get_connection()
        try:
            where_clause = "1=1"
            params = []
            
            if search:
                where_clause += " AND (vendor_name LIKE ? OR normalized_name LIKE ?)"
                search_term = f"%{search}%"
                params.extend([search_term, search_term])
            
            if filters:
                for key, value in filters.items():
                    if value is not None:
                        where_clause += f" AND {key} = ?"
                        params.append(value)
            
            df = pd.read_sql_query(f'''
                SELECT * FROM vendors 
                WHERE {where_clause}
                ORDER BY vendor_name
            ''', conn, params=params)
            
            return df
        finally:
            conn.close()
    
    def update_vendor(self, vendor_id: int, updates: Dict) -> bool:
        """Update a vendor"""
        if not updates:
            return False
        
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            set_clauses = []
            params = []
            
            for key, value in updates.items():
                if key != 'vendor_id':  # Prevent updating the ID
                    set_clauses.append(f"{key} = ?")
                    params.append(value)
            
            if not set_clauses:
                return False
            
            # Add updated_at timestamp
            set_clauses.append("updated_at = CURRENT_TIMESTAMP")
            params.append(vendor_id)
            
            cursor.execute(f'''
                UPDATE vendors 
                SET {", ".join(set_clauses)}
                WHERE vendor_id = ?
            ''', params)
            
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def delete_vendor(self, vendor_id: int) -> bool:
        """Delete a vendor"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM vendors WHERE vendor_id = ?", (vendor_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    # ===================== CATEGORIES CRUD =====================
    
    def create_category(self, category_name: str, parent_category_id: Optional[int] = None,
                       category_type: Optional[str] = None) -> int:
        """Create a new category"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO categories (category_name, parent_category_id, category_type)
                VALUES (?, ?, ?)
            ''', (category_name, parent_category_id, category_type))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()
    
    def read_category(self, category_id: int) -> Dict:
        """Read a single category by ID"""
        conn = self.get_connection()
        try:
            df = pd.read_sql_query('''
                SELECT * FROM categories WHERE category_id = ?
            ''', conn, params=(category_id,))
            
            return df.iloc[0].to_dict() if not df.empty else {}
        finally:
            conn.close()
    
    def read_categories(self, filters: Dict = None, hierarchical: bool = False) -> pd.DataFrame:
        """Read multiple categories with optional filters"""
        conn = self.get_connection()
        try:
            where_clause = "1=1"
            params = []
            
            if filters:
                for key, value in filters.items():
                    if value is not None:
                        where_clause += f" AND {key} = ?"
                        params.append(value)
            
            query = f'''
                SELECT c1.*, 
                       c2.category_name as parent_category_name
                FROM categories c1
                LEFT JOIN categories c2 ON c1.parent_category_id = c2.category_id
                WHERE {where_clause}
                ORDER BY c1.category_name
            '''
            
            df = pd.read_sql_query(query, conn, params=params)
            
            return df
        finally:
            conn.close()
    
    def update_category(self, category_id: int, updates: Dict) -> bool:
        """Update a category"""
        if not updates:
            return False
        
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            set_clauses = []
            params = []
            
            for key, value in updates.items():
                if key != 'category_id':  # Prevent updating the ID
                    set_clauses.append(f"{key} = ?")
                    params.append(value)
            
            if not set_clauses:
                return False
            
            # Add updated_at timestamp
            set_clauses.append("updated_at = CURRENT_TIMESTAMP")
            params.append(category_id)
            
            cursor.execute(f'''
                UPDATE categories 
                SET {", ".join(set_clauses)}
                WHERE category_id = ?
            ''', params)
            
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def delete_category(self, category_id: int) -> bool:
        """Delete a category"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM categories WHERE category_id = ?", (category_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    # ===================== TRANSACTIONS CRUD =====================
    
    def create_transaction(self, data: Dict) -> int:
        """Create a new spend transaction"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            columns = ', '.join(data.keys())
            placeholders = ', '.join(['?' for _ in data.keys()])
            values = list(data.values())
            
            cursor.execute(f'''
                INSERT INTO spend_transactions ({columns})
                VALUES ({placeholders})
            ''', values)
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()
    
    def read_transaction(self, transaction_id: int) -> Dict:
        """Read a single transaction by ID"""
        conn = self.get_connection()
        try:
            df = pd.read_sql_query('''
                SELECT * FROM spend_transactions WHERE transaction_id = ?
            ''', conn, params=(transaction_id,))
            
            return df.iloc[0].to_dict() if not df.empty else {}
        finally:
            conn.close()
    
    def read_transactions(self, filters: Dict = None, limit: int = None) -> pd.DataFrame:
        """Read multiple transactions with optional filters and limit"""
        conn = self.get_connection()
        try:
            where_clause = "1=1"
            params = []
            
            if filters:
                for key, value in filters.items():
                    if value is not None:
                        if key in ['start_date', 'end_date']:
                            if key == 'start_date':
                                where_clause += " AND invoice_date >= ?"
                            else:
                                where_clause += " AND invoice_date <= ?"
                        else:
                            where_clause += f" AND {key} = ?"
                        params.append(value)
            
            limit_clause = f"LIMIT {limit}" if limit else ""
            
            df = pd.read_sql_query(f'''
                SELECT * FROM spend_transactions 
                WHERE {where_clause}
                ORDER BY invoice_date DESC
                {limit_clause}
            ''', conn, params=params)
            
            return df
        finally:
            conn.close()
    
    def update_transaction(self, transaction_id: int, updates: Dict) -> bool:
        """Update a transaction"""
        if not updates:
            return False
        
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            set_clauses = []
            params = []
            
            for key, value in updates.items():
                if key != 'transaction_id':  # Prevent updating the ID
                    set_clauses.append(f"{key} = ?")
                    params.append(value)
            
            if not set_clauses:
                return False
            
            # Add updated_at timestamp
            set_clauses.append("updated_at = CURRENT_TIMESTAMP")
            params.append(transaction_id)
            
            cursor.execute(f'''
                UPDATE spend_transactions 
                SET {", ".join(set_clauses)}
                WHERE transaction_id = ?
            ''', params)
            
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def delete_transaction(self, transaction_id: int) -> bool:
        """Delete a transaction"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM spend_transactions WHERE transaction_id = ?", (transaction_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def bulk_create_transactions(self, transactions: List[Dict]) -> List[int]:
        """Create multiple transactions in bulk"""
        if not transactions:
            return []
        
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            # Assume all transactions have the same structure
            columns = list(transactions[0].keys())
            placeholders = ', '.join(['?' for _ in columns])
            columns_str = ', '.join(columns)
            
            transaction_ids = []
            for data in transactions:
                values = [data.get(col) for col in columns]
                cursor.execute(f'''
                    INSERT INTO spend_transactions ({columns_str})
                    VALUES ({placeholders})
                ''', values)
                transaction_ids.append(cursor.lastrowid)
            
            conn.commit()
            return transaction_ids
        finally:
            conn.close()
    
    # ===================== ERROR LOGS CRUD =====================
    
    def create_error_log(self, error_type: str, error_description: str, 
                        transaction_id: Optional[int] = None, severity: str = 'Medium') -> int:
        """Create a new error log"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO error_logs (error_type, error_description, transaction_id, severity)
                VALUES (?, ?, ?, ?)
            ''', (error_type, error_description, transaction_id, severity))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()
    
    def read_error_log(self, error_id: int) -> Dict:
        """Read a single error log by ID"""
        conn = self.get_connection()
        try:
            df = pd.read_sql_query('''
                SELECT * FROM error_logs WHERE error_id = ?
            ''', conn, params=(error_id,))
            
            return df.iloc[0].to_dict() if not df.empty else {}
        finally:
            conn.close()
    
    def read_error_logs(self, filters: Dict = None) -> pd.DataFrame:
        """Read multiple error logs with optional filters"""
        conn = self.get_connection()
        try:
            where_clause = "1=1"
            params = []
            
            if filters:
                for key, value in filters.items():
                    if value is not None:
                        where_clause += f" AND {key} = ?"
                        params.append(value)
            
            df = pd.read_sql_query(f'''
                SELECT e.*, 
                       COALESCE(t.supplier_name, 'N/A') as supplier_name,
                       COALESCE(t.item_invoice_value, 0) as amount
                FROM error_logs e
                LEFT JOIN spend_transactions t ON e.transaction_id = t.transaction_id
                WHERE {where_clause}
                ORDER BY e.created_at DESC
            ''', conn, params=params)
            
            return df
        finally:
            conn.close()
    
    def update_error_log(self, error_id: int, updates: Dict) -> bool:
        """Update an error log"""
        if not updates:
            return False
        
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            set_clauses = []
            params = []
            
            for key, value in updates.items():
                if key != 'error_id':  # Prevent updating the ID
                    set_clauses.append(f"{key} = ?")
                    params.append(value)
            
            if not set_clauses:
                return False
            
            # Add updated_at timestamp
            set_clauses.append("updated_at = CURRENT_TIMESTAMP")
            params.append(error_id)
            
            cursor.execute(f'''
                UPDATE error_logs 
                SET {", ".join(set_clauses)}
                WHERE error_id = ?
            ''', params)
            
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def delete_error_log(self, error_id: int) -> bool:
        """Delete an error log"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM error_logs WHERE error_id = ?", (error_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def resolve_error(self, error_id: int, resolution_notes: str = None) -> bool:
        """Mark an error as resolved"""
        updates = {
            'status': 'Resolved',
            'resolution_notes': resolution_notes,
            'resolved_at': datetime.now().isoformat()
        }
        return self.update_error_log(error_id, updates)
    
    # ===================== RULES CRUD =====================
    
    def create_rule(self, rule_name: str, rule_type: str, conditions: str, 
                   actions: str, description: Optional[str] = None) -> int:
        """Create a new rule"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO rules (rule_name, rule_type, conditions, actions, description)
                VALUES (?, ?, ?, ?, ?)
            ''', (rule_name, rule_type, conditions, actions, description))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()
    
    def read_rule(self, rule_id: int) -> Dict:
        """Read a single rule by ID"""
        conn = self.get_connection()
        try:
            df = pd.read_sql_query('''
                SELECT * FROM rules WHERE rule_id = ?
            ''', conn, params=(rule_id,))
            
            return df.iloc[0].to_dict() if not df.empty else {}
        finally:
            conn.close()
    
    def read_rules(self, filters: Dict = None) -> pd.DataFrame:
        """Read multiple rules with optional filters"""
        conn = self.get_connection()
        try:
            where_clause = "1=1"
            params = []
            
            if filters:
                for key, value in filters.items():
                    if value is not None:
                        where_clause += f" AND {key} = ?"
                        params.append(value)
            
            df = pd.read_sql_query(f'''
                SELECT * FROM rules 
                WHERE {where_clause}
                ORDER BY rule_name
            ''', conn, params=params)
            
            return df
        finally:
            conn.close()
    
    def update_rule(self, rule_id: int, updates: Dict) -> bool:
        """Update a rule"""
        if not updates:
            return False
        
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            set_clauses = []
            params = []
            
            for key, value in updates.items():
                if key != 'rule_id':  # Prevent updating the ID
                    set_clauses.append(f"{key} = ?")
                    params.append(value)
            
            if not set_clauses:
                return False
            
            # Add updated_at timestamp
            set_clauses.append("updated_at = CURRENT_TIMESTAMP")
            params.append(rule_id)
            
            cursor.execute(f'''
                UPDATE rules 
                SET {", ".join(set_clauses)}
                WHERE rule_id = ?
            ''', params)
            
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def delete_rule(self, rule_id: int) -> bool:
        """Delete a rule"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM rules WHERE rule_id = ?", (rule_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def toggle_rule_status(self, rule_id: int) -> bool:
        """Toggle rule active/inactive status"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            # Get current status
            cursor.execute("SELECT is_active FROM rules WHERE rule_id = ?", (rule_id,))
            current_status = cursor.fetchone()
            
            if current_status:
                new_status = not bool(current_status[0])
                cursor.execute('''
                    UPDATE rules 
                    SET is_active = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE rule_id = ?
                ''', (new_status, rule_id))
                conn.commit()
                return True
            return False
        finally:
            conn.close()
    
    # ===================== ANALYTICS & REPORTING =====================
    
    def get_spend_summary(self, date_range: tuple = None) -> Dict:
        """Get spend summary analytics"""
        conn = self.get_connection()
        try:
            where_clause = "1=1"
            params = []
            
            if date_range:
                where_clause += " AND invoice_date BETWEEN ? AND ?"
                params.extend(date_range)
            
            query = f'''
                SELECT 
                    COUNT(*) as total_transactions,
                    SUM(item_invoice_value) as total_spend,
                    AVG(item_invoice_value) as avg_transaction_value,
                    COUNT(DISTINCT supplier_name) as unique_suppliers,
                    COUNT(DISTINCT bu_code) as unique_business_units
                FROM spend_transactions
                WHERE {where_clause}
            '''
            
            df = pd.read_sql_query(query, conn, params=params)
            return df.iloc[0].to_dict() if not df.empty else {}
        finally:
            conn.close()
    
    def get_top_suppliers(self, limit: int = 10, date_range: tuple = None) -> pd.DataFrame:
        """Get top suppliers by spend"""
        conn = self.get_connection()
        try:
            where_clause = "1=1"
            params = []
            
            if date_range:
                where_clause += " AND invoice_date BETWEEN ? AND ?"
                params.extend(date_range)
            
            query = f'''
                SELECT 
                    supplier_name,
                    SUM(item_invoice_value) as total_spend,
                    COUNT(*) as transaction_count,
                    AVG(item_invoice_value) as avg_transaction_value
                FROM spend_transactions
                WHERE {where_clause}
                GROUP BY supplier_name
                ORDER BY total_spend DESC
                LIMIT {limit}
            '''
            
            return pd.read_sql_query(query, conn, params=params)
        finally:
            conn.close()
    
    def get_spend_by_category(self, date_range: tuple = None) -> pd.DataFrame:
        """Get spend breakdown by category"""
        conn = self.get_connection()
        try:
            where_clause = "1=1"
            params = []
            
            if date_range:
                where_clause += " AND t.invoice_date BETWEEN ? AND ?"
                params.extend(date_range)
            
            query = f'''
                SELECT 
                    COALESCE(c.category_name, 'Uncategorized') as category,
                    SUM(t.item_invoice_value) as total_spend,
                    COUNT(t.transaction_id) as transaction_count
                FROM spend_transactions t
                LEFT JOIN categories c ON t.category_id = c.category_id
                WHERE {where_clause}
                GROUP BY c.category_name
                ORDER BY total_spend DESC
            '''
            
            return pd.read_sql_query(query, conn, params=params)
        finally:
            conn.close()
    
    def get_monthly_trend(self) -> pd.DataFrame:
        """Get monthly spend trend"""
        conn = self.get_connection()
        try:
            query = '''
                SELECT 
                    strftime('%Y-%m', invoice_date) as month,
                    SUM(item_invoice_value) as monthly_spend,
                    COUNT(*) as transaction_count
                FROM spend_transactions
                WHERE invoice_date IS NOT NULL
                GROUP BY strftime('%Y-%m', invoice_date)
                ORDER BY month
            '''
            
            return pd.read_sql_query(query, conn)
        finally:
            conn.close()
    
    def get_business_unit_analysis(self) -> pd.DataFrame:
        """Get spend analysis by business unit"""
        conn = self.get_connection()
        try:
            query = '''
                SELECT 
                    bu_code,
                    SUM(item_invoice_value) as total_spend,
                    COUNT(*) as transaction_count,
                    AVG(item_invoice_value) as avg_transaction_value,
                    COUNT(DISTINCT supplier_name) as unique_suppliers
                FROM spend_transactions
                WHERE bu_code IS NOT NULL AND bu_code != ''
                GROUP BY bu_code
                ORDER BY total_spend DESC
            '''
            
            return pd.read_sql_query(query, conn)
        finally:
            conn.close()
    
    def get_duplicate_analysis(self) -> pd.DataFrame:
        """Identify potential duplicate transactions"""
        conn = self.get_connection()
        try:
            query = '''
                SELECT 
                    supplier_name,
                    item_invoice_value,
                    invoice_date,
                    COUNT(*) as duplicate_count,
                    GROUP_CONCAT(transaction_id) as transaction_ids
                FROM spend_transactions
                WHERE supplier_name IS NOT NULL 
                  AND item_invoice_value IS NOT NULL 
                  AND invoice_date IS NOT NULL
                GROUP BY supplier_name, item_invoice_value, invoice_date
                HAVING COUNT(*) > 1
                ORDER BY duplicate_count DESC
            '''
            
            return pd.read_sql_query(query, conn)
        finally:
            conn.close()
    
    def get_data_quality_report(self) -> Dict:
        """Get comprehensive data quality metrics"""
        conn = self.get_connection()
        try:
            metrics = {}
            
            # Total records
            df = pd.read_sql_query("SELECT COUNT(*) as total FROM spend_transactions", conn)
            metrics['total_transactions'] = df.iloc[0]['total']
            
            # Missing data
            missing_query = '''
                SELECT 
                    SUM(CASE WHEN supplier_name IS NULL OR supplier_name = '' THEN 1 ELSE 0 END) as missing_suppliers,
                    SUM(CASE WHEN item_invoice_value IS NULL THEN 1 ELSE 0 END) as missing_amounts,
                    SUM(CASE WHEN invoice_date IS NULL THEN 1 ELSE 0 END) as missing_dates,
                    SUM(CASE WHEN bu_code IS NULL OR bu_code = '' THEN 1 ELSE 0 END) as missing_bu_codes
                FROM spend_transactions
            '''
            df = pd.read_sql_query(missing_query, conn)
            metrics.update(df.iloc[0].to_dict())
            
            # Error statistics
            error_query = '''
                SELECT 
                    COUNT(*) as total_errors,
                    SUM(CASE WHEN status = 'Open' THEN 1 ELSE 0 END) as open_errors
                FROM error_logs
            '''
            df = pd.read_sql_query(error_query, conn)
            metrics.update(df.iloc[0].to_dict())
            
            return metrics
        finally:
            conn.close()

# Global instance
crud = SpendPlatformCRUD()
