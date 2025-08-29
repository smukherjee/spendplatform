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
    
    # ===================== USERS CRUD =====================
    
    def create_user(self, username: str, password_hash: str, role: str) -> int:
        """Create a new user"""
        conn = self.get_connection()
               return grouped
        finally:
            conn.close()

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

# Global instance
crud = SpendPlatformCRUD()
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
            df = pd.read_sql_query(
                "SELECT * FROM users WHERE user_id = ?", 
                conn, params=(user_id,)
            )
            return df.to_dict('records')[0] if not df.empty else None
        finally:
            conn.close()
    
    def read_users(self, filters: Dict = None) -> pd.DataFrame:
        """Read all users with optional filters"""
        conn = self.get_connection()
        try:
            query = "SELECT * FROM users"
            params = []
            
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(f"{key} = ?")
                    params.append(value)
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
            
            return pd.read_sql_query(query, conn, params=params)
        finally:
            conn.close()
    
    def update_user(self, user_id: int, updates: Dict) -> bool:
        """Update user information"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            set_clauses = []
            params = []
            
            for key, value in updates.items():
                set_clauses.append(f"{key} = ?")
                params.append(value)
            
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
                     supplier_no: Optional[str] = None, contact_info: Optional[str] = None) -> int:
        """Create a new vendor"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO vendors (vendor_name, normalized_name, supplier_no, contact_info)
                VALUES (?, ?, ?, ?)
            ''', (vendor_name, normalized_name, supplier_no, contact_info))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()
    
    def read_vendor(self, vendor_id: int) -> Dict:
        """Read a single vendor by ID"""
        conn = self.get_connection()
        try:
            df = pd.read_sql_query(
                "SELECT * FROM vendors WHERE vendor_id = ?", 
                conn, params=(vendor_id,)
            )
            return df.to_dict('records')[0] if not df.empty else None
        finally:
            conn.close()
    
    def read_vendors(self, filters: Dict = None, search: str = None) -> pd.DataFrame:
        """Read all vendors with optional filters and search"""
        conn = self.get_connection()
        try:
            query = "SELECT * FROM vendors"
            params = []
            conditions = []
            
            if filters:
                for key, value in filters.items():
                    conditions.append(f"{key} = ?")
                    params.append(value)
            
            if search:
                conditions.append("(vendor_name LIKE ? OR normalized_name LIKE ?)")
                params.extend([f"%{search}%", f"%{search}%"])
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY vendor_name"
            
            return pd.read_sql_query(query, conn, params=params)
        finally:
            conn.close()
    
    def update_vendor(self, vendor_id: int, updates: Dict) -> bool:
        """Update vendor information"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            updates['updated_at'] = datetime.now()
            
            set_clauses = []
            params = []
            
            for key, value in updates.items():
                set_clauses.append(f"{key} = ?")
                params.append(value)
            
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
                       category_level: int = 1, description: Optional[str] = None) -> int:
        """Create a new category"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO categories (category_name, parent_category_id, category_level, description)
                VALUES (?, ?, ?, ?)
            ''', (category_name, parent_category_id, category_level, description))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()
    
    def read_category(self, category_id: int) -> Dict:
        """Read a single category by ID"""
        conn = self.get_connection()
        try:
            df = pd.read_sql_query(
                "SELECT * FROM categories WHERE category_id = ?", 
                conn, params=(category_id,)
            )
            return df.to_dict('records')[0] if not df.empty else None
        finally:
            conn.close()
    
    def read_categories(self, filters: Dict = None, hierarchical: bool = False) -> pd.DataFrame:
        """Read all categories with optional filters"""
        conn = self.get_connection()
        try:
            if hierarchical:
                query = '''
                    SELECT c.*, p.category_name as parent_name
                    FROM categories c
                    LEFT JOIN categories p ON c.parent_category_id = p.category_id
                    ORDER BY c.category_level, c.category_name
                '''
                params = []
            else:
                query = "SELECT * FROM categories"
                params = []
                
                if filters:
                    conditions = []
                    for key, value in filters.items():
                        conditions.append(f"{key} = ?")
                        params.append(value)
                    if conditions:
                        query += " WHERE " + " AND ".join(conditions)
                
                query += " ORDER BY category_name"
            
            return pd.read_sql_query(query, conn, params=params)
        finally:
            conn.close()
    
    def update_category(self, category_id: int, updates: Dict) -> bool:
        """Update category information"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            set_clauses = []
            params = []
            
            for key, value in updates.items():
                set_clauses.append(f"{key} = ?")
                params.append(value)
            
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
        """Delete a category (and update children to orphaned)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            # First update children to remove parent reference
            cursor.execute(
                "UPDATE categories SET parent_category_id = NULL WHERE parent_category_id = ?",
                (category_id,)
            )
            
            # Then delete the category
            cursor.execute("DELETE FROM categories WHERE category_id = ?", (category_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    # ===================== SPEND TRANSACTIONS CRUD =====================
    
    def create_transaction(self, transaction_data: Dict) -> int:
        """Create a new spend transaction"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            columns = list(transaction_data.keys())
            placeholders = ['?' for _ in columns]
            values = list(transaction_data.values())
            
            cursor.execute(f'''
                INSERT INTO spend_transactions ({", ".join(columns)})
                VALUES ({", ".join(placeholders)})
            ''', values)
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()
    
    def read_transaction(self, transaction_id: int) -> Dict:
        """Read a single transaction by ID"""
        conn = self.get_connection()
        try:
            df = pd.read_sql_query(
                "SELECT * FROM spend_transactions WHERE transaction_id = ?", 
                conn, params=(transaction_id,)
            )
            return df.to_dict('records')[0] if not df.empty else None
        finally:
            conn.close()
    
    def read_transactions(self, filters: Dict = None, date_range: tuple = None,
                         limit: int = None, offset: int = None) -> pd.DataFrame:
        """Read transactions with comprehensive filtering"""
        conn = self.get_connection()
        try:
            query = "SELECT * FROM spend_transactions"
            params = []
            conditions = []
            
            if filters:
                for key, value in filters.items():
                    if isinstance(value, list):
                        placeholders = ','.join(['?' for _ in value])
                        conditions.append(f"{key} IN ({placeholders})")
                        params.extend(value)
                    else:
                        conditions.append(f"{key} = ?")
                        params.append(value)
            
            if date_range:
                conditions.append("invoice_date BETWEEN ? AND ?")
                params.extend(date_range)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY upload_date DESC"
            
            if limit:
                query += f" LIMIT {limit}"
                if offset:
                    query += f" OFFSET {offset}"
            
            return pd.read_sql_query(query, conn, params=params)
        finally:
            conn.close()
    
    def update_transaction(self, transaction_id: int, updates: Dict) -> bool:
        """Update transaction information"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            set_clauses = []
            params = []
            
            for key, value in updates.items():
                set_clauses.append(f"{key} = ?")
                params.append(value)
            
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
    
    def bulk_update_transactions(self, transaction_ids: List[int], updates: Dict) -> int:
        """Bulk update multiple transactions"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            set_clauses = []
            params = []
            
            for key, value in updates.items():
                set_clauses.append(f"{key} = ?")
                params.append(value)
            
            placeholders = ','.join(['?' for _ in transaction_ids])
            params.extend(transaction_ids)
            
            cursor.execute(f'''
                UPDATE spend_transactions 
                SET {", ".join(set_clauses)}
                WHERE transaction_id IN ({placeholders})
            ''', params)
            
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()
    
    # ===================== ERROR LOGS CRUD =====================
    
    def create_error(self, transaction_id: int, error_type: str, description: str) -> int:
        """Create a new error log"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO error_logs (transaction_id, error_type, description)
                VALUES (?, ?, ?)
            ''', (transaction_id, error_type, description))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()
    
    def read_error(self, error_id: int) -> Dict:
        """Read a single error by ID"""
        conn = self.get_connection()
        try:
            df = pd.read_sql_query(
                "SELECT * FROM error_logs WHERE error_id = ?", 
                conn, params=(error_id,)
            )
            return df.to_dict('records')[0] if not df.empty else None
        finally:
            conn.close()
    
    def read_errors(self, filters: Dict = None, include_transaction_data: bool = False) -> pd.DataFrame:
        """Read all errors with optional filters"""
        conn = self.get_connection()
        try:
            if include_transaction_data:
                query = '''
                    SELECT e.*, t.supplier_name, t.item_invoice_value, t.bu_code
                    FROM error_logs e
                    LEFT JOIN spend_transactions t ON e.transaction_id = t.transaction_id
                '''
            else:
                query = "SELECT * FROM error_logs"
            
            params = []
            
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(f"e.{key} = ?" if include_transaction_data else f"{key} = ?")
                    params.append(value)
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY created_at DESC"
            
            return pd.read_sql_query(query, conn, params=params)
        finally:
            conn.close()
    
    def update_error(self, error_id: int, updates: Dict) -> bool:
        """Update error information"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            set_clauses = []
            params = []
            
            for key, value in updates.items():
                set_clauses.append(f"{key} = ?")
                params.append(value)
            
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
    
    def delete_error(self, error_id: int) -> bool:
        """Delete an error log"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM error_logs WHERE error_id = ?", (error_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    def resolve_errors(self, error_ids: List[int], resolved_by: str) -> int:
        """Bulk resolve multiple errors"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            placeholders = ','.join(['?' for _ in error_ids])
            cursor.execute(f'''
                UPDATE error_logs 
                SET status = 'Resolved', resolved_at = CURRENT_TIMESTAMP, resolved_by = ?
                WHERE error_id IN ({placeholders})
            ''', [resolved_by] + error_ids)
            
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()
    
    # ===================== RULES CRUD =====================
    
    def create_rule(self, rule_name: str, rule_description: str, rule_type: str,
                   rule_condition: str, created_by: str) -> int:
        """Create a new rule"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO rules (rule_name, rule_description, rule_type, rule_condition, created_by)
                VALUES (?, ?, ?, ?, ?)
            ''', (rule_name, rule_description, rule_type, rule_condition, created_by))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()
    
    def read_rule(self, rule_id: int) -> Dict:
        """Read a single rule by ID"""
        conn = self.get_connection()
        try:
            df = pd.read_sql_query(
                "SELECT * FROM rules WHERE rule_id = ?", 
                conn, params=(rule_id,)
            )
            return df.to_dict('records')[0] if not df.empty else None
        finally:
            conn.close()
    
    def read_rules(self, filters: Dict = None, active_only: bool = True) -> pd.DataFrame:
        """Read all rules with optional filters"""
        conn = self.get_connection()
        try:
            query = "SELECT * FROM rules"
            params = []
            conditions = []
            
            if active_only:
                conditions.append("active_flag = 1")
            
            if filters:
                for key, value in filters.items():
                    conditions.append(f"{key} = ?")
                    params.append(value)
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY created_at DESC"
            
            return pd.read_sql_query(query, conn, params=params)
        finally:
            conn.close()
    
    def update_rule(self, rule_id: int, updates: Dict) -> bool:
        """Update rule information"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            set_clauses = []
            params = []
            
            for key, value in updates.items():
                set_clauses.append(f"{key} = ?")
                params.append(value)
            
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
            cursor.execute('''
                UPDATE rules 
                SET active_flag = CASE WHEN active_flag = 1 THEN 0 ELSE 1 END
                WHERE rule_id = ?
            ''', (rule_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()
    
    # ===================== ANALYTICS & REPORTING METHODS =====================
    
    def get_spend_summary(self, date_range: tuple = None, group_by: str = 'region') -> pd.DataFrame:
        """Get spend summary grouped by specified field"""
        conn = self.get_connection()
        try:
            query = f'''
                SELECT {group_by}, 
                       COUNT(*) as transaction_count,
                       SUM(item_invoice_value) as total_spend,
                       AVG(item_invoice_value) as avg_spend,
                       COUNT(DISTINCT supplier_name) as unique_suppliers
                FROM spend_transactions
            '''
            params = []
            
            if date_range:
                query += " WHERE invoice_date BETWEEN ? AND ?"
                params.extend(date_range)
            
            query += f" GROUP BY {group_by} ORDER BY total_spend DESC"
            
            return pd.read_sql_query(query, conn, params=params)
        finally:
            conn.close()
    
    def get_top_suppliers(self, limit: int = 10, date_range: tuple = None) -> pd.DataFrame:
        """Get top suppliers by spend"""
        conn = self.get_connection()
        try:
            query = '''
                SELECT supplier_name,
                       COUNT(*) as transaction_count,
                       SUM(item_invoice_value) as total_spend,
                       AVG(item_invoice_value) as avg_spend
                FROM spend_transactions
            '''
            params = []
            
            if date_range:
                query += " WHERE invoice_date BETWEEN ? AND ?"
                params.extend(date_range)
            
            query += f" GROUP BY supplier_name ORDER BY total_spend DESC LIMIT {limit}"
            
            return pd.read_sql_query(query, conn, params=params)
        finally:
            conn.close()
    
    def get_error_summary(self) -> pd.DataFrame:
        """Get error summary statistics"""
        conn = self.get_connection()
        try:
            query = '''
                SELECT error_type,
                       COUNT(*) as error_count,
                       SUM(CASE WHEN status = 'Open' THEN 1 ELSE 0 END) as open_errors,
                       SUM(CASE WHEN status = 'Resolved' THEN 1 ELSE 0 END) as resolved_errors
                FROM error_logs
                GROUP BY error_type
                ORDER BY error_count DESC
            '''
            
            return pd.read_sql_query(query, conn)
        finally:
            conn.close()
    
    def get_data_quality_metrics(self) -> Dict:
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
