"""
Database initialization script
"""

import sqlite3

def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect('spend_platform.db')
    cursor = conn.cursor()
    
    print("Initializing database tables...")
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'Data Analyst',
            last_login TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    print("✅ Users table created")
    
    # Vendors table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vendors (
            vendor_id INTEGER PRIMARY KEY AUTOINCREMENT,
            vendor_name TEXT NOT NULL,
            normalized_name TEXT,
            supplier_no TEXT,
            contact_info TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    print("✅ Vendors table created")
    
    # Categories table (hierarchical)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT NOT NULL,
            parent_category_id INTEGER,
            category_level INTEGER DEFAULT 1,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (parent_category_id) REFERENCES categories (category_id)
        )
    ''')
    print("✅ Categories table created")
    
    # Spend transactions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS spend_transactions (
            transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            bu_code TEXT,
            bu_name TEXT,
            region TEXT,
            po_no TEXT,
            po_item_no TEXT,
            invoice_date DATE,
            supplier_no TEXT,
            supplier_name TEXT,
            invoice_amount REAL,
            currency_type TEXT,
            unit_of_purch TEXT,
            item_invoice_value REAL,
            material_code TEXT,
            material_item_name TEXT,
            order_description TEXT,
            inv_item_desc TEXT,
            po_price REAL,
            po_purchase_qty REAL,
            unit_price REAL,
            tower_practice TEXT,
            category TEXT,
            subcategory TEXT,
            vendor_id INTEGER,
            category_id INTEGER,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'Active',
            FOREIGN KEY (vendor_id) REFERENCES vendors (vendor_id),
            FOREIGN KEY (category_id) REFERENCES categories (category_id)
        )
    ''')
    print("✅ Spend transactions table created")
    
    # Error logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS error_logs (
            error_id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id INTEGER,
            error_type TEXT NOT NULL,
            description TEXT,
            status TEXT DEFAULT 'Open',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved_at TIMESTAMP,
            resolved_by TEXT,
            FOREIGN KEY (transaction_id) REFERENCES spend_transactions (transaction_id)
        )
    ''')
    print("✅ Error logs table created")
    
    # Rules table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rules (
            rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
            rule_name TEXT NOT NULL,
            rule_description TEXT,
            rule_type TEXT,
            rule_condition TEXT,
            active_flag BOOLEAN DEFAULT TRUE,
            created_by TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    print("✅ Rules table created")
    
    # User settings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_settings (
            setting_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            setting_name TEXT NOT NULL,
            setting_value TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    print("✅ User settings table created")
    
    conn.commit()
    conn.close()
    
    print("\n🎉 Database initialization completed successfully!")

if __name__ == "__main__":
    init_database()
