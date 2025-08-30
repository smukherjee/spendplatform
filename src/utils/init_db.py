"""Database initialization and setup module."""
import sqlite3
from src.utils.crypto import hash_password as crypto_hash_password
from src.utils.debug import debug_logger
from pathlib import Path
from typing import Optional
from src.config import config
from src.exceptions.base import DatabaseError


def init_database(database_path: Optional[str] = None) -> None:
    """Initialize the SQLite database with required tables.
    
    Args:
        database_path: Optional path to database file
        
    Raises:
        DatabaseError: If database initialization fails
    """
    db_path = database_path or config.database.path
    
    try:
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(db_path, check_same_thread=False)
        cursor = conn.cursor()
        
        # Create tables
        create_tables(cursor)

        # Create default users
        create_default_users(cursor)

        conn.commit()
        conn.close()

    except sqlite3.Error as e:
        raise DatabaseError(f"Failed to initialize database: {str(e)}")


def create_tables(cursor: sqlite3.Cursor) -> None:
    """Create all required database tables."""
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')
    
    # Vendors table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vendors (
            vendor_id INTEGER PRIMARY KEY AUTOINCREMENT,
            vendor_name TEXT UNIQUE NOT NULL,
            vendor_code TEXT,
            contact_email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Categories table (use `description` column)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT NOT NULL,
            parent_category_id INTEGER,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (parent_category_id) REFERENCES categories (category_id)
        )
    ''')

    # Ensure backward-compatible schema migrations (add `description` if missing)
    try:
        cursor.execute("PRAGMA table_info(categories)")
        existing_cols = [row[1] for row in cursor.fetchall()]
        if 'description' not in existing_cols:
            debug_logger.info("Migrating categories table: adding description column")
            try:
                cursor.execute("ALTER TABLE categories ADD COLUMN description TEXT")
            except Exception as me:
                debug_logger.exception("Failed to add description column", me)
    except Exception as e:
        debug_logger.exception("Failed to check/perform categories table migration", e)
    
    # Spend transactions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS spend_transactions (
            transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            bu_code TEXT,
            bu_name TEXT,
            region TEXT,
            po_no TEXT,
            po_item_no TEXT,
            invoice_date TEXT,
            supplier_no TEXT,
            supplier_name TEXT,
            item_invoice_value REAL,
            currency_type TEXT,
            material_code TEXT,
            material_item_name TEXT,
            tower_practice TEXT,
            category TEXT,
            subcategory TEXT,
            category_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (category_id) REFERENCES categories (category_id)
        )
    ''')
    
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
    
    # User settings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_settings (
            setting_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            setting_key TEXT NOT NULL,
            setting_value TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (user_id),
            UNIQUE(user_id, setting_key)
        )
    ''')


def create_default_users(cursor: sqlite3.Cursor) -> None:
    """Create default demo users if they don't exist."""
    
    # Check if users already exist
    # Default demo users
    # Passwords follow the pattern '{username}1234' for easy demo access.
    default_password_pattern = "{username}1234"
    default_users = [
        ('admin', 'Admin'),
        ('manager', 'Spend Manager'),
        ('analyst', 'Data Analyst')
    ]

    for username, role in default_users:
        # Compute secure Argon2 hash for the demo password using pattern
        plain_password = default_password_pattern.format(username=username)
        password_hash = crypto_hash_password(plain_password)

        # Insert new user or update existing demo user to use secure hash
        cursor.execute("SELECT user_id FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        if row:
            cursor.execute(
                "UPDATE users SET password_hash = ?, role = ? WHERE username = ?",
                (password_hash, role, username)
            )
        else:
            cursor.execute(
                '''
                INSERT INTO users (username, password_hash, role)
                VALUES (?, ?, ?)
                ''',
                (username, password_hash, role)
            )


def reset_all_user_passwords(cursor: sqlite3.Cursor, pattern: str = "{username}1234") -> int:
    """Reset every user's password to the given pattern (formatted with username).

    Args:
        cursor: SQLite cursor
        pattern: A format string where '{username}' will be replaced by username

    Returns:
        Number of users updated
    """
    cursor.execute("SELECT username FROM users")
    rows = cursor.fetchall()
    updated = 0
    for (username,) in rows:
        try:
            new_plain = pattern.format(username=username)
            new_hash = crypto_hash_password(new_plain)
            cursor.execute(
                "UPDATE users SET password_hash = ? WHERE username = ?",
                (new_hash, username)
            )
            updated += 1
        except Exception as e:
            debug_logger.exception("Failed to reset password for user", e, {"username": username})
    debug_logger.info("Passwords reset for users", {"count": updated})
    return updated


def create_indexes(cursor: sqlite3.Cursor) -> None:
    """Create database indexes for better performance."""
    
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_transactions_supplier ON spend_transactions(supplier_name)",
        "CREATE INDEX IF NOT EXISTS idx_transactions_date ON spend_transactions(invoice_date)",
        "CREATE INDEX IF NOT EXISTS idx_transactions_region ON spend_transactions(region)",
        "CREATE INDEX IF NOT EXISTS idx_transactions_category ON spend_transactions(category_id)",
        "CREATE INDEX IF NOT EXISTS idx_errors_transaction ON error_logs(transaction_id)",
        "CREATE INDEX IF NOT EXISTS idx_errors_status ON error_logs(status)",
        "CREATE INDEX IF NOT EXISTS idx_vendors_name ON vendors(vendor_name)",
        "CREATE INDEX IF NOT EXISTS idx_categories_parent ON categories(parent_category_id)"
    ]
    
    for index_sql in indexes:
        cursor.execute(index_sql)


if __name__ == "__main__":
    # Initialize database when run directly
    init_database()
