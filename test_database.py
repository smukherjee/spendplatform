"""Test script to debug vendor addition issues."""
import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.db_simple import get_db_connection
from src.utils.init_db import init_database

def test_database():
    """Test database operations."""
    print("Testing database operations...")
    
    # Initialize database
    try:
        init_database()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return
    
    # Test vendor addition
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check if vendors table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vendors'")
            if cursor.fetchone():
                print("✅ Vendors table exists")
            else:
                print("❌ Vendors table does not exist")
                return
            
            # Check current vendors
            cursor.execute("SELECT COUNT(*) FROM vendors")
            count = cursor.fetchone()[0]
            print(f"📊 Current vendor count: {count}")
            
            # Add a test vendor using actual column names
            cursor.execute("""
                INSERT INTO vendors (vendor_name, supplier_no, contact_info, normalized_name, created_at)
                VALUES (?, ?, ?, ?, datetime('now'))
            """, ("Test Vendor", "TV001", "test@vendor.com", "test vendor"))
            conn.commit()
            print("✅ Test vendor added successfully")
            
            # Check count again
            cursor.execute("SELECT COUNT(*) FROM vendors")
            new_count = cursor.fetchone()[0]
            print(f"📊 New vendor count: {new_count}")
            
            # List all vendors (using actual column names)
            cursor.execute("SELECT vendor_name, supplier_no FROM vendors")
            vendors = cursor.fetchall()
            print("📋 All vendors:")
            for vendor in vendors:
                print(f"  - {vendor[0]} ({vendor[1] or 'No code'})")
                
    except Exception as e:
        print(f"❌ Database operation failed: {e}")

if __name__ == "__main__":
    test_database()
