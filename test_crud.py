"""
Test script to validate all CRUD operations
"""

from crud_operations import crud
import pandas as pd
from datetime import datetime

def test_crud_operations():
    """Test all CRUD operations"""
    print("🧪 Testing CRUD Operations...")
    
    # Test Users CRUD
    print("\n👥 Testing Users CRUD...")
    try:
        # Create user
        user_id = crud.create_user("test_user", "hashed_password", "Data Analyst")
        print(f"✅ Created user with ID: {user_id}")
        
        # Read user
        user = crud.read_user(user_id)
        print(f"✅ Read user: {user['username']}")
        
        # Update user
        updated = crud.update_user(user_id, {"role": "Spend Manager"})
        print(f"✅ Updated user: {updated}")
        
        # Read all users
        users_df = crud.read_users()
        print(f"✅ Read {len(users_df)} users")
        
        # Delete user
        deleted = crud.delete_user(user_id)
        print(f"✅ Deleted user: {deleted}")
        
    except Exception as e:
        print(f"❌ Users CRUD error: {e}")
    
    # Test Vendors CRUD
    print("\n🏢 Testing Vendors CRUD...")
    try:
        # Create vendor
        vendor_id = crud.create_vendor("Test Vendor Corp", "TEST VENDOR CORP", "V001", "Contact info")
        print(f"✅ Created vendor with ID: {vendor_id}")
        
        # Read vendor
        vendor = crud.read_vendor(vendor_id)
        print(f"✅ Read vendor: {vendor['vendor_name']}")
        
        # Update vendor
        updated = crud.update_vendor(vendor_id, {"contact_info": "Updated contact"})
        print(f"✅ Updated vendor: {updated}")
        
        # Search vendors
        search_results = crud.read_vendors(search="Test")
        print(f"✅ Search found {len(search_results)} vendors")
        
        # Delete vendor
        deleted = crud.delete_vendor(vendor_id)
        print(f"✅ Deleted vendor: {deleted}")
        
    except Exception as e:
        print(f"❌ Vendors CRUD error: {e}")
    
    # Test Categories CRUD
    print("\n📁 Testing Categories CRUD...")
    try:
        # Create parent category
        parent_id = crud.create_category("Test Parent Category", description="Parent category")
        print(f"✅ Created parent category with ID: {parent_id}")
        
        # Create child category
        child_id = crud.create_category("Test Child Category", parent_category_id=parent_id, 
                                      category_level=2, description="Child category")
        print(f"✅ Created child category with ID: {child_id}")
        
        # Read category
        category = crud.read_category(parent_id)
        print(f"✅ Read category: {category['category_name']}")
        
        # Read hierarchical categories
        hierarchical = crud.read_categories(hierarchical=True)
        print(f"✅ Read {len(hierarchical)} hierarchical categories")
        
        # Update category
        updated = crud.update_category(child_id, {"description": "Updated description"})
        print(f"✅ Updated category: {updated}")
        
        # Delete categories
        deleted_child = crud.delete_category(child_id)
        deleted_parent = crud.delete_category(parent_id)
        print(f"✅ Deleted categories: {deleted_child and deleted_parent}")
        
    except Exception as e:
        print(f"❌ Categories CRUD error: {e}")
    
    # Test Transactions CRUD
    print("\n💰 Testing Transactions CRUD...")
    try:
        # Create transaction
        transaction_data = {
            'supplier_name': 'Test Supplier',
            'item_invoice_value': 1000.00,
            'bu_code': 'TEST001',
            'region': 'Global',
            'currency_type': 'USD',
            'category': 'Test Category',
            'status': 'Active'
        }
        
        txn_id = crud.create_transaction(transaction_data)
        print(f"✅ Created transaction with ID: {txn_id}")
        
        # Read transaction
        transaction = crud.read_transaction(txn_id)
        print(f"✅ Read transaction: {transaction['supplier_name']}")
        
        # Update transaction
        updated = crud.update_transaction(txn_id, {"item_invoice_value": 1500.00})
        print(f"✅ Updated transaction: {updated}")
        
        # Read transactions with filters
        filtered = crud.read_transactions(filters={"region": "Global"}, limit=10)
        print(f"✅ Filtered transactions: {len(filtered)}")
        
        # Bulk update
        bulk_updated = crud.bulk_update_transactions([txn_id], {"status": "Inactive"})
        print(f"✅ Bulk updated {bulk_updated} transactions")
        
        # Delete transaction
        deleted = crud.delete_transaction(txn_id)
        print(f"✅ Deleted transaction: {deleted}")
        
    except Exception as e:
        print(f"❌ Transactions CRUD error: {e}")
    
    # Test Errors CRUD
    print("\n❌ Testing Errors CRUD...")
    try:
        # First create a transaction for the error
        transaction_data = {
            'supplier_name': 'Error Test Supplier',
            'item_invoice_value': -100.00,  # Negative amount for error
            'bu_code': 'ERR001',
            'region': 'Global',
            'status': 'Active'
        }
        
        txn_id = crud.create_transaction(transaction_data)
        
        # Create error
        error_id = crud.create_error(txn_id, "Negative Amount", "Transaction has negative amount")
        print(f"✅ Created error with ID: {error_id}")
        
        # Read error
        error = crud.read_error(error_id)
        print(f"✅ Read error: {error['error_type']}")
        
        # Read errors with transaction data
        errors_with_data = crud.read_errors(include_transaction_data=True)
        print(f"✅ Read {len(errors_with_data)} errors with transaction data")
        
        # Update error
        updated = crud.update_error(error_id, {"description": "Updated error description"})
        print(f"✅ Updated error: {updated}")
        
        # Resolve errors
        resolved = crud.resolve_errors([error_id], "test_user")
        print(f"✅ Resolved {resolved} errors")
        
        # Clean up
        crud.delete_error(error_id)
        crud.delete_transaction(txn_id)
        print("✅ Cleaned up test error and transaction")
        
    except Exception as e:
        print(f"❌ Errors CRUD error: {e}")
    
    # Test Rules CRUD
    print("\n📏 Testing Rules CRUD...")
    try:
        # Create rule
        rule_id = crud.create_rule(
            rule_name="Test Validation Rule",
            rule_type="Validation", 
            rule_condition="amount > 0",
            created_by="test_user",
            rule_description="Test rule for validation"
        )
        print(f"✅ Created rule with ID: {rule_id}")
        
        # Read rule
        rule = crud.read_rule(rule_id)
        print(f"✅ Read rule: {rule['rule_name']}")
        
        # Update rule
        updated = crud.update_rule(rule_id, {"rule_description": "Updated rule description"})
        print(f"✅ Updated rule: {updated}")
        
        # Toggle rule status
        toggled = crud.toggle_rule_status(rule_id)
        print(f"✅ Toggled rule status: {toggled}")
        
        # Read all rules
        rules_df = crud.read_rules(active_only=False)
        print(f"✅ Read {len(rules_df)} rules")
        
        # Delete rule
        deleted = crud.delete_rule(rule_id)
        print(f"✅ Deleted rule: {deleted}")
        
    except Exception as e:
        print(f"❌ Rules CRUD error: {e}")
    
    # Test Analytics Methods
    print("\n📊 Testing Analytics Methods...")
    try:
        # Get spend summary
        spend_summary = crud.get_spend_summary(group_by='region')
        print(f"✅ Spend summary: {len(spend_summary)} regions")
        
        # Get top suppliers
        top_suppliers = crud.get_top_suppliers(limit=5)
        print(f"✅ Top suppliers: {len(top_suppliers)} suppliers")
        
        # Get error summary
        error_summary = crud.get_error_summary()
        print(f"✅ Error summary: {len(error_summary)} error types")
        
        # Get data quality metrics
        quality_metrics = crud.get_data_quality_metrics()
        print(f"✅ Data quality metrics: {len(quality_metrics)} metrics")
        
    except Exception as e:
        print(f"❌ Analytics error: {e}")
    
    print("\n🎉 CRUD Operations Testing Complete!")

if __name__ == "__main__":
    test_crud_operations()
