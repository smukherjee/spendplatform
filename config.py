"""
Configuration settings for the Spend Platform application
"""

import os
from datetime import timedelta

class Config:
    # Database settings
    DATABASE_PATH = 'spend_platform.db'
    
    # Application settings
    APP_TITLE = "Spend Data Management Platform"
    APP_ICON = "💰"
    
    # Authentication settings
    SESSION_TIMEOUT = timedelta(hours=8)
    
    # File upload settings
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS = ['.xlsx', '.xls', '.csv']
    
    # Dashboard settings
    DEFAULT_CHART_HEIGHT = 400
    RECENT_TRANSACTIONS_LIMIT = 10
    
    # Pagination settings
    DEFAULT_RECORDS_PER_PAGE = 15
    PAGINATION_OPTIONS = [10, 15, 25, 50, 100]
    
    # Error management
    ERROR_TYPES = [
        'Negative Amount',
        'Missing Supplier',
        'Invalid Date',
        'Duplicate Transaction',
        'Missing Category',
        'Invalid Currency',
        'Missing BU Code',
        'Data Quality Issue'
    ]
    
    # User roles and permissions
    USER_ROLES = {
        'Admin': {
            'can_upload': True,
            'can_manage_master_data': True,
            'can_resolve_errors': True,
            'can_view_all_data': True,
            'can_export': True
        },
        'Spend Manager': {
            'can_upload': True,
            'can_manage_master_data': False,
            'can_resolve_errors': True,
            'can_view_all_data': True,
            'can_export': True
        },
        'Data Analyst': {
            'can_upload': False,
            'can_manage_master_data': False,
            'can_resolve_errors': False,
            'can_view_all_data': True,
            'can_export': True
        }
    }
    
    # Demo credentials
    DEMO_USERS = {
        'admin': {'password': 'admin123', 'role': 'Admin'},
        'manager': {'password': 'manager123', 'role': 'Spend Manager'},
        'analyst': {'password': 'analyst123', 'role': 'Data Analyst'}
    }
    
    # Data validation rules
    VALIDATION_RULES = {
        'required_fields': [
            'supplier_name',
            'item_invoice_value',
            'invoice_date',
            'bu_code'
        ],
        'numeric_fields': [
            'item_invoice_value',
            'invoice_amount',
            'po_price',
            'unit_price'
        ],
        'date_fields': [
            'invoice_date',
            'po_creation_date'
        ]
    }
    
    # Chart colors
    CHART_COLORS = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

# Environment-specific configurations
class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    DEBUG = False
    TESTING = False
    DATABASE_PATH = os.environ.get('DATABASE_PATH', 'spend_platform.db')

class TestingConfig(Config):
    DEBUG = True
    TESTING = True
    DATABASE_PATH = ':memory:'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
