# Hybrid Classification System v0.6 Configuration

# Model Configuration
MODEL_ACCURACY = {
    'l2': 0.8028,  # Production SVM Enhanced model accuracy
    'l4': 0.5469,  # L4 Random Forest model accuracy  
    'l5': 0.5030   # L5 SVM model accuracy
}

# Classification Thresholds
CONFIDENCE_THRESHOLDS = {
    'high': 0.75,      # High confidence threshold (use L2 directly)
    'medium': 0.50,    # Medium confidence threshold (try enhancement)
    'l5_override': 0.65 # L5 confidence threshold for category override
}

# Enhancement Configuration
ENHANCEMENT_CONFIG = {
    'minimum_improvement': 0.05,  # Minimum improvement required for enhancement
    'prefer_l5_over_l4': True,    # Prefer L5 enhancement over L4 when both available
    'enable_category_override': True,  # Enable L5-based category override
    'enable_l4_l2_mapping': True  # Enable L4→L2 mapping for L4 enhancements
}

# File Paths
MODEL_PATHS = {
    'l2_model': 'models/production_svm_enhanced.pkl',
    'l4_model': 'models/l4_simplified_model.pkl', 
    'l5_model': 'models/l5_simplified_model.pkl',
    'taxonomy_file': 'data/Categorization File.xlsx'
}

# Taxonomy Configuration
TAXONOMY_CONFIG = {
    'sheet_name': 'Taxonomy',
    'l2_column': 'Category 2',
    'l4_column': 'Category 4', 
    'l5_column': 'Category 5'
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'batch_size': 1000,        # Default batch size for processing
    'max_description_length': 500,  # Maximum description length
    'enable_caching': False,   # Enable prediction caching (for future implementation)
    'log_predictions': True    # Log prediction details for monitoring
}

# API Configuration
API_CONFIG = {
    'default_timeout': 30,     # Default API timeout in seconds
    'max_batch_size': 5000,    # Maximum batch size for API requests
    'enable_metadata': True,   # Include metadata in API responses
    'enable_health_check': True # Enable health check endpoint
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'logs/classification.log',
    'max_size': '10MB',
    'backup_count': 5
}

# Version Information
VERSION = '0.6'
BUILD_DATE = '2025-09-09'
DESCRIPTION = 'Production Hybrid Classification System with L5-based L2 Category Override'
