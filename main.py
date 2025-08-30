"""
Main application entry point for the Spend Platform.
This file serves as the entry point and delegates to the refactored src module.
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.app import main

if __name__ == "__main__":
    main()
