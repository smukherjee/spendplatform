#!/bin/bash

# Setup script for Spend Data Management Platform

echo "🚀 Setting up Spend Data Management Platform..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install uv first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✅ uv is installed"

# Install dependencies
echo "📦 Installing dependencies..."
uv sync

# Initialize database
echo "🗄️ Initializing database..."
uv run python scripts/init_db.py

# Load sample data
echo "📊 Loading sample data..."
uv run python scripts/load_sample_data.py

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "To start the application:"
echo "   uv run streamlit run src/app.py"
echo ""
echo "Then open your browser and go to: http://localhost:8501"
echo ""
echo "Demo Credentials:"
echo "   Admin: admin / admin123"
echo "   Spend Manager: manager / manager123" 
echo "   Data Analyst: analyst / analyst123"
echo ""
