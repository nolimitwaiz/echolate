#!/bin/bash

# Echo Web Application Startup Script
# This script ensures reliable startup and handles common issues

set -e  # Exit on any error

echo "🚀 Starting Echo Web Application..."

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed or not in PATH"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "launch_web.py" ]; then
    echo "❌ launch_web.py not found. Are you in the Echo directory?"
    exit 1
fi

# Install dependencies if needed (skip if already installed)
echo "📦 Checking dependencies..."
python -c "import gradio" 2>/dev/null || {
    echo "Installing dependencies..."
    pip install -r requirements.txt
}

# Kill any existing instances
echo "🔄 Stopping any existing instances..."
pkill -f "launch_web.py" 2>/dev/null || true

# Wait a moment for processes to stop
sleep 2

# Start the application
echo "🌐 Launching Echo Web Interface..."
echo "📍 Server will be available at: http://127.0.0.1:7861"
echo "🔧 Press Ctrl+C to stop the server"
echo ""

python launch_web.py