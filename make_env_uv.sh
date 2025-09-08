#!/bin/bash

# Script to set up the environment using UV instead of Conda
# This replaces the conda-based make_env.sh

# Change to the script's directory or exit if unsuccessful
cd "$(dirname "$0")" || { echo "Unable to change directory. Exiting."; exit 1; }

echo "Setting up environment with UV..."

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "UV is not installed. Installing UV..."
    pip install uv
fi

# Create a virtual environment with the required Python version
echo "Creating virtual environment with Python 3.11..."
uv venv --python 3.11 .venv

# Activate the virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install the project dependencies
echo "Installing dependencies..."
uv pip install -e .

# Install system dependencies note
echo ""
echo "⚠️  IMPORTANT: System dependencies required:"
echo "   - FFmpeg (for video processing)"
echo "   - On Ubuntu/Debian: sudo apt-get install ffmpeg"
echo "   - On macOS: brew install ffmpeg"
echo "   - On Windows: Download from https://ffmpeg.org/download.html"
echo ""

# Install ffmpeg-python separately (pip package)
echo "Installing ffmpeg-python..."
uv pip install ffmpeg-python==0.2.0

echo ""
echo "✅ Environment setup complete!"
echo "To activate the environment in the future, run:"
echo "   source .venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "   deactivate"