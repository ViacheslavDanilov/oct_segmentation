#!/bin/bash

# Define variables
ENV_FILE="environment.yaml"

# Change to the script's directory or exit if unsuccessful
cd "$(dirname "$0")" || { echo "Unable to change directory. Exiting."; exit 1; }

# Get environment name
ENV_NAME=$(awk '/name:/ {print $2}' "$ENV_FILE")

# Create environment
conda env create --file "${ENV_FILE}" --verbose

# Activate environment and install the project as a package
conda run -n "${ENV_NAME}" pip install -e .

# Informative message
echo "Environment setup complete"
