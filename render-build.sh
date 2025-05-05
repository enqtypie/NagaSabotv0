#!/bin/bash
set -e  # Exit on error

echo "Starting build process..."

# Install dependencies
echo "Installing dependencies..."
npm install

# Build the application
echo "Building the application..."
npm run build

# Verify the build output
if [ ! -d "dist/naga-sabot/browser" ]; then
    echo "Error: Build output directory not found!"
    exit 1
fi

echo "Build completed successfully!"
echo "Output directory: dist/naga-sabot/browser" 