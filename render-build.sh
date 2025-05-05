#!/bin/bash
# Build script for Render

# Install dependencies
npm install

# Build the application
npm run build

# Output the build directory for Render
echo "Build completed. Output directory: dist/naga-sabot/browser" 