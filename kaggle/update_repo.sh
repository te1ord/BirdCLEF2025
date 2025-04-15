#!/bin/bash

# Define variables
DATA_DIR="data/kaggle_datasets/birdclef2025_code"
MAIN_FOLDER="$DATA_DIR/main_folder"
ZIP_FILE="$DATA_DIR/main_folder.zip"
CODE_BASE="src"
TOML_FILE="pyproject.toml"
LOCK_FILE="poetry.lock"
KAGGLE_MSG="new code version"

# Clean up any existing files and folders
rm -rf "$ZIP_FILE"
rm -rf "$MAIN_FOLDER"

# Create the main folder
mkdir -p "$MAIN_FOLDER"

# Copy necessary files and directories
cp -r "$CODE_BASE" "$MAIN_FOLDER/"
cp "$TOML_FILE" "$MAIN_FOLDER/"
cp "$LOCK_FILE" "$MAIN_FOLDER/"

# Navigate to the target directory
cd "$DATA_DIR" || exit

# Zip the main folder
zip -r "$(basename "$ZIP_FILE")" "$(basename "$MAIN_FOLDER")"

# Remove the unzipped folder
rm -rf main_folder

# Upload the new version to Kaggle
kaggle datasets version -p ./ -m "$KAGGLE_MSG"