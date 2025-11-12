#!/bin/bash
set -e

# Navigate to the script's directory to ensure all relative paths work correctly.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"

# Directories for the output files
DIR_ORIG="output_orig"
DIR_FAST="output_fast"

# Path to the batch processing and comparison scripts
BATCH_LOADER="batch_load_ForAINetV2_data.py"
COMPARATOR="compare_outputs.py"

echo "--- Starting data generation with original loader ---"
python $BATCH_LOADER --loader orig --output_folder $DIR_ORIG
echo "--- Original data generation complete ---"

echo ""

echo "--- Starting data generation with fast loader ---"
python $BATCH_LOADER --loader fast --output_folder $DIR_FAST
echo "--- Fast data generation complete ---"

echo ""

echo "--- Comparing outputs ---"
python $COMPARATOR $DIR_ORIG $DIR_FAST --verbose
echo "--- Comparison complete ---"
