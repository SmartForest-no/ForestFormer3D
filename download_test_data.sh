#!/bin/bash
# download_test_data.sh
# Download and extract ForestFormer3D test data into the correct directory, after mounting the volume.

set -e

cd "$(dirname "$0")"

mkdir -p data/ForAINetV2/test_data
wget "https://zenodo.org/records/16742708/files/test_data.zip?download=1" -O test_data.zip
unzip test_data.zip -d data/ForAINetV2
rm test_data.zip

mkdir -p data/ForAINetV2/train_val_data
wget "https://zenodo.org/records/16742708/files/train_val_data.zip?download=1" -O train_val_data.zip
unzip train_val_data.zip -d data/ForAINetV2
rm train_val_data.zip

echo "Test data downloaded and extracted to data/ForAINetV2/test_data"