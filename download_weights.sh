#!/bin/bash
# download_weights.sh
# Download and extract ForestFormer3D weights into the correct directory, after mounting the volume.

set -e

cd "$(dirname "$0")"

mkdir -p work_dirs/clean_forestformer
wget "https://zenodo.org/records/16742708/files/clean_forestformer.zip?download=1" -O clean_forestformer.zip
unzip clean_forestformer.zip -d work_dirs
rm clean_forestformer.zip

echo "Weights downloaded and extracted to work_dirs/clean_forestformer"