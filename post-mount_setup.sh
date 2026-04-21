#!/bin/bash
# post-mount_setup.sh
# This script fixes some installation issues, it bundles fixes proposed in the original ForestFormer3D repository.
set -e

cd "$(dirname "$0")"

pip uninstall torch-points-kernels -y
pip install --no-deps --no-cache-dir torch-points-kernels==0.7.0

pip uninstall torch-cluster
pip install torch-cluster --no-cache-dir --no-deps

pip show mmengine
cp replace_mmdetection_files/loops.py /opt/conda/lib/python3.10/site-packages/mmengine/runner/
cp replace_mmdetection_files/base_model.py /opt/conda/lib/python3.10/site-packages/mmengine/model/base_model/
cp replace_mmdetection_files/transforms_3d.py /opt/conda/lib/python3.10/site-packages/mmdet3d/datasets/transforms/

git clone https://github.com/Karbo123/segmentator.git segmentator \
    && cd segmentator/csrc \
    && git reset --hard 76efe46d03dd27afa78df972b17d07f2c6cfb696 \
    && mkdir build \
    && cd build \
    && cmake .. \
        -DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path())') \
        -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
        -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR') + '/libpython3.10.so')") \
        -DCMAKE_INSTALL_PREFIX=$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())') \
    && make \
    && make install