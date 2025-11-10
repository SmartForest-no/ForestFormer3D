FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# 更新和安装必要的依赖
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub \
    && apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libxrender-dev cmake \
    && apt-get install -y build-essential software-properties-common \
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt-get install -y gcc-9 g++-9 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 60 \
    && apt-get install -y python3-dev python3-pip \
    && apt-get install -y --no-install-recommends libopenblas-dev nvidia-utils-530

# 设置环境变量以确保 CUDA 工具的可用性
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 安装调试工具
RUN pip install debugpy

# 安装 OpenMMLab 项目
RUN pip install --no-deps \
    mmengine==0.7.3 \
    mmdet==3.0.0 \
    mmsegmentation==1.0.0 \
    git+https://github.com/open-mmlab/mmdetection3d.git@22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61
RUN pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13.0/index.html --no-deps

# 安装 MinkowskiEngine
RUN apt-get update \
    && apt-get -y install libopenblas-dev nvidia-cuda-dev
#RUN TORCH_CUDA_ARCH_LIST="6.1 7.0 8.6" \  A10
#A100
RUN TORCH_CUDA_ARCH_LIST="8.0" \ 
    pip install git+https://github.com/NVIDIA/MinkowskiEngine.git@02fc608bea4c0549b0a7b00ca1bf15dee4a0b228 -v --no-deps \
    --install-option="--blas=openblas" \
    --install-option="--force_cuda"

# 手动编译 torch-scatter，确保 CUDA 支持
RUN git clone https://github.com/rusty1s/pytorch_scatter.git \
    && cd pytorch_scatter \
    && git checkout tags/2.0.9 -b v2.0.9 \
    && TORCH_CUDA_ARCH_LIST="6.1;7.0;8.0" FORCE_CUDA=1 pip install .

# 单独安装 ScanNet superpoint segmentator
RUN git clone https://github.com/Karbo123/segmentator.git segmentator \
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

# 安装剩余的 Python 包
RUN pip install --no-deps \
    spconv-cu116==2.3.6 \
    addict==2.4.0 \
    yapf==0.33.0 \
    termcolor==2.3.0 \
    packaging==23.1 \
    numpy==1.24.1 \
    rich==13.3.5 \
    opencv-python==4.7.0.72 \
    pycocotools==2.0.6 \
    Shapely==1.8.5 \
    scipy==1.10.1 \
    terminaltables==3.1.10 \
    numba==0.57.0 \
    llvmlite==0.40.0 \
    pccm==0.4.7 \
    ccimport==0.4.2 \
    pybind11==2.10.4 \
    ninja==1.11.1 \
    lark==1.1.5 \
    cumm-cu116==0.4.9 \
    pyquaternion==0.9.9 \
    lyft-dataset-sdk==0.0.8 \
    pandas==2.0.1 \
    python-dateutil==2.8.2 \
    matplotlib==3.5.2 \
    pyparsing==3.0.9 \
    cycler==0.11.0 \
    kiwisolver==1.4.4 \
    scikit-learn==1.2.2 \
    joblib==1.2.0 \
    threadpoolctl==3.1.0 \
    cachetools==5.3.0 \
    nuscenes-devkit==1.1.10 \
    trimesh==3.21.6 \
    open3d==0.17.0 \
    plotly==5.18.0 \
    dash==2.14.2 \
    plyfile==1.0.2 \
    flask==3.0.0 \
    werkzeug==3.0.1 \
    click==8.1.7 \
    blinker==1.7.0 \
    itsdangerous==2.1.2 \
    importlib_metadata==2.1.2 \
    zipp==3.17.0 \
    tensorboard==2.15.1 \
    tensorboard-data-server==0.7.2 \
    protobuf \
    absl-py \
    future \
    MarkupSafe==2.0.1 \
    markdown \
    grpcio \
    google-auth-oauthlib \
    google-auth \
    requests-oauthlib \
    oauthlib

RUN apt-get update && apt-get install -y nvidia-utils-530

# 设置 PYTHONPATH 环境变量
ENV PYTHONPATH=/workspace

# 保持容器运行
CMD ["bash", "-c", "while true; do sleep 1000; done"]

RUN pip install --no-deps --no-cache-dir\
    torch-points-kernels==0.7.0

RUN pip uninstall torch-cluster

RUN pip install --no-deps --no-cache-dir\
    torch-cluster
