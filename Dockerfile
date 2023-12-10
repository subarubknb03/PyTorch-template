FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# How to build python from source
# https://devguide.python.org/getting-started/setup-building/
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    libbz2-dev \
    libffi-dev \
    libgdbm-dev \
    libdb-dev \
    liblzma-dev \
    libncursesw5-dev \
    libreadline6-dev \
    libsqlite3-dev \
    libssl-dev \
    tk-dev \
    uuid-dev \
    zlib1g-dev \
    sudo \
    vim \
    wget \
    && rm -rf /var/lib/apt/lists/*

# install python3.10.13
RUN wget --no-check-certificate https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tar.xz \
    && tar -xf Python-3.10.13.tar.xz \
    && cd Python-3.10.13 \
    && ./configure --enable-optimizations \
    && make -s -j4 \
    && make altinstall \
    && cd .. \
    && rm -f Python-3.10.13.tar.xz \
    && rm -rf Python-3.10.13

# create a symbolic link to use python3 with python
RUN ln -s /usr/local/bin/python3.10 /usr/local/bin/python

# install pip & install python package
RUN python -m pip install --upgrade --no-cache-dir pip \
    && python -m pip install --no-cache-dir \
    ipywidgets \
    jupyterlab \
    matplotlib \
    numpy \
    optuna \
    pandas \
    plotly \
    scikit-learn \
    scipy \
    seaborn \
    tqdm \
    # install PyTorch & PyG
    && python -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    && python -m pip install --no-cache-dir pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html \
    && python -m pip install --no-cache-dir torch_geometric \
    && python -m pip install --no-cache-dir torchinfo

# Reproducibility of cuBLAS
# https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

# add user
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN useradd -u $USER_ID -o -m user && groupmod -g $GROUP_ID user
USER ${USER_ID}

WORKDIR /mnt/workspace

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--LabApp.token=''"]
