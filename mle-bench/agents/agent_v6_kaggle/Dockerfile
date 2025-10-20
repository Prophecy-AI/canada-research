FROM mlebench-env

# Build args from mle-bench
ARG SUBMISSION_DIR
ENV SUBMISSION_DIR=${SUBMISSION_DIR}
ARG LOGS_DIR
ENV LOGS_DIR=${LOGS_DIR}
ARG CODE_DIR
ENV CODE_DIR=${CODE_DIR}
ARG AGENT_DIR
ENV AGENT_DIR=${AGENT_DIR}

# CRITICAL: Force Python unbuffered output so background scripts show real-time output
ENV PYTHONUNBUFFERED=1

# Create required directories
RUN mkdir -p ${LOGS_DIR} ${CODE_DIR} ${AGENT_DIR}

ARG CONDA_ENV_NAME=agent

# Copy requirements first (for Docker layer caching)
COPY mle-bench/agents/agent_v6_kaggle/requirements.txt ${AGENT_DIR}/requirements.txt

# Fix protobuf version conflict (google-genai requires 3.x, base image may have 4.x)
RUN conda run -n ${CONDA_ENV_NAME} pip uninstall -y protobuf && \
    conda run -n ${CONDA_ENV_NAME} pip install 'protobuf>=3.20.2,<4.0.0' && \
    conda clean -afy

# Install agent dependencies
RUN conda run -n ${CONDA_ENV_NAME} pip install -r ${AGENT_DIR}/requirements.txt && \
    conda clean -afy

# Pre-install common ML packages to save time during competition (WITHOUT lightgbm - built separately)
RUN conda run -n ${CONDA_ENV_NAME} pip install \
    xgboost>=2.0.0 \
    catboost>=1.2.0 \
    scikit-learn>=1.3.0 \
    pandas>=2.0.0 \
    numpy>=1.24.0 \
    matplotlib>=3.7.0 \
    seaborn>=0.12.0 \
    scipy>=1.11.0 \
    && conda clean -afy

# Install build tools for LightGBM CUDA compilation
# Need CMake 3.28+ (base image has 3.16)
# Also install OpenGL libraries required by OpenCV (used by albumentations, cv2, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    ripgrep \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install newer CMake (3.28+ required by LightGBM)
RUN wget -qO- https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-linux-x86_64.tar.gz | tar -xz -C /opt && \
    ln -sf /opt/cmake-3.28.3-linux-x86_64/bin/cmake /usr/local/bin/cmake && \
    cmake --version

# Build LightGBM from source with CUDA support
# Uses CUDA 11.8 toolkit from base image
# scikit-build-core (LightGBM 4.0+) runs cmake with -DUSE_CUDA=ON via config-settings
RUN conda run -n ${CONDA_ENV_NAME} bash -c " \
    export PATH=/usr/local/cuda-11.8/bin:\$PATH && \
    export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:\$LD_LIBRARY_PATH && \
    pip install --no-binary lightgbm --no-cache-dir \
        --config-settings=cmake.define.USE_CUDA=ON \
        'lightgbm>=4.0.0' \
    " && \
    conda clean -afy

# Install PyTorch with CUDA support for GPU
RUN conda run -n ${CONDA_ENV_NAME} pip install \
    torch>=2.0.0 \
    torchvision>=0.15.0 \
    torchaudio>=2.0.0 \
    --index-url https://download.pytorch.org/whl/cu118 \
    && conda clean -afy

# ------------------------------------------------------------------
# Install RAPIDS cuML (GPU-accelerated scikit-learn replacement)
# CUDA 11.8 build – matches PyTorch wheel above
# ------------------------------------------------------------------
RUN conda run -n ${CONDA_ENV_NAME} \
    conda install -y -c rapidsai -c nvidia -c conda-forge \
        cuml=24.04 python=${PYTHON_VERSION} cudatoolkit=11.8 && \
    conda clean -afy

# Install TensorFlow with GPU support (optional, for flexibility)
RUN conda run -n ${CONDA_ENV_NAME} pip install \
    tensorflow>=2.13.0 \
    && conda clean -afy

# Copy agent files (including symlinked directories)
# Build context is canada-research root, so paths are relative to that
COPY mle-bench/agents/agent_v6_kaggle/ ${AGENT_DIR}/
COPY agent_v5/ ${AGENT_DIR}/agent_v5/
COPY agent_v6/ ${AGENT_DIR}/agent_v6/
COPY debug.py ${AGENT_DIR}/debug.py
COPY security/ ${AGENT_DIR}/security/
COPY observability/ ${AGENT_DIR}/observability/

# Set working directory to CODE_DIR (agent's workspace)
WORKDIR ${CODE_DIR}
