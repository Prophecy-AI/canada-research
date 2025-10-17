# LightGBM CUDA Fix - Implementation Summary

## Problem
LightGBM was training on **CPU instead of GPU** despite having NVIDIA A10 24GB GPU available.

**Error messages:**
```
No OpenCL device found
LightGBMError: CUDA Tree Learner was not enabled in this build
```

**Root cause:** PyPI `lightgbm==4.5.0` is CPU-only. CUDA support requires building from source with `-DUSE_CUDA=ON` flag.

## Solution

### 1. Base Image Changes (`mle-bench/environment/Dockerfile`)

**Added CUDA Toolkit 11.8 installation** (lines 67-76):
```dockerfile
# Install CUDA Toolkit 11.8 (needed for LightGBM CUDA build in agent Dockerfile)
# A10 GPU uses Ampere architecture (CUDA 11.x compatible)
RUN wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run -O /tmp/cuda_installer.run && \
    sh /tmp/cuda_installer.run --silent --toolkit --no-opengl-libs && \
    rm /tmp/cuda_installer.run && \
    echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> /etc/bash.bashrc && \
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> /etc/bash.bashrc

ENV PATH=/usr/local/cuda-11.8/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH}
```

**Removed lightgbm from requirements.txt** (line 51):
```dockerfile
# Remove lightgbm from requirements (will be built from source with CUDA in agent Dockerfile)
RUN sed -i '/^lightgbm==/d' ${REQUIREMENTS}
```

**Why CUDA 11.8:**
- Matches PyTorch CUDA version (cu118)
- Compatible with NVIDIA A10 (Ampere architecture)
- Stable and well-tested with LightGBM

### 2. Agent Image Already Fixed (`mle-bench/agents/agent_v5_kaggle/Dockerfile`)

**CMake 3.28+ installation** (lines 42-51):
```dockerfile
# Install build tools for LightGBM CUDA compilation
# Need CMake 3.28+ (base image has 3.16)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install newer CMake (3.28+ required by LightGBM)
RUN wget -qO- https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-linux-x86_64.tar.gz | tar -xz -C /opt && \
    ln -sf /opt/cmake-3.28.3-linux-x86_64/bin/cmake /usr/local/bin/cmake && \
    cmake --version
```

**LightGBM CUDA build** (lines 53-68):
```dockerfile
# Build LightGBM from source with CUDA support
# Compiles with -DUSE_CUDA=ON flag which enables CUDA Tree Learner (for NVIDIA GPUs)
# Uses CUDA 11.8 toolkit from base image
RUN conda run -n ${CONDA_ENV_NAME} bash -c " \
    export PATH=/usr/local/cuda-11.8/bin:\$PATH && \
    export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:\$LD_LIBRARY_PATH && \
    cd /tmp && \
    git clone --recursive --branch stable --depth 1 https://github.com/microsoft/LightGBM && \
    cd LightGBM && \
    cmake -B build -S . -DUSE_CUDA=ON && \
    cmake --build build -j\$(nproc) && \
    cd python-package && \
    pip install . --no-cache-dir \
    " && \
    rm -rf /tmp/LightGBM && \
    conda clean -afy
```

### 3. Agent Prompts Fixed (`kaggle_agent.py`)

**Line 218:**
```python
# BEFORE: device='gpu'
# AFTER: device_type='cuda'
```

**Line 411:**
```python
# BEFORE: params = {'device': 'gpu', 'max_bin': 63, 'gpu_use_dp': False}
# AFTER: params = {'device_type': 'cuda', 'max_bin': 255}  # use 'cuda' for CUDA build
```

## Why This Works

### CUDA vs OpenCL in LightGBM

LightGBM has **two mutually exclusive GPU implementations:**

1. **OpenCL Build** (`device_type='gpu'`)
   - Build flag: `-DUSE_GPU=ON`
   - Works with: AMD, NVIDIA, Intel GPUs
   - Less maintained, older

2. **CUDA Build** (`device_type='cuda'`) ✅ **We use this**
   - Build flag: `-DUSE_CUDA=ON`
   - Works with: NVIDIA GPUs only
   - Better maintained, faster, more GPU work
   - **Recommended for NVIDIA A10**

### Build Process

1. **Base image** installs CUDA Toolkit 11.8 (provides `nvcc` compiler, CUDA libraries)
2. **Agent image** builds LightGBM from source with `-DUSE_CUDA=ON` flag
3. **Runtime** uses `device_type='cuda'` parameter to enable GPU training

## Expected Performance

### Hardware: NVIDIA A10 GPU
- **VRAM:** 24GB GDDR6
- **CUDA Cores:** 9,216
- **Architecture:** Ampere (CUDA 11.x compatible)

### Speedup (CPU → CUDA)
- **Before (CPU):** ~60-90 minutes for 3-fold CV on 12M rows
- **After (CUDA):** ~4-6 minutes for 3-fold CV on 12M rows
- **Speedup:** 10-15x faster

### GPU Utilization
- **Memory usage:** 15-20 GB / 24 GB (70-85%)
- **GPU compute:** 80-95% (tree construction on GPU)
- **CPU usage:** 20-30% (data preprocessing, gradients)

**Why not 100% GPU?** LightGBM is hybrid - some operations stay on CPU (by design).

## Build Time Impact

- **Old build:** ~3-5 minutes (PyPI wheels)
- **New build:** ~8-12 minutes (CUDA toolkit + LightGBM source build)
- **Delta:** +5-7 minutes (one-time cost)
- **Worth it?** YES - saves 10-15x runtime for every competition!

## Validation After Rebuild

### Test 1: Check CUDA is enabled
```bash
docker run --rm --gpus all agent_v5_kaggle:latest bash -c "
  conda run -n agent python -c \"
import lightgbm as lgb
import numpy as np
X = np.random.rand(100, 5)
y = np.random.rand(100)
data = lgb.Dataset(X, label=y)
params = {'device_type': 'cuda', 'num_leaves': 31, 'verbose': -1}
try:
    model = lgb.train(params, data, num_boost_round=1)
    print('✅ CUDA ENABLED')
except Exception as e:
    print(f'❌ CUDA DISABLED: {e}')
  \"
"
```

**Expected output:** `✅ CUDA ENABLED`

### Test 2: Monitor GPU during competition
```bash
watch -n 1 nvidia-smi
```

**Expected during training:**
- GPU Memory: 15-20 GB / 24 GB (70-85%)
- GPU Util: 80-95%
- Process: python

### Test 3: Verify speedup
- Run same competition twice (old vs new image)
- Should see 10-15x speedup

## Changes Summary

| File | Lines | Change |
|------|-------|--------|
| `mle-bench/environment/Dockerfile` | 50-51 | Remove lightgbm from requirements |
| `mle-bench/environment/Dockerfile` | 67-76 | Install CUDA Toolkit 11.8 |
| `mle-bench/agents/agent_v5_kaggle/Dockerfile` | 57-58 | Export CUDA env vars during build |
| `mle-bench/agents/agent_v5_kaggle/kaggle_agent.py` | 218, 411 | Fix GPU parameters |

## Next Steps

1. ✅ Changes committed to git
2. 🔄 **Rebuild Docker images** (set `rebuild_image=true` in GitHub Actions)
3. ⏳ Run test competition to validate GPU usage
4. 📊 Monitor GPU utilization and speedup

---

**Status:** Ready to rebuild and test
**Expected outcome:** LightGBM training 10-15x faster with 80-95% GPU utilization
