# Vesuvius Training Container (PyTorch + Keras)
# Includes topometrics for competition metric evaluation

FROM nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV KERAS_BACKEND=torch

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    cmake \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    libpython3.11-dev \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

RUN python -m pip install --upgrade pip

WORKDIR /workspace

# Copy topometrics from eval (for competition metrics)
COPY eval/data/topological-metrics-kaggle /workspace/topological-metrics-kaggle
COPY eval/data/wheels /workspace/wheels

# Install dependencies from wheels
RUN pip install --no-index --find-links=/workspace/wheels \
    numpy scipy scikit_image pillow tifffile imagecodecs \
    connected_components_3d surface_distance pybind11 pybind11_global cmake

# Build Betti-Matching-3D
WORKDIR /workspace/topological-metrics-kaggle
ENV pybind11_DIR=/usr/local/lib/python3.11/dist-packages/pybind11/share/cmake/pybind11
RUN chmod +x scripts/build_betti.sh scripts/setup_submodules.sh \
    && ./scripts/build_betti.sh

# Install topometrics
RUN pip install -e . --no-deps

# Install PyTorch
WORKDIR /workspace
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install TensorFlow (for tf.data only)
RUN pip install --no-cache-dir tensorflow

# Install medicai and Keras nightly
RUN pip install --no-cache-dir \
    git+https://github.com/innat/medic-ai.git \
    && pip install --force-reinstall keras-nightly==3.12.0.dev2025100703

# Install additional dependencies
RUN pip install --no-cache-dir \
    tqdm \
    numpy \
    matplotlib \
    pyyaml

# Copy training script and configs
COPY train/train.py /workspace/
COPY train/configs /workspace/configs

CMD ["python", "train.py", "--help"]
