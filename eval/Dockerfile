# Vesuvius Competition Metric Evaluation Container
# Uses official topometrics library from Kaggle dataset

FROM nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV KERAS_BACKEND=jax

# Install system dependencies (no system pybind11 - use pip version)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    libpython3.11-dev \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Create working directory
WORKDIR /workspace

# Copy metric resources (topometrics library + wheels)
COPY data/topological-metrics-kaggle /workspace/topological-metrics-kaggle
COPY data/wheels /workspace/wheels

# Install dependencies from wheels (offline)
RUN pip install --no-index --find-links=/workspace/wheels \
    numpy scipy scikit_image pillow tifffile imagecodecs \
    connected_components_3d surface_distance pybind11 pybind11_global cmake

# Build Betti-Matching-3D (C++ extension)
# Use pip-installed pybind11 instead of system version
WORKDIR /workspace/topological-metrics-kaggle
ENV pybind11_DIR=/usr/local/lib/python3.11/dist-packages/pybind11/share/cmake/pybind11
RUN chmod +x scripts/build_betti.sh scripts/setup_submodules.sh \
    && ./scripts/build_betti.sh

# Install topometrics package
RUN pip install -e . --no-deps

# Install additional dependencies for model loading
# Use the exact Keras nightly version from training (3.12.0.dev2025100703)
# Install keras-nightly LAST so it overwrites the keras module from tensorflow
WORKDIR /workspace
RUN pip install --no-cache-dir \
    jax[cuda12] \
    tensorflow \
    git+https://github.com/innat/medic-ai.git \
    && pip install --force-reinstall keras-nightly==3.12.0.dev2025100703

# Copy evaluation script
COPY evaluate.py /workspace/

# Default command
CMD ["python", "evaluate.py", "--help"]
