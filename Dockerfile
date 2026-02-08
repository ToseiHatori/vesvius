# Vesuvius nnUNet Training Environment
# Based on Kaggle GPU image with nnUNet dependencies pre-installed

FROM gcr.io/kaggle-gpu-images/python:latest

# Install nnUNet and dependencies
RUN pip install --no-cache-dir \
    nnunetv2 \
    nibabel \
    tifffile \
    tqdm \
    scikit-image

# Set working directory
WORKDIR /workspace

# Default command: keep container running for interactive use
CMD ["bash"]
