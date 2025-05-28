# Spirit LM Docker Setup for CUDA 12.6
FROM python:3.10.17-bookworm

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies with Python 3.10 specifically
# Also add the libnvidia keys and whatnot.
# Why haven't they 
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libsox-fmt-all \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python and pip to use 3.10 specifically
RUN ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Verify Python 3.10 installation
RUN python --version && python -c "import sys; print(f'Python version: {sys.version}')"

# Upgrade pip for Python 3.10
RUN python -m pip install --upgrade pip

# Install PyTorch with CUDA 12.6 support using Python 3.10
RUN python -m pip install --root-user-action ignore torch==2.6.0+cu126 torchaudio==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126

# Clone Spirit LM repository
RUN git clone -b main https://github.com/A-Vaillant/spiritlm.git /app/spiritlm

WORKDIR /app/spiritlm
### WORKDIR IS NOW: /app/spiritlm ###

# Copy entrypoint script
RUN cp deployment/docker-entrypoint.sh /app/docker-entrypoint.sh && \
    chmod +x /app/docker-entrypoint.sh

# Install Spirit LM package using Python 3.10
RUN python -m pip install --root-user-action ignore -e '.[eval]'

# Set environment variables for audio libraries
ENV TORCHAUDIO_USE_SOX=1
ENV TORIO_USE_FFMPEG=1

# Expose port for Gradio interface
EXPOSE 7860

# Set the entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]