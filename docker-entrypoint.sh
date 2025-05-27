#!/bin/bash

# Docker entrypoint script for Spirit LM
# Ensure this file has Unix line endings (LF) not Windows (CRLF)

set -e

echo "Starting Spirit LM Docker container..."

# Check if NVIDIA GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "Warning: NVIDIA GPU not detected. Spirit LM will run on CPU only."
fi

# Check if checkpoints directory is mounted and populated
if [ -d "/app/checkpoints" ] && [ "$(ls -A /app/checkpoints)" ]; then
    echo "Checkpoints directory found and populated."
    echo `ls /app/checkpoints/`
else
    echo "Warning: Checkpoints directory is empty or not mounted."
    echo "Please mount your Spirit LM checkpoints to /app/checkpoints"
    echo "Expected structure:"
    echo "/app/checkpoints/"
    echo "├── speech_tokenizer/"
    echo "└── spiritlm_model/"
fi

python -m venv .venv

# Verify Python installation
echo "Python version: $(python3 --version)"
echo "Working directory: $(pwd)"
echo "Available files in /app:"
ls -la /app/

# Set default command if none provided
if [ $# -eq 0 ]; then
    echo "No command provided. Starting interactive Gradio interface..."
    exec python3 /app/spiritlm_demo.py
else
    # Execute the provided command
    exec "$@"
fi