# Dockerfile to build image for MNIST training
# Author: Tai Wan Kim
# Date: November, 2025

# Base image
FROM python:3.10-slim

# Set working directory and env variables
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# Install CPU-only versions of PyTorch and torchvision
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision

# (Optional)
# This is not required because PyTorch and torchvision are our only dependencies
# But keep here just in case we add other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || true

# Copy local directory to container and run our program
COPY . .
ENTRYPOINT ["python3", "-u", "main.py"]