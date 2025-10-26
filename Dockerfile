FROM python:3.10-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# Install CPU-only PyTorch/torchvision
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision

# (Optional) your own deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || true

COPY . .
ENTRYPOINT ["python3", "-u", "main.py"]