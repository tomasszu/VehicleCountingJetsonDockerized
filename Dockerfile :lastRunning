# Use CUDA as the base image
FROM nvidia/cuda:11.4.3-cudnn8-runtime-ubuntu20.04


# Set non-interactive frontend for apt and configure timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1



# Install Python 3.10 and necessary utilities
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    build-essential \
    screen \
    libgl1 \
    libglib2.0-0 && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-distutils python3.10-venv && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python && \
    rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

RUN pip install --no-cache-dir torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# Copy requirements and install dependencies
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple -r requirements.txt

# Copy the application code
COPY . /app

# Create a non-root user
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# Default command
CMD ["nvidia-smi"]
