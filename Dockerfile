FROM --platform=linux/arm64 nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.9-py3

# Set non-interactive frontend for apt and configure timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 16FAAD7AF99A65E2


# Install system dependencies (except opencv-python which will be installed with pip)
RUN apt-get update && apt-get install -y \
    sudo \
    git \
    libx11-6 \
    build-essential \
    screen \
    libgl1 \
    libglib2.0-0

RUN apt-get install -y python3-opencv

# Set up working directory
WORKDIR /app

# # Copy requirements and install dependencies
# COPY requirements.txt .
# RUN python -m pip install --no-cache-dir --upgrade pip && \
#     python -m pip install --no-cache-dir -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple -r requirements.txt

# Copy the application code
COPY . /app

# Create a non-root user
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# Default command to open bash
CMD ["bash", "-c", "cd /app && exec bash"]
