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
    libglib2.0-0 \
    libfreetype6-dev \
    pkg-config

RUN apt-get install -y python3-opencv
RUN apt-get install -y nano
RUN apt-get install python-pip python3-pip -y

# Set up working directory
WORKDIR /app

# # Copy requirements and install dependencies
# COPY requirements.txt .
# RUN python -m pip install --no-cache-dir --upgrade pip && \
#     python -m pip install --no-cache-dir -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple -r requirements.txt

RUN pip3 install --no-cache-dir -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple psutil==5.9.8 pillow==5.3.0 matplotlib PyYAML tqdm requests

RUN apt install python3-pandas -y

# maybe varetu pillow==8.4.0, kas pedejais prieks 3.6 python

# RUN python -m pip install --no-cache-dir --upgrade pip && \
#     python -m pip install --no-cache-dir -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple psutil==5.9.8

# PRIEKŠ ONNX

# RUN wget https://nvidia.box.com/shared/static/jy7nqva7l88mq9i8bw3g3sklzf4kccn2.whl -O onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl
# RUN pip3 install onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl

# Iespejams ka vajag priekš ONNX
# RUN pip3 install protobuf=3.19.6
# RUN apt install protobuf-compiler
# RUN apt install libprotoc-dev
# RUN pip3 install onnx==1.11.0


# Copy the application code
COPY . /app

# Create a non-root user
# RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
# USER appuser

# Default command to open bash
CMD ["bash", "-c", "cd /app && exec bash"]
