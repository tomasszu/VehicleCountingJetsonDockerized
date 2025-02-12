FROM nvcr.io/nvidia/cuda:10.2-devel-ubuntu18.04

#RUN add-apt-repository ppa:deadsnakes/ppa

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

# Set non-interactive frontend for apt and configure timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get install -y python3.6

RUN apt-get install -y python3-opencv
RUN apt-get install -y nano
RUN apt-get install python-pip python3-pip -y

# Set up working directory
WORKDIR /app

RUN pip3 install --no-cache-dir -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple psutil==5.9.8 pillow==5.3.0 matplotlib PyYAML tqdm requests

RUN pip3 install numpy=1.19.5

RUN pip3 install 'typing-extensions<4.0'

RUN apt install python3-pandas -y

RUN pip3 install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html


COPY . /app

# Default command to open bash
CMD ["bash", "-c", "cd /app && exec bash"]