# Linux NVIDIA/CUDA training image.
# Physics backends: the scoped tensor Manager runtime (MuJoCo and MJWarp).
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1
ENV NV_VARIANT_PROVIDER_FORCE_CUDA_DRIVER_VERSION=12.8
ENV PATH="/root/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python-is-python3 \
    git \
    curl \
    ca-certificates \
    build-essential \
    cmake \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/UniLab

COPY . /workspace/UniLab

RUN scripts/tools/install_uv_wheelnext.sh /root/.local/bin

RUN uv sync --dev --extra mujoco --extra mjwarp \
    && uv cache clean \
    && rm -rf /root/.cache/uv

CMD ["uv", "run", "train", "--help"]
