#
# Copyright (C) 2023 Apple Inc. All rights reserved.
#

FROM ubuntu:20.04
LABEL maintainer=kmetcalf@apple.com

# Useful variables to improve readability
ARG MUJOCO_URI=https://github.com/deepmind/mujoco/releases/download/2.1.1/mujoco-2.1.1-linux-x86_64.tar.gz
ARG MUJOCO_DIR=/root/.mujoco/mujoco-2.1.1
ARG NVIDIA_DUMMY_DIR=/usr/lib/nvidia-000
ARG VENV=/opt/venv

## Set up the environment variables for Mujoco and mujoco_py
ENV MUJOCO_PY_MUJOCO_PATH="${MUJOCO_DIR}"
ENV PATH="${VENV}/bin:$PATH"
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${MUJOCO_DIR}/lib:${MUJOCO_DIR}/bin:${NVIDIA_DUMMY_DIR}
# Tell NVIDIA to mount graphics (libEGL_nvidia.so.0), compute (libcuda.so.0) and utility (nvidia-smi)
# when using this docker
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility
# Use EGL (headless) for rendering
ENV MUJOCO_GL=egl

## Compatibility with bolt (and python3.9)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        iputils-ping \
        curl \
        openssh-server \
        sudo \
        build-essential \
        git \
        fuse \
        libglvnd-core-dev  \
        libosmesa6-dev \
        libglew-dev  \
        libsndfile1 \
        python3.9 \
        python3.9-venv \
        python3.9-dev \
        rsync \
        software-properties-common \
        tmux \
        unzip \
        vim \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Configure libvngl (vendor-neutral gl)
RUN echo '{"file_format_version" : "1.0.0","ICD" : {"library_path" : "libEGL_nvidia.so.0"}}' ] \
    >> "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"

#### Install Mujoco
RUN curl -L -o /mujoco2.1.1.tar.gz ${MUJOCO_URI} && \
    mkdir /root/.mujoco/ && \
    tar -xzf /mujoco2.1.1.tar.gz --directory /root/.mujoco/ && \
    rm -rf /mujoco2.1.1.tar.gz && \
    ln -s /root/.mujoco/mujoco-2.1.1 /root/.mujoco/mujoco2.1.1_linux && \
    ln -s /root/.mujoco/mujoco-2.1.1/lib/libmujoco.so /root/.mujoco/mujoco-2.1.1/lib/libmujoco210.so

### Create venv
RUN python3.9 -m venv $VENV && \
    ${VENV}/bin/pip install --upgrade --no-cache-dir pip wheel && \
    ${VENV}/bin/pip install --upgrade --no-cache-dir \
        "toml" \
        "flit" \
        "typing-inspect" \
        "matplotlib" \
        "numpy" \
        "scipy" \
        "torch >= 1.9.0" \
        "torchvision" \
        "torchinfo" \
        "scikit-image" \
        "tensorboard" \
        "cython<3.0" \
        "mujoco_py<2.2>=2.1"  \
        "dm_control" \
        "stable-baselines3" \
        "gym" \
        "pyyaml" \
        "patchelf"

### Build mujoco_py cython extensions and install metaworld
RUN MUJOCO_PY_FORCE_CPU=1 LDFLAGS="-L${MUJOCO_DIR}/lib" python3.9 -c "import mujoco_py" && \
    # Clean up cpu build artifacts
    rm ${VENV}/lib/python3.9/site-packages/mujoco_py/generated/_pyxbld_2.1.2.14_39_linuxcpuextensionbuilder/temp.linux-x86_64-3.9/opt/venv/lib/python3.9/site-packages/mujoco_py/cymj.o && \
    rm ${VENV}/lib/python3.9/site-packages/mujoco_py/generated/_pyxbld_2.1.2.14_39_linuxcpuextensionbuilder/temp.linux-x86_64-3.9/opt/venv/lib/python3.9/site-packages/mujoco_py/gl/osmesashim.o && \
    # BUILD gpu version of mujoco-py  \
    # NVIDIA_DUMMY_DIR needed because mujoco_py has limited search over possible locations for nvidia libs)
    mkdir ${NVIDIA_DUMMY_DIR} && \
    # Create a dummy nvidia-smi (not available during docker-build)
    touch /usr/bin/nvidia-smi && chmod +x /usr/bin/nvidia-smi && \
    LDFLAGS="-L${MUJOCO_DIR}/lib" python3.9 -c "import mujoco_py" && \
    # Clean up gpu build artifacts
    rm ${VENV}/lib/python3.9/site-packages/mujoco_py/generated/_pyxbld_2.1.2.14_39_linuxgpuextensionbuilder/temp.linux-x86_64-3.9/opt/venv/lib/python3.9/site-packages/mujoco_py/cymj.o && \
    rm ${VENV}/lib/python3.9/site-packages/mujoco_py/generated/_pyxbld_2.1.2.14_39_linuxgpuextensionbuilder/temp.linux-x86_64-3.9/opt/venv/lib/python3.9/site-packages/mujoco_py/gl/eglshim.o && \
    rm /usr/bin/nvidia-smi && \
    # Install metaworld
    ${VENV}/bin/pip install --no-dependencies --no-cache-dir git+https://github.com/rlworkgroup/metaworld.git@master#egg=metaworld
