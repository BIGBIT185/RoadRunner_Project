ARG BASE_IMAGE=nvidia/cuda:11.8.0-base-ubuntu20.04@sha256:8803790082168d6fc95089288e8debb8e8318386658d10deb5c7175ec83b5e14
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    QT_X11_NO_MITSHM=1 ROS_DISTRO=noetic \
    PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=5 PIP_DISABLE_PIP_VERSION_CHECK=1
SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

# Install only runtime math libraries, headers and NVRTC needed by CuPy RawKernel.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates curl gnupg2 build-essential cmake pkg-config \
       python3-dev python3-pip libgl1-mesa-glx libglib2.0-0 \
       libxkbcommon-x11-0 xauth \
       cuda-nvrtc-11-8 cuda-nvrtc-dev-11-8 cuda-cudart-dev-11-8 \
       libcublas-11-8 libcusolver-11-8 libcurand-11-8 libcusparse-11-8 libcufft-11-8 \
    && curl -fsSL --retry 5 https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
       -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo 'deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros/ubuntu focal main' \
       > /etc/apt/sources.list.d/ros1.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       ros-noetic-ros-base ros-noetic-catkin ros-noetic-message-generation \
       ros-noetic-message-filters ros-noetic-tf2-ros ros-noetic-diagnostic-msgs \
       ros-noetic-sensor-msgs ros-noetic-std-msgs ros-noetic-std-srvs \
       ros-noetic-grid-map-msgs ros-noetic-grid-map-rviz-plugin \
       ros-noetic-rviz ros-noetic-rosbag python3-catkin-pkg python3-rospkg \
    && rm -rf /var/lib/apt/lists/*

# Fixed installers avoid a live conda solver and preserve the existing launch paths.
RUN curl -fsSL --retry 5 \
       https://repo.anaconda.com/miniconda/Miniconda3-py38_23.1.0-1-Linux-x86_64.sh \
       -o /tmp/mapping-python.sh \
    && echo '640b7dceee6fad10cb7e7b54667b2945c4d6f57625d062b2b0952b7f3a908ab7  /tmp/mapping-python.sh' | sha256sum -c - \
    && bash /tmp/mapping-python.sh -b -p /opt/conda \
    && rm /tmp/mapping-python.sh \
    && /opt/conda/bin/conda clean -afy
RUN curl -fsSL --retry 5 \
       https://repo.anaconda.com/miniconda/Miniconda3-py310_24.7.1-0-Linux-x86_64.sh \
       -o /tmp/semantic-python.sh \
    && echo '684cda724bc37e3bbbb342e440fc4cac515c92e91a489eb4359feca35382894b  /tmp/semantic-python.sh' | sha256sum -c - \
    && bash /tmp/semantic-python.sh -b -p /opt/eomt \
    && rm /tmp/semantic-python.sh \
    && /opt/eomt/bin/conda clean -afy

COPY env/requirements-mapping.txt env/requirements-eomt.lock docker/requirements-*.txt /opt/roadrunner/env/
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/conda/bin/python3 -m pip install pip==24.2 setuptools==70.3.0 wheel==0.44.0 \
    && /opt/conda/bin/python3 -m pip install \
       -r /opt/roadrunner/env/requirements-mapping.txt \
       -r /opt/roadrunner/env/requirements-mapping-runtime.txt \
       -r /opt/roadrunner/env/requirements-ros-python.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/eomt/bin/python -m pip install pip==24.2 setuptools==70.3.0 wheel==0.44.0 \
    && /opt/eomt/bin/python -m pip install --require-hashes \
       -r /opt/roadrunner/env/requirements-eomt.lock \
    && /opt/eomt/bin/python -m pip install \
       -r /opt/roadrunner/env/requirements-ros-python.txt

ENV PATH=/opt/conda/bin:${PATH} \
    PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages \
    PYTHONUTF8=1 \
    ROADRUNNER_MAPPING_PYTHON=/opt/conda/bin/python3 \
    ROADRUNNER_SEMANTIC_PYTHON=/opt/eomt/bin/python
# Match the ROS library search paths used by container_env.bash at runtime.
# CuPy import needs the host driver, so its actual import/kernel test runs with --gpus all.
RUN source /opt/ros/noetic/setup.bash \
    && mkdir -p /opt/roadrunner/ros_python_compat \
    && /opt/conda/bin/python3 -c 'import cv2, numpy, rospy, scipy, tf2_ros, torch; from importlib.metadata import version; assert numpy.__version__ == "1.22.4"; assert version("cupy-cuda11x") == "12.3.0"' \
    && /opt/eomt/bin/python -c 'import rospy, torch, transformers; from transformers import AutoImageProcessor, AutoModelForUniversalSegmentation; assert transformers.__version__ == "4.56.1"'
COPY docker/verify_runtime.py /opt/roadrunner/verify_runtime.py
WORKDIR /code_ws_new/RoadRunner_Project
CMD ["sleep", "infinity"]
