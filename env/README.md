# 安装环境

代码和模型均来自本工程，可复用已有 `roadrunner_bevfusion_container` 的安装环境，或从公开基础镜像构建新环境；不 source 原 RoadRunner overlay。容器把宿主工程父目录挂载为 `/code_ws_new`，提供 NVIDIA GPU、host 网络和 X11。启动脚本默认使用本机 UID/GID 创建构建产物。

也可通过`./scripts/build_docker.bash`从公开CUDA11.8 / Ubuntu20.04基础镜像独立构建，再用`./scripts/create_container.bash`创建容器，详见[部署说明](../部署说明.md)。Dockerfile位于`docker/`。新镜像几何环境除下列数值依赖外，使用Torch2.4.1+cu118/torchvision0.19.1和Pillow10.4.0；语义环境继续使用原哈希锁。两个解释器分别安装`docker/requirements-ros-python.txt`，无需从旧容器复制兼容依赖目录。`verify_docker.bash`用于隔离验证新镜像。

新镜像显式启用`PYTHONUTF8=1`，防止CUDA原生库改变线程locale后，Python按ASCII读取中文源码或路径。CuPy依赖`fastrlock`固定为0.8.3，避免0.8.4源码与Python3.8不兼容。

- Ubuntu 20.04 / ROS Noetic；catkin、message_generation/runtime、rospy、message_filters、tf2_ros、sensor_msgs、std_msgs、std_srvs、diagnostic_msgs、grid_map_msgs、grid_map_rviz_plugin、RViz、rosbag。
- 几何解释器 `/opt/conda/bin/python3`：Python 3.8.16、NumPy 1.22.4、SciPy 1.7.3、CuPy 12.3.0 / CUDA 11.8；数值依赖见 `requirements-mapping.txt`。
- 语义解释器 `/opt/eomt/bin/python`：Python 3.10、torch 2.7.0+cu118、torchvision 0.22.0+cu118、transformers 4.56.1、NumPy 1.26.4。完整锁定依赖见 `requirements-eomt.lock`，便于在新环境中安装；不要把语义环境的 NumPy/CUDA 库路径加到几何进程。
- 语义 ROS 兼容模块：安装 rospkg/catkin_pkg/PyYAML 等纯 Python ROS 依赖；当前容器安装在 `/opt/roadrunner/ros_python_compat`。运行时使用本工程 `devel` 中生成的消息。

建图和语义进程默认使用宿主UID/GID创建文件；当前容器没有对应用户的home记录，RViz启动使用容器已有root用户，避免Qt尝试创建`/.rviz`。CuPy/ROS/模型缓存和日志使用工程内的显式目录；不改动宿主用户home或全局环境。

权重位于 `models/eomt_rugd24/`，原始 SHA256 清单位于包内 `config/eomt_rugd_model.sha256`，模型加载前校验。权重与 bag 均不进入 Git。`requirements-eomt.lock` 安装命令为目标 Python 的 `-m pip install --require-hashes -r ...`；CUDA/驱动和 ROS 需先安装。本工程不负责安装或修改全局驱动。

若在已具备这些依赖的本机直接运行，source ROS 后在工程根目录 `catkin_make -DPYTHON_EXECUTABLE=<几何Python>`，再 source 本工程 `devel/setup.bash`，设置 `ROADRUNNER_MAPPING_PYTHON`、`ROADRUNNER_SEMANTIC_PYTHON`，运行相同 launch。`scripts/*.bash` 是当前容器环境的便利入口。
