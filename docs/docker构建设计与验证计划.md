# Docker 镜像构建与验证

目标：其他 Linux x86_64 主机可以从公开基础镜像构建建图运行环境，不再要求提前取得本机容器的导出归档。

## 设计

选择公开 NVIDIA CUDA 11.8 base / Ubuntu 20.04 镜像，按需安装CUDA运行库、头文件和NVRTC，以及 ROS Noetic、GridMap/RViz、两个隔离 Python 环境。不需要完整devel镜像内的Nsight等工具。保持现有 `/opt/conda/bin/python3` 和 `/opt/eomt/bin/python` 路径，兼容现有启动脚本。另一种方式是继续导出当前容器，速度快但无法重建；直接用现有私有镜像作为 FROM 也无法解决首次部署问题。

几何保留 NumPy1.22.4、SciPy1.7.3、CuPy12.3.0；补齐源码实际导入的 OpenCV、Torch 和 ROS Python 依赖。几何 Torch 采用支持 Python3.8/CUDA11.8 的2.4.1，语义继续使用现有完整哈希锁（Torch2.7.0、Transformers4.56.1）。Miniconda安装器固定版本并校验SHA256，不在构建中进行conda求解。镜像只保存运行环境，不包含模型、bag、用户目录和本机编译结果。

`scripts/build_docker.bash` 调用 Dockerfile；`scripts/create_container.bash` 创建可被现有脚本使用的容器，支持无桌面模式，并拒绝覆盖已有容器；`scripts/verify_docker.bash` 创建临时容器、独立工作区及私有网络中的ROS master，完成构建、测试、GPU检查和可选GOOSE短片段验收，退出清理临时容器。验证报告保存在工程的reports中。验证使用`--network none`，即使端口与旧实例一致也无法注册到旧master，并检查自己启动的roscore仍存活。

## 实施与验证清单

- [x] 新增Dockerfile、最小构建上下文与锁定运行依赖。
- [x] 新增镜像构建、容器创建和隔离验证入口。
- [x] 公开基础镜像的负向检查按预期失败：没有`/opt/conda/bin/python3`；新镜像另行完整验收。
- [x] 实际构建新镜像，修复构建暴露的问题。
- [x] 在新镜像中全新catkin构建、649项核心回归、CuPy CUDA内核和Torch GPU运算。
- [x] 使用本地权重和GOOSE包验证模型就绪、100次语义预测/89次融合、高程及普通/CVaR各102帧地图。
- [x] 更新部署说明、启动说明、整体介绍和环境说明；发布按用户此前授权执行。验收证据见`reports/Docker镜像验证.md`。

构建需要可访问Docker Hub、Ubuntu/ROS软件源、Anaconda、PyPI及PyTorch下载源。主机驱动和NVIDIA Container Toolkit仍由主机提供，不在镜像中安装。镜像验证只能证明本机驱动/GPU和短片段数据条件下可运行，不等同于其他硬件或完整长时间数据验收。
