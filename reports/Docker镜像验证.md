# Docker 镜像构建与验证（2026-10-23）

结论：公开基础镜像构建、全新工作区编译、核心回归、真实GPU运算和GOOSE短片段语义建图均通过。原建图源码未修改。

## 镜像与构建

- 构建入口：`./scripts/build_docker.bash`，Dockerfile为`docker/Dockerfile`。
- 基础：公开`nvidia/cuda:11.8.0-base-ubuntu20.04`，固定摘要`sha256:8803790082168d6fc95089288e8debb8e8318386658d10deb5c7175ec83b5e14`；安装ROS Noetic、CUDA数学运行库、NVRTC/头文件与两个Python环境。
- 产物：`roadrunner-mapping:20261023`，本机image ID为`sha256:b3dfe5a66e62a83c4811197531dc150058d6def20181c6bac3c0303d30bebef5`。Docker报告大小8,374,912,410字节，约7.80GiB；镜像未上传到镜像仓库。
- 本次实际下载和安装依赖，中途主机重启后继续使用BuildKit缓存。最终构建退出码0；不依赖旧`roadrunner-bevfusion:eomt`镜像或旧容器文件系统。安装器SHA256和语义依赖哈希校验通过。
- 修正首次构建暴露的兼容问题：固定`fastrlock==0.8.3`以支持Python3.8；启用`PYTHONUTF8=1`，避免CuPy原生计算改变线程locale后按ASCII读取中文源码。构建期加载ROS库路径并检查软件导入，真实CuPy/GPU检查放在运行容器执行。

## 验收命令与结果

```bash
./scripts/verify_docker.bash --bag ~/code_ws_new/datasets/GOOSE/6_Gravel_Soil_Forest_Path.bag
```

临时容器使用独立网络和新工作区，重新生成build/devel；本机旧容器、宿主ROS master和工作区编译结果不参与验收。命令退出码0，容器和临时工作区已清理。

| 检查 | 实测结果 |
|---|---|
| 全新catkin构建 | 通过，生成本工程消息 |
| 核心回归 | 649通过、1项几何环境可选Transformers处理器用例跳过，31.21秒 |
| 几何环境 | Python3.8.16、NumPy1.22.4、SciPy1.7.3、CuPy12.3.0、Torch2.4.1+cu118；ROS/TF导入通过 |
| 语义环境 | Python3.10.14、NumPy1.26.4、Torch2.7.0+cu118、Transformers4.56.1；自定义ROS消息导入通过 |
| 实际GPU | RTX4060 Laptop；两环境CUDA张量运算正确，CuPy RawKernel编译执行正确，CUDA runtime11080 |
| 模型 | 三文件SHA256通过，就绪信号true，真实EoMT推理通过 |
| 输入 | GOOSE前25秒、0.5倍速；60秒墙钟观察中图像/点云/CameraInfo各238帧 |
| 语义 | 100次预测、89次融合；两个节点失败计数及语义融合失败计数均0 |
| 地图 | 普通/CVaR各102帧；base_link、40m×40m、0.2m，四图层均有有效值 |
| 数值契约 | 概率有限值均在[0,1]；高程有效格34321，几何未知格高程为NaN |
| 存活 | 末尾两个算法节点均通过rosnode ping，core和launch仍存活 |
| 容器创建 | 实际创建并检查GPU、host网络、父目录挂载、X11授权及ros.bash包装器，随后仅移除测试容器 |
| 保护路径 | 同名运行容器拒绝覆盖；模拟验证容器重名时原容器保留，清理仅针对本次CID |

详细证据：[完整运行报告目录](docker-verify-20261023-174637-12395/)、[GOOSE报告](docker-verify-20261023-174637-12395/goose.json)、[几何GPU检查](docker-verify-20261023-174637-12395/mapping-runtime.json)、[语义GPU检查](docker-verify-20261023-174637-12395/semantic-runtime.json)、[创建容器检查](docker-container-smoke.json)。

这是本机新镜像与GOOSE短片段验收；未在另一台全新主机完成驱动/Toolkit安装或长时间运行验证。GridMap消息和RViz插件安装已检查，本轮未单独验收RViz图形渲染。原README中的2.35/6.67Hz对比来自前序环境，本次没有重测该性能对比。
