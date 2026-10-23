"""Exercise real ROS imports and GPU kernels; used in both Python environments."""
import json
import sys

import numpy as np
import rospy
import torch
import yaml
from sensor_msgs.msg import Image, PointCloud2


def main():
    mode = sys.argv[1]
    assert mode in {"mapping", "semantic"}
    assert sys.flags.utf8_mode == 1, "Python UTF-8 mode must stay enabled with CUDA libraries"
    assert torch.cuda.is_available(), "Torch cannot access CUDA"
    value = torch.tensor([1., 2., 3.], device="cuda").square().sum().item()
    assert value == 14., value
    result = dict(environment=mode, python=sys.version.split()[0], numpy=np.__version__,
                  torch=torch.__version__, torch_cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0), torch_result=value, python_utf8_mode=sys.flags.utf8_mode)
    if mode == "mapping":
        import cv2
        import cupy as cp
        import scipy
        import tf2_ros
        import rospkg
        from grid_map_msgs.msg import GridMap
        values = cp.arange(8, dtype=cp.float32)
        # RawKernel also checks NVRTC/toolkit availability, not only the driver.
        kernel = cp.RawKernel('extern "C" __global__ void twice(float* x) '
                              '{ int i = threadIdx.x; x[i] *= 2.0f; }', 'twice')
        kernel((1,), (8,), (values,))
        cp.cuda.Stream.null.synchronize()
        np.testing.assert_array_equal(cp.asnumpy(values), np.arange(8) * 2)
        result.update(cupy=cp.__version__, scipy=scipy.__version__, opencv=cv2.__version__,
                      cuda_runtime=cp.cuda.runtime.runtimeGetVersion(), cupy_kernel=True)
    else:
        import transformers
        from transformers import AutoImageProcessor, AutoModelForUniversalSegmentation
        from roadrunner_mapping.msg import EomtSemanticPrediction
        result.update(transformers=transformers.__version__, semantic_ros_message=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
