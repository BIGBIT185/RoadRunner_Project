"""在线几何链的 GPU PointCloud2 字节解码和体素索引。

The ROS message itself is still received by Python, but field unpacking, finite
and range filtering, rigid transformation, and BEV cell indexing are performed
on CuPy arrays. The CPU decoder remains available for the semantic path and as
an explicit fallback when a bag uses an unsupported field layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_POINT_FIELD_FORMATS = {
    1: "i1",
    2: "u1",
    3: "i2",
    4: "u2",
    5: "i4",
    6: "u4",
    7: "f4",
    8: "f8",
}


@dataclass(frozen=True)
class GpuPointCloud:
    """Decoded GPU arrays in the source LiDAR frame."""

    xyz: object
    intensity: object
    height: int = 1
    width: int = 0

    @property
    def point_count(self) -> int:
        return int(self.xyz.shape[0])


def _decode_field(point_bytes, field, *, cp, big_endian: bool):
    format_code = _POINT_FIELD_FORMATS.get(int(field.datatype))
    if field.count != 1 or format_code is None:
        raise ValueError(f"unsupported PointCloud2 field: {field.name}")
    itemsize = np.dtype(format_code).itemsize
    offset = int(field.offset)
    if offset < 0 or offset + itemsize > point_bytes.shape[1]:
        raise ValueError(f"PointCloud2 field is outside point_step: {field.name}")
    # PointCloud2 points are already contiguous after removing row padding;
    # slicing the byte matrix avoids one large advanced-indexing allocation per
    # field.  The resulting view is decoded directly on the device.
    field_bytes = point_bytes[:, offset : offset + itemsize]
    # A strided view beginning at a non-word field offset can be misaligned for
    # CuPy's dtype view on older CUDA/Python stacks.  One device-local compact
    # copy keeps the decode safe and is still cheaper than CPU unpacking.
    field_bytes = field_bytes[:, ::-1] if big_endian and itemsize > 1 else field_bytes
    field_bytes = field_bytes.copy()
    native_dtype = cp.dtype(np.dtype(format_code))
    values = field_bytes.view(native_dtype).reshape(-1)
    return values.astype(cp.float32, copy=False)


def decode_pointcloud2_gpu(message, *, cp) -> GpuPointCloud:
    """Decode x/y/z/intensity from a ROS ``PointCloud2`` on the GPU.

    A NumPy ``uint8`` view is created only to expose the ROS byte buffer to
    CuPy; no point fields are unpacked or filtered on the CPU. Organized bags
    with row padding and both ROS byte orders are supported.
    """

    requested = ("x", "y", "z", "intensity")
    available = {field.name: field for field in message.fields}
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(f"PointCloud2 is missing fields: {missing}")
    height = int(message.height)
    width = int(message.width)
    point_step = int(message.point_step)
    row_step = int(message.row_step)
    if height < 0 or width < 0 or point_step <= 0 or row_step < width * point_step:
        raise ValueError("invalid PointCloud2 dimensions or strides")
    required_bytes = height * row_step
    host_bytes = np.frombuffer(message.data, dtype=np.uint8)
    if host_bytes.size < required_bytes:
        raise ValueError(
            "PointCloud2 data is shorter than height * row_step: "
            f"{host_bytes.size} < {required_bytes}"
        )
    if not height or not width:
        empty = cp.empty((0,), dtype=cp.float32)
        return GpuPointCloud(
            xyz=cp.empty((0, 3), dtype=cp.float32),
            intensity=empty,
            height=height,
            width=width,
        )

    raw = cp.asarray(host_bytes[:required_bytes])
    # Remove only row padding.  Each field is then a strided device view over
    # all points, so no CPU structured-array unpacking is needed.
    point_bytes = raw.reshape(height, row_step)[:, : width * point_step]
    point_bytes = point_bytes.reshape(-1, point_step)
    big_endian = bool(message.is_bigendian)
    x = _decode_field(point_bytes, available["x"], cp=cp, big_endian=big_endian)
    y = _decode_field(point_bytes, available["y"], cp=cp, big_endian=big_endian)
    z = _decode_field(point_bytes, available["z"], cp=cp, big_endian=big_endian)
    intensity = _decode_field(
        point_bytes,
        available["intensity"],
        cp=cp,
        big_endian=big_endian,
    )
    return GpuPointCloud(
        xyz=cp.stack((x, y, z), axis=1),
        intensity=intensity,
        height=height,
        width=width,
    )
