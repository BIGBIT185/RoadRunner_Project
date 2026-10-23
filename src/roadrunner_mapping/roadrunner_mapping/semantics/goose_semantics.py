"""GOOSE-64 固定语义配置。

类别顺序必须与 EoMT checkpoint 完全一致；风险字典、地面/动态/草地类别集合
都由类别 ID 派生，不能在运行时按名称重新排序。
"""

from __future__ import annotations

import numpy as np


# 模型输出的 label ID 直接索引这个元组，任何增删或重排都会破坏 checkpoint 契约。
GOOSE_CLASSES = (
    "undefined", "traffic_cone", "snow", "cobble", "obstacle", "leaves",
    "street_light", "bikeway", "ego_vehicle", "pedestrian_crossing",
    "road_block", "road_marking", "car", "bicycle", "person", "bus",
    "forest", "bush", "moss", "traffic_light", "motorcycle", "sidewalk",
    "curb", "asphalt", "gravel", "boom_barrier", "rail_track",
    "tree_crown", "tree_trunk", "debris", "crops", "soil", "rider",
    "animal", "truck", "on_rails", "caravan", "trailer", "building", "wall",
    "rock", "fence", "guard_rail", "bridge", "tunnel", "pole",
    "traffic_sign", "misc_sign", "barrier_tape", "kick_scooter", "low_grass",
    "high_grass", "scenery_vegetation", "sky", "water", "wire", "outlier",
    "heavy_machinery", "container", "hedge", "barrel", "pipe", "tree_root",
    "military_vehicle",
)

# 颜色仅供离线可视化或调试使用，在线 EoMT 节点不发布彩色 mask。
GOOSE_COLOR_PALETTE = (
    (0, 0, 0), (255, 255, 0), (209, 87, 160), (255, 52, 255),
    (255, 74, 70), (0, 137, 65), (0, 111, 166), (163, 0, 89),
    (255, 219, 229), (122, 73, 0), (0, 0, 166), (99, 255, 172),
    (183, 151, 98), (0, 77, 67), (143, 176, 255), (153, 125, 135),
    (90, 0, 7), (128, 150, 147), (180, 168, 189), (27, 68, 0),
    (79, 198, 1), (59, 93, 255), (74, 59, 83), (255, 47, 128),
    (97, 97, 90), (52, 54, 45), (107, 121, 0), (0, 194, 160),
    (255, 170, 146), (136, 111, 76), (0, 134, 237), (209, 97, 0),
    (221, 239, 255), (0, 0, 53), (123, 79, 75), (161, 194, 153),
    (48, 0, 24), (10, 166, 216), (1, 51, 73), (0, 132, 111),
    (55, 33, 1), (255, 181, 0), (194, 255, 237), (160, 121, 191),
    (204, 7, 68), (192, 185, 178), (194, 255, 153), (0, 30, 9),
    (190, 196, 89), (111, 0, 98), (12, 189, 102), (238, 195, 255),
    (69, 109, 117), (183, 123, 104), (122, 135, 161), (255, 140, 0),
    (120, 141, 102), (250, 208, 159), (255, 138, 154), (232, 211, 23),
    (208, 208, 0), (221, 0, 0), (196, 164, 132), (64, 64, 64),
)

# 风险定义：0 表示安全，1 表示危险；实时 GridMap 最终发布 1-risk 概率。
_GOOSE_RISKS = (
    1.00, 1.00, 0.60, 0.45, 1.00, 0.40, 1.00, 0.05,
    1.00, 0.05, 1.00, 0.05, 1.00, 1.00, 1.00, 1.00,
    0.90, 0.70, 0.35, 1.00, 1.00, 0.10, 0.65, 0.05,
    0.35, 1.00, 0.80, 1.00, 1.00, 0.90, 0.70, 0.30,
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
    1.00, 1.00, 1.00, 0.10, 0.40, 1.00, 1.00, 1.00,
    1.00, 1.00, 0.25, 0.55, 0.80, 1.00, 1.00, 1.00,
    1.00, 1.00, 1.00, 0.80, 1.00, 1.00, 0.80, 1.00,
)

GOOSE_SEMANTIC_RISK = {
    class_id: risk for class_id, risk in enumerate(_GOOSE_RISKS)
}
# 地面类用于语义投影中的地面概率和几何草地规则。
GOOSE_GROUND_CLASS_IDS = frozenset(
    {2, 3, 5, 7, 9, 11, 18, 21, 23, 24, 31, 43, 50, 51}
)
# 动态类不会直接删除几何点，只会影响语义栅格的最新观测策略。
GOOSE_DYNAMIC_CLASS_IDS = frozenset(
    {12, 13, 14, 15, 20, 32, 33, 34, 35, 36, 37, 49, 57, 63}
)
GOOSE_GRASS_CLASS_IDS = frozenset({50, 51})
GOOSE_INVALID_CLASS_IDS = frozenset({0, 8, 53, 56})
GOOSE_HARD_OBSTACLE_IDS = frozenset(
    class_id
    for class_id, risk in GOOSE_SEMANTIC_RISK.items()
    if risk >= 0.95 and class_id not in GOOSE_INVALID_CLASS_IDS
)


def colorize_goose_labels(labels: np.ndarray) -> np.ndarray:
    """把二维类别 ID mask 转换为官方调试颜色，仅用于可视化。"""

    label_map = np.asarray(labels)
    if label_map.ndim != 2 or not np.issubdtype(label_map.dtype, np.integer):
        raise ValueError("GOOSE labels must be a two-dimensional integer array")
    if np.any(label_map < 0) or np.any(label_map >= len(GOOSE_COLOR_PALETTE)):
        raise ValueError("GOOSE label id is outside [0,63]")
    palette = np.asarray(GOOSE_COLOR_PALETTE, dtype=np.uint8)
    return np.ascontiguousarray(palette[label_map])
