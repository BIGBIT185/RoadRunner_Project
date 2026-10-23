"""语义推理模型的类别与风险配置。

实时链通过 ``semantic_profile`` 选择模型输出的类别契约。RUGD EoMT
checkpoint 是 24 类（没有 void 类），因此不能直接复用旧的 25 类
DeepLab/RUGD 编号，也不能把它当成 GOOSE-64 输出。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

from .goose_semantics import (
    GOOSE_CLASSES,
    GOOSE_DYNAMIC_CLASS_IDS,
    GOOSE_GRASS_CLASS_IDS,
    GOOSE_GROUND_CLASS_IDS,
    GOOSE_INVALID_CLASS_IDS,
    GOOSE_SEMANTIC_RISK,
)


@dataclass(frozen=True)
class EomtSemanticProfile:
    """一个 EoMT checkpoint 的完整输出契约。"""

    name: str
    classes: Tuple[str, ...]
    semantic_risk: Mapping[int, float]
    ground_class_ids: frozenset
    dynamic_class_ids: frozenset
    invalid_class_ids: frozenset
    grass_class_ids: frozenset

    @property
    def class_count(self) -> int:
        return len(self.classes)


# 最新 RUGD EoMT checkpoint 的 config.json 中 id2label 顺序。
RUGD24_CLASSES = (
    "dirt",
    "sand",
    "grass",
    "tree",
    "pole",
    "water",
    "sky",
    "vehicle",
    "container/generic-object",
    "asphalt",
    "gravel",
    "building",
    "mulch",
    "rock-bed",
    "log",
    "bicycle",
    "person",
    "fence",
    "bush",
    "sign",
    "rock",
    "bridge",
    "concrete",
    "picnic-table",
)

# 风险定义与原 RUGD 语义链保持一致，但去掉旧 DeepLab 的 void=0，
# 所有 ID 因此整体前移一位。
RUGD24_SEMANTIC_RISK = {
    0: 0.30,  # dirt
    1: 0.50,  # sand
    2: 0.25,  # grass
    3: 1.00,  # tree
    4: 1.00,  # pole
    5: 1.00,  # water
    6: 1.00,  # sky
    7: 1.00,  # vehicle
    8: 1.00,  # container/generic-object
    9: 0.05,  # asphalt
    10: 0.35,  # gravel
    11: 1.00,  # building
    12: 0.40,  # mulch
    13: 0.70,  # rock-bed
    14: 1.00,  # log
    15: 1.00,  # bicycle
    16: 1.00,  # person
    17: 1.00,  # fence
    18: 0.70,  # bush
    19: 1.00,  # sign
    20: 1.00,  # rock
    21: 0.10,  # bridge
    22: 0.10,  # concrete
    23: 1.00,  # picnic-table
}
RUGD24_GROUND_CLASS_IDS = frozenset({0, 1, 2, 9, 10, 12, 13, 21, 22})
RUGD24_DYNAMIC_CLASS_IDS = frozenset({7, 15, 16})
RUGD24_GRASS_CLASS_IDS = frozenset({2})
RUGD24_INVALID_CLASS_IDS = frozenset()


GOOSE64_PROFILE = EomtSemanticProfile(
    name="goose64",
    classes=tuple(GOOSE_CLASSES),
    semantic_risk=GOOSE_SEMANTIC_RISK,
    ground_class_ids=frozenset(GOOSE_GROUND_CLASS_IDS),
    dynamic_class_ids=frozenset(GOOSE_DYNAMIC_CLASS_IDS),
    invalid_class_ids=frozenset(GOOSE_INVALID_CLASS_IDS),
    grass_class_ids=frozenset(GOOSE_GRASS_CLASS_IDS),
)

RUGD24_PROFILE = EomtSemanticProfile(
    name="rugd24",
    classes=RUGD24_CLASSES,
    semantic_risk=RUGD24_SEMANTIC_RISK,
    ground_class_ids=RUGD24_GROUND_CLASS_IDS,
    dynamic_class_ids=RUGD24_DYNAMIC_CLASS_IDS,
    invalid_class_ids=RUGD24_INVALID_CLASS_IDS,
    grass_class_ids=RUGD24_GRASS_CLASS_IDS,
)

EOMT_PROFILES = {
    GOOSE64_PROFILE.name: GOOSE64_PROFILE,
    RUGD24_PROFILE.name: RUGD24_PROFILE,
}


def get_eomt_profile(name: str) -> EomtSemanticProfile:
    """按名称取得 EoMT 类别契约，并拒绝未定义配置。"""

    profile_name = str(name).strip().lower()
    try:
        return EOMT_PROFILES[profile_name]
    except KeyError as error:
        supported = ", ".join(sorted(EOMT_PROFILES))
        raise ValueError(
            f"unknown semantic profile {name!r}; supported profiles: {supported}"
        ) from error
