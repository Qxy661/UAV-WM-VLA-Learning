"""π0.5 的无人机输入/输出变换。

照 `openpi/src/openpi/policies/libero_policy.py` 改：把「腕部相机 + 8 维状态 + 7 维动作」
换成「单目 + 10 维状态 + 4 维动作」。三处必须与官方一致，否则训练/推理会静默错位：

1. 图像键名固定为 ``observation/image``，进来可能是 float 或 CHW，先统一成 uint8 HWC。
2. ``image_mask`` 只对真正提供的视角为 True。我们只有一路相机，两路腕部相机传全零图并
   把 mask 置 False；模型不会去 attend 它们（pi0-fast 的约定不同，见官方注释）。
3. 动作只切前 ``action_dim_out`` 维。模型侧统一 pad 到 32 维，输出必须切回来，否则上层会
   拿到 28 个恒零的假动作。
"""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _load_if_path(x):
    """LeRobot 的 image 特征存的是每帧 PNG 路径，不是数组；必须先解码再进模型。"""
    if isinstance(x, (str, bytes)) or hasattr(x, "read"):
        import io
        import os

        from PIL import Image

        if isinstance(x, str) and not os.path.isabs(x):
            x = os.path.join(os.environ.get("UAV_DATASET_ROOT", "."), x)
        buf = io.BytesIO(x) if isinstance(x, bytes) else x
        return np.asarray(Image.open(buf).convert("RGB"))
    return x


def _parse_image(image) -> np.ndarray:
    image = np.asarray(_load_if_path(image))
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class UavInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        zeros = np.zeros_like(base_image)
        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": zeros,
                "right_wrist_0_rgb": zeros,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class UavOutputs(transforms.DataTransformFn):
    action_dim_out: int = 4

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim_out])}
