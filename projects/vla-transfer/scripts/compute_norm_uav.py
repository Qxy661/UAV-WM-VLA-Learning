"""给 pi05_uav 计算归一化统计量（官方脚本的注册版）。

官方的 ``scripts/compute_norm_stats.py`` 只认 ``_CONFIGS_DICT`` 里的配置，我们的
``pi05_uav`` 是运行时注册的，所以这里先注册再直接调用官方 main。
"""

from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/opt/robogo/tools/openpi/scripts")

from policy import uav_config  # noqa: E402
import compute_norm_stats as _cns  # noqa: E402


def main() -> None:
    cfg = uav_config.register(uav_config.build_config(
        repo_id=os.environ.get("UAV_REPO", "uav/inspect96"),
        num_train_steps=int(os.environ.get("UAV_STEPS", "30000")),
        batch_size=int(os.environ.get("UAV_BATCH", "32")),
        exp_name=os.environ.get("UAV_EXP", "uav_a3"),
    ))
    _cns.main(cfg.name, max_frames=int(os.environ.get("UAV_MAX_FRAMES", "200000")))


if __name__ == "__main__":
    main()
