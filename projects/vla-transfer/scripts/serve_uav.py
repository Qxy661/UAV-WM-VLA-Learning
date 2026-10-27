"""用我们自己的 TrainConfig 起一个 π0.5 策略服务。

官方的 `serve_policy.py` 只认注册在 `openpi.training.config._CONFIGS` 里的配置，
我们的 `pi05_uav` 是运行时注册的，所以这里自己走一遍 create_trained_policy。
归一化统计量显式从 `assets/<config>/<asset_id>/norm_stats.json` 读，而不是等
checkpoint 目录里那份（训练不会把 assets 拷进 ckpt）。
"""

from __future__ import annotations

import dataclasses
import logging
import os
import sys
from pathlib import Path

OPENPI = Path("/opt/robogo/tools/openpi")
sys.path.insert(0, str(OPENPI / "src"))
sys.path.insert(0, str(OPENPI))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "policy"))

import uav_config  # noqa: E402  注册 pi05_uav
from openpi.policies.policy_config import create_trained_policy  # noqa: E402
from openpi.serving.websocket_policy_server import WebsocketPolicyServer  # noqa: E402
from openpi.training import checkpoints as _checkpoints  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> None:
    repo = os.environ.get("UAV_REPO", "uav/inspect96")
    exp = os.environ.get("UAV_EXP", "a3_uav6000")
    step = os.environ.get("UAV_STEP", "6000")
    port = int(os.environ.get("UAV_PORT", "8000"))

    # tier 必须一起传：a1 冻结主干（ckpt 里没有 LoRA 参数），a3 主干与动作专家都挂 LoRA。
    # 漏传会退化成默认 a3，加载 a1 时报 "expected dict with 6 children, got 2"
    # （缺 gating_einsum_lora_a/b、linear_lora_a/b），看着像权重坏了，其实是结构不匹配。
    tier = os.environ.get("UAV_TIER", "a3")
    cfg = uav_config.build_config(repo_id=repo, tier=tier,
                                  num_train_steps=int(os.environ.get("UAV_STEPS", "6000")),
                                  batch_size=1, exp_name=exp)

    ckpt = OPENPI / "checkpoints" / cfg.name / exp / step
    if not ckpt.exists():
        raise SystemExit(f"checkpoint 不存在：{ckpt}")
    norm_stats = _checkpoints.load_norm_stats(cfg.assets_dirs, repo)
    logging.info("norm stats loaded from %s", cfg.assets_dirs)

    # 关键：推理时不要传 repack。repack 是给 LeRobot 列名做一次改名（image→observation/image），
    # 里面明确要求 "actions" 键；而推理请求里没有动作，会 KeyError: 'actions'。
    # 官方 serve_policy 也是不传。我们的客户端本来就发 observation/* 键，UavInputs 直接吃。
    policy = create_trained_policy(cfg, ckpt, norm_stats=norm_stats)
    logging.info("policy ready, listening on 0.0.0.0:%d", port)
    WebsocketPolicyServer(policy, host="0.0.0.0", port=port).serve_forever()


if __name__ == "__main__":
    main()
