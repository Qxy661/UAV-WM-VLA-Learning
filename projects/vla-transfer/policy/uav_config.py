"""把无人机巡检数据集注册成一个 openpi 的 TrainConfig。

用法（开发机上）::

    cd /opt/robogo/tools/openpi
    export HF_LEROBOT_HOME=/root/work/hf_lerobot
    export PYTHONPATH=/root/work/vla:$PYTHONPATH
    python /root/work/vla/scripts/train_uav.py pi05_uav --exp-name uav_a3 --num-train-steps 2 --batch-size 2

设计取舍（写进 PLAN.md 的 P3）：
- ``action_dim`` 保持官方默认 32，而不是改成 4。原因是 π0.5 的 action_in/out 投影层形状由它
  决定，改成 4 会让预训练权重全部 mismatch；10 维状态与 4 维动作由 ``PadStatesAndActions``
  补零到 32，输出再切回 4。代价是损失里有 28 维恒零目标（官方 LIBERO 配置同样如此）。
- 动作是**绝对量**（归一化推力/力矩），不是 delta，所以不加 ``DeltaActions``。
- ``prompt_from_task=True``：指令存在 LeRobot 的 tasks.jsonl 里，按 task_index 取回。
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import sys

import openpi.models.pi0_config as pi0_config
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from policy import uav_policy  # noqa: E402


@dataclasses.dataclass(frozen=True)
class LeRobotUavDataConfig(DataConfigFactory):
    action_dim_out: int = int(os.environ.get("UAV_ACTION_DIM", "4"))

    def create(self, assets_dirs: pathlib.Path, model_config) -> DataConfig:
        repack = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[uav_policy.UavInputs(model_type=model_config.model_type)],
            outputs=[uav_policy.UavOutputs(action_dim_out=self.action_dim_out)],
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
        )


# 三档消融：可训练参数的范围不同，其余（数据、步数、学习率、评测）完全一致。
#   a1 = 只训动作头（action_in/out 投影 + time_mlp），其余全冻结
#   a2 = 视觉语言主干挂 LoRA，动作专家冻结（对应"动作头 + 视觉 LoRA"）
#   a3 = 主干与动作专家都挂 LoRA（官方 low_mem_finetune 的写法）
HEAD_RE = r".*(action_in_proj|action_out_proj|time_mlp).*"


def _variants(tier: str) -> tuple[str, str]:
    if tier == "a1":
        return "gemma_2b", "gemma_300m"
    if tier == "a2":
        return "gemma_2b_lora", "gemma_300m"
    if tier == "a3":
        return "gemma_2b_lora", "gemma_300m_lora"
    raise ValueError(f"未知的消融档位 {tier!r}，应为 a1 / a2 / a3")


def build_config(*, repo_id: str, tier: str = "a3", num_train_steps: int = 30000,
                 batch_size: int = 32, exp_name: str | None = None,
                 action_horizon: int = 8) -> _config.TrainConfig:
    dummy = os.environ.get("UAV_DUMMY") == "1"
    from flax import nnx
    import openpi.shared.nnx_utils as nnx_utils

    paligemma, expert = (("dummy", "dummy") if dummy else _variants(tier))
    model = pi0_config.Pi0Config(
        pi05=True, action_horizon=action_horizon, discrete_state_input=False,
        paligemma_variant=paligemma, action_expert_variant=expert,
    )
    if dummy:
        freeze = nnx.Nothing
    elif tier == "a1":
        # 冻结除动作头以外的一切：trainable = Param ∧ ¬Not(head) = Param ∧ head
        freeze = nnx.Not(nnx_utils.PathRegex(HEAD_RE))
    else:
        freeze = model.get_freeze_filter()

    # 权重来源：dummy 用空加载器；设了 UAV_INIT 就从本项目自己的 ckpt 续训
    # （跨仓库微调用，传 .../<exp>/<step>/params）；否则从 pi05_base 起。
    if dummy:
        loader = weight_loaders.NoOpWeightLoader()
    elif os.environ.get("UAV_INIT"):
        loader = weight_loaders.CheckpointWeightLoader(os.environ["UAV_INIT"])
    else:
        loader = weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params")
    return _config.TrainConfig(
        # name 固定为 pi05_uav：归一化统计量按 name 存放在 assets/pi05_uav/ 下，
        # 三档消融共用同一份统计量；区分实验靠 exp_name。
        name="pi05_uav",
        model=model,
        data=LeRobotUavDataConfig(
            repo_id=repo_id,
            base_config=DataConfig(prompt_from_task=True),
        ),
        batch_size=batch_size,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(1_000, max(num_train_steps // 10, 1)),
            peak_lr=float(os.environ.get("UAV_LR", "5e-5")),
            decay_steps=max(num_train_steps, 2_000),
            decay_lr=float(os.environ.get("UAV_LR", "5e-5")),
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None if tier != "a1" or dummy else 0.999,
        weight_loader=loader,
        freeze_filter=freeze,
        num_train_steps=num_train_steps,
        exp_name=exp_name or f"uav_{tier}",
        # 数据侧瓶颈在这里：每样本要解 8 张 PNG，num_workers=0 时单进程解码，
        # 实测训练 3.4 s/步（batch 4）、算 norm stats 1.55 样本/秒。
        # 开发机 cgroup 只给 24 核，留 12 个给 DataLoader。
        num_workers=int(os.environ.get("UAV_WORKERS", "12")),
        wandb_enabled=False,   # 开发机没有 wandb key；指标看 stdout 与 checkpoints 目录
        overwrite=os.environ.get("UAV_RESUME") != "1",
    )


def register(cfg: _config.TrainConfig) -> _config.TrainConfig:
    _config._CONFIGS.append(cfg)
    _config._CONFIGS_DICT[cfg.name] = cfg
    return cfg


if __name__ == "__main__":
    # 官方训练入口是 scripts/train.py（不在包里），用文件路径挂进去后调它的 main。
    sys.path.insert(0, "/opt/robogo/tools/openpi/scripts")
    import train as _train  # noqa: E402

    cfg = register(build_config(
        repo_id=os.environ.get("UAV_REPO", "uav/inspect96"),
        tier=os.environ.get("UAV_TIER", "a3"),
        num_train_steps=int(os.environ.get("UAV_STEPS", "30000")),
        batch_size=int(os.environ.get("UAV_BATCH", "32")),
        exp_name=os.environ.get("UAV_EXP") or None,
    ))
    _train.main(cfg)