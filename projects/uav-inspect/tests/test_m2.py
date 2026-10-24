"""M2 的契约测试：相机几何、数据集格式、划分与统计口径。

跑法（CI 与本机）：
    python -m pytest projects/uav-inspect/tests -q
没有 pytest 时用等价的离线脚本：python projects/uav-inspect/scripts/check_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from uavinspect.config import load                                  # noqa: E402
from uavinspect.data.recorder import (EpisodeRecorder, compute_stats,  # noqa: E402
                                      write_manifest)
from uavinspect.sim import camera as cam                            # noqa: E402
from uavinspect.sim.rigid import RigidEngine                        # noqa: E402
from uavinspect.sim.scene import build_warehouse                    # noqa: E402

CFG = PKG / "configs" / "default.yaml"


def test_camera_down_ray_hits_ground():
    """正下方一条射线：地面命中距离应等于高度（地面在 z=0）。"""
    h = cam.raycast(np.array([0.0, 0.0, 2.0]), np.array([[0.0, 0.0, -1.0]]), boxes=[])
    assert h.box_id[0] == -1
    assert abs(float(h.t[0]) - 2.0) < 1e-6


def test_camera_box_blocks_target():
    """目标前放一块板：射线先打到板，目标判为不可见。"""
    engine = RigidEngine(load(CFG))
    engine.add_box((1.0, 0.1, 1.0), (0.0, 1.0, 1.0))
    engine.add_box((0.1, 0.1, 0.1), (0.0, 2.0, 1.0))
    boxes = engine.boxes
    eye = np.array([0.0, 0.0, 1.0])
    target = np.array([0.0, 2.0, 1.0])
    assert not cam.target_visible(eye, target, boxes=boxes)
    assert not cam.target_visible(eye, target, boxes=[boxes[1]])   # 没有板就能看见


def test_in_view_rejects_behind_camera():
    """目标在相机背后：方位角差 180°，不能算在视野内。"""
    ok = cam.in_view(np.array([0.0, 0.0, 1.0]), np.array([0.0, 2.0, 1.0]),
                     yaw=0.0, hfov_deg=70.0, boxes=[])
    assert not ok
    ok2 = cam.in_view(np.array([0.0, 0.0, 1.0]), np.array([0.0, 2.0, 1.0]),
                      yaw=np.pi / 2, hfov_deg=70.0, boxes=[])
    assert ok2


def test_scene_target_occluded_from_behind_rack():
    """仓储场景：目标从 behind-rack 侧看不见，从通道侧看得见。"""
    engine = RigidEngine(load(CFG))
    scene = build_warehouse(engine, 0)
    labels = [lab for _, _, lab in scene.viewpoints]
    visible = [cam.target_visible(np.asarray(p, float), scene.target_pos, boxes=scene.boxes)
               for p, _, _ in scene.viewpoints]
    assert "behind-rack" in labels
    assert visible[labels.index("behind-rack")] is False
    assert visible[labels.index("front")] is True


def test_recorder_roundtrip_and_split(tmp_path: Path):
    """写三集再读回：字段齐全、划分按 episode、stats 只用 train。"""
    for ep, (seed, z) in enumerate([(0, 1.0), (1, 1.0), (2, 3.0)]):
        rec = EpisodeRecorder(tmp_path / f"episode_{ep:04d}")
        for k in range(4):
            rec.add(pos=np.array([0.0, 0.0, z]), vel=np.zeros(3),
                    quat=np.array([1.0, 0.0, 0.0, 0.0]), action=np.zeros(4), t=0.02 * k,
                    phase=0, image=np.full((8, 8, 3), 10 * ep, np.uint8))
        rec.close(meta={"seed": seed, "coverage": 1.0})
    eps = [{"id": "e0", "dir": "episode_0000", "seed": 0, "frames": 4},
           {"id": "e1", "dir": "episode_0001", "seed": 1, "frames": 4},
           {"id": "e2", "dir": "episode_0002", "seed": 2, "frames": 4}]
    manifest = write_manifest(tmp_path, eps)
    splits = {e["seed"]: e["split"] for e in manifest["episodes"]}
    assert set(splits) == {0, 1, 2}
    assert splits[2] == "val"          # val_every=3 → 第 3 个进验证集
    stats = compute_stats(tmp_path, manifest)
    # 3.0 那集在 val 里；若泄漏进 train，均值会被拉到 2.0
    assert abs(stats["pos"]["mean"][2] - 1.0) < 1e-9
    assert stats["episodes"] == 2
    z = np.load(tmp_path / "episode_0000" / "obs.npz")
    for key in ("images", "pos", "vel", "quat", "action", "t", "phase", "frame_t"):
        assert key in z, f"obs.npz 缺字段 {key}"
    assert (tmp_path / "manifest.json").exists()
    assert json.loads((tmp_path / "stats.json").read_text(encoding="utf-8"))["n_episodes"] == 1


def _run(seed: int, out: Path) -> dict:
    from m2_inspect import run_episode      # scripts/ 已由 fixture 注入 sys.path
    return run_episode(load(CFG), seed, out / f"episode_{seed:04d}", "rigid")


def test_episode_is_deterministic(tmp_path: Path):
    """同一 seed 跑两遍：位姿与图像逐位相同。"""
    _run(3, tmp_path / "a")
    _run(3, tmp_path / "b")
    za = np.load(tmp_path / "a" / "episode_0003" / "obs.npz")
    zb = np.load(tmp_path / "b" / "episode_0003" / "obs.npz")
    assert np.array_equal(za["pos"], zb["pos"])
    assert np.array_equal(za["images"], zb["images"])


def test_two_episodes_cover_all_feasible_viewpoints(tmp_path: Path):
    """真实跑两集：覆盖率必须是 1.0（可行视点全拍到），route_time 有值。"""
    for seed in (0, 1):
        ep = _run(seed, tmp_path / f"run{seed}")
        assert ep["coverage"] == 1.0, ep
        assert ep["occluded_viewpoints"] == ["behind-rack"], ep
        assert ep["route_time_s"] and ep["route_time_s"] > 0
        assert ep["frames"] > 30


@pytest.fixture(scope="module", autouse=True)
def _path():
    scripts = PKG / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    yield
