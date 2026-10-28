"""多机任务分配：拍卖式出价 + 丢包下的重分配 + 与集中式最优的差距。

为什么不用「假装完美的集中式分配」：真机上通信会丢包、会延迟、会有机器掉线，
分配算法必须能在信息不全时给出一个可执行、可重分配的结果。这里把通信抽象成
`comm.Link`（丢包率、延迟、固定种子），分配器只看得到「成功收到的出价」。

与最优的对比用精确 DP（任务数 ≤ 16），不是启发式对启发式——否则差距里混着
两边的近似误差，说不清是算法差还是实现差。
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .comm import Link


@dataclass(frozen=True)
class Task:
    """一个待巡检点。`pos` 是三维坐标，`service_s` 是到点后悬停拍摄的固定耗时。"""

    tid: int
    pos: tuple[float, float, float]
    service_s: float = 3.0


@dataclass(frozen=True)
class Drone:
    did: int
    pos: tuple[float, float, float]
    speed: float = 1.6  # m/s，与 m1_multi 的巡航速度同量级


def travel_s(a: np.ndarray, b: np.ndarray, speed: float) -> float:
    return float(np.linalg.norm(a - b)) / max(speed, 1e-6)


def cost_matrix(tasks: list[Task], drones: list[Drone]) -> np.ndarray:
    c = np.zeros((len(drones), len(tasks)), np.float64)
    for i, d in enumerate(drones):
        for j, t in enumerate(tasks):
            c[i, j] = travel_s(np.asarray(d.pos), np.asarray(t.pos), d.speed) + t.service_s
    return c


def greedy_global(tasks: list[Task], drones: list[Drone], *, link: Link | None = None) -> Assignment:
    """全局贪心：每次在「所有 (机器, 任务) 对」里挑代价最小的那个。

    这是拍卖的天然对照组：如果两者差距很小，说明顺序拍卖的损失来自
    「逐任务决策」而不是「没有报价机制」；如果差距大，说明出价结构有用。
    """
    c = cost_matrix(tasks, drones)
    link = link or Link(loss=0.0, delay_s=0.0, seed=0)
    free = set(range(len(tasks)))
    used = np.zeros(len(drones), np.float64)
    loads: dict[int, list[int]] = {d.did: [] for d in drones}
    lost = msgs = 0
    while free:
        best = (-1, -1, float("inf"))
        for tid in sorted(free):
            for i, _ in enumerate(drones):
                msgs += 1
                if not link.send():
                    lost += 1
                    continue
                val = used[i] + c[i][tid]
                if val < best[2]:
                    best = (i, tid, val)
        if best[0] < 0:
            break
        i, tid, val = best
        loads[drones[i].did].append(tid)
        used[i] = val
        free.discard(tid)
    return Assignment(loads=loads, rounds=1, bids_lost=lost, messages=msgs)


def optimal_makespan(c: np.ndarray) -> float:
    """精确最优：把任务集合划分给各机，取各机总耗时最大值的最小值。

    任务数 ≤ 16 时用子集 DP：`f[mask]` = 前 k 台机器处理完 mask 的最小时刻。
    分给同一台机器的任务之间按最近邻串行，这里为了口径简单，直接用
    「该机所有任务耗时之和」——不含排序优化，与拍卖侧的估价口径一致。
    """
    n_d, n_t = c.shape
    full = (1 << n_t) - 1
    INF = float("inf")
    prev = [INF] * (full + 1)
    prev[0] = 0.0
    for i in range(n_d):
        cur = [INF] * (full + 1)
        for mask in range(full + 1):
            if prev[mask] == INF:
                continue
            rest = full ^ mask
            sub = rest
            while True:
                load = float(c[i][[j for j in range(n_t) if sub >> j & 1]].sum()) if sub else 0.0
                t = max(prev[mask], load)
                if t < cur[mask | sub]:
                    cur[mask | sub] = t
                if sub == 0:
                    break
                sub = (sub - 1) & rest
        prev = cur
    return float(prev[full])


@dataclass
class Assignment:
    """分配结果：每台机器拿到哪些任务，以及出价轮次与耗时。"""

    loads: dict[int, list[int]] = field(default_factory=dict)
    rounds: int = 0
    bids_lost: int = 0
    messages: int = 0

    def makespan(self, c: np.ndarray, tasks: list[Task]) -> float:
        worst = 0.0
        for did, tids in self.loads.items():
            s = 0.0
            for tid in tids:
                s += float(c[did][tid])
            worst = max(worst, s)
        return worst


def auction_allocate(tasks: list[Task], drones: list[Drone], *, link: Link | None = None,
                     max_rounds: int = 8) -> Assignment:
    """顺序拍卖：每轮把剩余任务按「当前估价最低的机器」依次分配。

    丢包体现在出价上：机器对某任务的出价以概率 `loss` 丢掉；丢掉的机器这一轮
    不参与该任务的竞争——于是分配会比无丢包时差一点，这正是要测的东西。
    """
    c = cost_matrix(tasks, drones)
    link = link or Link(loss=0.0, delay_s=0.0, seed=0)
    free = set(range(len(tasks)))
    loads: dict[int, list[int]] = {d.did: [] for d in drones}
    used = np.zeros(len(drones), np.float64)
    rounds = 0
    lost = 0
    msgs = 0

    while free and rounds < max_rounds:
        rounds += 1
        progressed = False
        for tid in sorted(free):
            best_i, best_val = -1, float("inf")
            for i, _ in enumerate(drones):
                msgs += 1
                arrived = link.send()
                if not arrived:
                    lost += 1
                    continue
                val = used[i] + c[i][tid]
                if val < best_val:
                    best_i, best_val = i, val
            if best_i < 0:
                continue
            loads[drones[best_i].did].append(tid)
            used[best_i] += c[best_i][tid]
            free.discard(tid)
            progressed = True
        if not progressed:
            break
    return Assignment(loads=loads, rounds=rounds, bids_lost=lost, messages=msgs)


def reassign_after_failure(tasks: list[Task], drones: list[Drone], dead: int, *,
                          assignment: Assignment, link: Link | None = None) -> tuple[Assignment, float]:
    """一台机器掉线后，把它名下的任务在同一轮里重新拍卖。

    返回 (新分配, 重分配耗时估计)。耗时估计只算通信与计算，不含飞行——
    飞行时间已经在 makespan 里体现。
    """
    keep = [d for d in drones if d.did != dead]
    orphan = list(assignment.loads.get(dead, []))
    if not orphan or not keep:
        return assignment, 0.0
    c = cost_matrix(tasks, keep)
    link = link or Link(loss=0.0, delay_s=0.0, seed=0)
    used = {}
    for d in keep:
        used[d.did] = sum(float(c[i][tid]) for i, dd in enumerate(keep) for tid in assignment.loads.get(d.did, []))
    loads = {d.did: list(assignment.loads.get(d.did, [])) for d in keep}
    t_comm = 0.0
    for tid in orphan:
        best_i, best_val = -1, float("inf")
        for i, d in enumerate(keep):
            t_comm += link.delay_s
            if not link.send():
                continue
            val = used[d.did] + c[i][tid]
            if val < best_val:
                best_i, best_val = i, val
        if best_i >= 0:
            loads[keep[best_i].did].append(tid)
            used[keep[best_i].did] += c[best_i][tid]
    return Assignment(loads=loads, rounds=assignment.rounds + 1,
                      bids_lost=assignment.bids_lost), t_comm


def random_problem(n_tasks: int, n_drones: int, seed: int,
                   bound: float = 4.0) -> tuple[list[Task], list[Drone]]:
    rng = np.random.default_rng(seed)
    tasks = [Task(tid=j, pos=tuple(rng.uniform(-bound, bound, 3))) for j in range(n_tasks)]
    drones = [Drone(did=i, pos=tuple(rng.uniform(-bound, bound, 3))) for i in range(n_drones)]
    return tasks, drones


def all_assignments(n_tasks: int, n_drones: int):
    """小规模穷举（只用于交叉验证 DP 最优是对的）。"""
    for combo in itertools.product(range(n_drones), repeat=n_tasks):
        yield combo
