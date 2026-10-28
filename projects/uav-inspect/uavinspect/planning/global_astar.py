"""二维全局栅格 A*：把货架当矩形障碍，膨胀一个机体半径再搜。

为什么要有它：`avoidance.py` 的斥力是反应式的，只看当前几米，遇到「货架在中间、
目标和自己在两侧」这种需要绕远的局面，斥力会把飞机顶在货架前直到超时。全局规划
补的正是这段「先退后绕」的能力。

约定：平面是 (x, y)，高度固定为悬停高度；障碍用轴对齐矩形 (cx, cy, hx, hy) 表示。
不做三维是因为本任务货架贯通到 0.75 m×2 之外、机体也只在这一层活动，加 z 维度只会
让格子数乘十倍而不改变结论。
"""

from __future__ import annotations

import heapq
import math

import numpy as np

Cell = tuple[int, int]


def racks_2d(rack_x, rack_y: float, half_x: float, half_y: float) -> list[tuple[float, float, float, float]]:
    """把货架列表摊成二维矩形：中心 (x, 0)、半长 (half_x, half_y)。"""
    return [(float(x), 0.0, float(half_x), float(half_y)) for x in rack_x]


def clearance_2d(p: np.ndarray, boxes: list[tuple[float, float, float, float]]) -> float:
    """点到最近矩形表面的距离；点在矩形内部时返回负值（用来判碰撞）。"""
    best = float("inf")
    for cx, cy, hx, hy in boxes:
        dx = abs(float(p[0]) - cx) - hx
        dy = abs(float(p[1]) - cy) - hy
        if dx <= 0 and dy <= 0:
            best = min(best, max(dx, dy))
        else:
            best = min(best, math.hypot(max(dx, 0.0), max(dy, 0.0)))
    return best


class Grid:
    """房间内的占用栅格；`inflate` 是机体半径加安全余量。"""

    def __init__(self, room, boxes, *, res: float = 0.10, inflate: float = 0.30) -> None:
        self.res = res
        self.xmin, self.ymin = -room["x"] / 2 + 0.20, -room["y"] / 2 + 0.20
        self.xmax, self.ymax = room["x"] / 2 - 0.20, room["y"] / 2 - 0.20
        self.nx = int((self.xmax - self.xmin) / res) + 1
        self.ny = int((self.ymax - self.ymin) / res) + 1
        self.occ = np.zeros((self.nx, self.ny), dtype=bool)
        for cx, cy, hx, hy in boxes:
            i0, i1 = self._span(cx - hx - inflate, self.xmin, self.nx)
            j0, j1 = self._span(cy - hy - inflate, self.ymin, self.ny)
            i2, i3 = self._span(cx + hx + inflate, self.xmin, self.nx)
            j2, j3 = self._span(cy + hy + inflate, self.ymin, self.ny)
            self.occ[min(i0, i2):max(i1, i3) + 1, min(j0, j2):max(j1, j3) + 1] = True

    def _span(self, v: float, lo: float, n: int) -> tuple[int, int]:
        a = int(round((v - lo) / self.res))
        return max(0, min(n - 1, a)), max(0, min(n - 1, a))

    def to_cell(self, p) -> Cell:
        i = int(round((float(p[0]) - self.xmin) / self.res))
        j = int(round((float(p[1]) - self.ymin) / self.res))
        return max(0, min(self.nx - 1, i)), max(0, min(self.ny - 1, j))

    def to_xy(self, c: Cell) -> np.ndarray:
        return np.array([self.xmin + c[0] * self.res, self.ymin + c[1] * self.res])

    def free(self, c: Cell) -> bool:
        return not self.occ[c[0], c[1]]

    def nearest_free(self, c: Cell, *, max_ring: int = 20) -> Cell:
        """起终点可能落在膨胀层里：向外找最近的空格。"""
        if self.free(c):
            return c
        for r in range(1, max_ring):
            for di in range(-r, r + 1):
                for dj in (-r, r) if abs(di) != r else range(-r, r + 1):
                    ci, cj = c[0] + di, c[1] + dj
                    if 0 <= ci < self.nx and 0 <= cj < self.ny and self.free((ci, cj)):
                        return ci, cj
        return c


def astar(grid: Grid, start, goal) -> list[np.ndarray]:
    """返回从 start 到 goal 的折线（世界坐标，含两端）。无解时返回空列表。"""
    s = grid.nearest_free(grid.to_cell(start))
    g = grid.nearest_free(grid.to_cell(goal))
    if s == g:
        return [grid.to_xy(s)]

    def h(c: Cell) -> float:
        dx, dy = abs(c[0] - g[0]), abs(c[1] - g[1])
        return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)

    openh: list[tuple[float, Cell]] = [(h(s), s)]
    came: dict[Cell, Cell] = {}
    cost: dict[Cell, float] = {s: 0.0}
    neigh = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]
    seen = set()
    while openh:
        _, c = heapq.heappop(openh)
        if c in seen:
            continue
        seen.add(c)
        if c == g:
            path = [c]
            while c in came:
                c = came[c]
                path.append(c)
            return _smooth(grid, [grid.to_xy(x) for x in reversed(path)])
        for di, dj, w in neigh:
            nc = (c[0] + di, c[1] + dj)
            if not (0 <= nc[0] < grid.nx and 0 <= nc[1] < grid.ny) or not grid.free(nc):
                continue
            if di and dj and (not grid.free((c[0] + di, c[1])) or not grid.free((c[0], c[1] + dj))):
                continue  # 禁止贴角切过去
            nc_cost = cost[c] + w
            if nc_cost < cost.get(nc, float("inf")):
                cost[nc] = nc_cost
                came[nc] = c
                heapq.heappush(openh, (nc_cost + h(nc), nc))
    return []


def _smooth(grid: Grid, path: list[np.ndarray]) -> list[np.ndarray]:
    """视线剪枝：能直连就跳过中间点。"""
    if len(path) <= 2:
        return path
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if _line_free(grid, path[i], path[j]):
                break
            j -= 1
        out.append(path[j])
        i = j
    return out


def _line_free(grid: Grid, a: np.ndarray, b: np.ndarray) -> bool:
    n = int(np.linalg.norm(b - a) / (grid.res * 0.5)) + 1
    for k in range(n + 1):
        p = a + (b - a) * (k / n)
        if not grid.free(grid.to_cell(p)):
            return False
    return True


def path_length(path: list[np.ndarray]) -> float:
    return float(sum(np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1)))
