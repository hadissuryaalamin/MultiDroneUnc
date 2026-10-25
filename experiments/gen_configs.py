"""Generate full-factorial configs into experiments/configs/complexity.

Factors:
    state space : small (5x5x3), medium (10x10x5), large (20x20x10)
    action space: 8 (change_altitude=False), 26 (change_altitude=True)
    obstacles   : sparse clutter, dense clutter
    drones      : 1, 4

Obstacles are full-height pillars (span every z level), so the same
layout blocks both the 8-action and the 26-action drones; flying over an
obstacle is never a shortcut. Clutter layouts use a fixed seed per grid size,
so the 8- and 26-action configs of the same size share identical obstacles.
"""
import argparse
import itertools
from collections import deque
from pathlib import Path

import numpy as np
import yaml

OUT_DIR = Path(__file__).resolve().parent / 'configs' / 'complexity'

GRIDS = {
    "small": (5, 5, 3),
    "medium": (10, 10, 5),
    "large": (20, 20, 10),
}
ACTIONS = {8: False, 26: True}
CLUTTER_DENSITY = {"sparse": 0.02, "dense": 0.10}
NUM_DRONES = [1, 4]
SEED = 0

BASE = dict(
    step_cost=-1.0,
    collision_penalty=-50.0,
    goal_reward=100.0,
    discount_factor=0.98,
    max_num_steps=100,
    alpha=0.5,
)


def lanes(Y, n):
    """y-coordinates for n drones, spread evenly across the grid width."""
    if n == 1:
        return [Y // 2]
    return [int(round(v)) for v in np.linspace(0, Y - 1, n)]


def starts_goals(grid, n):
    X, Y, Z = grid
    z = Z // 2
    ys = lanes(Y, n)
    return [[0, y, z] for y in ys], [[X - 1, y, z] for y in ys]


def reachable(blocked_xy, start, goal):
    """BFS on the XY footprint with 8-connectivity (pillars are full height)."""
    X, Y = blocked_xy.shape
    seen = np.zeros_like(blocked_xy)
    q = deque([start])
    seen[start] = True
    while q:
        x, y = q.popleft()
        if (x, y) == goal:
            return True
        for dx, dy in itertools.product((-1, 0, 1), repeat=2):
            nx, ny = x + dx, y + dy
            if 0 <= nx < X and 0 <= ny < Y and not seen[nx, ny] and not blocked_xy[nx, ny]:
                seen[nx, ny] = True
                q.append((nx, ny))
    return False


def clutter_footprint(grid, density, rng):
    """Random pillar footprint over interior columns x=1..X-2.

    Start column (x=0) and goal plane (x=X-1) stay free, and every drone
    lane (for both 1 and 4 drones) keeps a start-to-goal path.
    """
    X, Y, _ = grid
    candidates = [(x, y) for x in range(1, X - 1) for y in range(Y)]
    k = max(1, int(round(density * X * Y)))
    pairs = [(tuple(s[:2]), tuple(g[:2]))
             for n in NUM_DRONES
             for s, g in zip(*starts_goals(grid, n))]

    for _ in range(1000):
        idx = rng.choice(len(candidates), size=k, replace=False)
        blocked = np.zeros((X, Y), dtype=bool)
        for i in idx:
            blocked[candidates[i]] = True
        if all(reachable(blocked, s, g) for s, g in pairs):
            return blocked
    raise RuntimeError(f"no feasible clutter layout for grid {grid}, density {density}")


def to_cells(blocked_xy, Z):
    xs, ys = np.nonzero(blocked_xy)
    return [[int(x), int(y), int(z)] for x, y in zip(xs, ys) for z in range(Z)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT_DIR,
                        help='Directory for the 24 generated YAML configs')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    footprints = {}
    for size, grid in GRIDS.items():
        rng = np.random.default_rng(SEED)
        for name, dens in CLUTTER_DENSITY.items():
            footprints[size, name] = clutter_footprint(grid, dens, rng)

    count = 0
    for (size, grid), (n_act, alt), obs, n in itertools.product(
            GRIDS.items(), ACTIONS.items(), CLUTTER_DENSITY, NUM_DRONES):
        starts, goals = starts_goals(grid, n)
        cfg = dict(
            grid_size=list(grid),
            start_positions=starts,
            goal_positions=goals,
            obstacle_cells=to_cells(footprints[size, obs], grid[2]),
            change_altitude=alt,
            **BASE,
        )
        fname = f"S{size}_A{n_act}_O{obs}_n{n}.yaml"
        with (args.output / fname).open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, default_flow_style=None, sort_keys=False)
        count += 1
    print(f"wrote {count} configs to {args.output}")


if __name__ == "__main__":
    main()
