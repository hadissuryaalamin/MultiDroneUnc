"""Run every config in experiments/configs for N headless episodes and summarise.

Each config gets its own folder with config, per-run CSV and trajectories.
The top-level folder holds all_runs.csv and summary_by_environment.csv with
mean, sample std and 95% confidence intervals. Configs whose runs.csv is
already complete are skipped, so an interrupted batch can be resumed with
--output pointing at the same folder.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multi_drone import MultiDroneUnc  # noqa: E402
from planner import Planner  # noqa: E402
from run_u8050066 import run  # noqa: E402

NAME_RE = re.compile(r'S(?P<state>\w+?)_A(?P<actions>\d+)_O(?P<obstacles>\w+?)_n(?P<drones>\d+)$')
RUN_FIELDS = ['environment', 'state', 'actions', 'obstacles', 'drones', 'run', 'seed',
              'success', 'discounted_reward', 'steps', 'drones_reached', 'reached_fraction',
              'obstacle_collisions', 'vehicle_collision_drones', 'termination',
              'simulations_per_step', 'seconds']
METRICS = ['discounted_reward', 'steps', 'reached_fraction', 'simulations_per_step', 'seconds']


class HeadlessEnv(MultiDroneUnc):
    """No vedo window; counts generative-model calls made by the planner."""

    def __init__(self, path):
        super().__init__(path)
        self.sim_calls = 0

    def simulate(self, state, action):
        self.sim_calls += 1
        return super().simulate(state, action)

    def _init_plot(self):
        pass

    def _update_plot(self):
        pass


def factors(name):
    match = NAME_RE.match(name)
    if not match:
        return {'state': '', 'actions': '', 'obstacles': '', 'drones': ''}
    return match.groupdict()


def run_episode(config_path, seed, planning_time):
    env = HeadlessEnv(str(config_path))
    env.cfg.seed = seed
    env.rng = np.random.default_rng(seed)
    planner = Planner(env, a_param=1.0, b_param=2)
    started = perf_counter()
    reward, history = run(env, planner, planning_time_per_step=planning_time)
    seconds = perf_counter() - started

    final_info = history[-1][5]
    success = bool(final_info['success'])
    obstacle_collisions = sum(int(item[5]['num_collisions']) for item in history)
    vehicle_collisions = sum(int(item[5]['num_vehicle_collisions']) for item in history)
    if obstacle_collisions or vehicle_collisions:
        termination = 'collision'
    elif success:
        termination = 'success'
    else:
        termination = 'max_steps'
    reached = int(history[-1][3][:, 3].sum())

    trajectory = [{
        'step': step, 'state': state.tolist(), 'action': int(action),
        'next_state': next_state.tolist(), 'reward': float(r), 'done': bool(done),
        'num_collisions': int(info['num_collisions']),
        'num_vehicle_collisions': int(info['num_vehicle_collisions']),
    } for step, (state, action, r, next_state, done, info) in enumerate(history, 1)]

    row = {
        'seed': seed, 'success': success, 'discounted_reward': float(reward),
        'steps': len(history), 'drones_reached': reached,
        'reached_fraction': reached / env.N,
        'obstacle_collisions': obstacle_collisions,
        'vehicle_collision_drones': vehicle_collisions,
        'termination': termination,
        'simulations_per_step': env.sim_calls / len(history),
        'seconds': seconds,
    }
    return row, trajectory


def run_config(config_path, out_dir, runs, seed_start, planning_time):
    name = config_path.stem
    env_dir = out_dir / name
    runs_csv = env_dir / 'runs.csv'
    if runs_csv.exists():
        with runs_csv.open(encoding='utf-8') as stream:
            done_rows = list(csv.DictReader(stream))
        if len(done_rows) >= runs:
            return name, done_rows, True

    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / 'config.yaml').write_text(config_path.read_text(encoding='utf-8'), encoding='utf-8')
    rows = []
    with runs_csv.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=RUN_FIELDS)
        writer.writeheader()
        for index in range(1, runs + 1):
            seed = seed_start + index - 1
            row, trajectory = run_episode(env_dir / 'config.yaml', seed, planning_time)
            row = {'environment': name, **factors(name), 'run': index, **row}
            writer.writerow(row)
            stream.flush()
            rows.append(row)
            (env_dir / f'run_{index:02d}_trajectory.json').write_text(
                json.dumps(trajectory), encoding='utf-8')
            print(f'[{name}] run {index}/{runs}: {row["termination"]}, '
                  f'reward={row["discounted_reward"]:.2f}, steps={row["steps"]}, '
                  f'sims/step={row["simulations_per_step"]:.0f}', flush=True)
    return name, rows, False


def mean_ci(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    n = len(values)
    m = float(values.mean())
    if n < 2:
        return m, 0.0, m, m
    sd = float(values.std(ddof=1))
    half = float(stats.t.ppf((1 + confidence) / 2, n - 1) * sd / np.sqrt(n))
    return m, sd, m - half, m + half


def wilson_ci(successes, n, confidence=0.95):
    z = stats.norm.ppf((1 + confidence) / 2)
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def summarise(name, rows):
    def as_bool(v):
        return v if isinstance(v, bool) else str(v) == 'True'

    n = len(rows)
    successes = sum(as_bool(r['success']) for r in rows)
    lo, hi = wilson_ci(successes, n)
    out = {'environment': name, **factors(name), 'runs': n,
           'success_rate': successes / n, 'success_ci_low': lo, 'success_ci_high': hi,
           'collision_episodes': sum(r['termination'] == 'collision' for r in rows),
           'max_steps_episodes': sum(r['termination'] == 'max_steps' for r in rows)}
    for metric in METRICS:
        m, sd, low, high = mean_ci([float(r[metric]) for r in rows])
        out.update({f'{metric}_mean': m, f'{metric}_std': sd,
                    f'{metric}_ci_low': low, f'{metric}_ci_high': high})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs', type=Path, default=ROOT / 'experiments' / 'configs')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--planning-time', type=float, default=8.0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()

    configs = sorted(args.configs.glob('*.yaml'))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d')
    out_dir = args.output or ROOT / 'experiments' / 'results' / f'complexity_{args.runs}runs_{stamp}'
    out_dir.mkdir(parents=True, exist_ok=True)

    planner_bytes = (ROOT / 'planner.py').read_bytes()
    (out_dir / 'planner.py').write_bytes(planner_bytes)
    (out_dir / 'metadata.json').write_text(json.dumps({
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'configs': [c.name for c in configs],
        'runs_per_config': args.runs,
        'seeds': list(range(args.seed_start, args.seed_start + args.runs)),
        'planning_time_per_step': args.planning_time,
        'planner_params': {'a_param': 1.0, 'b_param': 2},
        'workers': args.workers,
        'planner_sha256': hashlib.sha256(planner_bytes).hexdigest(),
    }, indent=2), encoding='utf-8')
    print(f'Results: {out_dir} ({len(configs)} configs x {args.runs} runs)', flush=True)

    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_config, c, out_dir, args.runs, args.seed_start,
                               args.planning_time) for c in configs]
        for future in as_completed(futures):
            name, rows, skipped = future.result()
            results[name] = rows
            print(f'== {name} {"skipped (complete)" if skipped else "done"} '
                  f'[{len(results)}/{len(configs)}]', flush=True)

    names = sorted(results)
    with (out_dir / 'all_runs.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=RUN_FIELDS)
        writer.writeheader()
        for name in names:
            writer.writerows(results[name])

    summary = [summarise(name, results[name]) for name in names]
    with (out_dir / 'summary_by_environment.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f'Wrote {out_dir / "summary_by_environment.csv"}', flush=True)


if __name__ == '__main__':
    main()
