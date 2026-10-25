"""Turn a run_all.py results folder into B4 figures and LaTeX table rows.

Usage: python experiments/analyze.py experiments/results/complexity_10runs_20261024
Writes figures to report/figures/ and table rows to <results>/b4_tables.tex.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
STATES = ['small', 'medium', 'large']
ACTIONS = ['8', '26']
OBSTACLES = ['sparse', 'dense']
DRONES = ['1', '4']

# One-factor-at-a-time sets, all drawn from the full factorial.
SETS = {
    'S': ('state', STATES, {'actions': '8', 'obstacles': 'sparse'}),
    'A': ('actions', ACTIONS, {'state': 'medium', 'obstacles': 'sparse'}),
    'C': ('obstacles', OBSTACLES, {'state': 'medium', 'actions': '8'}),
}
LABELS = {'state': 'State space (grid)', 'actions': 'Actions per drone',
          'obstacles': 'Clutter'}
GRID = {'small': r'$5{\times}5{\times}3$', 'medium': r'$10{\times}10{\times}5$',
        'large': r'$20{\times}20{\times}10$'}


def t_ci(values):
    values = np.asarray(values, dtype=float)
    m = values.mean()
    if len(values) < 2:
        return m, 0.0
    return m, stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))


def wilson(successes, n):
    z = stats.norm.ppf(0.975)
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    # Clamp so float rounding at p=0 or p=1 cannot put p outside its interval.
    return p, min(p, max(0.0, centre - half)), max(p, min(1.0, centre + half))


def select(rows, **fixed):
    return [r for r in rows if all(r[k] == v for k, v in fixed.items())]


def stat_cells(rows):
    succ = [r['success'] == 'True' for r in rows]
    p, lo, hi = wilson(sum(succ), len(succ))
    rew, rew_h = t_ci([float(r['discounted_reward']) for r in rows])
    reach, reach_h = t_ci([float(r['reached_fraction']) for r in rows])
    steps, steps_h = t_ci([float(r['steps']) for r in rows])
    sims, sims_h = t_ci([float(r['simulations_per_step']) for r in rows])
    ends = defaultdict(int)
    for r in rows:
        ends[r['termination']] += 1
    return dict(p=p, lo=lo, hi=hi, rew=rew, rew_h=rew_h, reach=reach, reach_h=reach_h,
                steps=steps, steps_h=steps_h, sims=sims, sims_h=sims_h,
                col=ends['collision'], cap=ends['max_steps'], n=len(rows))


def fmt_row(label, s):
    return (f"{label} & {s['p']:.2f} [{s['lo']:.2f}, {s['hi']:.2f}] & "
            f"${s['rew']:.1f} \\pm {s['rew_h']:.1f}$ & ${s['reach']:.2f} \\pm {s['reach_h']:.2f}$ & "
            f"${s['steps']:.1f} \\pm {s['steps_h']:.1f}$ & ${s['sims']:.0f} \\pm {s['sims_h']:.0f}$ & "
            f"{s['col']}/{s['cap']} \\\\")


def level_name(factor, level):
    if factor == 'state':
        return f"{level} ({GRID[level]})"
    return level


def plot_sets(rows, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), sharey='row')
    for col, (set_id, (factor, levels, fixed)) in enumerate(SETS.items()):
        for n, marker in zip(DRONES, ['o', 's']):
            succ, s_lo, s_hi, rew, rew_h = [], [], [], [], []
            for level in levels:
                s = stat_cells(select(rows, drones=n, **fixed, **{factor: level}))
                succ.append(s['p']); s_lo.append(s['p'] - s['lo']); s_hi.append(s['hi'] - s['p'])
                rew.append(s['rew']); rew_h.append(s['rew_h'])
            x = np.arange(len(levels)) + (0.06 if n == '4' else -0.06)
            axes[0, col].errorbar(x, succ, yerr=[s_lo, s_hi], marker=marker, capsize=3,
                                  label=f'{n} drone' + ('s' if n == '4' else ''))
            axes[1, col].errorbar(x, rew, yerr=rew_h, marker=marker, capsize=3)
        fixed_txt = ', '.join(f'{k}={v}' for k, v in fixed.items())
        axes[0, col].set_title(f'Set {set_id}: {LABELS[factor]}\n({fixed_txt})', fontsize=9)
        for row in range(2):
            axes[row, col].set_xticks(range(len(levels)))
            axes[row, col].set_xticklabels(levels)
            axes[row, col].grid(alpha=0.3)
        axes[1, col].axhline(0, color='grey', lw=0.8)
    axes[0, 0].set_ylabel('Success rate (Wilson 95% CI)')
    axes[1, 0].set_ylabel('Discounted reward (95% CI)')
    axes[0, 0].set_ylim(-0.05, 1.05)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / 'b4_sets.pdf')
    fig.savefig(out_dir / 'b4_sets.png', dpi=150)


def plot_marginals(rows, out_dir):
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.8), sharey=True)
    for ax, (factor, levels) in zip(axes, [('state', STATES), ('actions', ACTIONS),
                                          ('obstacles', OBSTACLES), ('drones', DRONES)]):
        ps, lo, hi = [], [], []
        for level in levels:
            s = stat_cells(select(rows, **{factor: level}))
            ps.append(s['p']); lo.append(s['p'] - s['lo']); hi.append(s['hi'] - s['p'])
        ax.errorbar(range(len(levels)), ps, yerr=[lo, hi], marker='o', capsize=3)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(levels)
        ax.set_title({'drones': 'Number of drones'}.get(factor, LABELS.get(factor)), fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('Success rate')
    axes[0].set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_dir / 'b4_marginals.pdf')
    fig.savefig(out_dir / 'b4_marginals.png', dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path, help='Full-factorial run_all.py results folder')
    parser.add_argument('--figures-dir', type=Path, default=ROOT / 'report' / 'figures',
                        help='Output directory for figures')
    args = parser.parse_args()
    results = args.results
    runs_csv = results / 'all_runs.csv'
    if not runs_csv.is_file():
        parser.error(f'missing results CSV: {runs_csv}')
    with runs_csv.open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    expected = {(state, actions, obstacles, drones)
                for state in STATES for actions in ACTIONS
                for obstacles in OBSTACLES for drones in DRONES}
    present = {(row.get('state'), row.get('actions'), row.get('obstacles'), row.get('drones'))
               for row in rows}
    if present != expected:
        parser.error('analysis requires exactly the 24 sparse/dense complexity scenarios; for custom/Mukhit configs, '
                     'use summary_by_environment.csv from run_all.py')
    fig_dir = args.figures_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_sets(rows, fig_dir)
    plot_marginals(rows, fig_dir)

    lines = ['% Set tables (one factor at a time)']
    for set_id, (factor, levels, fixed) in SETS.items():
        for n in DRONES:
            for i, level in enumerate(levels, 1):
                s = stat_cells(select(rows, drones=n, **fixed, **{factor: level}))
                lines.append(fmt_row(f"{set_id}{i}-n{n} & {level_name(factor, level)}", s))
        lines.append(r'\midrule')

    lines.append('% Marginals over the full factorial')
    for factor, levels in [('state', STATES), ('actions', ACTIONS),
                           ('obstacles', OBSTACLES), ('drones', DRONES)]:
        for level in levels:
            lines.append(fmt_row(f"{factor} & {level}", stat_cells(select(rows, **{factor: level}))))
        lines.append(r'\midrule')

    lines.append('% Full factorial')
    for n in DRONES:
        for obs in OBSTACLES:
            for act in ACTIONS:
                for state in STATES:
                    s = stat_cells(select(rows, drones=n, obstacles=obs, actions=act, state=state))
                    lines.append(fmt_row(f"{n} & {obs} & {act} & {state}", s))
        lines.append(r'\midrule')

    (results / 'b4_tables.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f"Wrote {fig_dir} figures and {results / 'b4_tables.tex'}")


if __name__ == '__main__':
    main()
