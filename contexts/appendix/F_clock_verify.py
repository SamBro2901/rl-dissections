"""Appendix F: recompute the numbers of the clock-experiment appendix from the two run directories."""
import csv, glob, json, os, re
import pandas as pd

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(R, 'contexts/appendix/appendix_data')
RUNS = {'R2000': 'results/mbpo/HalfCheetah-v5/seed_0/20260904_122958',
        'R200': 'results/mbpo/HalfCheetah-v5/seed_0/20260904_130915'}
KWH = 3.6e6
res = {}
for name, rel in RUNS.items():
    d = os.path.join(R, rel)
    seg = json.load(open(os.path.join(d, 'segment_energy.json')))
    meta = json.load(open(os.path.join(d, 'metadata.json')))
    base = pd.read_csv(sorted(glob.glob(os.path.join(d, 'emissions_*.csv')))[0], float_precision='round_trip')
    ex = meta['experiment_config']
    sub = seg['_sub_segment_wall_time_seconds']
    subs = ['buffer_sample', 'critic_update', 'actor_update', 'target_update']
    E = {s: seg[s] * KWH for s in subs}
    gu_rows = base[base.task_name.str.fullmatch(r'gradient_updates_\d+')]
    E_gu = gu_rows.energy_consumed.sum() * KWH
    T_gu = gu_rows.duration.sum()
    P_gu_meas = E_gu / T_gu
    alloc_power = {s: E[s] / sub[s] for s in subs}
    alloc_power_all = sum(E.values()) / sum(sub.values())
    # idle windows (window-corrected: energy / (ram_energy/ram_power)), as in contexts/step1 and sec:units
    idle = {}
    for w in ('idle_baseline_head', 'idle_baseline_tail'):
        row = base[base.task_name == w].iloc[0]
        e_j = row.energy_consumed * KWH
        win = row.ram_energy * KWH / row.ram_power
        idle[w] = dict(P_recorded=e_j / row.duration, P_corrected=e_j / win, duration=float(row.duration), window=float(win))
    res[name] = dict(
        run=rel, commit=meta['git_commit'], clock=(ex.get('gpu_min_clock_mhz'), ex.get('gpu_max_clock_mhz')),
        hidden=meta['algo_config']['hidden_sizes'], torch=meta['torch_version'], codecarbon=meta['codecarbon_version'],
        E_sub_kJ={s: E[s] / 1e3 for s in subs}, E_gu_kJ=E_gu / 1e3, sum_sub_kJ=sum(E.values()) / 1e3,
        T_timers={s: sub[s] for s in subs}, T_gu_task=T_gu, P_gu_measured=P_gu_meas, P_alloc=alloc_power, P_alloc_all=alloc_power_all,
        idle=idle, idle_mean_corrected=0.5 * (idle['idle_baseline_head']['P_corrected'] + idle['idle_baseline_tail']['P_corrected']),
        n_tasks=len(base))
for k in ('buffer_sample', 'critic_update', 'actor_update', 'target_update'):
    res.setdefault('ratio', {})[k] = res['R200']['E_sub_kJ'][k] / res['R2000']['E_sub_kJ'][k]
res['ratio']['gu'] = res['R200']['E_gu_kJ'] / res['R2000']['E_gu_kJ']
json.dump(res, open(os.path.join(OUT, 'F_clock_numbers.json'), 'w'), indent=1, default=float)
print(json.dumps(res, indent=1, default=float)[:6000])
