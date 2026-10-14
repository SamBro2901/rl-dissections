"""Appendix B: independent recomputation (from the raw run files) of the fits and adjacent-pair costs of the training total
and of the cross-seed table, compared with the context CSVs that the Section 5 tables are generated from."""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(R, 'flop_analysis'))
import appendix_tables as a

runs = a.load_runs()
A = a.agg(runs)
fits = pd.read_csv(os.path.join(R, 'contexts/section_5.2_data/a1_linear_fits.csv'))
fits = fits[(fits.fit_basis == 'config_means')]
pairs = pd.read_csv(os.path.join(R, 'contexts/section_5.2_data/a2_delta_pairs.csv'))
SWEEPS = {
    'UTD': ['baseline', 'UTD 2', 'UTD 4'], 'width': ['width 256', 'width 512', 'baseline'], 'batch': ['baseline', 'batch 512', 'batch 1024'],
    'batch_td3': ['baseline', 'batch 256', 'batch 512', 'batch 1024'], 'num_q': ['num_q 3', 'baseline', 'num_q 7'], 'horizon': ['horizon 1', 'baseline', 'horizon 5'],
}
rows = []
worst_a = worst_b = worst_r2 = worst_pair = 0.0
n_fit = n_pair = 0
for algo in a.ALGOS:
    for env in a.ENVS:
        for sweep, cfgs in SWEEPS.items():
            key = 'batch_td3' if (sweep == 'batch_td3') else sweep
            if algo == 'td3' and sweep == 'batch':
                continue
            if algo != 'td3' and sweep == 'batch_td3':
                continue
            if algo == 'tdmpc2' and sweep not in ('num_q', 'horizon'):
                continue
            if algo != 'tdmpc2' and sweep in ('num_q', 'horizon'):
                continue
            if not all((algo, env, c) in A for c in cfgs):
                continue
            E = np.array([np.mean(A[(algo, env, c)]['TOTAL']['E']) for c in cfgs])
            F = np.array([np.mean(A[(algo, env, c)]['TOTAL']['F']) for c in cfgs])
            b, a0 = np.polyfit(F, E, 1)
            pred = a0 + b * F
            r2 = 1 - np.sum((E - pred) ** 2) / np.sum((E - E.mean()) ** 2)
            sw = {'batch_td3': 'batch'}.get(sweep, sweep)
            ref = fits[(fits.algo == algo) & (fits.env == env) & (fits.sweep == sw) & (fits.segment == 'total')]
            assert len(ref) == 1, (algo, env, sweep)
            ref = ref.iloc[0]
            da = abs(a0 - ref.a_J) / abs(ref.a_J)
            db = abs(b - ref.b_J_per_FLOP) / abs(ref.b_J_per_FLOP)
            dr = abs(r2 - ref.r2)
            worst_a, worst_b, worst_r2 = max(worst_a, da), max(worst_b, db), max(worst_r2, dr)
            n_fit += 1
            # adjacent pairs in sweep order of the values
            vals = [float(ref['values'].split('/')[i]) for i in range(len(cfgs))] if '/' in str(ref['values']) else None
            sub = pairs[(pairs.algo == algo) & (pairs.env == env) & (pairs.sweep == sw) & (pairs.segment == 'total') & (pairs.status == 'ok')]
            order = np.argsort([float(v) for v in str(ref['values']).split('/')]) if vals else range(len(cfgs))
            cs = [cfgs[i] for i in order]
            for k in range(len(cs) - 1):
                e0, e1 = np.mean(A[(algo, env, cs[k])]['TOTAL']['E']), np.mean(A[(algo, env, cs[k + 1])]['TOTAL']['E'])
                f0, f1 = np.mean(A[(algo, env, cs[k])]['TOTAL']['F']), np.mean(A[(algo, env, cs[k + 1])]['TOTAL']['F'])
                mine = (e1 - e0) / (f1 - f0) * 1e12
                m = sub[(np.isclose(sub.E_from_J, e0, rtol=1e-9)) & (np.isclose(sub.E_to_J, e1, rtol=1e-9))]
                if len(m):
                    worst_pair = max(worst_pair, abs(mine - m.iloc[0].dE_per_dF_pJ) / abs(m.iloc[0].dE_per_dF_pJ))
                    n_pair += 1
print(f'total-fits recomputed: {n_fit}; max rel dev intercept {worst_a:.2e}, slope {worst_b:.2e}, abs dev R2 {worst_r2:.2e}')
print(f'adjacent pairs of the total recomputed and matched: {n_pair}; max rel dev {worst_pair:.2e}')
# cross-seed table against cross_seed_energy_per_flop.csv
cs = pd.read_csv(os.path.join(R, 'flop_analysis/output/cross_seed_energy_per_flop.csv'))
cs = cs[cs.segment == 'TOTAL_MEASURED_TRAINING']
cs = cs[cs.env_id.isin(a.ENVS)]
worstE = worstJ = 0.0
n = 0
for (algo, env, cfg), d in A.items():
    t = d['TOTAL']
    E, F = np.mean(t['E']), np.mean(t['F'])
    # match by energy
    m = cs[(cs.algo == algo) & (cs.env_id == env) & np.isclose(cs.mean_energy_joules, E, rtol=1e-9)]
    if len(m) != 1:
        print('no unique match', algo, env, cfg, len(m))
        continue
    m = m.iloc[0]
    worstE = max(worstE, abs(E - m.mean_energy_joules) / E)
    worstJ = max(worstJ, abs(E / F - m.mean_energy_per_flop_j_per_flop) / (E / F))
    n += 1
print(f'cross-seed rows matched: {n}; max rel dev energy {worstE:.2e}, J/FLOP {worstJ:.2e}')
