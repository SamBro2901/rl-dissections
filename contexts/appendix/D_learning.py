"""Appendix D data: learning check (all 56 configurations), return curves, Spearman block.

Recomputed from results/*/training_metrics.json of the 280 canonical runs.
 * a run's "first 10 %" / "last 10 %" are its first / last k = max(1, floor(n/10)) training episodes
   (phase == 'train'; warm-up episodes and the trailing incomplete episode are excluded);
 * return = mean of the per-episode returns of those episodes; return per step = sum of returns / sum of lengths;
 * cross-seed values = mean over the five seeds, sd = sample sd (n-1).
Writes appendix_data/D_*.csv.
"""
import csv, json, math, os
import numpy as np
import pandas as pd
from scipy import stats

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(R, 'contexts/appendix/appendix_data')
runs = list(csv.DictReader(open(os.path.join(R, 'contexts/step1/04_canonical_run_list.csv'))))


def short(cfg):
    if cfg == 'baseline':
        return 'baseline'
    k, v = cfg.split('=')
    v = v.replace('[', '').replace(']', '').replace(' ', '').split(',')[0]
    return {'updates_per_env_step': 'UTD ', 'hidden_sizes': 'width ', 'batch_size': 'batch ', 'rollout_max_length': 'rollout ',
            'num_q': 'num_q ', 'horizon': 'horizon '}[k] + v


recs, curves = [], []
for r in runs:
    tm = json.load(open(os.path.join(R, 'results', r['run_dir'], 'training_metrics.json')))
    ep = [e for e in tm['episodes'] if e['phase'] == 'train']
    n = len(ep)
    k = max(1, n // 10)
    f, l = ep[:k], ep[-k:]
    recs.append(dict(algo=r['algo'], env=r['env'], config=r['config'], cfg=short(r['config']), seed=int(r['seed']), n_ep=n, k=k,
                     first_ret=np.mean([e['return'] for e in f]), last_ret=np.mean([e['return'] for e in l]),
                     first_len=np.mean([e['length'] for e in f]), last_len=np.mean([e['length'] for e in l]),
                     first_ps=sum(e['return'] for e in f) / sum(e['length'] for e in f), last_ps=sum(e['return'] for e in l) / sum(e['length'] for e in l),
                     last_early=np.mean([e['length'] < 1000 for e in l]), all_1000=all(e['length'] == 1000 for e in ep)))
    # curves: 10 windows of 10,000 training steps; episodes assigned by the step at which they ended
    for w in range(10):
        lo, hi = w * 10000, (w + 1) * 10000
        wr = [e['return'] for e in ep if lo < e['env_step'] <= hi]
        curves.append(dict(algo=r['algo'], env=r['env'], cfg=short(r['config']), seed=int(r['seed']), window=w, step_end=hi,
                           ret=(np.mean(wr) if wr else np.nan), n_ep=len(wr)))
P = pd.DataFrame(recs)
P.to_csv(os.path.join(OUT, 'D_per_run.csv'), index=False)
C = pd.DataFrame(curves)
C.to_csv(os.path.join(OUT, 'D_curves_per_run.csv'), index=False)

rows = []
for (a, e, c), g in P.groupby(['algo', 'env', 'cfg'], sort=False):
    rows.append(dict(algo=a, env=e, cfg=c, n_seeds=len(g), first_mean=g.first_ret.mean(), first_sd=g.first_ret.std(ddof=1),
                     last_mean=g.last_ret.mean(), last_sd=g.last_ret.std(ddof=1),
                     first_len=g.first_len.mean(), last_len=g.last_len.mean(),
                     first_ps=g.first_ps.mean(), last_ps=g.last_ps.mean(),
                     seeds_improving=int((g.last_ret > g.first_ret).sum()), seeds_improving_ps=int((g.last_ps > g.first_ps).sum()),
                     n_ep_min=int(g.n_ep.min()), n_ep_max=int(g.n_ep.max()), n_ep_mean=g.n_ep.mean(), k_min=int(g.k.min()), k_max=int(g.k.max()),
                     frac_early_last=g.last_early.mean(), improved=bool(g.last_ret.mean() > g.first_ret.mean()),
                     improved_ps=bool(g.last_ps.mean() > g.first_ps.mean()), all_ep_1000=bool(g.all_1000.all())))
L = pd.DataFrame(rows)
L.to_csv(os.path.join(OUT, 'D_learning_check_all56.csv'), index=False)

# curves aggregated over seeds
cg = C.groupby(['algo', 'env', 'cfg', 'window', 'step_end']).ret.agg(['mean', lambda x: x.std(ddof=1), 'count']).reset_index()
cg.columns = ['algo', 'env', 'cfg', 'window', 'step_end', 'mean', 'sd', 'n']
cg.to_csv(os.path.join(OUT, 'D_curves.csv'), index=False)

# summary facts
hc = L[L.env == 'HalfCheetah-v5']
ant = L[L.env == 'Ant-v5']
print('HC configs', len(hc), 'all improved (config mean)', hc.improved.all(), 'seeds improving min', hc.seeds_improving.min(), 'all 5/5:', (hc.seeds_improving == 5).all())
print('HC last range', hc.last_mean.min(), hc.last_mean.max(), 'k', hc.k_min.min(), hc.k_max.max(), 'n_ep', hc.n_ep_min.min(), hc.n_ep_max.max(), 'all ep 1000', hc.all_ep_1000.all())
print('Ant k range', ant.k_min.min(), ant.k_max.max(), 'n_ep', ant.n_ep_min.min(), ant.n_ep_max.max())
fl = L[~L.improved]
print('non improving', len(fl), fl[['algo', 'env', 'cfg']].values.tolist())
print('non improving per step', L[~L.improved_ps][['algo', 'env', 'cfg']].values.tolist())
print('flagged with ps rise (of 14):', int(fl.improved_ps.sum()))
print('mbpo ant ps rises', int(ant[ant.algo == 'mbpo'].improved_ps.sum()), 'of', len(ant[ant.algo == 'mbpo']))
print('len longer late among flagged:', int((fl.last_len > fl.first_len).sum()), 'of', len(fl))

# Spearman block from the 56 config means (mean over seeds of FLOPs/energy from the per-run table)
pr = pd.read_csv(os.path.join(R, 'flop_analysis/output/per_run_energy_per_flop.csv'), usecols=['algo', 'env_id', 'architecture_signature', 'hidden_sizes',
                                                                                          'updates_per_env_step', 'mbpo_rollout_regime', 'seed', 'run_dir',
                                                                                          'included_in_cross_seed_avg', 'segment', 'total_flops', 'total_energy_joules'],
                 dtype={'run_dir': str, 'segment': str})
canon = {r['run_dir'] for r in runs}
pr['rd'] = pr.run_dir.str.replace('results/', '', regex=False)
t = pr[(pr.segment == 'TOTAL_MEASURED_TRAINING') & pr.rd.isin(canon)].copy()
assert len(t) == 280, len(t)
m = {r['run_dir']: (r['algo'], r['env'], short(r['config'])) for r in runs}
t['cfg'] = t.rd.map(lambda x: m[x][2])
t['algo'] = t.rd.map(lambda x: m[x][0])
t['env'] = t.rd.map(lambda x: m[x][1])
t['flops'] = t.total_flops.astype(float)
cm = t.groupby(['env', 'algo', 'cfg'], sort=False).agg(flops=('flops', 'mean'), energy=('total_energy_joules', 'mean')).reset_index()
cm = cm.merge(L[['algo', 'env', 'cfg', 'last_mean']], on=['algo', 'env', 'cfg'])
cm.to_csv(os.path.join(OUT, 'D_config_means56.csv'), index=False)
sp = []
for env, g in cm.groupby('env'):
    for lab, y in (('return vs FLOPs', 'last_mean'), ('energy vs FLOPs', 'energy')):
        rho, p = stats.spearmanr(g.flops, g[y])
        sp.append(dict(scope='pooled', env=env, algo='all', analysis=lab, n=len(g), rho=rho, p=p))
    for algo, h in g.groupby('algo'):
        for lab, y in (('return vs FLOPs', 'last_mean'), ('energy vs FLOPs', 'energy')):
            rho, p = stats.spearmanr(h.flops, h[y])
            sp.append(dict(scope='per algorithm', env=env, algo=algo, analysis=lab, n=len(h), rho=rho, p=p))
pd.DataFrame(sp).to_csv(os.path.join(OUT, 'D_spearman.csv'), index=False)
print(pd.DataFrame(sp).round(4).to_string())
