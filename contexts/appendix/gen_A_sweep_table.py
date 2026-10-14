"""Appendix A: write the rows of the sweep/override table (LaTeX) from A_override_map.csv and the override files."""
import csv, json, os, collections
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
rows = list(csv.DictReader(open(os.path.join(R, 'contexts/appendix/appendix_data/A_override_map.csv'))))
byfile = collections.OrderedDict()
for r in rows:
    if r['override_file'].startswith('(none'):
        continue
    byfile.setdefault(r['override_file'], []).append(r)
ALG = {'sac': 'SAC', 'td3': 'TD3', 'mbpo': 'MBPO', 'tdmpc2': 'TD-MPC2'}
BS = chr(92)


def esc(s):
    return s.replace('_', BS + '_')


def tt(s):
    return BS + 'texttt{' + esc(s) + '}'


def key(fn):
    stem = fn[:-5]
    algo = stem.split('_')[0]
    ant = '_ant' in stem
    rest = stem.replace('_ant', '')
    pars = ('utd', 'width', 'batch', 'rollout', 'numq', 'horizon')
    p = -1
    for i, q in enumerate(pars):
        if q in rest:
            p = i
    return (['sac', 'td3', 'mbpo', 'tdmpc2'].index(algo), p, ant, rest)


out = []
for fn in sorted(byfile, key=key):
    rs = byfile[fn]
    ov = json.load(open(os.path.join(R, 'configs/overrides', fn)))
    algo = rs[0]['algo']
    envs = sorted({r['env'] for r in rs}, key=lambda e: e != 'HalfCheetah-v5')
    envname = ' and '.join(envs)
    n = sum(int(r['runs']) for r in rs)
    cfg = rs[0]['config']
    if cfg == 'baseline':
        param, val, swept = '---', '---', None
    else:
        k, v = cfg.split('=')
        swept = k
        param = tt(k)
        if '[' in v:
            val = '$(' + v.strip('[]').replace(' ', '') + ')$'
        else:
            val = v
    fixed = {k: v for k, v in ov.items() if k != swept}
    sched = {'rollout_min_epoch': 20, 'rollout_max_epoch': 100, 'rollout_min_length': 1, 'rollout_max_length': 25}
    if fixed == sched:
        also = 'Ant schedule$^{a}$'
    elif fixed and set(fixed) == set(sched) - {'rollout_max_length'} and all(sched[k] == v for k, v in fixed.items()):
        also = 'Ant schedule$^{a}$ with the maximum length above'
    elif fixed:
        also = ', '.join(tt(k) + ' ' + str(v).lower() for k, v in fixed.items())
    else:
        also = '---'
    out.append('    ' + ' & '.join([tt(fn), ALG[algo], envname, param, val, also, BS + 'num{%d}' % n]) + ' ' + BS + BS)
open(os.path.join(R, 'contexts/appendix/appendix_data/A_sweep_rows.tex'), 'w').write('\n'.join(out) + '\n')
print(len(out))
print('\n'.join(out[:2]))
