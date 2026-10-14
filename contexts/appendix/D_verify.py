"""Appendix D: compare the existing tables tab:app-learning and tab:app-config-means of thesis-main.tex with the recomputed values."""
import os, re, sys
import pandas as pd

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEX = os.path.join(os.path.dirname(os.path.dirname(R)), 'Thesis Report', 'thesis-main.tex')
OUT = os.path.join(R, 'contexts/appendix/appendix_data')
tex = open(TEX, encoding='utf-8').read()
L = pd.read_csv(os.path.join(OUT, 'D_learning_check_all56.csv'))
CM = pd.read_csv(os.path.join(OUT, 'D_config_means56.csv'))
ALG = {'SAC': 'sac', 'TD3': 'td3', 'MBPO': 'mbpo', 'TD-MPC2': 'tdmpc2'}


def num(s):
    s = s.strip().replace('$', '')
    m = re.search(r'\\num\{([^}]*)\}', s)
    if m:
        s = m.group(1)
    s = s.replace('^\\ast', '').replace('\\ast', '').replace('{,}', '')
    return float(s)


def cfgname(s):
    s = s.replace('\\texttt{num\\_q}', 'num_q').replace('$^\\ast$', '').replace('\\ast', '').strip()
    s = {'base': 'baseline'}.get(s, s)
    return s


# ---- tab:app-learning
i = tex.index('\\label{tab:app-learning}')
seg = tex[i:tex.index('\\end{tabular}', i)]
res1 = []
for line in seg.splitlines():
    if '&' not in line or 'Algorithm' in line or 'Return' in line:
        continue
    f = [x.strip().rstrip('\\').strip() for x in line.split('&')]
    if len(f) != 6:
        continue
    a, c = ALG[f[0]], cfgname(f[1])
    row = L[(L.algo == a) & (L.env == 'Ant-v5') & (L.cfg == c)]
    assert len(row) == 1, (a, c)
    r = row.iloc[0]
    vals = [num(x) for x in f[2:]]
    mine = [r.first_mean, r.last_mean, r.first_len, r.last_len]
    res1.append((a, c, [round(abs(v - m_), 3) for v, m_ in zip(vals, mine)], max(abs(v - m_) / max(abs(m_), 1e-9) for v, m_ in zip(vals, mine))))
print('tab:app-learning rows', len(res1), 'max abs diff', max(max(x[2]) for x in res1), 'max rel diff (display rounding 0.1)', max(x[3] for x in res1))
bad = [x for x in res1 if max(x[2]) > 0.051]
print('rows with abs diff > 0.05:', bad)

# ---- tab:app-config-means
i = tex.index('\\label{tab:app-config-means}')
seg = tex[i:tex.index('\\end{longtable}', i)]
env = algo = None
res2 = []
for line in seg.splitlines():
    if '&' not in line or 'Environment' in line:
        continue
    f = [x.strip().rstrip('\\').strip() for x in line.split('&')]
    if len(f) != 6:
        continue
    env = f[0] or env
    algo = f[1] or algo
    c = cfgname(f[2])
    key = (ALG[algo], env, c)
    row = CM[(CM.algo == key[0]) & (CM.env == key[1]) & (CM.cfg == key[2])]
    assert len(row) == 1, key
    r = row.iloc[0]
    fl, en, rt = num(f[3]), num(f[4]), num(f[5])
    res2.append((key, abs(fl - r.flops) / r.flops, abs(en - r.energy / 1e3) / (r.energy / 1e3), abs(rt - r.last_mean)))
print('tab:app-config-means rows', len(res2))
print('max rel dev FLOPs (4 digits)', max(x[1] for x in res2), ' energy (kJ, 1 decimal)', max(x[2] for x in res2), ' max abs dev return (1 decimal)', max(x[3] for x in res2))
print('rows with return abs dev > 0.051:', [x[0] for x in res2 if x[3] > 0.051])
print('rows with FLOPs rel dev > 5e-4:', [x[0] for x in res2 if x[1] > 5e-4], ' energy abs dev > 0.051 kJ:',
      [x[0] for x in res2 if abs(x[2]) * 1 > 0 and False])
# energy absolute deviation in kJ
res3 = []
for line in seg.splitlines():
    pass
