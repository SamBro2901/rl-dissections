"""Appendix D: assemble appendix_D.tex from the template, the recomputed tables and the existing text of thesis-main.tex."""
import os
import pandas as pd

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
THESIS = os.path.join(os.path.dirname(os.path.dirname(R)), 'Thesis Report')
D = os.path.join(R, 'contexts/appendix/appendix_data')
BS = chr(92)
tpl = open(os.path.join(R, 'contexts/appendix/appendix_D.tpl.tex'), encoding='utf-8').read()
_bak = os.path.join(THESIS, 'thesis-main.tex.bak-appendix')
tex = open(_bak if os.path.exists(_bak) else os.path.join(THESIS, 'thesis-main.tex'), encoding='utf-8').read()   # the text before the insertion
L = pd.read_csv(os.path.join(D, 'D_learning_check_all56.csv'))
SP = pd.read_csv(os.path.join(D, 'D_spearman.csv'))
NAME = {'sac': 'SAC', 'td3': 'TD3', 'mbpo': 'MBPO', 'tdmpc2': 'TD-MPC2'}
ORDER = {
    'sac': ['baseline', 'UTD 2', 'UTD 4', 'width 256', 'width 512', 'batch 512', 'batch 1024'],
    'td3': ['baseline', 'UTD 2', 'UTD 4', 'width 256', 'width 512', 'batch 256', 'batch 512', 'batch 1024'],
    'mbpo': ['baseline', 'UTD 2', 'UTD 4', 'width 256', 'width 512', 'batch 512', 'batch 1024', 'rollout 1', 'rollout 15'],
    'tdmpc2': ['baseline', 'num_q 3', 'num_q 7', 'horizon 1', 'horizon 5'],
}


def cfg_tex(c):
    return c.replace('num_q', BS + 'texttt{num' + BS + '_q}')


def ms(m, s):
    return BS + 'num{%.1f+-%.1f}' % (m, s)


def num(x, d=1):
    return BS + 'num{%.*f}' % (d, x)


# ---------------------------------------------------------------- existing blocks (verbatim)
i0 = tex.index(BS + 'chapter{Learning check details}')
i1 = tex.index(BS + 'chapter{Reproducibility details}')
ch = tex[i0:i1]
t0 = ch.index(BS + 'begin{table}[tb]')
t1 = ch.index(BS + 'Zcref{tab:app-config-means} lists the means')
ant_block = ch[t0:t1].rstrip('\n')            # table + three paragraphs on Ant-v5
m0 = t1
means_block = ch[m0:].rstrip('\n')            # intro paragraph + longtable
assert 'tab:app-learning}' in ant_block and 'tab:app-config-means}' in means_block and 'longtable' in means_block

# ---------------------------------------------------------------- HalfCheetah-v5 table
hc = L[L.env == 'HalfCheetah-v5']
rows = []
for algo in ('sac', 'td3', 'mbpo', 'tdmpc2'):
    first = True
    for c in ORDER[algo]:
        g = hc[(hc.algo == algo) & (hc.cfg == c)]
        if g.empty:
            continue
        r = g.iloc[0]
        rows.append('    ' + ' & '.join([NAME[algo] if first else '', cfg_tex(c), ms(r.first_mean, r.first_sd), ms(r.last_mean, r.last_sd), BS + 'num{%d}/' % r.seeds_improving + BS + 'num{5}']) + ' ' + BS + BS)
        first = False
    if algo != 'tdmpc2':
        rows.append('    ' + BS + 'midrule')
assert len(hc) == 27
tpl = tpl.replace('%%D_HC_ROWS%%', '\n'.join(rows))

# ---------------------------------------------------------------- per-step table for the 14 flagged Ant-v5 configurations
ant = L[(L.env == 'Ant-v5') & (~L.improved)]
assert len(ant) == 14
rows = []
for algo in ('sac', 'td3', 'mbpo'):
    for c in ORDER[algo]:
        g = ant[(ant.algo == algo) & (ant.cfg == c)]
        if g.empty:
            continue
        r = g.iloc[0]
        star = '$^' + BS + 'ast$' if not r.improved_ps else ''
        rows.append('    ' + ' & '.join([NAME[algo], cfg_tex(c) + star, num(r.first_ps, 3), num(r.last_ps, 3), BS + 'num{%d}/' % r.seeds_improving_ps + BS + 'num{5}']) + ' ' + BS + BS)
tpl = tpl.replace('%%D_PERSTEP_ROWS%%', '\n'.join(rows))

# ---------------------------------------------------------------- Spearman rows
sp_rows = []
for env in ('HalfCheetah-v5', 'Ant-v5'):
    first_env = True
    for scope, algo in [('pooled', 'all'), ('per algorithm', 'sac'), ('per algorithm', 'td3'), ('per algorithm', 'mbpo'), ('per algorithm', 'tdmpc2')]:
        a = SP[(SP.env == env) & (SP.scope == scope) & (SP.algo == algo) & (SP.analysis == 'return vs FLOPs')].iloc[0]
        b = SP[(SP.env == env) & (SP.scope == scope) & (SP.algo == algo) & (SP.analysis == 'energy vs FLOPs')].iloc[0]

        def pv(r):
            if abs(r.rho) == 1.0:
                return '---'
            return BS + 'num{%.3g}' % r.p

        label = 'all algorithms' if algo == 'all' else NAME[algo]
        sp_rows.append('    ' + ' & '.join([env if first_env else '', label, BS + 'num{%d}' % a.n, BS + 'num{%.3f}' % a.rho, pv(a), BS + 'num{%d}' % b.n, BS + 'num{%.3f}' % b.rho, pv(b)]) + ' ' + BS + BS)
        first_env = False
    if env == 'HalfCheetah-v5':
        sp_rows.append('    ' + BS + 'midrule')
tpl = tpl.replace('%%D_SPEARMAN_ROWS%%', '\n'.join(sp_rows))

# ---------------------------------------------------------------- figures
figs = []
for algo in ('sac', 'td3', 'mbpo', 'tdmpc2'):
    for env, es in (('HalfCheetah-v5', 'hc'), ('Ant-v5', 'ant')):
        panels = {'sac': 'update-to-data ratio, hidden width, batch size', 'td3': 'update-to-data ratio, hidden width, batch size',
                  'mbpo': 'update-to-data ratio, hidden width, batch size' + (', maximum rollout length' if env == 'Ant-v5' else ''),
                  'tdmpc2': 'size of the Q-ensemble, planning horizon'}[algo]
        figs.append(f'''\\begin{{figure}}[tbp]
  \\centering
  \\input{{figures/app-return-{algo}-{es}.tex}}
  \\caption{{{NAME[algo]} on {env}: mean return of the training episodes in ten windows of \\num{{10000}} steps (mean and standard deviation over five seeds). Panels: {panels}.}}
  \\label{{fig:app-return-{algo}-{es}}}
\\end{{figure}}''')
tpl = tpl.replace('%%D_FIGURES%%', '\n\n'.join(figs))
tpl = tpl.replace('%%D_EXISTING_ANT%%', ant_block).replace('%%D_EXISTING_MEANS%%', means_block)
open(os.path.join(R, 'contexts/appendix/appendix_D.tex'), 'w', encoding='utf-8').write(tpl)
print([l for l in tpl.splitlines() if '%%D_' in l], len(tpl.splitlines()))
