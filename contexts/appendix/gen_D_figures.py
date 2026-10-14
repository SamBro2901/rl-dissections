"""Appendix D: pgfplots return-curve figures (one per algorithm and environment) in the style of figures/ec-scatter.tex.

Reads appendix_data/D_curves.csv (D_learning.py).  Writes figures/app-return-<algo>-<env>.tex and .csv into the
thesis folder (Thesis Report/figures).
"""
import math, os
import pandas as pd

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
THESIS = os.path.join(os.path.dirname(os.path.dirname(R)), 'Thesis Report')
FIG = os.path.join(THESIS, 'figures')
D = os.path.join(R, 'contexts/appendix/appendix_data')
BS = chr(92)
C = pd.read_csv(os.path.join(D, 'D_curves.csv'))
COLORS = ['RoyalBlue', 'Orange', 'ForestGreen', 'BrickRed']
MARKS = ['*', 'square*', 'triangle*', 'diamond*']

UTD = ('Update-to-data ratio', [('baseline', 'UTD 1 (baseline)'), ('UTD 2', 'UTD 2'), ('UTD 4', 'UTD 4')])
WIDTH = ('Hidden width', [('width 256', '256'), ('width 512', '512'), ('baseline', '1024 (baseline)')])
BATCH = ('Batch size', [('baseline', '256 (baseline)'), ('batch 512', '512'), ('batch 1024', '1024')])
BATCH_TD3 = ('Batch size', [('baseline', '100 (baseline)'), ('batch 256', '256'), ('batch 512', '512'), ('batch 1024', '1024')])
ROLL = ('Maximum rollout length', [('rollout 1', '1'), ('rollout 15', '15'), ('baseline', '25 (baseline)')])
NUMQ = ('Q-ensemble size', [('num_q 3', '3'), ('baseline', '5 (baseline)'), ('num_q 7', '7')])
HOR = ('Planning horizon', [('horizon 1', '1'), ('baseline', '3 (baseline)'), ('horizon 5', '5')])
PANELS = {
    'sac': [UTD, WIDTH, BATCH], 'td3': [UTD, WIDTH, BATCH_TD3], 'mbpo': [UTD, WIDTH, BATCH], 'tdmpc2': [NUMQ, HOR],
}
ALGN = {'sac': 'SAC', 'td3': 'TD3', 'mbpo': 'MBPO', 'tdmpc2': 'TD-MPC2'}
ENVS = {'HalfCheetah-v5': 'hc', 'Ant-v5': 'ant'}
made = []
for algo in ('sac', 'td3', 'mbpo', 'tdmpc2'):
    for env, es in ENVS.items():
        panels = list(PANELS[algo])
        if algo == 'mbpo' and env == 'Ant-v5':
            panels.append(ROLL)
        g = C[(C.algo == algo) & (C.env == env)]
        rows, tex = [], []
        lo = math.inf
        hi = -math.inf
        for pi, (title, lines) in enumerate(panels):
            for li, (cfg, lab) in enumerate(lines):
                h = g[g.cfg == cfg].sort_values('window')
                assert len(h) == 10, (algo, env, cfg, len(h))
                for _, r in h.iterrows():
                    rows.append(dict(key=pi * 10 + li, panel=title, cfg=cfg, step_k=int(r.step_end // 1000), mean=r['mean'], sd=r['sd']))
                    lo = min(lo, r['mean'] - r['sd'])
                    hi = max(hi, r['mean'] + r['sd'])
        raw = hi - lo
        e = math.floor(math.log10(raw / 10))
        step = next(m * 10 ** e for m in (1, 2, 5, 10) if raw / (m * 10 ** e) <= 10)
        ymin, ymax = math.floor(lo / step) * step, math.ceil(hi / step) * step
        csvname = f'app-return-{algo}-{es}.csv'
        pd.DataFrame(rows).to_csv(os.path.join(FIG, csvname), index=False, float_format='%.6g')
        tex.append(f'% Data: figures/{csvname} (from contexts/appendix/appendix_data/D_curves.csv, Appendix D)')
        tex.append(BS + 'begin{tikzpicture}')
        for pi, (title, lines) in enumerate(panels):
            col, row = pi % 2, pi // 2
            at = f'{{({0.52 * col:.2f}{BS}textwidth,{-0.52 * row:.2f}{BS}textwidth)}}'
            opts = [f'width=0.47{BS}textwidth', f'height=0.33{BS}textwidth', 'grid=major', 'grid style={gray!25}',
                    f'tick label style={{font={BS}footnotesize}}', f'label style={{font={BS}footnotesize}}', f'title style={{font={BS}footnotesize}}',
                    f'xlabel={{Training step [{BS}num{{e3}}]}}'.replace(BS + 'num{e3}', 'thousand'), f'xmin=0, xmax=105, xtick={{0,20,40,60,80,100}}',
                    f'ymin={ymin:g}, ymax={ymax:g}', f'title={{{title}}}', f'at={at}',
                    'scaled y ticks=false',
                    f'legend style={{font={BS}tiny, at={{(0.5,-0.30)}}, anchor=north, legend columns=-1, draw=none, column sep=2pt}}']
            if col == 0:
                opts.append('ylabel={Mean return of training episodes}')
            else:
                opts.append('yticklabels={}')
            tex.append(BS + 'begin{axis}[' + ', '.join(opts) + ']')
            for li, (cfg, lab) in enumerate(lines):
                key = pi * 10 + li
                is_base = 'baseline' in lab
                color = 'black' if is_base else COLORS[li % 4]
                style = 'thick' if is_base else 'thin'
                tex.append(BS + f'addplot[{color}, {style}, mark={MARKS[li % 4]}, mark size=1pt, error bars/.cd, y dir=both, y explicit, '
                           f'error bar style={{line width=0.3pt}}] table[col sep=comma, x=step_k, y=mean, y error=sd, '
                           f'restrict expr to domain={{{BS}thisrow{{key}}}}{{{key}:{key}}}] {{figures/{csvname}}};')
                tex.append(BS + 'addlegendentry{' + lab + '}')
            tex.append(BS + 'end{axis}')
        tex.append(BS + 'end{tikzpicture}')
        open(os.path.join(FIG, f'app-return-{algo}-{es}.tex'), 'w', encoding='utf-8').write('\n'.join(tex) + '\n')
        made.append((algo, env, len(panels), round(ymin), round(ymax)))
print(made)
