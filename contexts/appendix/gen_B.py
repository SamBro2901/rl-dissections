"""Appendix B: assemble appendix_B.tex from the fragments written by flop_analysis/appendix_tables.py."""
import glob, os, re
import pandas as pd

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(R, 'contexts/appendix/appendix_data/B')
BS = chr(92)
tpl = open(os.path.join(R, 'contexts/appendix/appendix_B.tpl.tex'), encoding='utf-8').read()
frs = {os.path.basename(p)[2:-4]: open(p, encoding='utf-8').read().rstrip('\n') + '\n' for p in glob.glob(os.path.join(D, 'B_*.tex'))}

# verification summary table (groups of tables of the main text)
v = pd.read_csv(os.path.join(D, 'B_verification.csv'))
groups = [('SAC (Section 4.1)', 'tab:sac-'), ('TD3 (Section 4.2)', 'tab:td3-'), ('MBPO (Section 4.3)', 'tab:mbpo-'), ('TD-MPC2 (Section 4.4)', 'tab:tdmpc2-'),
          ('Energy versus compute (Section 5.2)', 'tab:ec-'), ('Robustness (Section 5.3)', 'tab:rob-')]
rows = []
for name, pre in groups:
    g = v[v.label.str.startswith(pre)]
    nums = int(g.tokens.sum())
    ok = int(g.matched.sum())
    rows.append('    ' + ' & '.join([name, BS + 'num{%d}' % len(g), BS + 'num{%d}' % nums, BS + 'num{%d}' % ok, BS + 'num{%d}' % (nums - ok),
                                     BS + 'num{%.2g}' % g.max_rel_dev.max()]) + ' ' + BS + BS)
tot = v.tokens.sum()
okt = v.matched.sum()
rows.append('    ' + BS + 'midrule')
rows.append('    Total & ' + BS + 'num{%d} & ' % len(v) + BS + 'num{%d} & ' % tot + BS + 'num{%d} & ' % okt + BS + 'num{%d} & ' % (tot - okt) + BS + 'num{%.2g} ' % v.max_rel_dev.max() + BS + BS)
table = '\n'.join([
    BS + 'begin{table}[tbp]', '  ' + BS + 'centering', '  ' + BS + 'footnotesize',
    '  ' + BS + 'caption{Agreement of the tables of the main text with the recomputed values. A number counts as found if the recomputed value, rounded to the digits printed, gives the printed number. '
    'The numbers not found are columns that are not tabulated here (ratios between the environments, the sweep values in the first columns and similar). '
    'The last column is the largest relative difference of a number found, over the numbers printed with at least four digits.}',
    '  ' + BS + 'label{tab:app-verification}',
    '  ' + BS + 'begin{tabular}{@{}lrrrrr@{}}', '    ' + BS + 'toprule',
    '    Tables of the main text & Tables & Numbers & Found & Not found & Max.\\ rel.\\ difference ' + BS + BS, '    ' + BS + 'midrule'] + rows +
    ['    ' + BS + 'bottomrule', '  ' + BS + 'end{tabular}', BS + 'end{table}']) + '\n'
frs['verification_table'] = table
for k, val in frs.items():
    tpl = tpl.replace('%%B_' + k + '%%', val)
left = [l for l in tpl.splitlines() if '%%B_' in l]
open(os.path.join(R, 'contexts/appendix/appendix_B.tex'), 'w', encoding='utf-8').write(tpl)
print('left:', left, 'lines:', len(tpl.splitlines()))
