"""Appendix F: fill the template from F_clock_numbers.json (run F_clock_verify.py first) and C_segment_summary.csv."""
import csv, json, math, os
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(R, 'contexts/appendix/appendix_data')
BS = chr(92)
num = json.load(open(os.path.join(D, 'F_clock_numbers.json')))
a, b = num['R2000'], num['R200']
tpl = open(os.path.join(R, 'contexts/appendix/appendix_F.tpl.tex'), encoding='utf-8').read()


def tt(s):
    return BS + 'texttt{' + s.replace('_', BS + '_') + '}'


def sig(x, n=4):
    if x == 0:
        return '0'
    d = n - 1 - int(math.floor(math.log10(abs(x))))
    return f'{round(x, d):.{max(d, 0)}f}'


sub = ['buffer_sample', 'critic_update', 'actor_update', 'target_update']
rows = []
for s in sub:
    rows.append(f'    {tt(s)} & {a["E_sub_kJ"][s]:.2f} & {b["E_sub_kJ"][s]:.2f} & {b["E_sub_kJ"][s] / a["E_sub_kJ"][s]:.2f} {BS}{BS}')
rows.append('    ' + BS + 'midrule')
rows.append(f'    Gradient updates (measured) & {a["E_gu_kJ"]:.2f} & {b["E_gu_kJ"]:.2f} & {b["E_gu_kJ"] / a["E_gu_kJ"]:.2f} {BS}{BS}')
tpl = tpl.replace('%%F_SUBALLOC_ROWS%%', '\n'.join(rows))
tpl = tpl.replace('%%F_PALLOC_2000%%', f'{a["P_alloc_all"]:.1f}').replace('%%F_PALLOC_200%%', f'{b["P_alloc_all"]:.1f}')
tpl = tpl.replace('%%F_PGU_2000%%', f'{a["P_gu_measured"]:.1f}').replace('%%F_PGU_200%%', f'{b["P_gu_measured"]:.1f}')

irows = []
for name, r in (('R2000', a), ('R200', b)):
    i = r['idle']
    h, t = i['idle_baseline_head'], i['idle_baseline_tail']
    irows.append(f'    {name} & {h["P_corrected"]:.2f} & {t["P_corrected"]:.2f} & {r["idle_mean_corrected"]:.2f} & {h["P_recorded"]:.2f} & {t["P_recorded"]:.2f} {BS}{BS}')
tpl = tpl.replace('%%F_IDLE_ROWS%%', '\n'.join(irows))

seg = {}
for r in csv.DictReader(open(os.path.join(R, 'contexts/Section_6_data/C_segment_summary.csv'))):
    seg[(r['clock'], r['segment'])] = r
order = ['rollout', 'dynamics_model_update', 'synthetic_rollout_generation', 'gradient_updates', 'buffer_sample', 'critic_update', 'actor_update',
         'target_update', 'TOTAL_MEASURED_TRAINING', 'warmup', 'idle_baseline_head', 'idle_baseline_tail']
kind = {'rollout': 'measured', 'dynamics_model_update': 'measured', 'synthetic_rollout_generation': 'measured', 'gradient_updates': 'measured',
        'buffer_sample': 'alloc.', 'critic_update': 'alloc.', 'actor_update': 'alloc.', 'target_update': 'alloc.', 'TOTAL_MEASURED_TRAINING': 'total',
        'warmup': 'other', 'idle_baseline_head': 'other', 'idle_baseline_tail': 'other'}
label = {'TOTAL_MEASURED_TRAINING': 'Training total', 'idle_baseline_head': tt('idle_baseline_head'), 'idle_baseline_tail': tt('idle_baseline_tail')}
frows = []
for s in order:
    x, y = seg[('2000MHz', s)], seg[('200MHz', s)]

    def f(r, k):
        v = r[k]
        return None if v in ('', None) else float(v)
    cells = [label.get(s, tt(s)), kind[s]]
    for k, n in (('energy_j', 6), ('duration_s', 5), ('mean_power_w', 5)):
        cells += [BS + 'num{%s}' % sig(f(x, k), n), BS + 'num{%s}' % sig(f(y, k), n)]
    jx, jy = f(x, 'energy_per_flop_j_per_flop'), f(y, 'energy_per_flop_j_per_flop')
    cells += [BS + 'num{%.3e}' % jx if jx else '---', BS + 'num{%.3e}' % jy if jy else '---']
    sh = f(x, 'share_of_training_total')
    cells += [BS + 'num{%s}' % sig(100 * sh, 3) if sh is not None else '---']
    frows.append('      ' + ' & '.join(cells) + ' ' + BS + BS)
    if s == 'target_update':
        frows.append('      ' + BS + 'midrule')
    if s == 'TOTAL_MEASURED_TRAINING':
        frows.append('      ' + BS + 'midrule')
tpl = tpl.replace('%%F_FULL_ROWS%%', '\n'.join(frows))
open(os.path.join(R, 'contexts/appendix/appendix_F.tex'), 'w', encoding='utf-8').write(tpl)
print([l for l in tpl.splitlines() if '%%F_' in l])
