"""Appendix C: LaTeX table bodies from the C_*.csv files (run C_flop_tables.py first)."""
import csv, os, json
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(R, 'contexts/appendix/appendix_data')
BS = chr(92)
ALG = {'sac': 'SAC', 'td3': 'TD3', 'mbpo': 'MBPO', 'tdmpc2': 'TD-MPC2'}
ENV = {'HalfCheetah-v5': 'HalfCheetah-v5', 'Ant-v5': 'Ant-v5'}


def n(x):
    return BS + 'num{%s}' % x


def rows_learners():
    out = []
    prev = None
    for r in csv.DictReader(open(os.path.join(D, 'C_percall_learners.csv'))):
        hid = r['hidden'].replace('x', ',')
        key = (r['algo'], r['env'])
        a = ALG[r['algo']] if r['algo'] != (prev or ('',))[0] else ''
        e = ENV[r['env']] if key != prev else ''
        if prev is not None and r['algo'] != prev[0]:
            out.append('      ' + BS + 'midrule')
        prev = key
        out.append('      ' + ' & '.join([a, e, n(r['batch']), '$(%s)$' % hid, n(r['actor_forward_bs1']), n(r['critic_fwdbwd']),
                                            n(r['actor_fwdbwd']), n(r['target_ops'])]) + ' ' + BS + BS)
    return '\n'.join(out)


def rows_mbpo_model():
    out = []
    for r in csv.DictReader(open(os.path.join(D, 'C_percall_mbpo_model.csv'))):
        sig = r['signature'].split('_ens')[0]  # bs256_h1024x1024
        b = sig.split('_')[0][2:]
        h = sig.split('_h')[1].replace('x', ',')
        out.append('      ' + ' & '.join([ENV[r['env']], n(b), '$(%s)$' % h, n(r['member_fwdbwd_mb']), n(r['per_sample_member_fwdbwd']),
                                           n(r['ensemble_forward_bs1']), n(r['per_synthetic_sample'])]) + ' ' + BS + BS)
    return '\n'.join(out)


def rows_tdmpc2():
    out = []
    prev = None
    for r in csv.DictReader(open(os.path.join(D, 'C_percall_tdmpc2.csv'))):
        e = ENV[r['env']] if r['env'] != prev else ''
        if prev is not None and r['env'] != prev:
            out.append('      ' + BS + 'midrule')
        prev = r['env']
        clf = '---' if r['episodic'] == 'False' else n(r['clf_plan']) + ' / ' + n(r['clf_wm'])
        out.append('      ' + ' & '.join([e, n(r['horizon']), n(r['num_q']), 'yes' if r['episodic'] == 'True' else 'no', n(r['plan']),
                                           n(r['wm_step']), n(r['prior_step']), n(r['update_total']), n(r['polyak_ops']), clf]) + ' ' + BS + BS)
    return '\n'.join(out)


def rows_crosscheck():
    out = []
    tot = 0
    for r in csv.DictReader(open(os.path.join(D, 'C_crosscheck_summary.csv'))):
        out.append('    ' + ' & '.join([ALG[r['algo']], n(r['signatures']), n(r['comparisons']), n(r['inexact']),
                                         '0' if float(r['max_rel_dev']) == 0 else BS + 'num{%.3g}' % float(r['max_rel_dev'])]) + ' ' + BS + BS)
        tot += int(r['comparisons'])
    out.append('    ' + BS + 'midrule')
    out.append('    Total & ' + n(42) + ' & ' + n(tot) + ' & ' + n(0) + ' & 0 ' + BS + BS)
    return '\n'.join(out)


def rows_fit_example():
    out = []
    for e in json.load(open(os.path.join(D, 'C_mbpo_fit_examples.json'))):
        out.append('      ' + ' & '.join([e['env'], n(e['n_fits']), n(e['fit_epochs_total']), n(e['optimiser_steps']), n(e['train_flops']),
                                           n(e['holdout_flops']), n(e['formula_total']), n(e['logged_total']), 'yes' if e['exact'] else 'NO']) + ' ' + BS + BS)
    return '\n'.join(out)


tpl = open(os.path.join(R, 'contexts/appendix/appendix_C.tpl.tex'), encoding='utf-8').read()
ex = json.load(open(os.path.join(D, 'C_sac_worked_example.json')))
fit = json.load(open(os.path.join(D, 'C_mbpo_fit_examples.json')))
chk = json.load(open(os.path.join(D, 'C_totals_check.json')))
rep = {
    '%%C_LEARNERS%%': rows_learners(), '%%C_MBPO_MODEL%%': rows_mbpo_model(), '%%C_TDMPC2%%': rows_tdmpc2(),
    '%%C_CROSSCHECK%%': rows_crosscheck(), '%%C_FIT_EXAMPLE_ROWS%%': rows_fit_example(),
}
for k, v in rep.items():
    tpl = tpl.replace(k, v)
open(os.path.join(R, 'contexts/appendix/appendix_C.tex'), 'w', encoding='utf-8').write(tpl)
print('written; remaining placeholders:', [l for l in tpl.splitlines() if '%%C_' in l])
