"""Appendix E: fill the template (sessions, commits, scripts, sweep scripts, tab:thr-provenance)."""
import collections, csv, json, os, subprocess

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(R, 'contexts/appendix/appendix_data')
BS = chr(92)
tpl = open(os.path.join(R, 'contexts/appendix/appendix_E.tpl.tex'), encoding='utf-8').read()


def git(*a):
    return subprocess.run(['git', '-C', R] + list(a), capture_output=True, text=True).stdout.strip()


def tt(s):
    return BS + 'texttt{' + s.replace('_', BS + '_' + BS + 'allowbreak ').replace('/', '/' + BS + 'allowbreak ') + '}'


def n(x):
    return BS + 'num{%s}' % x


# ---------------------------------------------------------------- tab:thr-provenance (verbatim from the existing chapter)
prov = open(os.path.join(D, 'E_thr_provenance_table.tex'), encoding='utf-8').read().rstrip('\n')
assert prov.lstrip().startswith(BS + 'begin{table}') and prov.rstrip().endswith(BS + 'end{table}')
tpl = tpl.replace('%%E_THR_PROVENANCE%%', prov)

# ---------------------------------------------------------------- sessions
content = {
    0: 'SAC baselines',
    1: 'SAC baseline (last two Ant-v5 runs); SAC UTD 2, 4 on HalfCheetah-v5',
    2: 'MBPO baselines',
    3: 'TD3 baselines; UTD 2, 4 of TD3 and of SAC on Ant-v5',
    4: 'TD-MPC2 baselines',
    5: 'Width sweeps of SAC, TD3, MBPO',
    6: 'UTD sweeps of MBPO',
    7: 'Batch-size sweeps on HalfCheetah-v5',
    8: 'Batch-size sweeps and MBPO rollout length on Ant-v5',
    9: 'TD-MPC2 num_q and horizon sweeps',
}
NAME = {'sac': 'SAC', 'td3': 'TD3', 'mbpo': 'MBPO', 'tdmpc2': 'TD-MPC2'}
rows = list(csv.DictReader(open(os.path.join(D, 'E_sessions.csv'))))
out = []
tot = 0
for r in rows:
    i = int(r['session'])
    per = collections.OrderedDict()
    for part in r['by_algo_env'].split('; '):
        ae, k = part.rsplit(':', 1)
        a, e = ae.split('/')
        per.setdefault(a, {})[('HC' if e.startswith('Half') else 'Ant')] = int(k)
    algs = '; '.join(NAME[a] + ' ' + ', '.join(f'{e} {k}' for e, k in sorted(v.items(), key=lambda t: t[0] != 'HC')) for a, v in per.items())
    c = content[i].replace('num_q', BS + 'texttt{num' + BS + '_q}')
    cm = ', '.join(tt(x) for x in r['commits'].split(','))
    gap = r['gap_before_h']
    out.append('      ' + ' & '.join([str(i), r['first_start'][:16].replace(' ', '~'), r['last_end'][:16].replace(' ', '~'), f"{float(r['span_h']):.1f}",
                                       ('---' if gap == '' else f'{float(gap):.1f}'), n(r['runs']), algs, c, cm]) + ' ' + BS + BS)
    tot += int(r['runs'])
summary = json.load(open(os.path.join(D, 'E_sessions_summary.json')))
out.append('      ' + BS + 'midrule')
out.append('      ' + ' & '.join(['', 'Total', '', f"{summary['sum_span_hours']:.1f}", '', n(tot), '', '', '']) + ' ' + BS + BS)
tpl = tpl.replace('%%E_SESSION_ROWS%%', '\n'.join(out))

# ---------------------------------------------------------------- commits
runs = list(csv.DictReader(open(os.path.join(D, 'E_sessions_runs.csv'))))
by = collections.defaultdict(list)
for r in runs:
    by[r['commit']].append(r)
order = sorted(by, key=lambda c: min(x['start_utc'] for x in by[c]))
change = {
    'f23f448': 'first recorded commit',
    '92dcee0': 'runner and thermal gate: start and end temperature and power added to the metadata; thermal-gate file name no longer written to the logged configuration (logging only)',
    'ecb98bb': 'none',
    '0bbbbb7': 'default hidden size of MBPO $(256,256)\\to(1024,1024)$ (one line of the configuration)',
    '58962d2': 'TD3 module (\\texttt{algorithms/td3.py}), \\texttt{TD3Config}, four lines of dispatch in the runner',
    'aa9186b': 'none',
    'b5b7e54': 'TD-MPC2 module (\\texttt{algorithms/tdmpc2.py}), \\texttt{TDMPC2Config}, four lines of dispatch in the runner',
    '7e672ef': 'none', 'ecb7135': 'none', '9b4b44c': 'none', '788ffcd': 'none', 'a3582b5': 'none',
}
crow = []
subjects_tex = lambda s: s.replace('&', BS + '&').replace('%', BS + '%').replace('_', BS + '_').replace('#', BS + '#')
for c in order:
    info = git('log', '-1', '--format=%ad|%s', '--date=format:%Y-%m-%d %H:%M', c)
    d, sub = info.split('|', 1)
    cnt = collections.Counter((x['algo'], x['env']) for x in by[c])
    ae = collections.OrderedDict()
    for (a, e), k in sorted(cnt.items(), key=lambda t: (['sac', 'td3', 'mbpo', 'tdmpc2'].index(t[0][0]), t[0][1] != 'HalfCheetah-v5')):
        ae.setdefault(NAME[a], []).append(('HC' if e.startswith('Half') else 'Ant') + ' ' + str(k))
    aes = '; '.join(a + ' ' + ', '.join(v) for a, v in ae.items())
    crow.append('      ' + ' & '.join([tt(c), d, subjects_tex(sub), n(len(by[c])), aes, change[c]]) + ' ' + BS + BS)
assert sum(len(v) for v in by.values()) == 280 and len(order) == 12
tpl = tpl.replace('%%E_COMMIT_ROWS%%', '\n'.join(crow))

# ---------------------------------------------------------------- scripts
scripts = [
    ('flop_analysis/measure_flops.py', 'per-call FLOPs with \\texttt{FlopCounterMode} on the real network classes', 'signatures found in the \\texttt{metadata.json} files', '\\texttt{flops\\_per\\_call.json}'),
    ('flop_analysis/compute_energy_per_flop.py', 'call counts, FLOPs, energy per FLOP per run and across the canonical seeds', 'run files, \\texttt{flops\\_per\\_call.json}', '\\texttt{per\\_run\\_energy\\_per\\_flop.csv}, \\texttt{cross\\_seed\\_energy\\_per\\_flop.csv}'),
    ('flop_analysis/check_totals.py', 'consistency checks of the training totals', 'run files, per-run table', 'console'),
    ('flop_analysis/verify_mbpo_fit_flops.py', 'MBPO fit formula against \\texttt{FlopCounterMode}', 'MBPO model code, PyTorch', 'console, \\texttt{verify\\_mbpo\\_fit\\_flops.log}'),
    ('flop_analysis/verify_analytic_flops.py', 'analytic dense-layer counts against the counted per-call FLOPs', '\\texttt{flops\\_per\\_call.json}', 'console'),
    ('flop_analysis/correlation_analysis.py', 'Spearman correlations of energy and return with FLOPs', 'run files, per-run table', '\\texttt{output/correlation/} (\\texttt{config\\_means.csv}, \\texttt{correlation\\_summary.csv})'),
    ('flop_analysis/section_5_2_analysis.py', 'FLOP predictability of energy (Spearman, log-log and linear fits, adjacent-pair costs)', 'per-run table (canonical seeds)', '\\texttt{contexts/section\\_5.2\\_data/}'),
    ('flop_analysis/section_5_2_render.py', 'renders the context file of Section 5.2', 'files above', '\\texttt{contexts/section\\_5.2.md}'),
    ('flop_analysis/section_5_3_analysis.py', 'orderings and swaps across the sweeps, seed-level orderings, TD-MPC2 ranges', 'per-run table (canonical seeds)', '\\texttt{contexts/section\\_5.3\\_data/}'),
    ('flop_analysis/section_5_3_render.py', 'renders the context file of Section 5.3', 'files above', '\\texttt{contexts/section\\_5.3.md}'),
    ('flop_analysis/section_5_checks.py', 'read-only checks of the numbers of Chapter 5', 'context CSVs, run files', 'console'),
    ('flop_analysis/section_6_checks.py', 'read-only checks of the numbers of Chapter 6 (the two clock runs)', 'run files', 'console'),
    ('flop_analysis/section_7_checks.py', 'read-only checks of the numbers of Chapter 7', 'run files, per-run table', 'console'),
    ('flop_analysis/section_8_checks.py', 'read-only checks of the numbers of Chapter 8', 'run files, per-run table', 'console'),
    ('flop_analysis/appendix_tables.py', 'LaTeX tables of \\zcref{app:full-results}', 'per-run table, context CSVs', '\\texttt{contexts/appendix/appendix\\_data/}'),
    ('contexts/step1/scripts/', 'idle drift, task durations, data-quality checks and execution timeline', 'run files', '\\texttt{contexts/step1/}'),
]
srow = []
for p, purpose, inp, outp in scripts:
    tracked = subprocess.run(['git', '-C', R, 'ls-files', '--error-unmatch', p], capture_output=True).returncode == 0
    if os.path.isdir(os.path.join(R, p)):
        tracked = bool(git('ls-files', p))
    if tracked:
        first = git('log', '--follow', '--diff-filter=A', '--format=%h', '--', p).splitlines()
        vc = 'tracked, first in ' + tt(first[-1]) if first else 'tracked'
    else:
        vc = 'not committed'
    srow.append('      ' + ' & '.join([tt(p), purpose, inp, outp, vc]) + ' ' + BS + BS)
tpl = tpl.replace('%%E_SCRIPT_ROWS%%', '\n'.join(srow))

# ---------------------------------------------------------------- sweep scripts
def tf(*paths):
    return ', '.join(tt(p) for p in paths)


sweeps = [
    ('(by hand)', 10, 'SAC baselines, both environments', '---', '---'),
    (tt('scripts/run_td3_env_sweep.sh'), 10, 'TD3 baselines (warmup ' + n(10000) + ')', tf('results/td3/_env_sweep_status.json', 'results/td3/_env_sweep.log'), 'no'),
    (tt('scripts/run_mbpo_env_sweep.sh'), 10, 'MBPO baselines', tf('results/mbpo/_env_sweep_status.json', 'results/mbpo/_env_sweep.log'), 'no'),
    (tt('scripts/run_tdmpc2_seed_sweep.sh'), 10, 'TD-MPC2 baselines', tf('results/tdmpc2/_seed_sweep_status.json', 'results/tdmpc2/_seed_sweep.log'), 'no'),
    (tt('scripts/run_utd_sweep.sh') + ' (earlier version)', 10, 'SAC UTD 2, 4 on HalfCheetah-v5', tf('results/sac/HalfCheetah-v5/_utd_sweep.log'), 'yes'),
    (tt('scripts/run_utd_sweep.sh') + ' (earlier version)', 30, 'UTD 2, 4 of SAC on Ant-v5 and of TD3 on both environments', tf('results/_utd_sweep.log') + ' (first ' + n(30) + ' blocks; status file overwritten)', 'yes'),
    (tt('scripts/run_utd_sweep.sh'), 20, 'UTD 2, 4 of MBPO on both environments', tf('results/_utd_sweep_status.json', 'results/_utd_sweep.log'), 'yes'),
    (tt('scripts/run_width_sweep.sh'), 60, 'width ' + n(256) + ', ' + n(512) + ' of SAC, TD3, MBPO on both environments', tf('results/_width_sweep_status.json', 'results/_width_sweep.log'), 'yes'),
    (tt('scripts/run_batch_size_sweep.sh'), 35, 'batch sizes of SAC, MBPO, TD3 on HalfCheetah-v5', tf('results/_batch_size_sweep_status.json', 'results/_batch_size_sweep.log'), 'yes'),
    (tt('scripts/run_ant_overnight_sweep.sh'), 45, 'batch sizes of SAC, MBPO, TD3 and MBPO rollout length on Ant-v5', tf('results/_ant_overnight_sweep_status.json', 'results/_ant_overnight_sweep.log'), 'yes'),
    (tt('scripts/run_tdmpc2_numq_horizon_sweep.sh'), 40, 'TD-MPC2 ' + tt('num_q') + ' and horizon on both environments', tf('results/tdmpc2/_numq_horizon_sweep_status.json', 'results/tdmpc2/_numq_horizon_sweep.log'), 'yes'),
]
wrow = []
for sc, k, cnt, files, off in sweeps:
    wrow.append('      ' + ' & '.join([sc, n(k), cnt, files, off]) + ' ' + BS + BS)
assert sum(k for _, k, *_ in sweeps) == 280
tpl = tpl.replace('%%E_SWEEP_ROWS%%', '\n'.join(wrow))
open(os.path.join(R, 'contexts/appendix/appendix_E.tex'), 'w', encoding='utf-8').write(tpl)
print([l for l in tpl.splitlines() if '%%E_' in l])
