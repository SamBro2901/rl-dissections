"""Appendix A: verify the logged configurations of the 280 canonical runs.

 (1) algo_config of every configuration differs from the baseline of its algorithm and environment only in the swept factor;
 (2) within a configuration, algo_config and experiment_config are identical for the five seeds (except the seed and one logging field);
 (3) experiment_config is the same in all runs except warmup_steps (TD3) and the logging field thermal_gate_reference_file;
 (4) every MBPO run has a model fit in each of its 100 epochs.
Writes appendix_data/A_config_diffs.csv and appendix_data/A_verify_summary.json.
"""
import collections, csv, json, os

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(R, 'contexts/appendix/appendix_data')
runs = list(csv.DictReader(open(os.path.join(R, 'contexts/step1/04_canonical_run_list.csv'))))
meta = {}
for r in runs:
    meta[(r['algo'], r['env'], r['config'], int(r['seed']))] = (r, json.load(open(os.path.join(R, 'results', r['run_dir'], 'metadata.json'))))
cfgs = collections.OrderedDict()
for (a, e, c, s), (r, m) in meta.items():
    cfgs.setdefault((a, e, c), []).append(m)
base = {(a, e): ms[0] for (a, e, c), ms in cfgs.items() if c == 'baseline'}
rows, bad_single = [], []
for (a, e, c), ms in cfgs.items():
    b = base[(a, e)]
    d = {k: (b['algo_config'].get(k), ms[0]['algo_config'].get(k)) for k in set(b['algo_config']) | set(ms[0]['algo_config']) if b['algo_config'].get(k) != ms[0]['algo_config'].get(k)}
    if c == 'baseline':
        ok = not d
    else:
        swept = c.split('=')[0]
        ok = set(d) == {swept}
    if not ok:
        bad_single.append((a, e, c, d))
    rows.append(dict(algo=a, env=e, config=c, seeds=len(ms), algo_diff_vs_baseline=json.dumps(d), single_factor=ok))
with open(os.path.join(OUT, 'A_config_diffs.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
within_bad = 0
log_field_diff = 0
for (a, e, c), ms in cfgs.items():
    for m in ms[1:]:
        if m['algo_config'] != ms[0]['algo_config']:
            within_bad += 1
        ex0 = {k: v for k, v in ms[0]['experiment_config'].items() if k not in ('seed', 'thermal_gate_reference_file')}
        ex1 = {k: v for k, v in m['experiment_config'].items() if k not in ('seed', 'thermal_gate_reference_file')}
        if ex0 != ex1:
            within_bad += 1
exp_vals = collections.defaultdict(collections.Counter)
for (a, e, c, s), (r, m) in meta.items():
    for k, v in m['experiment_config'].items():
        if k not in ('seed', 'algo_name', 'env_id'):
            exp_vals[k][json.dumps(v)] += 1
varying = {k: dict(v) for k, v in exp_vals.items() if len(v) > 1 or sum(v.values()) != 280}
fits = collections.Counter()
for (a, e, c, s), (r, m) in meta.items():
    if a == 'mbpo':
        tm = json.load(open(os.path.join(R, 'results', r['run_dir'], 'training_metrics.json')))
        fits[(sum(1 for x in tm['epochs'] if x['model_train_epochs'] is not None), len(tm['epochs']))] += 1
summary = dict(runs=len(meta), configurations=len(cfgs), configs_not_single_factor=bad_single, within_config_deviations=within_bad, varying_experiment_fields=varying,
               thermal_gate_reference_file_runs=exp_vals['thermal_gate_reference_file'].get('"results/_thermal_reference.json"', 0),
               mbpo_fits=[[list(k), v] for k, v in fits.items()], thermal_gate_reasons=collections.Counter(m['thermal_gate'].get('reason') for (r, m) in meta.values()))
json.dump(summary, open(os.path.join(OUT, 'A_verify_summary.json'), 'w'), indent=1, default=dict)
print(json.dumps(summary, indent=1, default=dict))
