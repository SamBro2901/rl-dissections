"""Appendix C: totals check restricted to the 280 canonical runs.

Restates checks 1 to 3 of flop_analysis/check_totals.py (definitions copied from it) for the canonical
runs only, and evaluates check 3 with two CSV parsers (Python float(), exact; pandas default C parser).
Also: one TOTAL row per run, no unexpected segment key.  Writes appendix_data/C_totals_check.json.
"""
import csv, glob, json, os, re
from collections import defaultdict

import pandas as pd

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(R, 'contexts/appendix/appendix_data')
runs = list(csv.DictReader(open(os.path.join(R, 'contexts/step1/04_canonical_run_list.csv'))))
TRAIN = {"rollout", "buffer_sample", "critic_update", "actor_update", "target_update", "dynamics_model_update",
         "synthetic_rollout_generation", "world_model_pretrain"}
EXCL = {"idle_baseline_head", "idle_baseline_tail", "warmup"}
SUB = ("buffer_sample", "critic_update", "actor_update", "target_update")
rows_by_run = defaultdict(list)
with open(os.path.join(R, 'flop_analysis/output/per_run_energy_per_flop.csv'), newline='') as f:
    for row in csv.DictReader(f):
        rows_by_run[row['run_dir']].append(row)


def rel(a, b):
    return abs(a - b) / abs(b) if b else abs(a - b)


res = dict(runs=0, check1_max=0.0, check2_exact=0, check2_bad=0, check3_py_max=0.0, check3_pd_max=0.0,
           one_total=0, unexpected_keys=0, missing_rows=0, total_flop_type_bad=0)
for r in runs:
    d = os.path.join(R, 'results', r['run_dir'])
    key = 'results/' + r['run_dir']
    seg = json.load(open(os.path.join(d, 'segment_energy.json')))
    res['runs'] += 1
    unexpected = [k for k in seg if not k.startswith('_') and k not in TRAIN and k not in EXCL]
    res['unexpected_keys'] += bool(unexpected)
    rows = rows_by_run.get(key)
    if not rows:
        res['missing_rows'] += 1
        continue
    tot = [x for x in rows if x['segment'] == 'TOTAL_MEASURED_TRAINING']
    res['one_total'] += len(tot) == 1
    res['total_flop_type_bad'] += tot[0]['flop_type'] != 'mixed_total'
    exp_e = sum(v for k, v in seg.items() if k in TRAIN)
    res['check1_max'] = max(res['check1_max'], rel(float(tot[0]['total_energy_kwh']), exp_e))
    msum = sum(int(x['total_flops']) for x in rows if x['flop_type'] == 'matmul')
    if int(tot[0]['total_flops']) == msum:
        res['check2_exact'] += 1
    else:
        res['check2_bad'] += 1
    csvp = sorted(glob.glob(os.path.join(d, 'emissions_*.csv')))[0]
    py = 0.0
    with open(csvp, newline='') as f:
        for row in csv.DictReader(f):
            if re.fullmatch(r'gradient_updates_\d+', row.get('task_name') or ''):
                py += float(row['energy_consumed'])
    b = pd.read_csv(csvp)
    pdv = b[b.task_name.str.fullmatch(r'gradient_updates_\d+')].energy_consumed.sum()
    alloc = sum(seg[s] for s in SUB)
    res['check3_py_max'] = max(res['check3_py_max'], rel(alloc, py))
    res['check3_pd_max'] = max(res['check3_pd_max'], rel(alloc, pdv))
json.dump(res, open(os.path.join(OUT, 'C_totals_check.json'), 'w'), indent=1)
print(json.dumps(res, indent=1))
