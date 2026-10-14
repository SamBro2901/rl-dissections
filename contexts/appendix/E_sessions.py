"""Appendix E: session table from the start/end times in the metadata of the 280 canonical runs.

A new session starts after a gap of more than GAP minutes between the end of one run and the start of the next
(runs sorted by start time).  Checks the 30 min / 2 h equivalence and the totals (280 runs, sum of run durations,
sum of session spans).  Writes appendix_data/E_sessions.csv / E_sessions_runs.csv / E_sessions_summary.json.
"""
import csv, json, os, datetime as dt, collections

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(R, 'contexts/appendix/appendix_data')
runs = list(csv.DictReader(open(os.path.join(R, 'contexts/step1/04_canonical_run_list.csv'))))
P = lambda s: dt.datetime.fromisoformat(s)
recs = []
for r in runs:
    m = json.load(open(os.path.join(R, 'results', r['run_dir'], 'metadata.json')))
    recs.append(dict(algo=r['algo'], env=r['env'], seed=int(r['seed']), config=r['config'], run_dir=r['run_dir'],
                     t0=P(m['start_time_utc']), t1=P(m['end_time_utc']), commit=m['git_commit'][:7]))
recs.sort(key=lambda x: x['t0'])


def sessions(gap_min):
    out = [[recs[0]]]
    for a, b in zip(recs, recs[1:]):
        if (b['t0'] - a['t1']).total_seconds() / 60 > gap_min:
            out.append([])
        out[-1].append(b)
    return out


S30, S60, S120 = sessions(30), sessions(60), sessions(120)
same = lambda A, B: [[x['run_dir'] for x in s] for s in A] == [[x['run_dir'] for x in s] for s in B]
summary = dict(n_runs=len(recs), sessions_30=len(S30), sessions_60=len(S60), sessions_120=len(S120), same_30_120=same(S30, S120), same_30_60=same(S30, S60))
summary['sum_run_hours'] = sum((x['t1'] - x['t0']).total_seconds() for x in recs) / 3600
spans = [(s[-1]['t1'] - s[0]['t0']).total_seconds() / 3600 for s in S30]
summary['sum_span_hours'] = sum(spans)
# min gap between sessions and max gap inside sessions
gaps_between = [(S30[i + 1][0]['t0'] - S30[i][-1]['t1']).total_seconds() / 3600 for i in range(len(S30) - 1)]
inside = [(b['t0'] - a['t1']).total_seconds() / 60 for s in S30 for a, b in zip(s, s[1:])]
summary['gaps_between_h'] = gaps_between
summary['max_gap_inside_min'] = max(inside)
summary['min_gap_between_min'] = min(gaps_between) * 60
rows = []
for i, s in enumerate(S30):
    cnt = collections.Counter((x['algo'], x['env']) for x in s)
    cfgs = collections.OrderedDict()
    for x in s:
        cfgs.setdefault((x['algo'], x['env'], x['config']), 0)
        cfgs[(x['algo'], x['env'], x['config'])] += 1
    rows.append(dict(session=i, date_first=s[0]['t0'].strftime('%Y-%m-%d'), date_last=s[-1]['t1'].strftime('%Y-%m-%d'), runs=len(s),
                     first_start=s[0]['t0'].strftime('%Y-%m-%d %H:%M:%S'), last_end=s[-1]['t1'].strftime('%Y-%m-%d %H:%M:%S'),
                     span_h=round(spans[i], 3), run_hours=round(sum((x['t1'] - x['t0']).total_seconds() for x in s) / 3600, 3),
                     gap_before_h=(round(gaps_between[i - 1], 3) if i else ''),
                     by_algo_env='; '.join(f'{a}/{e}:{n}' for (a, e), n in sorted(cnt.items())),
                     commits=','.join(sorted({x['commit'] for x in s})),
                     configs='; '.join(f'{a}/{e}/{c}:{n}' for (a, e, c), n in cfgs.items())))
with open(os.path.join(OUT, 'E_sessions.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
with open(os.path.join(OUT, 'E_sessions_runs.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['run_dir', 'algo', 'env', 'config', 'seed', 'start_utc', 'end_utc', 'commit', 'session'])
    for i, s in enumerate(S30):
        for x in s:
            w.writerow([x['run_dir'], x['algo'], x['env'], x['config'], x['seed'], x['t0'].isoformat(), x['t1'].isoformat(), x['commit'], i])
json.dump(summary, open(os.path.join(OUT, 'E_sessions_summary.json'), 'w'), indent=1)
print(json.dumps(summary, indent=1))
for r in rows:
    print(r['session'], r['first_start'], r['last_end'], r['runs'], r['span_h'], r['gap_before_h'], r['by_algo_env'], r['commits'])
