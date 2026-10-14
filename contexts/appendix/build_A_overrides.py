"""Appendix A: map every canonical configuration to its override file and verify
that dataclass default + override file == logged algo_config for all 280 runs."""
import json, os, sys, csv, collections, dataclasses
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, R)
from configs.config import ALGO_CONFIGS
runs = list(csv.DictReader(open(os.path.join(R,'contexts/step1/04_canonical_run_list.csv'))))
def slug(cfg):
    if cfg=='baseline': return None
    k,v = cfg.split('=')
    v = v.replace('[','').replace(']','').replace(' ','').split(',')[0]
    pre = {'updates_per_env_step':'utd','hidden_sizes':'width','batch_size':'batch','rollout_max_length':'rollout','num_q':'numq','horizon':'horizon'}[k]
    return f'{pre}{v}'
def fname(algo, env, cfg):
    s = slug(cfg)
    ant = env=='Ant-v5'
    if algo=='sac' or algo=='td3':
        return None if s is None else f'{algo}_{s}.json'
    if algo=='mbpo':
        base = 'mbpo_ant' if ant else 'mbpo'
        if s is None: return 'mbpo_ant.json' if ant else None
        return f'{base}_{s}.json'
    if algo=='tdmpc2':
        base = 'tdmpc2_ant' if ant else 'tdmpc2'
        if s is None: return 'tdmpc2_ant.json' if ant else None
        return f'{base}_{s}.json'
rows = collections.OrderedDict(); bad=[]; n=0
for r in runs:
    m = json.load(open(os.path.join(R,'results',r['run_dir'],'metadata.json')))
    fn = fname(r['algo'], r['env'], r['config'])
    ov = json.load(open(os.path.join(R,'configs/overrides',fn))) if fn else {}
    cfg = ALGO_CONFIGS[r['algo']](**{k:(tuple(v) if isinstance(v,list) else v) for k,v in ov.items()})
    exp = {k:(list(v) if isinstance(v,tuple) else v) for k,v in dataclasses.asdict(cfg).items()}
    got = m['algo_config']
    if exp!=got: bad.append((r['run_dir'], {k:(exp.get(k),got.get(k)) for k in set(exp)|set(got) if exp.get(k)!=got.get(k)}))
    n+=1
    rows.setdefault((fn, r['algo'], r['env'], r['config']), 0); rows[(fn, r['algo'], r['env'], r['config'])]+=1
print('runs checked', n, 'mismatches', len(bad)); print(bad[:5])
used = sorted({k[0] for k in rows if k[0]})
print(len(used), 'override files used'); 
allf = sorted(os.listdir(os.path.join(R,'configs/overrides')))
print('unused', [f for f in allf if f not in used])
with open(os.path.join(R,'contexts/appendix/appendix_data/A_override_map.csv'),'w',newline='') as f:
    w=csv.writer(f); w.writerow(['override_file','algo','env','config','runs'])
    for (fn,a,e,c),k in rows.items(): w.writerow([fn or '(none: defaults)',a,e,c,k])
