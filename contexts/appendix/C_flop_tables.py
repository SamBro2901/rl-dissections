"""Appendix C data: per-call table, worked examples, classifier FLOPs, region sums, cross-check summary.

Everything is read from flop_analysis/flops_per_call.json, the per-run CSV and the run files; the analytic
functions are those of flop_analysis/verify_analytic_flops.py (imported, torch-free).
Writes CSV/JSON files to contexts/appendix/appendix_data/ (prefix C_).
"""
import csv, json, math, os, sys, importlib.util, collections

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(R, 'contexts/appendix/appendix_data')
sys.path.insert(0, R)
spec = importlib.util.spec_from_file_location('vaf', os.path.join(R, 'flop_analysis/verify_analytic_flops.py'))
vaf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vaf)

data = json.load(open(os.path.join(R, 'flop_analysis/flops_per_call.json')))
CANON_ENVS = ('HalfCheetah-v5', 'Ant-v5')

# ---------------------------------------------------------------- 1. canonical signatures actually used by the 280 runs
runs = list(csv.DictReader(open(os.path.join(R, 'contexts/step1/04_canonical_run_list.csv'))))
sys.path.insert(0, os.path.join(R, 'flop_analysis'))
import flop_keys
used = collections.OrderedDict()
for r in runs:
    m = json.load(open(os.path.join(R, 'results', r['run_dir'], 'metadata.json')))
    used.setdefault((r['algo'], r['env'], flop_keys.signature(r['algo'], m['algo_config'])), 0)
    used[(r['algo'], r['env'], flop_keys.signature(r['algo'], m['algo_config']))] += 1
json_sigs = {(a, e, s) for a in ('sac', 'td3', 'mbpo', 'tdmpc2') for e in data[a] if e in CANON_ENVS for s in data[a][e]}
print('signatures used by canonical runs:', len(used), ' in json (canonical envs):', len(json_sigs),
      ' used-not-in-json:', set(used) - json_sigs, ' json-not-used:', sorted(json_sigs - set(used)))
per_algo_sig = collections.Counter(a for a, e, s in used)
print(per_algo_sig)

# ---------------------------------------------------------------- 2. cross-check summary on canonical signatures
rows = []
for algo in ('sac', 'td3', 'mbpo'):
    for env in CANON_ENVS:
        for sig, rec in data[algo][env].items():
            if (algo, env, sig) not in used:
                continue
            fn = vaf.td3_analytic if algo == 'td3' else vaf.sac_analytic
            for k, v in fn(rec['obs_dim'], rec['act_dim'], rec['hidden_sizes'], rec['batch_size']).items():
                rows.append((algo, env, sig, k, v, rec[k]))
            if algo == 'mbpo':
                d = vaf.mbpo_dynamics_analytic(rec['obs_dim'], rec['act_dim'], rec['ensemble_size'], rec['model_hidden_sizes'], rec['model_train_batch_size'])
                for k, v in d.items():
                    rows.append((algo, env, sig, k, v, rec[k]))
for env in CANON_ENVS:
    for sig, rec in data['tdmpc2'][env].items():
        if ('tdmpc2', env, sig) not in used:
            continue
        for k, v in vaf.tdmpc2_analytic(rec, sig).items():
            rows.append(('tdmpc2', env, sig, k, v, rec[k]))
summ = collections.OrderedDict()
for algo, env, sig, k, a, m in rows:
    s = summ.setdefault(algo, dict(sigs=set(), comps=0, maxdev=0.0, inexact=0))
    s['sigs'].add((env, sig))
    s['comps'] += 1
    dev = abs(a - m) / m if m else abs(a - m)
    s['maxdev'] = max(s['maxdev'], dev)
    s['inexact'] += a != m
out = [dict(algo=a, signatures=len(v['sigs']), comparisons=v['comps'], inexact=v['inexact'], max_rel_dev=v['maxdev']) for a, v in summ.items()]
with open(os.path.join(OUT, 'C_crosscheck_summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
print(out, 'total comparisons', sum(o['comparisons'] for o in out))

# ---------------------------------------------------------------- 3. per-call tables
with open(os.path.join(OUT, 'C_percall_learners.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['algo', 'env', 'signature', 'batch', 'hidden', 'actor_forward_bs1', 'critic_fwdbwd', 'actor_fwdbwd', 'target_ops'])
    for algo in ('sac', 'td3', 'mbpo'):
        for env in CANON_ENVS:
            for sig, rec in data[algo][env].items():
                if (algo, env, sig) in used:
                    w.writerow([algo, env, sig, rec['batch_size'], 'x'.join(map(str, rec['hidden_sizes'])), rec['actor_forward_bs1'],
                                rec['critic_fwdbwd'], rec['actor_fwdbwd'], rec['target_update_elementwise_ops']])
with open(os.path.join(OUT, 'C_percall_mbpo_model.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['env', 'signature', 'member_fwdbwd_mb', 'per_sample_member_fwdbwd', 'ensemble_forward_bs1', 'actor_forward_bs1', 'per_synthetic_sample'])
    for env in CANON_ENVS:
        for sig, rec in data['mbpo'][env].items():
            if ('mbpo', env, sig) in used:
                mb = rec['model_train_batch_size']
                w.writerow([env, sig, rec['dynamics_member_fwdbwd'], rec['dynamics_member_fwdbwd'] // mb, rec['dynamics_ensemble_forward_all_bs1'],
                            rec['actor_forward_bs1'], rec['actor_forward_bs1'] + rec['dynamics_ensemble_forward_all_bs1']])


def classifier_flops(rec, sig):
    """FLOPs of the termination classifier per planning call and per world-model step (analytic)."""
    import re
    m = re.search(r'enc(\d+)x(\d+)_mlp(\d+)_lat(\d+)', sig)
    n_enc, enc_dim, _, lat = (int(x) for x in m.groups())
    mlp = rec['mlp_dim']
    term = vaf.dims(lat, mlp, mlp, 1)
    H, B, S, iters = rec['horizon'], rec['batch_size'], rec['num_samples'], rec['iterations']
    plan = iters * H * vaf.fwd(S, term)
    wm = vaf.fwd(H * B, term) + vaf.bwd(H * B, term, first_input_grad=True)
    return plan, wm


with open(os.path.join(OUT, 'C_percall_tdmpc2.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['env', 'signature', 'horizon', 'num_q', 'episodic', 'plan', 'wm_step', 'prior_step', 'update_total', 'polyak_ops',
                'clf_plan', 'clf_wm', 'blackbox_total', 'breakdown_total'])
    for env in CANON_ENVS:
        for sig, rec in data['tdmpc2'][env].items():
            cp, cw = classifier_flops(rec, sig) if rec['episodic'] else (0, 0)
            cc = rec['_cross_check_blackbox_vs_breakdown']
            w.writerow([env, sig, rec['horizon'], rec['num_q'], rec['episodic'], rec['rollout_per_env_step'], rec['gradient_update_critic'],
                        rec['gradient_update_actor'], rec['gradient_update_total'], rec['gradient_update_target_elementwise_ops'], cp, cw,
                        cc['blackbox_total'], cc['breakdown_total']])
            assert rec['gradient_update_total'] == rec['gradient_update_critic'] + rec['gradient_update_actor'] + rec['gradient_update_target']
print('tdmpc2 region sums equal blackbox:', all(
    data['tdmpc2'][e][s]['_cross_check_blackbox_vs_breakdown']['blackbox_total'] == data['tdmpc2'][e][s]['_cross_check_blackbox_vs_breakdown']['breakdown_total']
    for e in CANON_ENVS for s in data['tdmpc2'][e]))

# ---------------------------------------------------------------- 4. classifier totals for the Ant-v5 baseline (per run totals)
rec = data['tdmpc2']['Ant-v5']['bs256_h3_ns512_it6_pit24_ne64_nq5_enc2x256_mlp512_lat512_epTrue']
cp, cw = classifier_flops(rec, 'bs256_h3_ns512_it6_pit24_ne64_nq5_enc2x256_mlp512_lat512_epTrue')
n_env, n_upd, warm = 100000, 100000, 5000
clf_total = cp * n_env + cw * (n_upd + warm)
print('classifier plan', cp, 'wm', cw, 'total', clf_total)

# ---------------------------------------------------------------- 5. worked examples: SAC critic/actor terms (Ant-v5 baseline)
obs, act, B = 105, 8, 256
h1 = h2 = 1024
trunk = vaf.dims(obs, h1, h2)
heads = vaf.dims(h2, act)
q = vaf.dims(obs + act, h1, h2, 1)
af = lambda rows: vaf.fwd(rows, trunk) + 2 * vaf.fwd(rows, heads)
ex = dict(
    actor_fwd_B=af(B), q_fwd_B=vaf.fwd(B, q), q_bwd_wo_x1=vaf.bwd(B, q, first_input_grad=False),
    critic=af(B) + 4 * vaf.fwd(B, q) + 2 * vaf.bwd(B, q, first_input_grad=False),
    q_back_x=vaf.bwd(B, q, first_input_grad=True, weights=False),
    pol_back=2 * vaf.bwd(B, heads, first_input_grad=True) + vaf.bwd(B, trunk, first_input_grad=False),
)
ex['actor'] = af(B) + 2 * vaf.fwd(B, q) + 2 * ex['q_back_x'] + ex['pol_back']
ex['measured_critic'] = data['sac']['Ant-v5']['bs256_h1024x1024']['critic_fwdbwd']
ex['measured_actor'] = data['sac']['Ant-v5']['bs256_h1024x1024']['actor_fwdbwd']
ex['n_params_q'] = vaf.n_params(q)
ex['n_params_actor'] = vaf.n_params(trunk) + 2 * vaf.n_params(heads)
ex['polyak'] = 2 * 2 * vaf.n_params(q)
ex['textbook_3x'] = 3 * (vaf.fwd(B, q) * 2)  # three times the forward cost of both critics
ex['textbook_critic_approx_json'] = data['sac']['Ant-v5']['bs256_h1024x1024']['_analytic_cross_check']['critic_fwdbwd_analytic_approx']
json.dump(ex, open(os.path.join(OUT, 'C_sac_worked_example.json'), 'w'), indent=1)
print(ex)

# ---------------------------------------------------------------- 6. MBPO fit worked example (baseline run seed 331 per env)
perrun = collections.defaultdict(list)
with open(os.path.join(R, 'flop_analysis/output/per_run_energy_per_flop.csv'), newline='') as f:
    for row in csv.DictReader(f):
        perrun[row['run_dir']].append(row)
cap = 1_000_000
examples = []
for env in CANON_ENVS:
    r = next(x for x in runs if x['algo'] == 'mbpo' and x['env'] == env and x['config'] == 'baseline' and x['seed'] == '331')
    d = os.path.join(R, 'results', r['run_dir'])
    meta = json.load(open(os.path.join(d, 'metadata.json')))
    tm = json.load(open(os.path.join(d, 'training_metrics.json')))
    ac, ec = meta['algo_config'], meta['experiment_config']
    sig = flop_keys.signature('mbpo', ac)
    fk = data['mbpo'][env][sig]
    per_member_sample = fk['dynamics_member_fwdbwd'] // ac['model_train_batch_size']
    fits = [(e['epoch'], e['model_train_epochs']) for e in tm['epochs'] if e['model_train_epochs']]
    tr = ho = steps = 0
    for ep, er in fits:
        n_total = min(ec['warmup_steps'] + ep * ec['steps_per_epoch'], cap)
        n_h = max(1, int(n_total * ac['model_holdout_ratio']))
        n_t = n_total - n_h
        tr += er * ac['ensemble_size'] * n_t * per_member_sample
        ho += er * n_h * fk['dynamics_ensemble_forward_all_bs1']
        steps += er * math.ceil(n_t / ac['model_train_batch_size'])
    n_syn = sum(e['synthetic_transitions_generated'] for e in tm['epochs'])
    syn = n_syn * (fk['actor_forward_bs1'] + fk['dynamics_ensemble_forward_all_bs1'])
    rows_r = {x['segment']: x for x in perrun['results/' + r['run_dir']]}
    logged_dyn = int(rows_r['dynamics_model_update']['total_flops'])
    logged_steps = int(rows_r['dynamics_model_update']['call_count'])
    logged_syn = int(rows_r['synthetic_rollout_generation']['total_flops'])
    examples.append(dict(env=env, run=r['run_dir'], fit_epochs_total=sum(e for _, e in fits), n_fits=len(fits), optimiser_steps=steps,
                         train_flops=tr, holdout_flops=ho, formula_total=tr + ho, logged_total=logged_dyn, logged_steps=logged_steps,
                         holdout_share=100 * ho / (tr + ho), synthetic_samples=n_syn, synth_formula=syn, synth_logged=logged_syn,
                         exact=(tr + ho == logged_dyn and steps == logged_steps and syn == logged_syn),
                         per_member_sample=per_member_sample, ens_fwd=fk['dynamics_ensemble_forward_all_bs1']))
for e in examples:
    print(e)
json.dump(examples, open(os.path.join(OUT, 'C_mbpo_fit_examples.json'), 'w'), indent=1)

# holdout share over all 80 MBPO runs (min/max)
shares = []
for r in runs:
    if r['algo'] != 'mbpo':
        continue
    rr = {x['segment']: x for x in perrun['results/' + r['run_dir']]}
    t = float(rr['dynamics_model_update']['dynamics_train_flops'])
    h = float(rr['dynamics_model_update']['dynamics_holdout_flops'])
    shares.append((r['env'], 100 * h / (t + h)))
for env in CANON_ENVS:
    v = [s for e, s in shares if e == env]
    print(env, len(v), min(v), max(v), sum(v) / len(v))
