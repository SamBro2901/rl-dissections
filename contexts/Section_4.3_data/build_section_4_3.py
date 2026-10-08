"""
Generator for contexts/Section_4.3.md (MBPO) + companion CSVs in contexts/Section_4.3_data/.
Read-only on results/, flop_analysis/ and git; writes only under contexts/.
Run from the repo root:  .venv/Scripts/python.exe contexts/Section_4.3_data/build_section_4_3.py
Shared modules live in contexts/Section_4.2_data/ (s4_data.py, s4_blocks.py, s4_audit.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Section_4.2_data"))
from s4_audit import *  # noqa: F401,F403,E402
from s4_audit import Src, versus_rows, srt  # noqa: E402
from configs.config import MBPOConfig, SACConfig  # noqa: E402

OUT_MD = REPO / "contexts" / "Section_4.3.md"
OUT_DIR = REPO / "contexts" / "Section_4.3_data"
OUT_DIR.mkdir(exist_ok=True)
PREFIX = "mbpo"

TAGS = ["base", "utd2", "utd4", "w256", "w512", "b512", "b1024", "rollout1", "rollout15"]
SPEC = Spec(
    algo="mbpo", section="4.3", name="MBPO", config_cls=MBPOConfig, tags=TAGS,
    tag_desc={"base": "SAC hidden (1024,1024), batch 256, UTD 1, real_ratio 0.05, model (200×4)×7; HC: fixed rollout length 1; Ant: rollout length 1→25 (mbpo_ant.json)",
              "utd2": "UTD 2", "utd4": "UTD 4", "w256": "SAC hidden (256,256)", "w512": "SAC hidden (512,512)",
              "b512": "SAC batch 512", "b1024": "SAC batch 1024",
              "rollout1": "Ant only: rollout_max_length 1", "rollout15": "Ant only: rollout_max_length 15"},
    tag_factor={"base": {}, "utd2": {"updates_per_env_step": 2}, "utd4": {"updates_per_env_step": 4},
                "w256": {"hidden_sizes": [256, 256]}, "w512": {"hidden_sizes": [512, 512]},
                "b512": {"batch_size": 512}, "b1024": {"batch_size": 1024},
                "rollout1": {"rollout_max_length": 1}, "rollout15": {"rollout_max_length": 15}},
    env_ref_override={"HC": None, "Ant": "configs/overrides/mbpo_ant.json"},
    extra_measured=["dynamics_model_update", "synthetic_rollout_generation"],
    measured_order=["dynamics_model_update", "rollout", "synthetic_rollout_generation", "gradient_updates"],
    sweeps=[("UTD", "updates_per_env_step ∈ {1,2,4}", ["base", "utd2", "utd4"], "updates_per_env_step"),
            ("width", "SAC hidden_sizes ∈ {(256,256),(512,512),(1024,1024)}", ["w256", "w512", "base"], "hidden_sizes"),
            ("batch size", "SAC batch_size ∈ {256,512,1024}", ["base", "b512", "b1024"], "batch_size"),
            ("maximum rollout length (Ant-v5 only)", "rollout_max_length ∈ {1,15,25}", ["rollout1", "rollout15", "base"], "mbpo_rollout_regime")],
    tag_envs={"rollout1": ["Ant"], "rollout15": ["Ant"]},
    expected_runs={"HC": 35, "Ant": 45},
    per_epoch_tasks=["dynamics_model_update", "rollout", "synthetic_rollout_generation", "gradient_updates"])

SUFFIX = {"base": "", "utd2": "_utd2", "utd4": "_utd4", "w256": "_width256", "w512": "_width512", "b512": "_batch512", "b1024": "_batch1024",
          "rollout1": "_rollout1", "rollout15": "_rollout15"}


def override_file(es, t):
    if es == "HC":
        return "none (MBPOConfig defaults)" if t == "base" else f"configs/overrides/mbpo{SUFFIX[t]}.json"
    return f"configs/overrides/mbpo_ant{SUFFIX[t]}.json"


SPEC.tag_override_file = override_file
LOG_TO_SCRIPT = {"results/mbpo/_env_sweep.log": "scripts/run_mbpo_env_sweep.sh",
                 "results/_utd_sweep.log": "scripts/run_utd_sweep.sh (current MBPO UTD sweep)",
                 "results/_width_sweep.log": "scripts/run_width_sweep.sh",
                 "results/_batch_size_sweep.log": "scripts/run_batch_size_sweep.sh",
                 "results/_ant_overnight_sweep.log": "scripts/run_ant_overnight_sweep.sh (= run_batch_size_sweep_ant.sh + run_mbpo_rollout_length_sweep.sh)"}
DS = Dataset(SPEC)


def launcher(es, t):
    logs = sorted({lf for r in DS.runs_of(es, t) for lf, _ in SWEEP_BLOCKS.get(r["run_dir"], [])})
    return "; ".join(f"{LOG_TO_SCRIPT.get(l, l)} (log `{l}`)" for l in logs) or "NO LOG BLOCK FOUND"


SPEC.tag_script = launcher
SAC = Dataset(sac_spec())
pe = per_epoch_frame(DS)
IDLE = export_csvs(DS, OUT_DIR, pe, PREFIX)
mb = Src("algorithms/mbpo.py")
dm = Src("algorithms/dynamics_model.py")
tf = Src("algorithms/termination_fns.py")
cepf = Src("flop_analysis/compute_energy_per_flop.py")
from algorithms.mbpo import _rollout_length  # noqa: E402

CFGD = {es: json.loads(json.dumps(dataclasses.asdict(MBPOConfig()))) for es, _ in ENVS}
with open("configs/overrides/mbpo_ant.json", encoding="utf-8") as _f:
    CFGD["Ant"].update(json.load(_f))


def regime_cfg(es, tag):
    d = dict(DS.ref_ac[es])
    d.update(SPEC.tag_factor[tag])
    return MBPOConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()})


REGIMES = [("HC fixed 1 (HC base: min_epoch 20, max_epoch 150, lengths 1→1)", "HC", "base"), ("Ant 1→25 (Ant base, mbpo_ant.json)", "Ant", "base"),
           ("Ant max 15 (rollout15)", "Ant", "rollout15"), ("Ant max 1 (rollout1)", "Ant", "rollout1")]


# =============================================================================== Section 2
def section2():
    import torch
    import gymnasium as gym
    from algorithms.sac import GaussianPolicy, QNetwork
    from algorithms.dynamics_model import GaussianEnsembleMLP
    out = []
    out.append("**2.1 Configuration objects** (dumped from `configs/config.py` and `configs/overrides/` at HEAD):\n")
    out.append("`MBPOConfig` defaults (`dataclasses.asdict(MBPOConfig())`; the HalfCheetah baseline uses exactly these):\n")
    out.append(code(json.dumps(dataclasses.asdict(MBPOConfig()), indent=2, default=list), "json"))
    out.append("`configs/overrides/mbpo_ant.json` (the Ant baseline = defaults + this file):\n")
    out.append(code(Path("configs/overrides/mbpo_ant.json").read_text(encoding="utf-8"), "json"))
    out.append("Override files of the sweeps (content): " + "; ".join(
        f"`{Path(p).name}` = {json.dumps(json.loads(Path(p).read_text(encoding='utf-8')))}" for p in sorted(glob.glob("configs/overrides/mbpo_*.json")) if "hopper" not in p and "walker" not in p and "humanoid" not in p))
    bad = []
    for es, _ in ENVS:
        for t in TAGS:
            if not DS.has(es, t):
                continue
            f = override_file(es, t)
            if f.startswith("none"):
                continue
            ov = json.loads(Path(f).read_text(encoding="utf-8"))
            ac = DS.runs_of(es, t)[0]["meta"]["algo_config"]
            for k, v in ov.items():
                if ac.get(k) != v and not (isinstance(v, (list, tuple)) and list(v) == list(ac.get(k))):
                    bad.append((label(es, t), k, v, ac.get(k)))
    out.append("- " + check("every override file listed in the inventory equals the corresponding fields of the logged `algo_config` of its runs", not bad, str(bad) if bad else ""))
    out.append("`ExperimentConfig` defaults (protocol; MBPO runs use them unchanged, `warmup_steps` = 5000):\n")
    out.append(code(json.dumps(dataclasses.asdict(ExperimentConfig(algo_name="mbpo", env_id="HalfCheetah-v5")), indent=2), "json"))
    for es in ("HC", "Ant"):
        r = srt(DS.runs_of(es, "base"))[0]
        out.append(f"\nComplete `metadata.json` of `{r['run_dir']}` ({label(es, 'base')}, seed {r['seed']}):\n")
        out.append(code(json.dumps(r["meta"], indent=2), "json"))
    out.append("\n**2.2 Verbatim code** (`algorithms/*.py` at HEAD; each file is identical at every run commit, Section 0).\n\n`algorithms/mbpo.py` (complete):\n")
    out.append(code(mb.excerpt(1, len(mb.lines)), "python"))
    out.append("`algorithms/dynamics_model.py` (complete):\n")
    out.append(code(dm.excerpt(1, len(dm.lines)), "python"))
    out.append("`algorithms/termination_fns.py` (complete):\n")
    out.append(code(tf.excerpt(1, len(tf.lines)), "python"))
    rb = Src("algorithms/replay_buffer.py")
    out.append("`algorithms/replay_buffer.py` (complete; used for the real buffer and the model buffer):\n")
    out.append(code(rb.excerpt(1, len(rb.lines)), "python"))
    sac = Src("algorithms/sac.py")
    a, b, blk = sac.block(r"^class SACAgent", r"^def train\(")
    out.append(f"`algorithms/sac.py` `SACAgent` (lines {a}–{b}; the unmodified policy optimiser MBPO reuses; `train()` of sac.py is not used by MBPO):\n")
    out.append(blk)
    # params
    envd = {}
    for es, env_id in ENVS:
        e = gym.make(env_id)
        envd[es] = (e.observation_space.shape[0], e.action_space.shape[0])
        e.close()
    rows = []
    for es, env_id in ENVS:
        o, a_ = envd[es]
        for w in (256, 512, 1024):
            pol = GaussianPolicy(o, a_, (w, w), 1.0)
            q = QNetwork(o, a_, (w, w))
            npol = sum(x.numel() for x in pol.parameters())
            nq = sum(x.numel() for x in q.parameters())
            rows.append([env_id, f"({w},{w})", f"{o}/{a_}", npol, nq, 2 * nq, 2 * nq, npol + 4 * nq])
    out.append("\n**2.3 Policy / critic network sizes** (real `GaussianPolicy` / `QNetwork` of `sac.py`, as used through `SACAgent`):\n")
    out.append(table(["env", "hidden", "obs/act dim", "actor (incl. μ and log σ heads)", "one Q-network", "both critics", "both target critics", "all 5 networks"], rows))
    rows = []
    for es, env_id in ENVS:
        o, a_ = envd[es]
        net = GaussianEnsembleMLP(o, a_, 7, (200, 200, 200, 200))
        per_member = sum(x.numel() for x in net.trunks[0].parameters()) + sum(x.numel() for x in net.mean_heads[0].parameters()) + sum(x.numel() for x in net.logvar_heads[0].parameters())
        bounds = net.max_logvar.numel() + net.min_logvar.numel()
        tot = sum(x.numel() for x in net.parameters())
        trunk = sum(x.numel() for x in net.trunks[0].parameters())
        rows.append([env_id, f"{o}+{a_}={o + a_} → {o + 1}", trunk, sum(x.numel() for x in net.mean_heads[0].parameters()), sum(x.numel() for x in net.logvar_heads[0].parameters()),
                     per_member, 7, bounds, tot, 5, 5 * per_member + 0])
    out.append("**2.4 Dynamics ensemble sizes** (`GaussianEnsembleMLP(obs_dim, act_dim, ensemble_size=7, hidden_sizes=(200,200,200,200))` instantiated; "
               "input = obs+act, output = obs+1 (Δobs and reward) per head; per member: trunk + mean head + log-variance head; plus the shared learned "
               "`max_logvar`/`min_logvar` vectors (2 × (obs+1)); the running normaliser has buffers, no parameters; elites = 5 of 7 members, chosen by holdout MSE):\n")
    out.append(table(["env", "input → output dim", "trunk (one member)", "mean head", "logvar head", "parameters per member", "members", "shared logvar bounds",
                      "total parameters (all 7 members + bounds)", "elites", "parameters of the 5 elites (members only)"], rows))
    ok = all(rw[8] == 7 * rw[5] + rw[7] for rw in rows)
    out.append("- " + check("total = 7 × per-member + shared bounds", ok))
    # FLOP constants
    keys = [("actor_forward_bs1 (rollout call)", lambda fk: fk["actor_forward_bs1"]),
            ("critic_fwdbwd (per update() call)", lambda fk: fk["critic_fwdbwd"]),
            ("actor_fwdbwd (per update() call)", lambda fk: fk["actor_fwdbwd"]),
            ("target_update_elementwise_ops (per update() call)", lambda fk: fk["target_update_elementwise_ops"]),
            ("dynamics_member_fwdbwd (one member, batch 256)", lambda fk: fk["dynamics_member_fwdbwd"]),
            ("dynamics_ensemble_forward_all_bs1 (7 members, 1 sample)", lambda fk: fk["dynamics_ensemble_forward_all_bs1"]),
            ("SAC batch / model batch", lambda fk: f"{fk['batch_size']}/{fk['model_train_batch_size']}"),
            ("per synthetic sample = actor_forward_bs1 + ensemble_forward_all_bs1", lambda fk: fk["actor_forward_bs1"] + fk["dynamics_ensemble_forward_all_bs1"])]
    out.append("\n**2.5 Per-call FLOPs from `flop_analysis/flops_per_call.json`** (`measure_flops.measure_sac` for the SAC part, `measure_mbpo_dynamics` for the model; "
               f"measured on `{FLOPS_JSON['_device_used_for_measurement']}`; matmul FLOPs = 2·m·n·k). Cell `HC ‖ Ant` (rollout1/rollout15 share the Ant baseline's signature, "
               "rollout length is not part of the architecture signature):\n")
    out.append(flop_constant_table(DS, keys))
    # analytic dynamics check
    rows = []
    allok = True
    for es, env_id in ENVS:
        o, a_ = envd[es]
        sig = sorted({r["sig"] for r in DS.runs if r["es"] == es})[0]
        fk = FLOPS_JSON["mbpo"][env_id][sig]
        B = fk["model_train_batch_size"]
        hs = [o + a_, 200, 200, 200, 200]
        outd = o + 1
        fwd = lambda b: sum(2 * b * i * j for i, j in zip(hs[:-1], hs[1:])) + 2 * 2 * b * 200 * outd  # noqa: E731
        first = 2 * B * (o + a_) * 200
        member = fwd(B) + (2 * fwd(B) - first)       # forward + weight grads (all) + input grads (all but first)
        ens1 = 7 * fwd(1)
        ex = (member, ens1) == (fk["dynamics_member_fwdbwd"], fk["dynamics_ensemble_forward_all_bs1"])
        allok &= ex
        rows.append([env_id, fk["dynamics_member_fwdbwd"], member, fk["_analytic_cross_check"]["dynamics_member_fwdbwd_analytic_approx"],
                     fk["dynamics_ensemble_forward_all_bs1"], ens1, "exact" if ex else "MISMATCH"])
    out.append("\n**2.6 Analytic cross-check of the dynamics constants** (derived by this generator; forward = 2·B·n_in·n_out per layer over the trunk and the two heads; "
               "member fwd+bwd = forward + weight-grad (all layers) + input-grad (all layers except the first, whose input is data); the stored `_analytic_cross_check` "
               "uses 3 × forward and is an approximation):\n")
    out.append(table(["env", "dynamics_member_fwdbwd measured", "derived", "stored analytic approx (3×fwd)", "ensemble_forward_all_bs1 measured", "derived (7 × member fwd at B=1)", "derived vs measured"], rows))
    out.append("- " + check("derived dense-layer counts equal FlopCounterMode exactly for the dynamics member and the ensemble forward", allok))
    # SAC part analytic
    rows = []
    allok2 = True
    for es, env_id in ENVS:
        o, a_ = envd[es]
        for sig in sorted({r["sig"] for r in DS.runs if r["es"] == es}):
            fk = FLOPS_JSON["mbpo"][env_id][sig]
            h1, h2 = fk["hidden_sizes"]
            B = fk["batch_size"]
            fa = lambda b: 2 * b * (o * h1 + h1 * h2 + 2 * h2 * a_)  # noqa: E731
            fq = lambda b: 2 * b * ((o + a_) * h1 + h1 * h2 + h2)  # noqa: E731
            first_q = 2 * B * (o + a_) * h1
            first_a = 2 * B * o * h1
            crit = fa(B) + 2 * fq(B) + 2 * (fq(B) + (2 * fq(B) - first_q))
            act = fa(B) + (2 * fa(B) - first_a) + 2 * fq(B) + 2 * fq(B)
            ex = (fa(1), crit, act) == (fk["actor_forward_bs1"], fk["critic_fwdbwd"], fk["actor_fwdbwd"])
            allok2 &= ex
            rows.append([env_id, f"`{sig}`", "exact" if ex else "MISMATCH"])
    out.append("- " + check("the SAC-part constants (rollout, critic, actor) of every MBPO signature equal the dense-layer derivation of Section 4.1 exactly", allok2))
    # call counts
    out.append(f"""
**2.7 Call counts and where they are defined** (`compute_energy_per_flop.rows_for_mbpo`, which calls `rows_for_sac_or_td3("sac", …)` for the SAC part, and `mbpo_fit_flops`):

```
n_env    = 100 * 1000 = 100000                      # rollout calls (actor forward at batch 1)
n_upd    = n_env * updates_per_env_step             # buffer_sample = critic_update = actor_update = target_update calls (policy_update_delay = 1)
dynamics_model_update: call_count = Σ over fits of epochs_run × ceil(n_train / 256)   # optimiser steps; epochs_run = logged model_train_epochs
synthetic_rollout_generation: call_count = Σ_epochs synthetic_transitions_generated    # samples; FLOPs = samples × (actor_forward_bs1 + dynamics_ensemble_forward_all_bs1)
```
`mbpo_fit_flops` (verbatim from `flop_analysis/compute_energy_per_flop.py`):
""")
    a, b, blk = cepf.block(r"^def mbpo_fit_flops", r"^def rows_for_mbpo")
    out.append(blk)
    a, b, blk = cepf.block(r"^def rows_for_mbpo", r"^def rows_for_tdmpc2")
    out.append(f"`rows_for_mbpo` (lines {a}–{b}):\n")
    out.append(blk)
    out.append("The generator recomputes both from the logged `training_metrics.json` epoch rows (`s4_data.mbpo_extras`): per epoch with a non-null `model_train_epochs`, "
               "n_total = min(warmup + epoch × 1000, capacity), n_holdout = max(1, int(0.2 · n_total)), n_train = n_total − n_holdout, train FLOPs = epochs_run × 7 × n_train × (dynamics_member_fwdbwd/256), "
               "holdout FLOPs = epochs_run × n_holdout × dynamics_ensemble_forward_all_bs1, steps = epochs_run × ceil(n_train/256). Call counts per configuration (mean over seeds; `(min, max)` where seeds differ):\n")
    out.append(call_count_table(DS))
    t1, t2 = total_flops_table(DS)
    out.append("\n**2.8 Total FLOPs per run and segment** (`dyn_model` and `synth_rollout` vary between seeds: early-stopped fits, Ant terminations):\n")
    out.append(t1)
    out.append("Ant / HalfCheetah ratios of the (seed-mean) quantities:\n")
    out.append(t2)
    return "\n".join(out)


# =============================================================================== Section 6
def section6():
    out = []
    spe = 1000
    # ---- 6.1 code facts
    L = {k: mb.find(p) for k, p in {"mixed": r"^class _MixedReplayBuffer", "sched": r"^def _rollout_length", "gen": r"^def _generate_model_rollouts", "train": r"^def train\(",
                                    "agent": r"agent = SACAgent\(", "model": r"model = EnsembleDynamicsModel", "term": r"termination_fn = get_termination_fn",
                                    "mbuf": r"model_buffer_capacity = max", "mixedbuf": r"mixed_buffer = _MixedReplayBuffer", "last_fit": r"last_model_train_step = -",
                                    "fit_cond": r"if \(global_step - last_model_train_step\)", "fit_task": r'TrackerTask\(tracker, "dynamics_model_update"', "roll_task": r'TrackerTask\(tracker, "rollout"',
                                    "synth_task": r'TrackerTask\(tracker, "synthetic_rollout_generation"', "gu_task": r'TrackerTask\(tracker, "gradient_updates"', "recon": r"# -+ reconcile",
                                    "nstart": r"n_start = min", "predict": r"next_obs_t, rew_t = model.predict", "done": r"done = termination_fn", "addb": r"model_buffer.add_batch", "keep": r"keep = ~done",
                                    "nreal": r"n_real = min", "emptyfb": r"if len\(self.model_buffer\) == 0", "sub": r"epoch_sub_times\[\"buffer_sample\"\] \+= info"}.items()}
    D = {k: dm.find(p) for k, p in {"fit": r"    def fit\(", "holdout": r"def _holdout_mse", "predict": r"    def predict\(", "boot": r"bootstrap_idx = np.random.randint", "stop": r"epochs_since_improved >= self.cfg.model_train_patience",
                                    "improve": r"mean_holdout_mse < best_holdout_mse", "elite": r"self.elite_indices = list\(", "nll": r"nll = ", "split": r"n_holdout = max"}.items()}
    T_ = {k: tf.find(p) for k, p in {"ant": r"^def _ant_done", "never": r"^def _never_done", "reg": r"^_REGISTRY", "get": r"^def get_termination_fn"}.items()}
    # sac cfg fields used by SACAgent
    sacsrc = Src("algorithms/sac.py")
    a = sacsrc.find(r"^class SACAgent")
    b = sacsrc.find(r"^def train\(")
    used = sorted(set(re.findall(r"cfg\.(\w+)", "\n".join(sacsrc.lines[a - 1:b]))))
    fields = [f.name for f in dataclasses.fields(MBPOConfig)]
    unused = [f for f in fields if f not in used]
    mbsrc = "\n".join(mb.lines)
    subclass = bool(re.search(r"class\s+\w+\(SACAgent\)", mbsrc))
    attr_assign = re.findall(r"^\s*agent\.\w+\s*=[^=]", mbsrc, re.M) + re.findall(r"setattr\(", mbsrc)
    agent_uses = sorted(set(re.findall(r"\bagent\.(\w+)", mbsrc)))
    out.append("**6.1 Code facts** (`algorithms/mbpo.py`, `algorithms/dynamics_model.py`, `algorithms/termination_fns.py`, `configs/config.py`, `configs/overrides/mbpo_ant*.json`; HEAD, identical at all run commits):\n")
    out.append(f"""
**(a) Order of the blocks within an epoch and their task names** (`train()`, lines {L['train']}–{len(mb.lines)}; `TrackerTask(tracker, prefix, epoch, …)` names the task `f"{{prefix}}_{{epoch}}"`, `algorithms/tracker_utils.py`):
1. `dynamics_model_update_{{epoch}}` — only if `(global_step − last_model_train_step) ≥ model_train_freq` (line {L['fit_cond']}; `last_model_train_step` starts at −model_train_freq, line {L['last_fit']}, so a fit happens in epoch 0): `model.fit(real_buffer)` (task at line {L['fit_task']});
2. `rollout_{{epoch}}` (line {L['roll_task']}): 1000 env steps with `agent.select_action(obs, deterministic=False)` (stochastic SAC policy), `env.step`, `real_buffer.add`, episode bookkeeping/reset — identical in structure to SAC;
3. `synthetic_rollout_generation_{{epoch}}` (line {L['synth_task']}): `_generate_model_rollouts(...)` with `rollout_length = _rollout_length(epoch, cfg)`;
4. `gradient_updates_{{epoch}}` (line {L['gu_task']}): `steps_per_epoch × updates_per_env_step` calls of `SACAgent.update(mixed_buffer, batch_size)` with the four `perf_counter` sub-timers (`buffer_sample`, `critic_update`, `actor_update`, `target_update`), allocated from the measured GU energy at the end of the run (line {L['recon']}).
`warmup_0` (random actions into the real buffer) precedes epoch 0 and is excluded from every total. The per-task CSV of every run has exactly 100 tasks of each of the four per-epoch prefixes (Section 1.3 check), i.e. a fit in every epoch.

**(b) Model fit** (`EnsembleDynamicsModel.fit`, `dynamics_model.py` lines {D['fit']}–{D['holdout'] - 2}):
- `model_train_freq` = {CFGD['HC']['model_train_freq']} real env steps; `steps_per_epoch` = 1000 ≥ `model_train_freq`, so the condition holds at the start of every epoch ⇒ **100 fits per run** (1 per epoch; verified from the logs: Section 6.2). The fit retrains the ensemble **from the current weights** on all real data in the buffer (`n = buffer.size`; the weights are not re-initialised; the Adam optimiser state persists), inputs normalised by a freshly fitted `RunningNormalizer`.
- Fit batch size `model_train_batch_size` = {CFGD['HC']['model_train_batch_size']}; holdout ratio `model_holdout_ratio` = {CFGD['HC']['model_holdout_ratio']} → `n_holdout = max(1, int(n · 0.2))` (line {D['split']}), `n_train = n − n_holdout`.
- Bootstrap: `bootstrap_idx = np.random.randint(0, n_train, size=(ensemble_size, n_train))` (line {D['boot']}) — each member gets its own resample **with replacement** of the training split, fixed for the fit call and reshuffled every epoch; every member sees `n_train` samples per epoch.
- Per optimiser step all 7 members are evaluated on their own minibatch (`for i in range(ensemble_size)`), one summed Gaussian-NLL loss + 0.01·(Σmax_logvar − Σmin_logvar), one backward, one Adam step (`model_lr` = {CFGD['HC']['model_lr']}, per-layer weight decay {CFGD['HC']['model_weight_decays']}). One *epoch* = ceil(n_train / 256) optimiser steps.
- Early stopping (lines {D['improve']}–{D['stop']}): after each epoch the per-member holdout MSE is computed (`_holdout_mse`, one `forward_all` over the holdout set); training stops when the **mean** holdout MSE has not improved by more than 1e-4 for `model_train_patience` = {CFGD['HC']['model_train_patience']} consecutive epochs, or after `model_max_train_epochs` = {CFGD['HC']['model_max_train_epochs']} epochs. After the loop the `num_elites` = {CFGD['HC']['num_elites']} members with the lowest per-member holdout MSE **of the best epoch** become the elites (line {D['elite']}). The weights are *not* restored to the best epoch.
- `ensemble_size` = {CFGD['HC']['ensemble_size']}, `num_elites` = {CFGD['HC']['num_elites']}, `model_hidden_sizes` = {CFGD['HC']['model_hidden_sizes']} (Swish activations), `deterministic_model` = {CFGD['HC']['deterministic_model']}. Parameters: Section 2.4.

**(c) Synthetic rollouts** (`_generate_model_rollouts`, lines {L['gen']}–{L['train'] - 2}):
- `rollout_batch_size` = {CFGD['HC']['rollout_batch_size']}: `n_start = min(rollout_batch_size, len(real_buffer))` (line {L['nstart']}) start states drawn uniformly (with replacement) from the real buffer; since the real buffer holds warmup + (epoch+1)·1000 transitions at that point, n_start = 6000, 7000, 8000, 9000 for epochs 0–3 and 10000 from epoch 4 on.
- Per step (loop `for _ in range(rollout_length)`): actor forward (stochastic, `with_logprob=False`) on all live states; `model.predict` (line {L['predict']}): normalise, **all 7 members' forward** (`forward_all`), pick one random elite member per row, sample next-Δobs and reward from its Gaussian (`deterministic_model` = False); `done = termination_fn(next_obs)` (line {L['done']}); `model_buffer.add_batch` of every generated transition (line {L['addb']}); `keep = ~done`, the loop stops if no state is kept, otherwise continues with `next_obs[keep]` (lines {L['keep']}). Terminated rows are dropped from further steps; the terminating transition itself **is** stored (with `done` = True).
- Termination function (`get_termination_fn`, line {T_['get']}; `termination_fns.py`): HalfCheetah-v5 → `_never_done` (line {T_['never']}; HC never terminates), Ant-v5 → `_ant_done` (line {T_['ant']}): `healthy = isfinite(next_obs).all & (z ≥ 0.2) & (z ≤ 1.0)` with z = `next_obs[:, 0]`, `done = ~healthy`.
- Rollout-length schedule (`_rollout_length`, line {L['sched']}): `epoch ≤ rollout_min_epoch → rollout_min_length`; `epoch ≥ rollout_max_epoch → rollout_max_length`; else `int(round(min_len + (epoch − min_epoch)/max(1, max_epoch − min_epoch) · (max_len − min_len)))`. Per-epoch table below (Section 6.1e).
- Model buffer: capacity = `max(10000, rollout_batch_size · rollout_max_length · model_retain_epochs)` (line {L['mbuf']}), `model_retain_epochs` = {CFGD['HC']['model_retain_epochs']}; a circular `ReplayBuffer` (`add_batch` overwrites the oldest entries). HC: 10000 · 1 · 1 = 10000; Ant L=25: 250000; Ant L=15: 150000; Ant L=1: 10000. Retained epochs of model data per regime: Section 6.1f.

**(d) `real_ratio` and the minibatch composition** (`_MixedReplayBuffer.sample`, lines {L['mixed']}–{L['sched'] - 2}; created at line {L['mixedbuf']}): `real_ratio` = {CFGD['HC']['real_ratio']}. `if len(model_buffer) == 0` the batch is purely real (line {L['emptyfb']}); this never happens during the measured `gradient_updates` because the model buffer is filled by the epoch's synthetic generation before the first gradient update of epoch 0. Otherwise `n_real = min(max(int(round(batch_size · real_ratio)), 0), batch_size)` (line {L['nreal']}), `n_model = batch_size − n_real`; two separate host-side `ReplayBuffer.sample` calls (real, model) and a NumPy `concatenate` of the five arrays; the concatenated host arrays are then moved to the GPU inside `SACAgent.update`'s `buffer_sample` timer. batch 256 → {int(round(256 * 0.05))} real + {256 - int(round(256 * 0.05))} model; batch 512 → {int(round(512 * 0.05))} + {512 - int(round(512 * 0.05))}; batch 1024 → {int(round(1024 * 0.05))} + {1024 - int(round(1024 * 0.05))}. (`len(mixed_buffer)` = real + model sizes, so the `len < batch_size` guard in the update loop never triggers.)

**(e) Is `SACAgent` used unmodified, and which quantities does each sweep touch?**
- `mbpo.py` imports `SACAgent` from `algorithms.sac` and instantiates it as `SACAgent(obs_dim, act_dim, act_limit, mbpo_cfg, device)` (line {L['agent']}); there is no subclass, wrapper, monkey-patch or attribute assignment on it in `mbpo.py` (regex search for `class …(SACAgent)`: {'found' if subclass else 'none'}; for `agent.<attr> = …` assignments and `setattr(`: {len(attr_assign)} found); the only members of the agent that `mbpo.py` touches are: {', '.join('`agent.' + u + '`' for u in agent_uses)}; `algorithms/sac.py` is byte-identical at all run commits (Section 0). `SACAgent` reads these config fields (parsed from its source): {', '.join(f'`{u}`' for u in used)}; MBPOConfig fields **not** read by `SACAgent` (consumed by `mbpo.py`/`dynamics_model.py`): {', '.join(f'`{u}`' for u in unused)}.
- **UTD** (`updates_per_env_step`): used only for `n_updates = steps_per_epoch × updates_per_env_step` in the `gradient_updates` loop (mbpo.py). It does **not** change the number of model fits (decided by `model_train_freq` vs `global_step`: 100 per run in every configuration — verified in Section 6.2) nor the number of synthetic-rollout calls (one per epoch = 100); the number of synthetic *transitions* and the fit length can still differ between runs because the policy and the data differ (Section 6.2/6.4 give the measured values per configuration).
- **Width** (`hidden_sizes`): read only by `SACAgent` (actor, two critics, two targets). `model_hidden_sizes` (200×4) and the model batch (256) are unchanged: per-call dynamics FLOPs are identical across the width configurations ({', '.join(sorted({str(FLOPS_JSON['mbpo']['HalfCheetah-v5'][s]['dynamics_member_fwdbwd']) for s in FLOPS_JSON['mbpo']['HalfCheetah-v5']}))} for the member fwd+bwd at HC); the synthetic-sample cost changes through `actor_forward_bs1` only.
- **Batch size** (`batch_size`): the SAC minibatch (`agent.update(mixed_buffer, batch_size)` and `_MixedReplayBuffer.sample`); `model_train_batch_size` stays 256, so the model-fit cost per optimiser step is unchanged; the real/model split changes (13/243 → 26/486 → 51/973).
- **Maximum rollout length** (`rollout_max_length`, Ant only): changes the schedule values (Section 6.1e), the number of synthetic transitions per epoch and per run, the model-buffer capacity (line {L['mbuf']}), hence the content of the model buffer that GU samples; it does not change the per-call FLOP constants (`flop_keys.mbpo_rollout_regime` tracks it separately from the architecture signature).
""")
    # ---- (e) per-epoch rollout length table
    rows = []
    sched = {}
    for name, es, tg in REGIMES:
        cfg = regime_cfg(es, tg)
        sched[name] = [_rollout_length(e, cfg) for e in range(100)]
    warm = 5000
    # verify against logs
    bad = []
    for r in DS.runs:
        cfg = regime_cfg(r["es"], r["tag"] if r["tag"] in ("rollout1", "rollout15") else "base")
        logged = [e["rollout_length"] for e in r["tm"]["epochs"]]
        if logged != [_rollout_length(e, cfg) for e in range(100)]:
            bad.append(r["run_dir"])
    out.append("**6.1e Rollout length used in every epoch (0–99) for each regime** — computed with `mbpo._rollout_length` from the logged configuration of each regime; "
               "`n_start` = min(10000, warmup + (epoch+1)·1000) start states; upper bound = n_start × L transitions (no early termination). "
               "The logged `rollout_length` of all 80 runs equals this schedule: " + check("logged rollout_length per epoch equals the schedule in every run", not bad, str(bad[:3]) if bad else "") + "\n")
    rows = []
    for e in range(100):
        ns = min(10000, warm + (e + 1) * spe)
        rows.append([e] + [sched[n][e] for n, _, _ in REGIMES] + [ns] + [ns * sched[n][e] for n, _, _ in REGIMES])
    out.append(table(["epoch", "L: HC fixed 1", "L: Ant 1→25", "L: Ant max 15", "L: Ant max 1", "n_start", "upper bound HC (n_start·L)", "upper bound Ant 25", "upper bound Ant 15", "upper bound Ant 1"], rows))
    # regime parameters
    rr = []
    for name, es, tg in REGIMES:
        c = regime_cfg(es, tg)
        rr.append([name, c.rollout_min_epoch, c.rollout_max_epoch, c.rollout_min_length, c.rollout_max_length, mbpo_rollout_regime(dataclasses.asdict(c)),
                   max(10000, c.rollout_batch_size * c.rollout_max_length * c.model_retain_epochs), sum(sched[name]), sum(min(10000, warm + (e + 1) * spe) * sched[name][e] for e in range(100))])
    out.append("Regime parameters (from the logged `algo_config`), model-buffer capacity, Σ of the schedule over the 100 epochs, and the upper bound on transitions per run:\n")
    out.append(table(["regime", "rollout_min_epoch", "rollout_max_epoch", "rollout_min_length", "rollout_max_length", "`mbpo_rollout_regime` key", "model buffer capacity", "Σ_epochs L", "upper bound transitions per run"], rr))
    # ---- (f) transitions generated, model buffer retention
    rows, ret_rows, csvr = [], [], []
    cap_of = {}
    for es, t in DS.cfgs():
        rs = srt(DS.runs_of(es, t))
        cfg = regime_cfg(es, t if t in ("rollout1", "rollout15") else "base")
        cap = max(10000, cfg.rollout_batch_size * cfg.rollout_max_length * cfg.model_retain_epochs)
        ub = sum(min(10000, warm + (e + 1) * spe) * _rollout_length(e, cfg) for e in range(100))
        gen = [sum(e["synthetic_transitions_generated"] for e in r["tm"]["epochs"]) for r in rs]
        rows.append([label(es, t), SPEC.tag_override_file(es, t) if False else "", msd(gen), g6(ub), msd([100 * g / ub for g in gen]), g6(min(gen)), g6(max(gen))])
        # model buffer retention from logged sizes
        mis = 0
        ret_all = []
        for r in rs:
            cs = 0
            counts = [e["synthetic_transitions_generated"] for e in r["tm"]["epochs"]]
            for i, e in enumerate(r["tm"]["epochs"]):
                cs += counts[i]
                if e["model_buffer_size"] != min(cap, cs):
                    mis += 1
                # number of most recent epochs whose data fit entirely in the buffer at the end of epoch i
                k, tot = 0, 0
                for j in range(i, -1, -1):
                    if tot + counts[j] <= cap:
                        tot += counts[j]
                        k += 1
                    else:
                        break
                ret_all.append(k)
        ret_rows.append([label(es, t), cap, mis, g6(min(ret_all)), g6(float(np.median(ret_all))), g6(max(ret_all)), g6(np.mean(ret_all))])
        for r in rs:
            for e in r["tm"]["epochs"]:
                csvr.append(dict(config=t, env=r["env"], seed=r["seed"], epoch=e["epoch"], rollout_length=e["rollout_length"], synthetic_transitions_generated=e["synthetic_transitions_generated"],
                                 model_buffer_size=e["model_buffer_size"], model_train_epochs=e["model_train_epochs"], model_holdout_mse=e["model_holdout_mse"]))
    pd.DataFrame(csvr).to_csv(OUT_DIR / "mbpo_model_per_epoch.csv", index=False, float_format="%.17g")
    out.append("\n**6.1f Synthetic transitions generated per run and model-buffer retention** — `synthetic_transitions_generated` is logged per epoch (`training_metrics.json`); the upper bound assumes no early termination (Σ_epochs n_start·L). "
               "Explicit early-termination counts are **not logged**; the shortfall relative to the upper bound is the only (indirect) record. Retention: with capacity C and logged per-epoch counts n_e, "
               "the number of most recent epochs whose data fit completely into the circular model buffer at the end of each epoch (min / median / max / mean over all epochs and seeds); the "
               "logged `model_buffer_size` equals min(C, cumulative generated) at every epoch (column 'size mismatches' counts epochs where it does not). Per-epoch values: `contexts/Section_4.3_data/mbpo_model_per_epoch.csv`.\n")
    out.append(table(["config", "", "transitions generated per run (mean ± sd)", "upper bound (no termination)", "generated / upper bound [%]", "min", "max"], rows))
    out.append(table(["config", "model buffer capacity", "size mismatches (epochs)", "retained epochs min", "median", "max", "mean"], ret_rows))
    na("Explicit early-termination counts of the synthetic rollouts (number of rows terminated per epoch / per step): not logged by `mbpo.py`; only the shortfall of `synthetic_transitions_generated` against n_start·L is available (Section 6.1f / 6.4).")
    # ---- 6.2 FLOP definitions
    out.append("\n**6.2 FLOP definitions and the model-fit statistics** — formulas verbatim in Section 2.7. Per configuration, from the logged epoch rows of every run (`mbpo_extras`):\n")
    rows, rows2, rows3 = [], [], []
    for es, t in DS.cfgs():
        rs = srt(DS.runs_of(es, t))
        Fd = np.array([r["mbpo"]["F_dyn"] for r in rs], float)
        hs = np.array([100 * r["mbpo"]["dyn_hold"] / r["mbpo"]["F_dyn"] for r in rs])
        nf = [len(r["mbpo"]["fits"]) for r in rs]
        first = [r["mbpo"]["fits"][0]["fit_epochs"] for r in rs]
        later = [np.mean([f["fit_epochs"] for f in r["mbpo"]["fits"][1:]]) for r in rs]
        allf = [np.mean([f["fit_epochs"] for f in r["mbpo"]["fits"]]) for r in rs]
        steps_first = [r["mbpo"]["fits"][0]["steps"] for r in rs]
        steps_later = [np.mean([f["steps"] for f in r["mbpo"]["fits"][1:]]) for r in rs]
        rows.append([label(es, t), g6(Fd.min()), g6(Fd.mean()), g6(Fd.max()), g6(Fd.std(ddof=1)), msd(hs), msd([r["mbpo"]["dyn_train"] for r in rs]), msd([r["mbpo"]["dyn_hold"] for r in rs])])
        rows2.append([label(es, t), f"{min(nf)}–{max(nf)}", msd(allf), msd(first), msd(later), msd([r["mbpo"]["dyn_steps"] for r in rs]), msd(steps_first), msd(steps_later),
                      msd([max(f["fit_epochs"] for f in r["mbpo"]["fits"]) for r in rs])])
        rows3.append([label(es, t), msd([r["mbpo"]["synth_samples"] for r in rs]), g6(rs[0]["fk"]["actor_forward_bs1"]), g6(rs[0]["fk"]["dynamics_ensemble_forward_all_bs1"]),
                      g6(rs[0]["fk"]["actor_forward_bs1"] + rs[0]["fk"]["dynamics_ensemble_forward_all_bs1"]), msd([r["mbpo"]["F_synth"] for r in rs]),
                      msd([r["F"]["TOTAL_MEASURED_TRAINING"] for r in rs])])
    out.append(table(["config", "dyn FLOPs min over seeds", "mean", "max", "sd", "holdout-forward share of dyn FLOPs [%] (mean ± sd)", "train FLOPs (mean ± sd)", "holdout FLOPs (mean ± sd)"], rows))
    out.append("The layout documents say the holdout forward is 'about 8 %' of the fit FLOPs; the actual values are 7.77581 % (HC) and 7.99274 % (Ant) in every run of every configuration (column above, sd ≈ 0). "
               "This share is a constant by construction: both terms scale linearly with epochs_run × n_total (n_holdout = 0.2·n and n_train = 0.8·n for n a multiple of 1000), so it depends only on the per-sample constants "
               "(7 × dynamics_member_fwdbwd/256 for training, dynamics_ensemble_forward_all_bs1 for the holdout forward), i.e. only on the environment's obs/act dimensions.\n")
    out.append(table(["config", "fits per run (min–max)", "fit epochs per fit, mean over fits [epochs]", "first fit (epoch 0) epochs", "later fits (epochs 1–99), mean epochs", "optimiser steps per run", "steps of the first fit", "steps per later fit (mean)", "longest fit of a run [epochs]"], rows2))
    out.append("Synthetic rollouts: per-sample FLOPs = `actor_forward_bs1` + `dynamics_ensemble_forward_all_bs1` — **independent of the rollout length L** (the cost of a step per sample is constant); the number of samples is Σ_epochs generated transitions (≈ n_start·L when nothing terminates):\n")
    out.append(table(["config", "samples per run (mean ± sd)", "actor_forward_bs1", "ensemble_forward_all_bs1", "per-sample FLOPs", "synthetic FLOPs per run", "TOTAL FLOPs per run"], rows3))
    ok = all(len(r["mbpo"]["fits"]) == 100 for r in DS.runs)
    out.append("- " + check("every run has exactly 100 fits (a non-null `model_train_epochs` in all 100 epochs)", ok))
    # J/FLOP both ways -> table
    rows = []
    for es, t in DS.cfgs():
        rs = srt(DS.runs_of(es, t))
        cells = [label(es, t)]
        for s in ("dynamics_model_update", "synthetic_rollout_generation", "TOTAL_MEASURED_TRAINING"):
            Em = np.mean(vals(rs, "E", s))
            Fm = np.mean(vals(rs, "F", s))
            per = np.array([r["E"][s] / r["F"][s] for r in rs])
            cells.append(f"{g6(Em / Fm)} ‖ {g6(per.mean())} (sd {g6(per.std(ddof=1))})")
        rows.append(cells)
    out.append("\nJ/FLOP of the two MBPO-specific segments and of TOTAL, **ratio of means ‖ mean of per-seed ratios (sd of per-seed ratios)** (FLOPs differ between seeds, so the two statistics differ slightly):\n")
    out.append(table(["config", "dynamics_model_update [J/FLOP]", "synthetic_rollout_generation [J/FLOP]", "TOTAL [J/FLOP]"], rows))
    # ---- 6.3 segment table remarks
    out.append("\n**6.3 Segment tables.** The common blocks of Sections 3–4 contain all seven segments in every table: `rollout`, `dynamics_model_update`, `synthetic_rollout_generation` "
               "(measured, one CodeCarbon task per epoch), `gradient_updates` (measured) with its four allocated sub-segments `buffer_sample`, `critic_update`, `actor_update`, `target_update`; shares of TOTAL; J/FLOP "
               "of the model fit and of the synthetic rollouts (both ratio-of-means and mean-of-ratios, Section 6.2); the hardware composition (CPU/GPU/RAM) and the count of tasks shorter than 1 s of each "
               "algorithm-specific task are in the 'Hardware composition' tables (one per measured task).")
    # ---- 6.4 rollout-length sweep
    out.append("\n**6.4 Rollout-length sweep (Ant-v5 only)** — the common block for lengths 1, 15 and 25 is Section 4.4. In addition, per configuration (Ant-v5), mean ± sd over seeds:\n")
    rows = []
    for t in ("rollout1", "rollout15", "base"):
        rs = srt(DS.runs_of("Ant", t))
        cfg = regime_cfg("Ant", t if t != "base" else "base")
        ub = sum(min(10000, warm + (e + 1) * spe) * _rollout_length(e, cfg) for e in range(100))
        gen = [r["mbpo"]["synth_samples"] for r in rs]
        rows.append([f"L_max = {cfg.rollout_max_length} ({t})", msd(vals(rs, "E", "synthetic_rollout_generation")), msd(vals(rs, "E", "dynamics_model_update")), msd(vals(rs, "E", "gradient_updates")),
                     msd(vals(rs, "E", "rollout")), msd(vals(rs, "E", "TOTAL_MEASURED_TRAINING")), msd(gen), g6(ub), msd([100 * (1 - g / ub) for g in gen]),
                     msd(vals(rs, "D", "synthetic_rollout_generation")), msd(vals(rs, "D", "dynamics_model_update"))])
    out.append(table(["config", "synthetic-rollout energy [J]", "model-fit energy [J]", "GU energy [J]", "rollout energy [J]", "TOTAL energy [J]", "transitions generated per run", "upper bound (no termination)",
                      "shortfall vs upper bound [%] (derived; early terminations)", "synthetic-rollout duration [s]", "model-fit duration [s]"], rows))
    out.append("Early-termination information that is logged: **none explicitly** (see Section 6.1f: NOT AVAILABLE as a count); the derived shortfall above is the fraction of the upper-bound transitions that were not generated "
               "because states terminated (Ant `_ant_done`), HC being exactly 0 (check below).")
    hcb = max(abs(r["mbpo"]["synth_samples"] - sum(min(10000, warm + (e + 1) * spe) * 1 for e in range(100))) for r in DS.runs if r["es"] == "HC")
    out.append("- " + check("HalfCheetah (never terminates, L = 1): generated transitions equal Σ n_start·1 exactly in all 35 HC runs", hcb == 0, f"max |difference| = {hcb}"))
    # ---- 6.5 env confound
    out.append("\n**6.5 Environment confound — complete diff of the logged configurations of the HC and the Ant baseline** (`metadata.json`, run seed 331 of each; every field that differs):\n")
    mh, ma = srt(DS.runs_of("HC", "base"))[0]["meta"], srt(DS.runs_of("Ant", "base"))[0]["meta"]
    rows = []

    def flat(d, pfx=""):
        o = {}
        for k, v in d.items():
            if isinstance(v, dict):
                o.update(flat(v, pfx + k + "."))
            else:
                o[pfx + k] = v
        return o
    fh, fa = flat({"algo_config": mh["algo_config"], "experiment_config": mh["experiment_config"]}), flat({"algo_config": ma["algo_config"], "experiment_config": ma["experiment_config"]})
    for k in sorted(set(fh) | set(fa)):
        if fh.get(k) != fa.get(k) and k not in ("experiment_config.seed",):
            rows.append([f"`{k}`", json.dumps(fh.get(k)), json.dumps(fa.get(k))])
    out.append(table(["field", "HalfCheetah-v5 baseline", "Ant-v5 baseline"], rows))
    out.append("Other (non-config) differences that are properties of the environments: observation/action dims 17/6 vs 105/8 (Section 2.3), termination function `_never_done` vs `_ant_done` "
               "(6.1c), episodes (HC 1000-step truncation only; Ant terminates early, Section 5.3). The rollout-length schedule per epoch of both baselines is the table in 6.1e "
               "(HC: constant 1; Ant: 1 for epochs ≤ 20, then linear to 25 at epoch 100).")
    # ---- 6.6 versus SAC
    out.append("\n**6.6 MBPO versus SAC side by side** at matched settings — all 7 configurations that exist for both algorithms (`base` = hidden (1024,1024), batch 256, UTD 1; `utd2`, `utd4`, `w256`, `w512`, `b512`, `b1024`). "
               "Columns: HC MBPO, HC SAC, Ant MBPO, Ant SAC. SAC values come from the same recomputation method as Section 4.1 (`s4_data.Dataset(sac_spec())`); cross-check against "
               "`contexts/section_4.1_data/sac_cross_seed_summary.csv`. Batch composition: SAC samples 256/512/1024 transitions all from the real buffer; MBPO samples 13 real + 243 model (batch 256), 26 + 486 (512), 51 + 973 (1024) "
               "via two host-side samples and a NumPy concatenate (6.1d). The policy-optimisation code path (`SACAgent.update`) is the same object code in both.\n")
    sc = pd.read_csv("contexts/section_4.1_data/sac_cross_seed_summary.csv")
    mx = 0
    for es, _ in ENVS:
        for t in SAC.spec.tags:
            for s in ["rollout", "buffer_sample", "critic_update", "actor_update", "target_update", "TOTAL_MEASURED_TRAINING"]:
                row = sc[(sc.tag == t) & (sc.env == dict(ENVS)[es]) & (sc.segment == s)].iloc[0]
                rs = srt(SAC.runs_of(es, t))
                mx = max(mx, abs(np.mean(vals(rs, "E", s)) - row.energy_j_mean) / row.energy_j_mean)
    out.append("- " + check("SAC energies recomputed here equal `sac_cross_seed_summary.csv` of Section 4.1 for all 7 configurations and both environments", mx < 1e-9, f"max relative deviation {mx:.2e}"))
    matched = ["base", "utd2", "utd4", "w256", "w512", "b512", "b1024"]

    def vs_table(title, getter, fmt=msd):
        rows = []
        for t in matched:
            cells = [t]
            for es, _ in ENVS:
                for dsx in (DS, SAC):
                    rs = srt(dsx.runs_of(es, t))
                    cells.append(fmt(getter(rs)))
            rows.append(cells)
        out.append(f"\n{title}\n")
        out.append(table(["config", "HC MBPO", "HC SAC", "Ant MBPO", "Ant SAC"], rows))
    vs_table("GU (`gradient_updates`, measured) energy [J]:", lambda rs: vals(rs, "E", "gradient_updates"))
    for s in SUBS:
        vs_table(f"`{s}` (allocated) energy [J]:", lambda rs, s=s: vals(rs, "E", s))
    for s in SUBS:
        vs_table(f"`{s}` share of GU [%] (wall-clock time share):", lambda rs, s=s: vals(rs, "share_gu", s))
    vs_table("`buffer_sample` share of TOTAL energy [%]:", lambda rs: vals(rs, "share", "buffer_sample"))
    vs_table("`buffer_sample` perf_counter time [s]:", lambda rs: [r["T"]["buffer_sample"] for r in rs])
    for s, u, nm in (("critic_update", 1, "J/FLOP"), ("actor_update", 1, "J/FLOP"), ("target_update", 1e9, "nJ per Polyak op")):
        vs_table(f"`{s}` energy per FLOP [{nm}] (ratio of means ± sd of per-seed ratios):", lambda rs, s=s, u=u: jpf_cell(rs, s, u), lambda v: v)
    vs_table("Per-call FLOPs critic / actor (identical code path; `flops_per_call.json`):", lambda rs: f"{g6(rs[0]['fk']['critic_fwdbwd'])} / {g6(rs[0]['fk']['actor_fwdbwd'])}", lambda v: v)
    vs_table("GU measured duration [s] and sub-timer coverage [%]:", lambda rs: f"{msd(vals(rs, 'D', 'gradient_updates'))}; cov {msd([100 * r['coverage'] for r in rs])}", lambda v: v)
    # ---- open points
    out.append("\n**6.7 Open points from the instructions (reported, not resolved):**\n")
    allr = pe[pe.task == "rollout"].duration_s
    out.append(f"- *Task durations versus the 1 s polling interval:* `rollout`: {int((allr < 1).sum())} of {len(allr)} tasks < 1 s (min/median/max {g6(allr.min())}/{g6(allr.median())}/{g6(allr.max())} s). "
               + "; ".join(f"`{k}`: {int((pe[pe.task == k].duration_s < 1).sum())} of {len(pe[pe.task == k])} < 1 s (min/median/max {g6(pe[pe.task == k].duration_s.min())}/{g6(pe[pe.task == k].duration_s.median())}/{g6(pe[pe.task == k].duration_s.max())} s)" for k in ("dynamics_model_update", "synthetic_rollout_generation", "gradient_updates"))
               + ". Per configuration: Section 5.2a.")
    out.append("- *Host-side work counted as FLOPs:* none; only matmul FLOPs (`FlopCounterMode`). **Counted:** `rollout` = actor forward at batch 1; `critic_update`/`actor_update` = as SAC (Section 4.1); `dynamics_model_update` = per fit epoch every member's fwd+bwd over its n_train bootstrap samples + the full-ensemble holdout forward (from per-sample constants; Adam, the NLL, `softplus` log-variance bounds, bootstrap resampling, `np.random.shuffle`, input normalisation, per-epoch `.cpu()` syncs are **not** counted); "
               "`synthetic_rollout_generation` = per generated sample actor forward + full 7-member ensemble forward (the random elite choice, Gaussian sampling, `termination_fn` (NumPy), `add_batch`, `.cpu().numpy()` transfers are **not** counted); `buffer_sample` = 0 (host sampling of two buffers + concatenate + H2D copies); `target_update` = Polyak operations. Counting uses all 7 members although only the 5 elites are used for the prediction (every member is scored, `predict` picks afterwards).")
    out.append("- *Segments whose timers or FLOPs are produced differently from SAC:* `dynamics_model_update` and `synthetic_rollout_generation` are directly measured CodeCarbon tasks (not time-allocated); the FLOPs of `dynamics_model_update` depend on the data-dependent early stopping (fit epochs per fit logged as `model_train_epochs`), and those of `synthetic_rollout_generation` on terminations; "
               "`buffer_sample` is two host samples + concatenate (SAC: one sample); everything inside `gradient_updates` is otherwise the SAC code path.")
    return "\n".join(out)


# =============================================================================== assemble
def main():
    spec = DS.spec
    MD = []
    P = MD.append
    recon_txt, recon_stats = recon(DS)
    prov = provenance(DS, ["build_section_4_3.py (in Section_4.3_data/)", "s4_data.py", "s4_blocks.py", "s4_audit.py (in Section_4.2_data/)"],
                      ["algorithms/mbpo.py", "algorithms/dynamics_model.py", "algorithms/termination_fns.py", "algorithms/sac.py", "algorithms/replay_buffer.py", "algorithms/tracker_utils.py"])
    inv_keys = ["hidden_sizes", "batch_size", "updates_per_env_step", "rollout_min_epoch", "rollout_max_epoch", "rollout_min_length", "rollout_max_length"]
    extra = []
    aud = audit(DS, inv_keys, extra)
    exec_txt, _ = exec_facts(DS, ["results/mbpo/_env_sweep_status.json", "results/_utd_sweep_status.json", "results/_width_sweep_status.json", "results/_batch_size_sweep_status.json",
                                   "results/_ant_overnight_sweep_status.json"],
                             ["`scripts/run_mbpo_env_sweep.sh` (baselines, 10 runs)", "`scripts/run_utd_sweep.sh` (as it exists now: MBPO UTD ∈ {2,4}, 20 runs)", "`scripts/run_width_sweep.sh`",
                              "`scripts/run_batch_size_sweep.sh` (HC)", "`scripts/run_ant_overnight_sweep.sh` (Ant batch sizes + rollout-length sweep; standalone `run_batch_size_sweep_ant.sh`, `run_mbpo_rollout_length_sweep.sh`)"])
    sec2 = section2()
    sec6 = section6()
    blocks = {"baseline": common_block(DS, ["base"], with_response=False, with_env=True)}
    for name, text, tags, xd in spec.sweeps:
        blocks[name] = common_block(DS, tags, with_response=True, with_env=name != spec.sweeps[3][0], base_tag="base")
    ols = ols_block(DS, OUT_DIR, PREFIX)
    dlt = delta_block(DS, OUT_DIR, PREFIX, [("hidden_sizes", "width"), ("batch_size", "batch size"), ("updates_per_env_step", "UTD"), ("mbpo_rollout_regime", "max rollout length")])
    qual = quality_block(DS, pe, IDLE, OUT_DIR, PREFIX)
    learn = learning_block(DS, OUT_DIR, PREFIX)
    common_section7_items(DS, IDLE)
    disc("`configs/config.py` documents `model_train_freq` (250) as 'retrain the ensemble once real env steps since the last retrain reach this', and README.md says the ensemble is 'retrained periodically'. "
         "`mbpo.train()` evaluates the condition only once per epoch (1000 real steps), so the effective schedule is one fit per epoch = 100 fits per run in every configuration (Section 6.2), not one per 250 steps.")
    disc("The docstring of `EnsembleDynamicsModel.fit()` says it 'trains the full ensemble from scratch'; the code never re-initialises the network or the Adam optimiser between fits (`fit()` only refits the input normaliser and resamples the bootstrap), "
         "so every fit continues from the previous weights (warm start). Early stopping consequently ends after on average "
         f"{g6(np.mean([np.mean([f['fit_epochs'] for f in r['mbpo']['fits'][1:]]) for r in DS.runs if r['es'] == 'HC']))} (HC) / "
         f"{g6(np.mean([np.mean([f['fit_epochs'] for f in r['mbpo']['fits'][1:]]) for r in DS.runs if r['es'] == 'Ant']))} (Ant) epochs per later fit, versus "
         f"{g6(np.mean([r['mbpo']['fits'][0]['fit_epochs'] for r in DS.runs if r['es'] == 'HC']))} (HC) / {g6(np.mean([r['mbpo']['fits'][0]['fit_epochs'] for r in DS.runs if r['es'] == 'Ant']))} (Ant) epochs in the first fit (means over all runs; Section 6.2).")
    question("`rollout_min_epoch`/`rollout_max_epoch` differ between the HC baseline (20/150) and the Ant baseline (20/100) in addition to `rollout_max_length` (1 vs 25). With L_max = 1 on HC the schedule is constant, "
             "so the epoch bounds have no effect there; whether the thesis should describe the HC regime as 'no scheduling' is for the author to decide.")
    question("The model is refit in every epoch (1000 real steps ≥ `model_train_freq` = 250), i.e. 100 fits per run instead of the 400 implied by 'retrain every 250 steps'. Whether this is the intended MBPO setting for the thesis description is not determined by the repository.")
    n_pass = sum(1 for _, ok, _ in CHECKS if ok)
    n_fail = len(CHECKS) - n_pass
    P("# Section 4.3 MBPO — context\n")
    P(f"Data context for thesis Section 4.3 (MBPO): {len(DS.runs)} canonical runs (HC: 7 configurations, Ant: 9 configurations, 5 seeds {CANON} each). "
      "Facts, numbers and provenance only — no interpretation, no LaTeX. All energies are **gross** (idle floor included, nothing subtracted); mean ± sample sd (ddof = 1) over the five seeds; 6 significant digits; "
      "every table cell shows `HalfCheetah-v5 ‖ Ant-v5` (left ‖ right) unless a table says otherwise (`n/a` = configuration does not exist in that environment). Conventions of Section 4.1 apply (`measured` vs `allocated` segments; "
      "`target_update` in nJ per Polyak operation, never J/FLOP; `buffer_sample` has no FLOPs; J/FLOP = ratio of means with the sd of the five per-seed ratios — MBPO FLOPs differ between seeds, so the mean of the per-seed ratios is given as well; "
      "relative changes computed per seed then averaged; environment comparison paired by seed, Ant relative to HC). Training total = `TOTAL_MEASURED_TRAINING` = rollout + dynamics_model_update + synthetic_rollout_generation + gradient_updates; warmup and idle windows are excluded.\n")
    P(f"Checks run by the generator: {len(CHECKS)}; passed {n_pass}; failed {n_fail}. Independent recomputation vs the pipeline CSVs: {recon_stats['n_compared']} values compared, "
      f"{recon_stats['n_fail']} deviating by more than 1e-9, maximum relative deviation {recon_stats['max_dev']:.3e}.\n")
    P("**Configuration tags:** `base` = SAC hidden (1024,1024), batch 256, UTD 1, real_ratio 0.05, model ensemble 7 × (200,200,200,200), warmup 5000; HC baseline = `MBPOConfig` defaults (rollout length fixed 1), Ant baseline = defaults + "
      "`configs/overrides/mbpo_ant.json` (rollout length 1→25 between epochs 20 and 100). `utd2`/`utd4`, `w256`/`w512`, `b512`/`b1024` change exactly that one factor of the same environment's baseline; `rollout1`/`rollout15` (Ant only) change `rollout_max_length` "
      "from 25 to 1 / 15. Abbreviations: HC = HalfCheetah-v5, Ant = Ant-v5, GU = `gradient_updates`, TOTAL = `TOTAL_MEASURED_TRAINING`, dyn_model = `dynamics_model_update`, synth_rollout = `synthetic_rollout_generation`, "
      "buf_sample = `buffer_sample`, critic = `critic_update`, actor = `actor_update`, target = `target_update`.\n")
    P("## 0. Provenance\n")
    P(prov)
    P("\n## 1. Data basis and audit\n")
    P(aud)
    P("\n" + exec_txt)
    P("\n" + recon_txt)
    P("\n## 2. Structural facts from code and FLOP files\n")
    P(sec2)
    P("\n## 3. Baseline decomposition and energy per FLOP (4.3.1)\n")
    P("Configuration `base` (see tags above; HC and Ant baselines differ in the rollout-length schedule, Section 6.5).\n")
    P(blocks["baseline"])
    for i, (name, text, tags, xd) in enumerate(spec.sweeps, 1):
        P(f"\n## 4.{i} Sweep: {name} — {text}\n")
        P(f"Configurations: {', '.join(f'`{t}` ({spec.tag_desc[t]})' for t in tags)}. Baseline for the response tables: `base` of the same environment."
          + (" Environment comparison omitted: the configurations exist only on Ant-v5 (except `base`)." if i == 4 else "") + "\n")
        P(blocks[name])
    P("\n## 5. Cross-configuration quantities\n")
    P(ols)
    P("\n" + dlt)
    P("\n" + qual)
    P("\n" + learn)
    P("\n## 6. Algorithm-specific items (MBPO)\n")
    P(sec6)
    P("\n## 7. Discrepancies, NOT AVAILABLE items, questions\n")
    P("**(a) Where data or code contradict CLAUDE.md, README.md or `thesis-layout.md` (the latter is not present in the repository):**\n")
    P("\n".join(f"- {d}" for d in DISCREPANCIES) if DISCREPANCIES else "- none found by the generator's checks")
    P("- `thesis-layout.md` is referenced by the instructions but does not exist in the repository (searched the repo and the Downloads folder); the layout expectations (80 MBPO runs: 7 + 9 configurations) were taken from the instruction file and verified against `results/`.")
    P("\n**(b) Quantities that could not be obtained (`NOT AVAILABLE`):**\n")
    P("\n".join(f"- NOT AVAILABLE: {d}" for d in NOT_AVAILABLE) if NOT_AVAILABLE else "- none")
    P("\n**(c) Questions for the author that the repository cannot answer:**\n")
    P("\n".join(f"- {d}" for d in QUESTIONS) if QUESTIONS else "- none")
    P("\n**Index of produced files** (`contexts/Section_4.3_data/`): `mbpo_segments_long.csv`, `mbpo_cross_seed_summary.csv`, `mbpo_run_inventory.csv`, `mbpo_per_epoch.csv`, `mbpo_model_per_epoch.csv`, `mbpo_idle_floor.csv`, "
      "`mbpo_ols_fits.csv`, `mbpo_delta_pairs.csv`, `mbpo_mad_flags.csv`, `mbpo_learning_performance.csv`; scripts `build_section_4_3.py` (here) and the shared modules in `contexts/Section_4.2_data/`.")
    OUT_MD.write_text("\n".join(MD) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD} ({len(MD)} blocks); checks {len(CHECKS)} pass {n_pass} fail {n_fail}")
    for n_, ok, d in CHECKS:
        if not ok:
            print("FAIL:", n_, d)


if __name__ == "__main__":
    main()
