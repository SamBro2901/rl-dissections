"""
Generator for contexts/Section_4.4.md (TD-MPC2) + companion CSVs in contexts/Section_4.4_data/.
Read-only on results/, flop_analysis/ and git; writes only under contexts/.
Run from the repo root:  .venv/Scripts/python.exe contexts/Section_4.4_data/build_section_4_4.py
Shared modules live in contexts/Section_4.2_data/ (s4_data.py, s4_blocks.py, s4_audit.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Section_4.2_data"))
from s4_audit import *  # noqa: F401,F403,E402
from s4_audit import Src, srt  # noqa: E402
from configs.config import TDMPC2Config  # noqa: E402

OUT_MD = REPO / "contexts" / "Section_4.4.md"
OUT_DIR = REPO / "contexts" / "Section_4.4_data"
OUT_DIR.mkdir(exist_ok=True)
PREFIX = "tdmpc2"

TAGS = ["base", "numq3", "numq7", "hor1", "hor5"]
SPEC = Spec(
    algo="tdmpc2", section="4.4", name="TD-MPC2", config_cls=TDMPC2Config, tags=TAGS,
    tag_desc={"base": "paper hyperparameters: num_q 5, horizon 3, 512 samples (24 policy-seeded), 6 iterations, 64 elites, batch 256, UTD 1; Ant: episodic = true (tdmpc2_ant.json)",
              "numq3": "num_q 3", "numq7": "num_q 7", "hor1": "horizon 1", "hor5": "horizon 5"},
    tag_factor={"base": {}, "numq3": {"num_q": 3}, "numq7": {"num_q": 7}, "hor1": {"horizon": 1}, "hor5": {"horizon": 5}},
    env_ref_override={"HC": None, "Ant": "configs/overrides/tdmpc2_ant.json"},
    extra_measured=["world_model_pretrain"], measured_order=["world_model_pretrain", "rollout", "gradient_updates"],
    sweeps=[("num_q", "num_q ∈ {3,5,7} (Q-ensemble size)", ["numq3", "base", "numq7"], "tdmpc2_num_q"),
            ("horizon", "horizon ∈ {1,3,5} (MPPI planning horizon / training sequence length)", ["hor1", "base", "hor5"], "tdmpc2_horizon")],
    expected_runs={"HC": 25, "Ant": 25}, per_epoch_tasks=["rollout", "gradient_updates"])

OVERRIDE = {"HC": {"base": "none (TDMPC2Config defaults)", "numq3": "configs/overrides/tdmpc2_numq3.json", "numq7": "configs/overrides/tdmpc2_numq7.json",
                   "hor1": "configs/overrides/tdmpc2_horizon1.json", "hor5": "configs/overrides/tdmpc2_horizon5.json"},
            "Ant": {"base": "configs/overrides/tdmpc2_ant.json", "numq3": "configs/overrides/tdmpc2_ant_numq3.json", "numq7": "configs/overrides/tdmpc2_ant_numq7.json",
                    "hor1": "configs/overrides/tdmpc2_ant_horizon1.json", "hor5": "configs/overrides/tdmpc2_ant_horizon5.json"}}
SPEC.tag_override_file = lambda es, t: OVERRIDE[es][t]
LOG_TO_SCRIPT = {"results/tdmpc2/_seed_sweep.log": "scripts/run_tdmpc2_seed_sweep.sh", "results/tdmpc2/_numq_horizon_sweep.log": "scripts/run_tdmpc2_numq_horizon_sweep.sh"}
DS = Dataset(SPEC)


def launcher(es, t):
    logs = sorted({lf for r in DS.runs_of(es, t) for lf, _ in SWEEP_BLOCKS.get(r["run_dir"], [])})
    return "; ".join(f"{LOG_TO_SCRIPT.get(l, l)} (log `{l}`)" for l in logs) or "NO LOG BLOCK FOUND"


SPEC.tag_script = launcher
pe = per_epoch_frame(DS)
IDLE = export_csvs(DS, OUT_DIR, pe, PREFIX)
tm_src = Src("algorithms/tdmpc2.py")
cepf = Src("flop_analysis/compute_energy_per_flop.py")
mfl = Src("flop_analysis/measure_flops.py")


def f_lin(dims):
    """matmul FLOPs per sample of an MLP with the given layer widths (2·n_in·n_out per Linear)."""
    return sum(2 * a * b for a, b in zip(dims[:-1], dims[1:]))


def parts(o, a, cfg, ep):
    lat, mlp, nb = cfg["latent_dim"], cfg["mlp_dim"], cfg["num_bins"]
    return dict(enc=f_lin([o, cfg["enc_dim"], lat]), dyn=f_lin([lat + a, mlp, mlp, lat]), rew=f_lin([lat + a, mlp, mlp, nb]),
                pi=f_lin([lat, mlp, mlp, 2 * a]), q=f_lin([lat + a, mlp, mlp, nb]), term=f_lin([lat, mlp, mlp, 1]) if ep else 0)


def analytic(o, a, cfg, ep, iterations=6):
    """Exact dense-layer FLOP model of agent.act() and agent.update() (derived by this generator; it reproduces every
    flops_per_call.json entry exactly, see Section 2.6)."""
    p = parts(o, a, cfg, ep)
    H, nq, B, ns, npi = cfg["horizon"], cfg["num_q"], cfg["batch_size"], cfg["num_samples"], cfg["num_pi_trajs"]
    roll_terms = dict(encode=p["enc"], pi_seed=npi * ((H - 1) * (p["pi"] + p["dyn"]) + p["pi"]),
                      plan_reward=iterations * ns * H * p["rew"], plan_dynamics=iterations * ns * H * p["dyn"],
                      plan_termination=iterations * ns * H * p["term"], plan_terminal_pi=iterations * ns * p["pi"],
                      plan_terminal_Q=iterations * ns * nq * p["q"])
    roll = sum(roll_terms.values())
    nog = H * B * (p["enc"] + p["pi"] + nq * p["q"])
    fg = B * p["enc"] + H * B * p["dyn"] + H * B * nq * p["q"] + H * B * p["rew"] + H * B * p["term"]
    crit = nog + 3 * fg - 2 * B * o * cfg["enc_dim"]
    act = (H + 1) * B * (3 * p["pi"] + 2 * nq * p["q"] - 2 * cfg["latent_dim"] * cfg["mlp_dim"])
    crit_terms = dict(target_encode_pi_Q_nograd=nog, forward_grad=fg, backward_grad=2 * fg - 2 * B * o * cfg["enc_dim"],
                      termination_part=3 * H * B * p["term"])
    return roll, crit, act, roll_terms, crit_terms, p


# =============================================================================== Section 2
def section2():
    import torch
    import gymnasium as gym
    from algorithms.tdmpc2 import WorldModel
    out = []
    out.append("**2.1 Configuration objects** (dumped from `configs/config.py` / `configs/overrides/` at HEAD):\n")
    out.append("`TDMPC2Config` defaults (`dataclasses.asdict(TDMPC2Config())`; the HalfCheetah baseline uses exactly these):\n")
    out.append(code(json.dumps(dataclasses.asdict(TDMPC2Config()), indent=2), "json"))
    out.append("Override files (content): " + "; ".join(f"`{Path(p).name}` = {json.dumps(json.loads(Path(p).read_text(encoding='utf-8')))}" for p in sorted(glob.glob("configs/overrides/tdmpc2_*.json"))))
    bad = []
    for es, _ in ENVS:
        for t in TAGS:
            f = OVERRIDE[es][t]
            if f.startswith("none"):
                continue
            ov = json.loads(Path(f).read_text(encoding="utf-8"))
            ac = DS.runs_of(es, t)[0]["meta"]["algo_config"]
            for k, v in ov.items():
                if ac.get(k) != v:
                    bad.append((label(es, t), k, v, ac.get(k)))
    out.append("- " + check("every override file listed in the inventory equals the corresponding fields of the logged `algo_config` of its runs", not bad, str(bad) if bad else ""))
    out.append("`ExperimentConfig` defaults (protocol; TD-MPC2 runs use them unchanged, `warmup_steps` = 5000):\n")
    out.append(code(json.dumps(dataclasses.asdict(ExperimentConfig(algo_name="tdmpc2", env_id="HalfCheetah-v5")), indent=2), "json"))
    for es in ("HC", "Ant"):
        r = srt(DS.runs_of(es, "base"))[0]
        out.append(f"\nComplete `metadata.json` of `{r['run_dir']}` ({label(es, 'base')}, seed {r['seed']}):\n")
        out.append(code(json.dumps(r["meta"], indent=2), "json"))
    out.append("\n**2.2 Verbatim code** — `algorithms/tdmpc2.py` (complete, HEAD; identical at every run commit, Section 0):\n")
    out.append(code(tm_src.excerpt(1, len(tm_src.lines)), "python"))
    out.append("`algorithms/tracker_utils.py` is shown in Section 4.2 / 4.1 (`TrackerTask` names tasks `f\"{prefix}_{counter}\"`; `world_model_pretrain` uses counter 0).\n")
    # params
    envd = {}
    for es, env_id in ENVS:
        e = gym.make(env_id)
        envd[es] = (e.observation_space.shape[0], e.action_space.shape[0], getattr(e.spec, "max_episode_steps", None))
        e.close()
    rows = []
    P = {}
    for es, env_id in ENVS:
        o, a, _ = envd[es]
        for nq in (3, 5, 7):
            for ep in ((False, True) if es == "Ant" else (False,)):
                cfg = TDMPC2Config(num_q=nq, episodic=ep)
                wm = WorldModel(o, a, cfg)
                cnt = lambda m: sum(x.numel() for x in m.parameters())  # noqa: E731
                enc, dyn, rew = cnt(wm.encoder), cnt(wm.dynamics), cnt(wm.reward_net)
                term = cnt(wm.termination_net) if wm.termination_net is not None else 0
                pi = cnt(wm.pi_net)
                qs = cnt(wm.qs)
                qt = cnt(wm.qs_target)
                trainable = enc + dyn + rew + term + pi + qs
                P[(es, nq, ep)] = dict(enc=enc, dyn=dyn, rew=rew, term=term, pi=pi, qs=qs, q1=cnt(wm.qs[0]), qt=qt, trainable=trainable)
                used_by_runs = (es == "HC" and not ep) or (es == "Ant" and ep)
                rows.append([env_id, f"{o}/{a}", nq, ep, "yes" if used_by_runs else "(not run)", enc, dyn, rew, term if ep else "— (not built)", pi, qs, cnt(wm.qs[0]), qt, trainable, trainable + qt])
    out.append("\n**2.3 Parameter counts** — real `WorldModel(obs_dim, act_dim, cfg)` instantiated on CPU (weights + biases + LayerNorm parameters; `sum(p.numel())`). Components: encoder (obs → 256 → 512, SimNorm), latent dynamics "
               "(512+act → 512 → 512 → 512, SimNorm), reward model (→ 101 bins), policy prior (→ 2·act), Q-ensemble `qs` (num_q networks, each → 101 bins), `qs_target` (a deep copy of `qs`, not trainable), "
               "termination classifier (512 → 512 → 512 → 1; built only if `episodic`). `trainable` = encoder + dynamics + reward + termination + policy prior + Q-ensemble (the target ensemble is excluded; the world-model optimiser covers "
               "encoder, dynamics, reward, Q-ensemble and termination, the policy prior has its own optimiser). 'used by runs' marks the combinations that appear in `results/` (HC: episodic false; Ant: episodic true):\n")
    out.append(table(["env", "obs/act", "num_q", "episodic", "in results/", "encoder", "dynamics", "reward model", "termination classifier", "policy prior", "Q-ensemble (all)", "per Q member", "target Q-ensemble", "trainable total", "trainable + target"], rows))
    ok = all(envd[es][:2] == (FLOPS_JSON["tdmpc2"][env_id][sorted(FLOPS_JSON["tdmpc2"][env_id])[0]]["obs_dim"], FLOPS_JSON["tdmpc2"][env_id][sorted(FLOPS_JSON["tdmpc2"][env_id])[0]]["act_dim"]) for es, env_id in ENVS)
    out.append("- " + check("local environment dims equal the dims stored in flops_per_call.json (HC 17/6, Ant 105/8)", ok) + f"; `env.spec.max_episode_steps` = {envd['HC'][2]} (HC) / {envd['Ant'][2]} (Ant) → discount = {TDMPC2Config().discount_max} for both (class docstring formula).")
    # planner settings
    rows = []
    for es, t in DS.cfgs():
        ac = srt(DS.runs_of(es, t))[0]["meta"]["algo_config"]
        a_ = envd[es][1]
        rows.append([label(es, t), ac["num_samples"], ac["num_pi_trajs"], ac["num_samples"] - ac["num_pi_trajs"], ac["horizon"], ac["iterations"] + 2 * int(a_ >= 20), ac["num_elites"], ac["temperature"],
                     f"{ac['min_std']}/{ac['max_std']}", ac["num_q"], ac["episodic"], ac["batch_size"], ac["updates_per_env_step"]])
    out.append("\n**2.4 Planner settings per configuration** (logged `algo_config`; `iterations` column = effective `agent.iterations` = `cfg.iterations + 2·(act_dim ≥ 20)` = 6 for both environments):\n")
    out.append(table(["config", "num_samples", "policy-seeded (num_pi_trajs)", "Gaussian-sampled", "horizon", "MPPI iterations", "num_elites", "temperature", "min_std/max_std", "num_q", "episodic", "batch", "UTD"], rows))
    # FLOP constants
    keys = [("rollout_per_env_step (one agent.act())", lambda fk: fk["rollout_per_env_step"]),
            ("gradient_update_critic (world-model step)", lambda fk: fk["gradient_update_critic"]),
            ("gradient_update_actor (policy-prior step)", lambda fk: fk["gradient_update_actor"]),
            ("gradient_update_target (FLOPs; Polyak is elementwise)", lambda fk: fk["gradient_update_target"]),
            ("gradient_update_target_elementwise_ops", lambda fk: fk["gradient_update_target_elementwise_ops"]),
            ("gradient_update_total (critic+actor+target)", lambda fk: fk["gradient_update_total"]),
            ("black-box agent.update() total", lambda fk: fk["_cross_check_blackbox_vs_breakdown"]["blackbox_total"]),
            ("breakdown vs black-box deviation [%]", lambda fk: fk["_cross_check_blackbox_vs_breakdown"]["pct_diff"]),
            ("iterations / horizon / num_q / episodic", lambda fk: f"{fk['iterations']}/{fk['horizon']}/{fk['num_q']}/{fk['episodic']}")]
    out.append("\n**2.5 Per-call FLOPs from `flop_analysis/flops_per_call.json`** (`measure_flops.measure_tdmpc2`: one unmodified `TDMPC2Agent.act(obs, t0=False, eval_mode=False)` for the rollout; one unmodified `agent.update()` black-box "
               f"for the cross-check; the 3-way breakdown replicates the body of `update()` in three `FlopCounterMode` windows; measured on `{FLOPS_JSON['_device_used_for_measurement']}`; matmul FLOPs = 2·m·n·k). Cell `HC ‖ Ant`:\n")
    out.append(flop_constant_table(DS, keys))
    mxdev = max(abs(r["fk"]["_cross_check_blackbox_vs_breakdown"]["pct_diff"]) for r in DS.runs)
    out.append("- " + check("the 3-way breakdown equals the black-box `agent.update()` FLOP count exactly in every signature", mxdev == 0, f"max |deviation| = {mxdev}%"))
    # analytic
    rows = []
    allok = True
    for es, env_id in ENVS:
        o, a, _ = envd[es]
        for t in TAGS:
            r = srt(DS.runs_of(es, t))[0]
            cfg = r["meta"]["algo_config"]
            roll, crit, act, rt, ct, p = analytic(o, a, cfg, cfg["episodic"])
            fk = r["fk"]
            ex = (roll, crit, act) == (fk["rollout_per_env_step"], fk["gradient_update_critic"], fk["gradient_update_actor"])
            allok &= ex
            rows.append([label(es, t), fk["rollout_per_env_step"], roll, fk["gradient_update_critic"], crit, fk["gradient_update_actor"], act, "exact" if ex else "MISMATCH"])
    out.append("\n**2.6 Analytic dense-layer model of the FLOP constants** (derived by this generator, not a pipeline output; Linear layers only, 2·n_in·n_out per sample; formulas below).\n"
               "Let enc, dyn, rew, π, Q, term be the per-sample matmul FLOPs of the encoder, latent dynamics, reward model, policy prior, one Q network and the termination classifier (term = 0 unless episodic); H = horizon, "
               "nq = num_q, B = batch (256), ns = num_samples (512), npi = num_pi_trajs (24), it = iterations (6):\n"
               "```\n"
               "act (rollout, one call) = enc + npi·[(H−1)(π+dyn) + π] + it·ns·[ H(rew + dyn + term) + π + nq·Q ]\n"
               "update critic_update    = H·B·(enc + π + nq·Q)                                  # no-grad: encode(obs[1:]), π(next_z), Q(target, all nq nets)\n"
               "                        + 3·[ B·enc + H·B·dyn + H·B·nq·Q + H·B·rew + H·B·term ] # grad forward + weight-grad + input-grad\n"
               "                        − 2·B·obs_dim·enc_dim                                    # the first encoder layer needs no input-grad\n"
               "update actor_update     = (H+1)·B·( 3π + 2·nq·Q − 2·latent·mlp )                # π fwd + π weight-grad + π input-grad (minus its first layer) + Q fwd + Q input-grad over all nq nets\n"
               "update target_update    = 0 matmul FLOPs; Polyak ops = 2·Σ params(Q-ensemble)\n"
               "```\n"
               "`Q(…, return_type=\"avg\"/\"all\"/\"min\", …)` evaluates **all** nq networks (`torch.stack([net(x) for net in nets])`) and only then selects/averages, so nq enters every Q term linearly. Comparison with the measured constants:\n")
    out.append(table(["config", "rollout measured", "rollout analytic", "critic measured", "critic analytic", "actor measured", "actor analytic", "analytic vs measured"], rows))
    out.append("- " + check("the analytic model equals the FlopCounterMode counts exactly for rollout, critic and actor in all 10 (env, configuration) entries", allok))
    t_calls = call_count_table(DS)
    out.append("\n**2.7 Call counts and where they are defined** (`compute_energy_per_flop.rows_for_tdmpc2`, called from `main()`; verbatim):\n")
    a, b, blk = cepf.block(r"^def rows_for_tdmpc2", r"^def parse_args")
    out.append(blk)
    out.append("```\n"
               "n_env     = max(1, train_steps // steps_per_epoch) * steps_per_epoch    # = 100000 rollout calls (one agent.act() per env step)\n"
               "n_upd     = n_env * updates_per_env_step                                # = 100000: buffer_sample = critic_update = actor_update = target_update calls\n"
               "pretrain  = warmup_steps                                                # = 5000 agent.update() calls in the world_model_pretrain task\n"
               "F_rollout = rollout_per_env_step * n_env;  F_pretrain = gradient_update_total * warmup_steps\n"
               "F_critic  = gradient_update_critic * n_upd;  F_actor = gradient_update_actor * n_upd;  OPS_target = gradient_update_target_elementwise_ops * n_upd\n"
               "```\n"
               "The loop issuing the calls is `train()` (`algorithms/tdmpc2.py`): `for _ in range(warmup_steps): agent.update(...)` inside `TrackerTask(tracker, \"world_model_pretrain\", 0, …)`, and per epoch "
               "`n_updates = steps_per_epoch * updates_per_env_step` calls of `agent.update` inside `gradient_updates_{epoch}`; unlike SAC/TD3/MBPO there is **no** `len(buffer) < batch_size` guard (`update()` returns early with all-zero timers only if `buffer.sample` returns `None`, i.e. no stored episode is at least `horizon` long). Call counts per run (identical across seeds):\n")
    out.append(t_calls)
    pr = pd.read_csv("flop_analysis/output/per_run_energy_per_flop.csv")
    ok = True
    for r in DS.runs:
        sub = pr[pr.run_dir == r["run_dir"]].set_index("segment")
        for s in ("rollout", "world_model_pretrain", "buffer_sample", "critic_update", "actor_update", "target_update"):
            ok &= int(sub.loc[s, "call_count"]) == r["calls"][s]
    out.append("- " + check(f"call counts equal the `call_count` column of per_run_energy_per_flop.csv for all {len(DS.runs)} runs", ok))
    bad = []
    for r in DS.runs:
        m = re.search(r"Pretrain burst done: (\d+) updates", r["log"])
        if not m or int(m.group(1)) != r["calls"]["world_model_pretrain"]:
            bad.append(r["run_dir"])
    out.append("- " + check("`run.log` reports `Pretrain burst done: 5000 updates` in every run (the only logged update count)", not bad, str(bad[:3]) if bad else ""))
    t1, t2 = total_flops_table(DS)
    out.append("\n**2.8 Total FLOPs per run and segment** (`wm_pretrain` = world_model_pretrain; `GU` = critic + actor; `TOTAL` = rollout + wm_pretrain + critic + actor; `target` = elementwise Polyak operations):\n")
    out.append(t1)
    out.append("Ant / HalfCheetah ratios (Ant uses `episodic` = true, so its constants include the termination classifier, Section 6.6):\n")
    out.append(t2)
    return "\n".join(out)


# =============================================================================== Section 6
def section6():
    out = []
    envd = {"HC": (17, 6), "Ant": (105, 8)}
    L = {k: tm_src.find(p) for k, p in {"wm": r"^class WorldModel", "q": r"    def Q\(self", "buf": r"^class _EpisodeSequenceBuffer", "sample": r"    def sample\(self, batch_size", "agent": r"^class TDMPC2Agent",
                                       "estimate": r"def _estimate_value", "act": r"    def act\(self", "pi": r"    def _update_pi", "update": r"    def update\(self", "bs0": r"batch = buffer.sample\(batch_size, cfg.horizon\)", "bs1": r"info.buffer_sample_s =",
                                       "cu0": r"# --- critic_update: world model", "cu1": r"info.critic_update_s =", "au0": r"# --- actor_update: policy prior", "au1": r"info.actor_update_s =", "tu0": r"# --- target_update: Polyak",
                                       "tu1": r"info.target_update_s =", "train": r"^def train\(", "pre": r'TrackerTask\(tracker, "world_model_pretrain"', "roll": r'TrackerTask\(tracker, "rollout"', "gu": r'TrackerTask\(tracker, "gradient_updates"',
                                       "recon": r"# -+ reconcile", "term_plan": r"termination = torch.clip", "term_loss": r"termination_loss = F.binary_cross_entropy", "td": r"td_targets = reward", "end_ep": r"buffer.end_episode\(\)",
                                       "act_call": r"agent.act\(obs, t0=is_t0", "inc": r"for _ in range\(n_updates\)", "none": r"if batch is None", "iters": r"self.iterations = cfg.iterations", "elig": r"eligible = \[ep", "pretrain_loop": r"for _ in range\(exp_cfg.warmup_steps\):\s*$"}.items()}
    out.append("**6.1 Code facts** (`algorithms/tdmpc2.py`, `configs/config.py`, `configs/overrides/tdmpc2_*.json`; HEAD, identical at all run commits):\n")
    out.append(f"""
**(a) Parameter counts** of every component per environment and for num_q = 3, 5, 7: Section 2.3 (the termination classifier exists only for `episodic` = true, i.e. on the Ant runs).

**(b) Planner settings per configuration:** Section 2.4 (512 samples of which 24 are seeded from the policy prior, horizon 3 in the baseline, 6 iterations (`self.iterations`, line {L['iters']}), 64 elites, temperature 0.5; the horizon sweep changes the horizon, the num_q sweep the Q-ensemble size; samples/iterations/elites are identical in all 50 runs).

**(c) What the `rollout` task contains** (`train()`, `TrackerTask(tracker, "rollout", epoch, …)`, line {L['roll']}): per env step: `agent.act(obs, t0=is_t0, eval_mode=False)` (line {L['act_call']}) = full MPPI planning in latent space (`act`, line {L['act']}): encode the observation, seed `num_pi_trajs` trajectories from the policy prior (π → dynamics for horizon−1 steps), then `iterations` MPPI iterations each of which samples the remaining trajectories from the Gaussian search distribution, scores all 512 trajectories with `_estimate_value` (line {L['estimate']}: reward model + latent dynamics per horizon step, [termination classifier if episodic], terminal π and **average over all num_q Q-networks**), takes the top-`num_elites`, refits mean/std; then one trajectory is picked by a Gumbel-softmax sample over the elite scores, noise `std·N(0,1)` is added, the action is clipped and `.cpu().numpy()` (a device sync every env step). Then `env.step` (MuJoCo), `buffer.add` (host appends to the open episode), reward bookkeeping and, on episode end, `buffer.end_episode()` (line {L['end_ep']}: `np.stack` of the whole episode into arrays, eviction of the oldest episodes if capacity is exceeded), the episode record, `env.reset()` and `buffer.start_episode`. Episode resets are therefore inside the `rollout` task. No evaluation episodes exist.

**(d) What `buffer_sample` does and what its timer covers** (`TDMPC2Agent.update`, lines {L['bs0']}–{L['bs1']}): `_EpisodeSequenceBuffer.sample(batch_size, horizon)` (line {L['sample']}): builds `eligible = [ep for ep in self._episodes if length ≥ horizon]` (a Python loop over **all stored closed episodes** on every call, line {L['elig']}), weights ∝ (length − horizon + 1), `np.random.choice` of `batch_size` episodes, then a **host-side Python loop over the 256 batch elements**, each slicing `horizon + 1` observations and `horizon` actions/rewards/terminations out of the episode arrays into preallocated arrays (sequence length = horizon + 1 observations, horizon transitions); then four `torch.as_tensor(..., device=cuda)` host→device copies. The timer covers exactly the sampling and these four copies; the open (current) episode is never sampled. The cost per call grows with the number of stored episodes (HC: 5 warmup + 100 training episodes ≈ 105 at the end; Ant: more, Section 5.3).

**(e) What `critic_update`, `actor_update` and `target_update` contain here** (confirming the mapping in README/CLAUDE.md against the code):
- `critic_update` (lines {L['cu0']}–{L['cu1']}) = the **world-model step**: no-grad real next-observation encodings `next_z = encode(obs[1:])`, policy-prior action at `next_z`, `target_q = min` over two randomly chosen target Q-networks (line {L['td']}: `td_targets = reward + discount·(1 − terminated)·target_q`, using the env's own `terminated` flag whether or not `episodic`); then the encoder on `obs[0]`, `horizon` unrolled latent-dynamics steps with the consistency (MSE) loss against `next_z` (ρ^t weighted), all num_q Q-networks and the reward model on the unrolled latents, soft two-hot cross-entropy reward and value losses, [termination BCE loss if `episodic`, line {L['term_loss']}], one weighted loss, backward, gradient-norm clipping, Adam step (encoder LR × 0.3). Dropout 0.01 is active (model in `train()` mode).
- `actor_update` (lines {L['au0']}–{L['au1']}) = `_update_pi` (line {L['pi']}) on `zs.detach()`: Q-ensemble frozen, π on all (H+1) latents, `Q(…, "avg")` over all num_q networks, `RunningScale` update, entropy-regularised loss, backward, clip, π Adam step. It is the policy-prior step (TD-MPC2's closest analogue of SAC's actor update).
- `target_update` (lines {L['tu0']}–{L['tu1']}) = Polyak averaging (τ = 0.01) of the Q-ensemble only (`qs` → `qs_target`): 2 elementwise ops per Q parameter, `gradient_update_target_elementwise_ops` = 2 × Σ params(`qs`).
- The mapping stated in the documents (critic = world-model step, actor = policy-prior step, target = Q-ensemble Polyak) **agrees with the code**. The encoder, dynamics, reward and termination networks are updated inside `critic_update` (the world-model optimiser), the policy prior inside `actor_update`.

**(f) Which FLOP quantities depend on `num_q` and `horizon`, and how** — from the exact analytic model of Section 2.6 (verified against every measured constant):
- *planning per call* (`rollout_per_env_step`): the planning cost is `enc + npi·[(H−1)(π+dyn)+π] + it·ns·[H(rew+dyn+term) + π + nq·Q]`: **linear in H** through the per-step reward + dynamics (+ termination) evaluations and the policy seeding; **linear in nq** through the terminal-value term `it·ns·nq·Q` (each Q-network has the same width as the reward/dynamics heads; the share of every term is in the decomposition table below);
- *world-model step* (`critic_update`): linear in H in all terms; linear in nq through the target-Q evaluation (all nq target nets, no-grad) and the Q forward/backward on the unrolled latents (3× forward cost);
- *policy-prior step* (`actor_update`): ∝ (H+1) (it runs on all H+1 latents) and linear in nq (all nets are evaluated for the average and back-propagated through);
- *Polyak operations*: ∝ nq (2 · Σ params(`qs`)), independent of H;
- the number of calls (n_env, n_upd, pretrain) does not depend on either parameter.
The per-call values and the factors against the baseline for every configuration are in Sections 2.5 and 4.1–4.2 (FLOP-factor tables).
""")
    # decomposition table
    rows = []
    for es, env_id in ENVS:
        o, a = envd[es]
        for t in TAGS:
            cfg = srt(DS.runs_of(es, t))[0]["meta"]["algo_config"]
            roll, crit, act, rt, ct, p = analytic(o, a, cfg, cfg["episodic"])
            rows.append([label(es, t), g6(roll)] + [f"{g6(v)} ({g6(100 * v / roll)} %)" for v in rt.values()])
    out.append("Decomposition of one planning call (`rollout_per_env_step`) into the terms of the analytic model (FLOPs and share of the call):\n")
    out.append(table(["config", "total"] + list(rt.keys()), rows))
    rows = []
    for es, env_id in ENVS:
        o, a = envd[es]
        for t in TAGS:
            cfg = srt(DS.runs_of(es, t))[0]["meta"]["algo_config"]
            roll, crit, act, rt, ct, p = analytic(o, a, cfg, cfg["episodic"])
            base_cfg = srt(DS.runs_of(es, "base"))[0]["meta"]["algo_config"]
            br, bc, ba, _, _, _ = analytic(o, a, base_cfg, base_cfg["episodic"])
            fk = srt(DS.runs_of(es, t))[0]["fk"]
            bfk = srt(DS.runs_of(es, "base"))[0]["fk"]
            rows.append([label(es, t), g6(fk["rollout_per_env_step"] / bfk["rollout_per_env_step"]), g6(fk["gradient_update_critic"] / bfk["gradient_update_critic"]),
                         g6(fk["gradient_update_actor"] / bfk["gradient_update_actor"]), g6(fk["gradient_update_total"] / bfk["gradient_update_total"]),
                         g6(fk["gradient_update_target_elementwise_ops"] / bfk["gradient_update_target_elementwise_ops"])])
    out.append("\nFactor of the per-call constants against the same environment's baseline (`flops_per_call.json`):\n")
    out.append(table(["config", "rollout (planning)", "critic (world-model step)", "actor (policy-prior step)", "update total", "Polyak ops"], rows))
    # 6.2 call counts
    out.append("\n**6.2 Call counts per run** (formulas and implementation in Section 2.7): rollout calls = 100000; update calls = `steps_per_epoch × updates_per_env_step × 100` = 100000 in every configuration (UTD 1; buffer_sample = critic = actor = target calls); "
               "pretraining updates = `warmup_steps` = 5000. Per configuration table: Section 2.7.")
    # 6.3 world_model_pretrain
    out.append("\n**6.3 `world_model_pretrain`** — a directly measured CodeCarbon task `world_model_pretrain_0`: a one-off burst of `warmup_steps` = 5000 `agent.update()` calls on the warmup data, right after warmup and before epoch 0.\n")
    rows = []
    for es, t in DS.cfgs():
        rs = srt(DS.runs_of(es, t))
        rows.append([label(es, t), msd(vals(rs, "E", "world_model_pretrain")), msd(vals(rs, "D", "world_model_pretrain")), msd(vals(rs, "P", "world_model_pretrain")), g6(rs[0]["F"]["world_model_pretrain"]),
                     g6(rs[0]["fk"]["gradient_update_total"]), jpf_cell(rs, "world_model_pretrain"), msd(vals(rs, "share", "world_model_pretrain")), rs[0]["calls"]["world_model_pretrain"]])
    out.append(table(["config", "energy [J]", "duration [s]", "mean power [W]", "FLOPs (5000 × update total)", "per-update FLOPs", "J/FLOP (ratio of means ± sd)", "share of TOTAL [%]", "updates"], rows))
    out.append(f"""
- *Timers:* the sub-timers (`buffer_sample`, `critic_update`, `actor_update`, `target_update`) of these 5000 updates are **discarded**: `train()` (lines {L['pre']}–{L['pre'] + 6}) keeps only `info.total_loss` of each pretraining update and does not add the `UpdateInfo` times to `sub_time_totals`; consequently the four allocated sub-segments and their wall-clock shares in `segment_energy.json` cover **only the steady-state epochs**, and the pretraining energy is a separate fused segment that is **not time-split**.
- *How it enters the training total:* `world_model_pretrain` is in `TRAINING_SEGMENTS` of `compute_energy_per_flop.py`; its energy is part of `TOTAL_MEASURED_TRAINING` (numerator), its FLOPs (`gradient_update_total × warmup_steps`) part of the matmul denominator; its duration is part of the TOTAL duration. It is not part of `gradient_updates`, and `warmup` (random actions) remains excluded.
- *Dependence on the sweeps:* the number of updates (5000) and the data (warmup episodes) do not change across the `num_q` / `horizon` configurations; its FLOPs change because the per-update FLOPs change (Section 6.1f factors); its energy per configuration is in the table above.
- *Update count evidence:* `run.log` of every run states `Pretrain burst done: 5000 updates` (checked, Section 2.7); no per-update count is stored elsewhere.
""")
    # 6.4 rollout dominant
    out.append("**6.4 Rollout as the dominant segment** — data only. Per configuration: rollout energy per env step (J, = rollout energy / 100000), planning FLOPs per call, rollout task duration (Σ over the 100 tasks; per-task statistics and the count of tasks < 1 s in Section 5.2a) and the share/power relations (full tables in Sections 3–4):\n")
    rows = []
    for es, t in DS.cfgs():
        rs = srt(DS.runs_of(es, t))
        sub = pe[(pe.env == dict(ENVS)[es]) & (pe.tag == t) & (pe.task == "rollout")].duration_s
        rows.append([label(es, t), msd([r["E"]["rollout"] / 100000 for r in rs]), g6(rs[0]["fk"]["rollout_per_env_step"]), msd(vals(rs, "D", "rollout")), f"{g6(sub.median())} [{g6(sub.min())}–{g6(sub.max())}]",
                     f"{int((sub < 1).sum())}/{len(sub)}", msd(vals(rs, "share", "rollout")), msd(vals(rs, "P", "rollout")), msd(vals(rs, "P", "gradient_updates")),
                     msd([r["P"]["rollout"] / r["P"]["gradient_updates"] for r in rs]), msd([100 * r["hw"]["rollout"]["gpu"] / r["hw"]["rollout"]["E"] for r in rs])])
    out.append(table(["config", "rollout energy per env step [J]", "planning FLOPs per call", "rollout task duration, Σ of 100 tasks [s]", "per-task median [min–max] [s]", "tasks < 1 s", "rollout share of TOTAL [%]", "P rollout [W]", "P GU [W]",
                      "P_rollout / P_GU", "GPU share of rollout energy [%]"], rows))
    out.append("Hardware composition (CPU/GPU/RAM) of the rollout task for every configuration: the 'Hardware composition of the measured `rollout` task' table in Sections 3–4.")
    # 6.5
    out.append("\n**6.5 Sweeps** — the common block for `num_q` {3,5,7} and `horizon` {1,3,5} on both environments is Section 4.1 and 4.2 (the baseline 5/3 is shared by both sweeps and appears in both sections and in Section 3). "
               "FLOP factors per segment against the baseline are in the 'FLOP factors' table of each sweep section and in the factor table of Section 6.1f.")
    # 6.6 env confound
    out.append("\n**6.6 Environment confound — complete diff of the logged configurations of the HC and the Ant baseline** (`metadata.json`, seed 331 of each; every field that differs):\n")
    mh, ma = srt(DS.runs_of("HC", "base"))[0]["meta"], srt(DS.runs_of("Ant", "base"))[0]["meta"]

    def flat(d, pfx=""):
        o = {}
        for k, v in d.items():
            if isinstance(v, dict):
                o.update(flat(v, pfx + k + "."))
            else:
                o[pfx + k] = v
        return o
    fh, fa = flat({"algo_config": mh["algo_config"], "experiment_config": mh["experiment_config"]}), flat({"algo_config": ma["algo_config"], "experiment_config": ma["experiment_config"]})
    rows = [[f"`{k}`", json.dumps(fh.get(k)), json.dumps(fa.get(k))] for k in sorted(set(fh) | set(fa)) if fh.get(k) != fa.get(k) and k != "experiment_config.seed"]
    out.append(table(["field", "HalfCheetah-v5 baseline", "Ant-v5 baseline"], rows))
    out.append("- " + check("`algo_config.episodic` is the only logged `algo_config` field that differs between the HC and the Ant baseline (expected by the instructions); the only other differing field is `experiment_config.env_id`",
                            [r_[0] for r_ in rows] == ["`algo_config.episodic`", "`experiment_config.env_id`"], str([r_[0] for r_ in rows])))
    # termination classifier
    import torch
    from algorithms.tdmpc2 import WorldModel
    cfgs = {nq: TDMPC2Config(num_q=nq, episodic=True) for nq in (3, 5, 7)}
    term_params = {nq: sum(x.numel() for x in WorldModel(105, 8, cfgs[nq]).termination_net.parameters()) for nq in (3, 5, 7)}
    trainable5 = sum(x.numel() for n, x in WorldModel(105, 8, cfgs[5]).named_parameters() if not n.startswith("qs_target"))
    out.append(f"\n**Termination classifier (Ant only, `episodic` = true).** Parameters: {term_params[5]} (independent of num_q; MLP 512 → 512 → 512 → 1) = {g6(100 * term_params[5] / trainable5)} % of the {trainable5} trainable parameters of the Ant world model (num_q 5).")
    out.append(f"- *Where it is used:* (i) **planning** — `_estimate_value` (line {L['estimate']}) accumulates `termination = clip(termination + (termination(z) > 0.5), max=1)` after every imagined step (line {L['term_plan']}) and masks the later rewards and the terminal value, so it is evaluated `iterations × num_samples × horizon` times per `act()` call; (ii) **world-model update** — in `critic_update`, `termination_loss = BCE(termination(zs[1:]), terminated)` (line {L['term_loss']}) is added to the loss with coefficient `termination_coef` = 1.0 and back-propagated; "
               f"(iii) it is **not** used in `_update_pi` and **not** in the TD-target (`td_targets` bootstraps from the env's own `terminated` flag, line {L['td']}).")
    # FLOPs of the classifier: analytic split + direct measurement
    from measure_flops import measure_tdmpc2 as _measure
    rows = []
    for es, env_id in [("Ant", "Ant-v5")]:
        o, a = envd[es]
        for t in TAGS:
            cfg = dict(srt(DS.runs_of(es, t))[0]["meta"]["algo_config"])
            roll, crit, act, rt, ct, p = analytic(o, a, cfg, True)
            cfg0 = dict(cfg)
            cfg0["episodic"] = False
            roll0, crit0, act0, _, _, _ = analytic(o, a, cfg0, False)
            m1 = _measure(o, a, 1000, cfg)
            m0 = _measure(o, a, 1000, cfg0)
            ok = (m1["rollout_per_env_step"], m1["gradient_update_critic"]) == (srt(DS.runs_of(es, t))[0]["fk"]["rollout_per_env_step"], srt(DS.runs_of(es, t))[0]["fk"]["gradient_update_critic"])
            d_roll, d_crit, d_act = m1["rollout_per_env_step"] - m0["rollout_per_env_step"], m1["gradient_update_critic"] - m0["gradient_update_critic"], m1["gradient_update_actor"] - m0["gradient_update_actor"]
            exact = (d_roll, d_crit, d_act) == (roll - roll0, crit - crit0, act - act0)
            rows.append([label(es, t), g6(d_roll), g6(100 * d_roll / m1["rollout_per_env_step"]), g6(d_crit), g6(100 * d_crit / m1["gradient_update_critic"]), g6(d_act),
                         g6(5000 * (d_crit + d_act)), g6(d_roll * 100000 + 100000 * (d_crit + d_act) + 5000 * (d_crit + d_act)), "yes" if ok and exact else "NO"])
    out.append("- *FLOPs attributable to the classifier* — separable: the episodic and non-episodic `TDMPC2Agent` can be built and measured with the same real classes at the Ant dimensions. Generator computation (not a pipeline output): `measure_flops.measure_tdmpc2` called with the Ant dimensions (105/8) and `episodic` = true and = false for each configuration, difference = classifier share; it equals the analytic term of Section 2.6 exactly (last column) and the `episodic` = true values reproduce `flops_per_call.json` exactly:\n")
    out.append(table(["Ant config", "Δ planning FLOPs per call", "Δ as share of the call [%]", "Δ critic (world-model step) FLOPs per update", "Δ as share of the update step [%]", "Δ actor FLOPs per update", "Δ over the 5000 pretrain updates",
                      "Δ over a whole run (100000 acts + 100000 updates + 5000 pretrain updates)", "equals analytic term and `flops_per_call.json`"], rows))
    out.append("The classifier's contribution to the parameter count, the planning FLOPs and the update FLOPs is therefore known exactly; its **energy** is not separable (it runs fused inside the measured `rollout` and `critic_update`/`gradient_updates` tasks).")
    na("Energy attributable to the termination classifier (inside the measured `rollout` and `gradient_updates` tasks; no run with `episodic` = false exists on Ant-v5).")
    ant = [r for r in DS.runs if r["es"] == "Ant"]
    out.append("\n*Completed training episodes per run on Ant-v5* (`training_metrics.json`, phase `train`; HC always 100): " + "; ".join(
        f"{t}: {min(sum(1 for e in r['tm']['episodes'] if e['phase'] == 'train') for r in DS.runs_of('Ant', t))}–{max(sum(1 for e in r['tm']['episodes'] if e['phase'] == 'train') for r in DS.runs_of('Ant', t))}" for t in TAGS)
        + f"; warmup episodes per run: {sorted({sum(1 for e in r['tm']['episodes'] if e['phase'] == 'warmup') for r in ant})} (Ant) / {sorted({sum(1 for e in r['tm']['episodes'] if e['phase'] == 'warmup') for r in DS.runs if r['es'] == 'HC'})} (HC). Per-configuration return statistics: Section 5.3.")
    # 6.7 open points
    allr = pe[pe.task == "rollout"].duration_s
    allg = pe[pe.task == "gradient_updates"].duration_s
    out.append("\n**6.7 Open points from the instructions (reported, not resolved):**\n")
    out.append(f"- *`rollout` task durations versus the 1 s polling interval:* {int((allr < 1).sum())} of {len(allr)} `rollout` tasks are shorter than 1 s (min/median/max {g6(allr.min())}/{g6(allr.median())}/{g6(allr.max())} s) — unlike SAC/TD3/MBPO, the TD-MPC2 rollout tasks are long because planning dominates; "
               f"{int((allg < 1).sum())} of {len(allg)} `gradient_updates` tasks are < 1 s (min/median/max {g6(allg.min())}/{g6(allg.median())}/{g6(allg.max())} s). `world_model_pretrain` (one task) lasts {msd([r['hw']['world_model_pretrain']['duration'] for r in DS.runs])} s. Per configuration: Section 5.2a.")
    out.append("- *Host-side work counted as FLOPs:* none; only matmul FLOPs (`FlopCounterMode`). **Counted:** `rollout` = everything inside one `agent.act()` that is a matmul (encoder, π, dynamics, reward, Q-networks, termination classifier) — the MPPI bookkeeping (`randn`, `clamp`, `topk`, `exp`, score normalisation, Gumbel-softmax, `index_select`, `.cpu()`) is not; "
               "`critic_update` / `actor_update` = matmuls of the world-model step / `_update_pi` incl. backward (soft two-hot, symlog, MSE/BCE losses, SimNorm/softmax, LayerNorm, Mish, dropout, grad clipping, Adam, `RunningScale` percentile sort are not); `buffer_sample` = 0 (host Python loop + four H2D copies); `target_update` = Polyak operation count; "
               "`world_model_pretrain` = 5000 × update total. Not counted: `env.step`, `buffer.add`, `end_episode` (`np.stack`), episode resets.")
    out.append("- *Segments whose timers or FLOPs are produced differently from SAC:* (i) `world_model_pretrain` exists only here (own measured task; sub-timers discarded, not time-split); (ii) `critic_update` is the world-model step and `actor_update` the policy-prior step (different quantities from SAC's critic/actor); "
               "(iii) the `rollout` is dominated by planning and its FLOPs come from one measured `agent.act()` call; (iv) `buffer_sample` is a host-side per-episode Python sampler (cost grows with the number of episodes) rather than a single NumPy gather; "
               "(v) the update counter is not guarded by `len(buffer) < batch_size`; (vi) the sub-timers' coverage of the measured GU task is reported in Sections 3–4.")
    return "\n".join(out)


# =============================================================================== assemble
def main():
    spec = DS.spec
    MD = []
    P = MD.append
    recon_txt, recon_stats = recon(DS)
    prov = provenance(DS, ["build_section_4_4.py (in Section_4.4_data/)", "s4_data.py", "s4_blocks.py", "s4_audit.py (in Section_4.2_data/)"],
                      ["algorithms/tdmpc2.py", "algorithms/tracker_utils.py"])
    inv_keys = ["num_q", "horizon", "episodic", "num_samples", "num_pi_trajs", "iterations", "num_elites", "batch_size", "updates_per_env_step"]
    aud = audit(DS, inv_keys, [])
    exec_txt, _ = exec_facts(DS, ["results/tdmpc2/_seed_sweep_status.json", "results/tdmpc2/_numq_horizon_sweep_status.json"],
                             ["`scripts/run_tdmpc2_seed_sweep.sh` (baselines, 10 runs; Ant with `configs/overrides/tdmpc2_ant.json`)", "`scripts/run_tdmpc2_numq_horizon_sweep.sh` (num_q ∈ {3,7}, horizon ∈ {1,5}, both envs, 40 runs; false start documented below)"])
    sec2 = section2()
    sec6 = section6()
    blocks = {"baseline": common_block(DS, ["base"], with_response=False, with_env=True)}
    for name, text, tags, xd in spec.sweeps:
        blocks[name] = common_block(DS, tags, with_response=True, with_env=True, base_tag="base")
    ols = ols_block(DS, OUT_DIR, PREFIX)
    dlt = delta_block(DS, OUT_DIR, PREFIX, [("tdmpc2_num_q", "num_q"), ("tdmpc2_horizon", "horizon")])
    qual = quality_block(DS, pe, IDLE, OUT_DIR, PREFIX)
    learn = learning_block(DS, OUT_DIR, PREFIX)
    common_section7_items(DS, IDLE)
    fs = [o for o in block_outcomes("tdmpc2") if o["outcome"] != "Finished OK"]
    failed = [o for o in fs if o["outcome"].startswith("FAILED")]
    unfinished = [o for o in fs if o["outcome"] == "no closing line"]
    disc("CLAUDE.md / README.md describe the TD-MPC2 num_q/horizon sweep's false start as 'the first 5 runs (num_q=3, HalfCheetah-v5, all 5 seeds)' crashing instantly with `FileNotFoundError: configs/overrides/tdmpc2_num_q3.json` "
         "before any run directory/tracker existed, followed by a clean relaunch. `results/tdmpc2/_numq_horizon_sweep.log` agrees on the 5 crashes "
         f"({len(failed)} `FAILED (exit 1)` blocks, indices {', '.join(o['index'] for o in failed)}, 12:18:39–12:18:42) and additionally contains a **sixth launch** with no closing line "
         f"({', '.join(o['start_line'][:60] for o in unfinished)}, started 12:18:42, followed directly by the relaunch's 'Priming sudo credentials' line at 12:21:00): the first attempt was stopped during that launch. "
         "No run directory was created by any of the six launches (all 50 analysed runs belong to the second attempt: `results/tdmpc2/_numq_horizon_sweep_status.json` 40/40 done, 5 + 5 + 5 + 5 + 5 runs per environment on disk), "
         "so nothing needs to be excluded, but the documents do not mention the sixth, interrupted launch. Whether it touched the GPU state (clock lock / persistence mode set by the runner before being killed) cannot be told from the log; "
         "the first run of the relaunch recorded a normal thermal-gate pass (Section 1.4).")
    question("The `rollout` energy includes the termination-classifier evaluations on Ant (planning) while HC has none; the FLOP share is known exactly (Section 6.6) but the energy share is not separable — whether to report Ant-vs-HC TD-MPC2 differences with or without this caveat is for the author to decide.")
    n_pass = sum(1 for _, ok, _ in CHECKS if ok)
    n_fail = len(CHECKS) - n_pass
    P("# Section 4.4 TD-MPC2 — context\n")
    P(f"Data context for thesis Section 4.4 (TD-MPC2): {len(DS.runs)} canonical runs (5 configurations × 2 environments × 5 seeds {CANON}). Facts, numbers and provenance only — no interpretation, no LaTeX. All energies are **gross** (idle floor included, nothing subtracted); "
      "mean ± sample sd (ddof = 1) over the five seeds; 6 significant digits; every table cell shows `HalfCheetah-v5 ‖ Ant-v5` (left ‖ right) unless a table says otherwise. Conventions of Section 4.1 apply (`measured` vs `allocated` segments; `target_update` in nJ per Polyak operation, never J/FLOP; "
      "`buffer_sample` has no FLOPs; J/FLOP = ratio of means with the sd of the five per-seed ratios (FLOPs are identical across seeds, so the mean of the per-seed ratios equals the ratio of means); relative changes computed per seed then averaged; environment comparison paired by seed, Ant relative to HC). "
      "Training total = `TOTAL_MEASURED_TRAINING` = rollout + world_model_pretrain + gradient_updates (critic = world-model step, actor = policy-prior step, target = Q-ensemble Polyak, buffer_sample); `warmup` (random actions) and the idle windows are excluded.\n")
    P(f"Checks run by the generator: {len(CHECKS)}; passed {n_pass}; failed {n_fail}. Independent recomputation vs the pipeline CSVs: {recon_stats['n_compared']} values compared, {recon_stats['n_fail']} deviating by more than 1e-9, maximum relative deviation {recon_stats['max_dev']:.3e}.\n")
    P("**Configuration tags:** `base` = TD-MPC2 paper hyperparameters (num_q 5, horizon 3, 512 samples of which 24 policy-seeded, 6 iterations, 64 elites, batch 256, UTD 1, warmup/pretrain 5000); HC baseline = `TDMPC2Config` defaults, Ant baseline = defaults + `configs/overrides/tdmpc2_ant.json` (`episodic` = true). "
      "`numq3`/`numq7` change `num_q` (5 → 3 / 7), `hor1`/`hor5` change `horizon` (3 → 1 / 5), each one at a time around the shared baseline. Abbreviations: HC = HalfCheetah-v5, Ant = Ant-v5, GU = `gradient_updates`, TOTAL = `TOTAL_MEASURED_TRAINING`, wm_pretrain = `world_model_pretrain`, "
      "buf_sample = `buffer_sample`, critic = `critic_update` (world-model step), actor = `actor_update` (policy-prior step), target = `target_update`.\n")
    P("## 0. Provenance\n")
    P(prov)
    P("\n## 1. Data basis and audit\n")
    P(aud)
    P("\n" + exec_txt)
    P("\n" + recon_txt)
    P("\n## 2. Structural facts from code and FLOP files\n")
    P(sec2)
    P("\n## 3. Baseline decomposition and energy per FLOP (4.4.1)\n")
    P("Configuration `base` (num_q 5, horizon 3; shared by both sweeps).\n")
    P(blocks["baseline"])
    for i, (name, text, tags, xd) in enumerate(spec.sweeps, 1):
        P(f"\n## 4.{i} Sweep: {name} — {text}\n")
        P(f"Configurations: {', '.join(f'`{t}` ({spec.tag_desc[t]})' for t in tags)}. Baseline for the response tables: `base` of the same environment (shared by both sweeps).\n")
        P(blocks[name])
    P("\n## 5. Cross-configuration quantities\n")
    P(ols)
    P("\n" + dlt)
    P("\n" + qual)
    P("\n" + learn)
    P("\n## 6. Algorithm-specific items (TD-MPC2)\n")
    P(sec6)
    P("\n## 7. Discrepancies, NOT AVAILABLE items, questions\n")
    P("**(a) Where data or code contradict CLAUDE.md, README.md or `thesis-layout.md` (the latter is not present in the repository):**\n")
    P("\n".join(f"- {d}" for d in DISCREPANCIES) if DISCREPANCIES else "- none found by the generator's checks")
    P("- `thesis-layout.md` is referenced by the instructions but does not exist in the repository; the layout expectation (50 TD-MPC2 runs: 5 configurations × 2 envs × 5 seeds) was taken from the instruction file and verified against `results/`.")
    P("\n**(b) Quantities that could not be obtained (`NOT AVAILABLE`):**\n")
    P("\n".join(f"- NOT AVAILABLE: {d}" for d in NOT_AVAILABLE) if NOT_AVAILABLE else "- none")
    P("\n**(c) Questions for the author that the repository cannot answer:**\n")
    P("\n".join(f"- {d}" for d in QUESTIONS) if QUESTIONS else "- none")
    P("\n**Index of produced files** (`contexts/Section_4.4_data/`): `tdmpc2_segments_long.csv`, `tdmpc2_cross_seed_summary.csv`, `tdmpc2_run_inventory.csv`, `tdmpc2_per_epoch.csv`, `tdmpc2_idle_floor.csv`, `tdmpc2_ols_fits.csv`, "
      "`tdmpc2_delta_pairs.csv`, `tdmpc2_mad_flags.csv`, `tdmpc2_learning_performance.csv`; scripts `build_section_4_4.py` (here) and the shared modules in `contexts/Section_4.2_data/`.")
    OUT_MD.write_text("\n".join(MD) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD} ({len(MD)} blocks); checks {len(CHECKS)} pass {n_pass} fail {n_fail}")
    for n_, ok, d in CHECKS:
        if not ok:
            print("FAIL:", n_, d)


if __name__ == "__main__":
    main()
