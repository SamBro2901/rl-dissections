"""
Generator for contexts/Section_4.2.md (TD3) + companion CSVs in contexts/Section_4.2_data/.
Read-only on results/, flop_analysis/ and git; writes only under contexts/.
Run from the repo root:  .venv/Scripts/python.exe contexts/Section_4.2_data/build_section_4_2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s4_audit import *  # noqa: F401,F403,E402
from s4_audit import Src, versus_rows, srt  # noqa: E402
from configs.config import TD3Config, SACConfig  # noqa: E402

OUT_MD = REPO / "contexts" / "Section_4.2.md"
OUT_DIR = REPO / "contexts" / "Section_4.2_data"
PREFIX = "td3"

TAGS = ["base", "utd2", "utd4", "w256", "w512", "b256", "b512", "b1024"]
SPEC = Spec(
    algo="td3", section="4.2", name="TD3", config_cls=TD3Config, tags=TAGS,
    tag_desc={"base": "hidden (1024,1024), batch 100 (paper default), UTD 1, policy delay 2",
              "utd2": "UTD 2", "utd4": "UTD 4", "w256": "hidden (256,256)", "w512": "hidden (512,512)",
              "b256": "batch 256 (SAC/MBPO default; the batch-matched comparison point)", "b512": "batch 512", "b1024": "batch 1024"},
    tag_factor={"base": {}, "utd2": {"updates_per_env_step": 2}, "utd4": {"updates_per_env_step": 4},
                "w256": {"hidden_sizes": [256, 256]}, "w512": {"hidden_sizes": [512, 512]},
                "b256": {"batch_size": 256}, "b512": {"batch_size": 512}, "b1024": {"batch_size": 1024}},
    env_ref_override={"HC": None, "Ant": None}, extra_measured=[], measured_order=["rollout", "gradient_updates"],
    sweeps=[("UTD", "updates_per_env_step ∈ {1,2,4}", ["base", "utd2", "utd4"], "updates_per_env_step"),
            ("width", "hidden_sizes ∈ {(256,256),(512,512),(1024,1024)}", ["w256", "w512", "base"], "hidden_sizes"),
            ("batch size", "batch_size ∈ {100,256,512,1024}", ["base", "b256", "b512", "b1024"], "batch_size")],
    expected_runs={"HC": 40, "Ant": 40}, per_epoch_tasks=["rollout", "gradient_updates"])

OVERRIDE_FILE = {"base": "none (TD3Config defaults)", "utd2": "configs/overrides/td3_utd2.json", "utd4": "configs/overrides/td3_utd4.json",
                 "w256": "configs/overrides/td3_width256.json", "w512": "configs/overrides/td3_width512.json",
                 "b256": "configs/overrides/td3_batch256.json", "b512": "configs/overrides/td3_batch512.json",
                 "b1024": "configs/overrides/td3_batch1024.json"}
SPEC.tag_override_file = lambda es, t: OVERRIDE_FILE[t]

LOG_TO_SCRIPT = {"results/td3/_env_sweep.log": "scripts/run_td3_env_sweep.sh",
                 "results/_utd_sweep.log": "scripts/run_utd_sweep.sh (at commit aa9186b / 24f41b0: SAC-Ant + TD3 UTD sweep; the file now holds the MBPO UTD sweep)",
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
td = TD3Config
td3src = Src("algorithms/td3.py")


# ----------------------------------------------------------------------------- Section 2
def section2():
    import torch
    import gymnasium as gym
    from algorithms.td3 import DeterministicPolicy, QNetwork
    out = []
    out.append("**2.1 Configuration objects** (dumped from `configs/config.py` at HEAD):\n")
    out.append("`TD3Config` defaults (`dataclasses.asdict(TD3Config())`):\n")
    out.append(code(json.dumps(dataclasses.asdict(TD3Config()), indent=2, default=list), "json"))
    bad = []
    for es, _ in ENVS:
        for t in TAGS:
            f = OVERRIDE_FILE[t]
            if f.startswith("none"):
                continue
            ov = json.loads(Path(f).read_text(encoding="utf-8"))
            ac = DS.runs_of(es, t)[0]["meta"]["algo_config"]
            for k, v in ov.items():
                if ac.get(k) != v:
                    bad.append((label(es, t), k, v, ac.get(k)))
    out.append("Override files of the sweeps (content): " + "; ".join(f"`{Path(p).name}` = {json.dumps(json.loads(Path(p).read_text(encoding='utf-8')))}" for p in sorted(glob.glob("configs/overrides/td3_*.json"))))
    out.append("- " + check("every override file listed in the inventory equals the corresponding fields of the logged `algo_config` of its runs", not bad, str(bad) if bad else ""))
    out.append("`ExperimentConfig` defaults (protocol, shared by all algorithms; `warmup_steps` is overridden to 10000 on the CLI for all TD3 runs):\n")
    out.append(code(json.dumps(dataclasses.asdict(ExperimentConfig(algo_name="td3", env_id="HalfCheetah-v5")), indent=2), "json"))
    for es in ("HC", "Ant"):
        r = srt(DS.runs_of(es, "base"))[0]
        out.append(f"\nComplete `metadata.json` of `{r['run_dir']}` ({label(es, 'base')}, seed {r['seed']}):\n")
        out.append(code(json.dumps(r["meta"], indent=2), "json"))
    # code excerpts
    a, b, blk = td3src.block(r"^class DeterministicPolicy", r"^@dataclass")
    out.append(f"\n**2.2 Verbatim code** — `algorithms/td3.py` at HEAD (identical at every run commit, Section 0). Networks (lines {a}–{b}):\n")
    out.append(blk)
    a, b, blk = td3src.block(r"    def __init__\(self, obs_dim, act_dim, act_limit, cfg: TD3Config", r"^def train\(")
    out.append(f"`TD3Agent` (`__init__`, `select_action`, `update`; lines {a}–{b}):\n")
    out.append(blk)
    a, b, blk = td3src.block(r"^def train\(")
    out.append(f"`train()` (lines {a}–{b}):\n")
    out.append(blk)
    out.append("`algorithms/replay_buffer.py` (complete):\n")
    rb = Src("algorithms/replay_buffer.py")
    out.append(code(rb.excerpt(1, len(rb.lines)), "python"))
    # network sizes
    envd = {}
    for es, env_id in ENVS:
        e = gym.make(env_id)
        envd[es] = (e.observation_space.shape[0], e.action_space.shape[0])
        e.close()
    rows = []
    for es, env_id in ENVS:
        o, a_ = envd[es]
        for w in (256, 512, 1024):
            pol = DeterministicPolicy(o, a_, (w, w), 1.0)
            q = QNetwork(o, a_, (w, w))
            npol = sum(x.numel() for x in pol.parameters())
            nq = sum(x.numel() for x in q.parameters())
            rows.append([env_id, f"({w},{w})", f"{o}/{a_}", npol, nq, 2 * nq, npol, 2 * nq, npol + 2 * nq, 2 * (npol + 2 * nq)])
    out.append("\n**2.3 Network sizes** — real `DeterministicPolicy` / `QNetwork` classes instantiated on CPU (parameter count = "
               "`sum(p.numel())`, weights + biases). TD3 holds six networks: actor, target actor, two critics, two target critics "
               "(`TD3Agent.__init__`):\n")
    out.append(table(["env", "hidden", "obs/act dim", "actor", "one Q-network", "both critics", "target actor", "both target critics",
                      "trainable (actor + 2 Q)", "all six networks"], rows))
    ok = all(envd[es] == (FLOPS_JSON["td3"][env_id]["bs100_h1024x1024"]["obs_dim"], FLOPS_JSON["td3"][env_id]["bs100_h1024x1024"]["act_dim"]) for es, env_id in ENVS)
    out.append("- " + check("local environment dims equal the dims stored in flops_per_call.json (HC 17/6, Ant 105/8)", ok))
    # FLOP constants
    keys = [("actor_forward_bs1 (rollout call)", lambda fk: fk["actor_forward_bs1"]),
            ("critic_fwdbwd (per update() call)", lambda fk: fk["critic_fwdbwd"]),
            ("actor_fwdbwd (per executed actor update)", lambda fk: fk["actor_fwdbwd"]),
            ("target_update_elementwise_ops (per executed target update)", lambda fk: fk["target_update_elementwise_ops"]),
            ("batch_size used for the measurement", lambda fk: fk["batch_size"])]
    out.append("\n**2.4 Per-call FLOPs from `flop_analysis/flops_per_call.json`** (`measure_flops.measure_td3`, torch `FlopCounterMode` on the real "
               f"`DeterministicPolicy`/`QNetwork`; measured on `{FLOPS_JSON['_device_used_for_measurement']}`; matmul FLOPs = 2·m·n·k; "
               "Adam / Polyak / clamp / noise operations are not matmuls and count 0 FLOPs). One column per constant; cell `HC ‖ Ant`:\n")
    out.append(flop_constant_table(DS, keys))
    # analytic
    rows = []
    allok = True
    for es, env_id in ENVS:
        o, a_ = envd[es]
        for sig in sorted({r["sig"] for r in DS.runs if r["es"] == es}):
            fk = FLOPS_JSON["td3"][env_id][sig]
            h1, h2 = fk["hidden_sizes"]
            B = fk["batch_size"]
            fa = lambda b: 2 * b * (o * h1 + h1 * h2 + h2 * a_)  # noqa: E731
            fq = lambda b: 2 * b * ((o + a_) * h1 + h1 * h2 + h2)  # noqa: E731
            first_q = 2 * B * (o + a_) * h1
            first_a = 2 * B * o * h1
            crit = fa(B) + 2 * fq(B) + 2 * (fq(B) + (2 * fq(B) - first_q))
            act = fa(B) + fq(B) + fq(B) + (2 * fa(B) - first_a)
            roll = fa(1)
            tgt = 2 * (sum(x.numel() for x in __import__("algorithms.td3", fromlist=["x"]).DeterministicPolicy(o, a_, (h1, h2), 1.0).parameters())
                       + 2 * sum(x.numel() for x in __import__("algorithms.td3", fromlist=["x"]).QNetwork(o, a_, (h1, h2)).parameters()))
            ex = (roll, crit, act, tgt) == (fk["actor_forward_bs1"], fk["critic_fwdbwd"], fk["actor_fwdbwd"], fk["target_update_elementwise_ops"])
            allok &= ex
            rows.append([env_id, f"`{sig}`", fk["actor_forward_bs1"], roll, fk["critic_fwdbwd"], crit, fk["actor_fwdbwd"], act,
                         fk["target_update_elementwise_ops"], tgt, "exact" if ex else "MISMATCH"])
    out.append("\n**2.5 Analytic cross-check** (derived by this generator, not a pipeline output): dense-layer count mirroring the code path — "
               "forward 2·B·n_in·n_out per layer; backward = weight-grad + input-grad (input-grad omitted for the first layer of a network fed "
               "by data); critic = target (target-actor fwd + 2 target-Q fwd) + 2 × (Q fwd + Q bwd); actor = actor fwd + Q1 fwd + Q1 "
               "input-grad bwd (frozen weights, all layers incl. the action input of layer 1) + actor bwd; target ops = 2 × (actor + 2 Q params). "
               "Exact integers:\n")
    out.append(table(["env", "signature", "rollout measured", "rollout analytic", "critic measured", "critic analytic", "actor measured",
                      "actor analytic", "target ops measured", "target ops analytic", "derived vs measured"], rows))
    out.append("- " + check("analytic dense-layer counts equal the FlopCounterMode counts exactly for rollout, critic, actor and target ops in every (env, signature)", allok))
    # call counts
    L = {k: td3src.find(p) for k, p in {"upd": r"    def update\(", "bs0": r"# --- buffer_sample ---", "cu0": r"# --- critic_update", "do": r"do_delayed_update =",
                                        "au_t0": r"t0 = time.perf_counter\(\)", "tgt": r"with torch.no_grad\(\):\s*$"}.items() if k in ("upd", "bs0", "cu0", "do")}
    out.append(f"""
**2.6 Call counts and where they are defined.** Per run (`compute_energy_per_flop.sac_like_call_counts` and `rows_for_sac_or_td3(algo="td3", …)`, called from `main()`):

```
n_epochs = max(1, train_steps // steps_per_epoch)               # = 100
n_env    = n_epochs * steps_per_epoch                           # = 100000 (warmup steps are NOT counted)
n_upd    = n_env * updates_per_env_step                         # update() calls = buffer_sample calls = critic_update calls
n_act    = ceil(n_upd / policy_update_delay)                    # executed actor updates (delay = 2)
target_update calls = n_act                                     # TD3 only (SAC: n_upd)
F_rollout = actor_forward_bs1 * n_env
F_critic  = critic_fwdbwd    * n_upd        # per update() call
F_actor   = actor_fwdbwd     * n_act        # per EXECUTED actor update
OPS_target = target_update_elementwise_ops * n_act
```
The loop that issues the calls is `train()` (`algorithms/td3.py`): `n_updates = steps_per_epoch * updates_per_env_step` calls of `agent.update()` per epoch (lines near `for _ in range(n_updates)`), breaking only if `len(buffer) < batch_size` (never the case: buffer ≥ 10000 > 1024 after the 10000 warmup steps). The delayed branch is `do_delayed_update = (self._update_counter % policy_update_delay) == 0` (line {L['do']}), `_update_counter` starts at 0 and increments once per `update()` call, so the actor and target updates run on update() calls 0, 2, 4, … (ceil(n_upd/2) times per run). Call counts per configuration (mean over seeds; identical across seeds):
""")
    out.append(call_count_table(DS))
    ok = True
    pr = pd.read_csv("flop_analysis/output/per_run_energy_per_flop.csv")
    for r in DS.runs:
        sub = pr[pr.run_dir == r["run_dir"]].set_index("segment")
        for s in ("rollout", "buffer_sample", "critic_update", "actor_update", "target_update"):
            ok &= int(sub.loc[s, "call_count"]) == r["calls"][s]
    out.append("- " + check(f"call counts equal the `call_count` column of per_run_energy_per_flop.csv for all {len(DS.runs)} runs", ok))
    out.append("- The actual number of `update()` calls is **not logged**; it follows from the loop (indirect evidence: every run has 100 epochs with "
               "`buffer_size` = warmup + 1000·(epoch+1) and a non-null `critic_loss_mean` and `actor_loss_mean` in every epoch).")
    bad = []
    for r in DS.runs:
        ep = r["tm"]["epochs"]
        w = r["meta"]["experiment_config"]["warmup_steps"]
        if [e["buffer_size"] for e in ep] != [w + 1000 * (i + 1) for i in range(100)] or any(e["critic_loss_mean"] is None or e["actor_loss_mean"] is None for e in ep):
            bad.append(r["run_dir"])
    out.append("- " + check("buffer_size per epoch = warmup + 1000(e+1) and critic/actor loss logged in all 100 epochs of every run", not bad, str(bad[:3]) if bad else ""))
    t1, t2 = total_flops_table(DS)
    out.append("\n**2.7 Total FLOPs per run and segment** (call count × per-call FLOPs; `target` = elementwise Polyak operations, not FLOPs; `GU` = critic + actor; "
               "`TOTAL` = rollout + critic + actor; `buffer_sample` = 0 FLOPs):\n")
    out.append(t1)
    out.append("Ant / HalfCheetah ratios of the same quantities:\n")
    out.append(t2)
    out.append("The factor of each configuration against its sweep baseline is given in the 'FLOP factors' table of each sweep section (Sections 3–4).")
    return "\n".join(out)


# ----------------------------------------------------------------------------- Section 6
def section6():
    out = []
    # 6.1 code facts
    L = {}
    L["upd"] = td3src.find(r"    def update\(")
    L["bs"] = td3src.find(r"# --- buffer_sample ---")
    L["bs_end"] = td3src.find(r"info.buffer_sample_s =")
    L["cu"] = td3src.find(r"# --- critic_update")
    L["noise"] = td3src.find(r"noise = \(torch.randn_like\(act_t\)")
    L["tn_act"] = td3src.find(r"next_action = \(self.actor_targ")
    L["cu_end"] = td3src.find(r"info.critic_update_s =")
    L["cu_item"] = td3src.find(r"info.critic_loss =")
    L["do"] = td3src.find(r"do_delayed_update =")
    L["au_t0"] = td3src.find(r"t0 = time.perf_counter\(\)", L["do"])
    L["au_end"] = td3src.find(r"info.actor_update_s =")
    L["au_item"] = td3src.find(r"info.actor_loss =")
    L["tu_t0"] = td3src.find(r"t0 = time.perf_counter\(\)", L["au_item"])
    L["tu_end"] = td3src.find(r"info.target_update_s =")
    L["cnt"] = td3src.find(r"self._update_counter \+= 1")
    L["explore"] = td3src.find(r"def _explore_action")
    L["roll"] = td3src.find(r'TrackerTask\(tracker, "rollout"')
    L["gu"] = td3src.find(r'TrackerTask\(tracker, "gradient_updates"')
    L["recon"] = td3src.find(r"# -+ reconcile")
    L["warm"] = td3src.find(r'TrackerTask\(tracker, "warmup"')
    L["loop"] = td3src.find(r"for _ in range\(n_updates\)")
    L["brk"] = td3src.find(r"if len\(buffer\) < td3_cfg.batch_size")
    L["sel"] = td3src.find(r"def select_action")
    L["init_t"] = td3src.find(r"self.target_policy_noise =")
    out.append("**6.1 Code facts** (file `algorithms/td3.py` at HEAD unless stated; identical at all run commits):\n")
    out.append(f"""
- **What one `TD3Agent.update()` call does** (lines {L['upd']}–{L['cnt']}): (1) `buffer_sample` timer (lines {L['bs']}–{L['bs_end']}): `ReplayBuffer.sample(batch_size)` on the host (NumPy fancy indexing) and five `torch.as_tensor(..., device=...)` host→device copies; (2) `critic_update` timer (lines {L['cu']}–{L['cu_end']}): under `no_grad` the clipped-noise target action from the **target actor** (target policy smoothing, lines {L['noise']}–{L['tn_act']}), both target critics, `min`, Bellman target; then Q1 and Q2 forward, MSE losses, `zero_grad`/`backward`/Adam step of Q1 and then of Q2; `critic_loss.item()` is on line {L['cu_item']}, after the timer; (3) **only if** `do_delayed_update = (self._update_counter % policy_update_delay) == 0` (line {L['do']}): `actor_update` timer (lines {L['au_t0']}–{L['au_end']}): freeze Q1, `actor_loss = -Q1(obs, actor(obs)).mean()`, backward, actor Adam step, unfreeze Q1 (`actor_loss.item()` on line {L['au_item']}, after the timer), then `target_update` timer (lines {L['tu_t0']}–{L['tu_end']}): Polyak averaging (`mul_(1 − τ)`, `add_(τ·p)`) of **three** networks — actor → target actor, Q1 → target Q1, Q2 → target Q2; (4) `self._update_counter += 1` (line {L['cnt']}).
- **Networks updated on actor-update iterations vs. others:** on iterations with `_update_counter % 2 == 0` all of: Q1, Q2 (critic step), the actor (actor step, gradient through Q1 only; Q2 is not used in the actor loss) and the three target networks. On the other iterations only Q1 and Q2 are updated; the actor, the target actor and both target critics are untouched.
- **`perf_counter` timers on iterations where the actor is skipped:** `buffer_sample_s` and `critic_update_s` are measured on every call; `actor_update_s` and `target_update_s` keep their dataclass default 0.0 (`UpdateInfo`), i.e. they are never started, so the summed `_sub_segment_wall_time_seconds` for `actor_update` and `target_update` cover only the executed (even-counter) iterations. The `_sub_segment_wall_time_seconds` share used for the energy allocation is therefore Σ over executed iterations for these two sub-segments and Σ over all `update()` calls for the other two. The remaining time inside the measured `gradient_updates` task (the `.item()` host syncs on lines {L['cu_item']} and {L['au_item']}, the loop overhead, the odd iterations' `if`) is in no timer (coverage < 100 %, Section 3).
- **Number of executions per run** (formula in Section 2.6): `update()` calls / `critic_update` / `buffer_sample` = n_upd = 100000 × UTD; `actor_update` and `target_update` = ceil(n_upd / 2) = 50000 × UTD (policy_update_delay = 2 in all 80 runs). Not logged; derived from the loop (`for _ in range(n_updates)`, line {L['loop']}, with a `break` only if `len(buffer) < batch_size`, line {L['brk']}).
- **How `flop_analysis` defines the per-call FLOPs and counts of these two segments** (`compute_energy_per_flop.rows_for_sac_or_td3`): `critic_update` = `critic_fwdbwd` × n_upd, per `update()` call (it includes the no-grad target computation: target-actor forward + two target-Q forwards); `actor_update` = `actor_fwdbwd` × n_act, **per executed actor update** (n_act = ceil(n_upd/delay)); `target_update` = `target_update_elementwise_ops` × n_act, per executed target update, where the constant is 2 × (parameters of actor + Q1 + Q2) (`measure_flops.measure_td3`), i.e. the Polyak operations of the **three** updated networks (the target networks themselves are the destination, not counted). (SAC, by contrast, updates its two target critics on every call and counts `target_update` per `update()` call.)
- **Number of networks in `target_update` / Polyak operations per run:** 3 networks (actor, Q1, Q2). Operations per run = `target_update_elementwise_ops` × n_act — see the per-configuration values in Section 2.7 (`F target`; e.g. HC baseline {g6(FLOPS_JSON['td3']['HalfCheetah-v5']['bs100_h1024x1024']['target_update_elementwise_ops'])} ops per call × 50000 calls).
- **Exploration noise:** added in `train()` by the local function `_explore_action` (line {L['explore']}): deterministic actor action from `select_action` (line {L['sel']}) + `np.random.normal(0, exploration_noise · act_limit)` (host-side NumPy), clipped to ±act_limit. It runs inside the measured `rollout` task (line {L['roll']}) for every env step; warmup (line {L['warm']}) uses `env.action_space.sample()` instead. Not a matmul → 0 FLOPs in the pipeline.
- **Target policy smoothing:** inside `update()`, `critic_update` timer, lines {L['noise']}–{L['tn_act']}: `noise = clamp(randn_like(act) · target_policy_noise, ±target_noise_clip)` added to the target actor's action, then clamped to ±act_limit. `target_policy_noise = cfg.target_policy_noise · act_limit` and `target_noise_clip = cfg.target_noise_clip · act_limit` are set in `__init__` (line {L['init_t']}).
- **Absence of the temperature step:** `td3.py` has no entropy temperature, no `log_alpha`, no alpha optimiser and no alpha loss; the actor is deterministic (`DeterministicPolicy`, tanh output × act_limit); `select_action` has no stochastic sampling; epoch rows have no `alpha_end` field (`training_metrics.json` keys: {list(DS.runs[0]['tm']['epochs'][0].keys())}). SAC's `actor_update` additionally contains the α loss/step and a log-prob term (see Section 4.1).
- **Reconcile / allocation** (line {L['recon']} ff.): identical to SAC — one split of the whole run's summed `gradient_updates` energy by the whole run's summed sub-timer shares.
""")
    # 6.2 baseline settings
    r_hc, r_ant = srt(DS.runs_of("HC", "base"))[0], srt(DS.runs_of("Ant", "base"))[0]
    ac = r_hc["meta"]["algo_config"]
    out.append("**6.2 Baseline settings** read from `metadata.json` (identical in all 10 baseline runs, Section 1.3b):\n")
    out.append(table(["setting", "value", "source"], [
        ["batch_size", ac["batch_size"], "algo_config"], ["actor_lr / critic_lr", f"{ac['actor_lr']} / {ac['critic_lr']}", "algo_config"],
        ["warmup_steps", r_hc["meta"]["experiment_config"]["warmup_steps"], "experiment_config (CLI `--warmup-steps 10000`)"],
        ["policy_update_delay", ac["policy_update_delay"], "algo_config"],
        ["exploration_noise (std, × act_limit)", ac["exploration_noise"], "algo_config"],
        ["target_policy_noise (std, × act_limit)", ac["target_policy_noise"], "algo_config"],
        ["target_noise_clip (× act_limit)", ac["target_noise_clip"], "algo_config"],
        ["gamma / tau", f"{ac['gamma']} / {ac['tau']}", "algo_config"],
        ["hidden_sizes / buffer_capacity / UTD", f"{ac['hidden_sizes']} / {ac['buffer_capacity']} / {ac['updates_per_env_step']}", "algo_config"],
        ["action bounds", "±1 in both environments (`act_limit = env.action_space.high[0]`)", "`td3.train()`"]]))
    rows = []
    for nm, dsx in (("TD3 baseline (batch 100)", DS), ("SAC baseline (batch 256)", SAC)):
        rs = {es: srt(dsx.runs_of(es, "base")) for es, _ in ENVS}
        rows.append([nm, SEP.join(str(rs[es][0]["meta"]["experiment_config"]["warmup_steps"]) for es, _ in ENVS),
                     SEP.join(msd(vals(rs[es], "E", "warmup")) for es, _ in ENVS),
                     SEP.join(msd(vals(rs[es], "D", "warmup")) for es, _ in ENVS),
                     SEP.join(msd(vals(rs[es], "P", "warmup")) for es, _ in ENVS),
                     SEP.join(msd([100 * r["E"]["warmup"] / r["E"]["TOTAL_MEASURED_TRAINING"] for r in rs[es]]) for es, _ in ENVS)])
    out.append("\nWarmup (random-action buffer fill; its own CodeCarbon task `warmup_0`, key `warmup` of `segment_energy.json`) — "
               "**excluded from every total** in both algorithms (`TRAINING_SEGMENTS` in `compute_energy_per_flop.py` does not contain "
               "`warmup`, and the generator's TOTAL = Σ training segments; checked: TOTAL ≠ TOTAL + warmup in the CSV, "
               "Section 1.5):\n")
    out.append(table(["algorithm", "warmup steps HC ‖ Ant", "warmup energy [J]", "warmup duration [s]", "warmup mean power [W]", "warmup energy / TOTAL training energy [%]"], rows))
    ok = all(abs(pr_ - r_["E"]["TOTAL_MEASURED_TRAINING"]) < 1e-6 * r_["E"]["TOTAL_MEASURED_TRAINING"] for r_, pr_ in
             [(r_, float(__import__("pandas").read_csv("flop_analysis/output/per_run_energy_per_flop.csv").query("run_dir == @r_['run_dir'] and segment == 'TOTAL_MEASURED_TRAINING'").total_energy_joules.iloc[0])) for r_ in DS.runs[:3] + SAC.runs[:3]])
    out.append("- " + check("warmup excluded from TOTAL_MEASURED_TRAINING (pipeline TOTAL = Σ training segments without warmup; spot-checked on 3 TD3 and 3 SAC runs; "
                            "all runs in Section 1.5)", ok))
    # 6.3 batch sweep
    out.append("\n**6.3 Batch-size sweep (four values: 100, 256, 512, 1024).** Baseline is batch 100 (`base`); the batch-256 run (`b256`) is the "
               "batch-matched comparison point to SAC (SAC's default batch is 256). The batch-size sweep section (Section 4.3) carries the common block for all four values; "
               "the full common-block values of `b256` alone, including the response against the batch-100 baseline, are repeated here:\n")
    out.append(common_block(DS, ["b256"], with_response=True, with_env=True, base_tag="base", prefix="[b256] "))
    # 6.4 TD3 vs SAC
    out.append("\n**6.4 TD3 versus SAC side by side** at hidden (1024,1024), UTD 1, batch 256, both environments. TD3 = `b256` (TD3Config otherwise: lr 1e-3, "
               "policy delay 2, warmup 10000); SAC = `base` (SACConfig: lr 3e-4, α tuning, warmup 5000). SAC values come from the same recomputation method as "
               "Section 4.1 (`s4_data.Dataset(sac_spec())`, same definitions as `build_section_4_1.py`); cross-check against "
               "`contexts/section_4.1_data/sac_cross_seed_summary.csv` below. Columns: HC TD3, HC SAC, Ant TD3, Ant SAC.\n")
    sc = pd.read_csv("contexts/section_4.1_data/sac_cross_seed_summary.csv")
    mx_e = mx_j = 0
    for es, _ in ENVS:
        for s in ["rollout", "buffer_sample", "critic_update", "actor_update", "target_update", "TOTAL_MEASURED_TRAINING"]:
            row = sc[(sc.tag == "base") & (sc.env == dict(ENVS)[es]) & (sc.segment == s)].iloc[0]
            rs = srt(SAC.runs_of(es, "base"))
            mx_e = max(mx_e, abs(np.mean(vals(rs, "E", s)) - row.energy_j_mean) / row.energy_j_mean)
            if s in rs[0]["F"] and s != "buffer_sample":
                mx_j = max(mx_j, abs(np.mean(vals(rs, "E", s)) / np.mean(vals(rs, "F", s)) - row.j_per_flop_ratio_of_means) / row.j_per_flop_ratio_of_means)
    out.append("- " + check("SAC baseline energies and J/FLOP recomputed here equal `sac_cross_seed_summary.csv` of Section 4.1", mx_e < 1e-9 and mx_j < 1e-9,
                            f"max relative deviation energy {mx_e:.2e}, J/FLOP {mx_j:.2e}"))
    segs = ["rollout", "buffer_sample", "critic_update", "actor_update", "target_update", "gradient_updates", "TOTAL_MEASURED_TRAINING"]
    hdr = ["segment", "HC TD3", "HC SAC", "Ant TD3", "Ant SAC"]

    def vt(title, getter, segs_=segs, fmt=msd):
        out.append(f"\n{title}\n")
        out.append(table(hdr, versus_rows(DS, "b256", SAC, "base", getter, segs_, fmt)))
    vt("Energy [J] (measured: rollout, GU; allocated: the four sub-segments):", lambda rs, s: vals(rs, "E", s))
    vt("Share of TOTAL energy [%]:", lambda rs, s: vals(rs, "share", s), [s for s in segs if s != "TOTAL_MEASURED_TRAINING"])
    vt("Share of GU energy [%] (= wall-clock time share of each sub-phase):", lambda rs, s: vals(rs, "share_gu", s), SUBS)
    vt("J/FLOP (ratio of means ± sd of per-seed ratios; GU derived; `target_update` in nJ per Polyak op):",
       lambda rs, s: (jpf_cell(rs, s, 1e9) if s == "target_update" else jpf_cell(rs, s)) if s != "buffer_sample" else "no FLOPs", segs, lambda v: v)
    vt("Per-call FLOPs (`flops_per_call.json`; critic per update() call, actor/target per executed update; target in elementwise ops):",
       lambda rs, s: {"rollout": rs[0]["fk"]["actor_forward_bs1"], "critic_update": rs[0]["fk"]["critic_fwdbwd"], "actor_update": rs[0]["fk"]["actor_fwdbwd"],
                      "target_update": rs[0]["fk"]["target_update_elementwise_ops"], "buffer_sample": 0}[s],
       ["rollout", "buffer_sample", "critic_update", "actor_update", "target_update"], lambda v: g6(v))
    vt("Call counts per run:", lambda rs, s: rs[0]["calls"][s], ["rollout", "buffer_sample", "critic_update", "actor_update", "target_update"], lambda v: g6(v))
    vt("Total FLOPs per run (`target`: ops):", lambda rs, s: rs[0]["F"][s], [s for s in segs if s != "buffer_sample"], lambda v: g6(v))
    vt("Duration [s] (measured: rollout, GU; sub-segments = perf_counter time):", lambda rs, s: vals(rs, "D", s), segs)
    vt("Mean power [W] (energy / duration; allocated sub-segments: E_k/T_k):", lambda rs, s: vals(rs, "P", s), segs)
    vt("Hardware composition of the measured tasks — CPU / GPU / RAM share [%]:",
       lambda rs, s: " / ".join(g6(np.mean([100 * r["hw"][s][c] / r["hw"][s]["E"] for r in rs])) for c in ("cpu", "gpu", "ram")), ["rollout", "gradient_updates"], lambda v: v)
    vt("Sub-timer coverage ΣT / D_GU [%] and idle-floor fraction of GU / TOTAL [%]:",
       lambda rs, s: {"cov": msd([100 * r["coverage"] for r in rs]), "idleGU": msd([r["idle_frac"]["gradient_updates"] for r in rs]),
                      "idleTOT": msd([r["idle_frac"]["TOTAL_MEASURED_TRAINING"] for r in rs])}[s], ["cov", "idleGU", "idleTOT"], lambda v: v)
    # 6.5 actor update
    out.append("\n**6.5 Actor update** — for every TD3 configuration: energy, FLOPs, J/FLOP and share of GU of `actor_update` and `critic_update` "
               "(allocated values), with the SAC counterpart of the same swept factor. **Matched settings exist exactly** for `b256`↔SAC `base`, "
               "`b512`↔SAC `b512`, `b1024`↔SAC `b1024` (hidden (1024,1024), UTD 1, same batch). For the TD3 configurations `base`, `utd2`, `utd4`, `w256`, `w512` the batch size is 100 "
               "whereas the SAC counterpart (same UTD / width) has batch 256, so those SAC values are **not** batch-matched (marked ≠batch). Cell `HC ‖ Ant`; "
               "`SAC counterpart` is the SAC configuration named in the second column:\n")
    pairs = [("base", "base", "≠batch (100 vs 256)"), ("utd2", "utd2", "≠batch"), ("utd4", "utd4", "≠batch"), ("w256", "w256", "≠batch"),
             ("w512", "w512", "≠batch"), ("b256", "base", "matched"), ("b512", "b512", "matched"), ("b1024", "b1024", "matched")]
    rows = []
    for tt, st, flag in pairs:
        for nm, dsx, tg in (("TD3", DS, tt), ("SAC", SAC, st)):
            cells = [f"{nm} {tg}", flag if nm == "SAC" else ""]
            for s in ("actor_update", "critic_update"):
                cells.append(envcell(dsx, tg, lambda rs, s=s: msd(vals(rs, "E", s))))
                cells.append(envcell(dsx, tg, lambda rs, s=s: g6(np.mean(vals(rs, "F", s)))))
                cells.append(envcell(dsx, tg, lambda rs, s=s: jpf_cell(rs, s)))
                cells.append(envcell(dsx, tg, lambda rs, s=s: msd(vals(rs, "share_gu", s))))
            rows.append(cells)
    out.append(table(["config", "matching", "E actor [J]", "F actor", "J/FLOP actor", "actor share of GU [%]", "E critic [J]", "F critic",
                      "J/FLOP critic", "critic share of GU [%]"], rows))
    out.append("Per-call constants for the same pairs are in Section 2.4 (TD3) and Section 4.1 Part 3 (SAC); the actor-update FLOPs per executed update differ "
               "between TD3 and SAC by construction of the two actor losses (TD3: actor fwd + Q1 only; SAC: actor fwd + Q1 + Q2 + log-prob/α), and the number of "
               "executed updates differs (TD3 ceil(n_upd/2), SAC n_upd).")
    # 6.6 open points
    out.append("\n**6.6 Open points from the instructions (reported, not resolved):**\n")
    allr = pe[pe.task == "rollout"].duration_s
    allg = pe[pe.task == "gradient_updates"].duration_s
    out.append(f"- *`rollout` task durations versus the 1 s polling interval:* across all {len(DS.runs)} TD3 runs {int((allr < 1.0).sum())} of {len(allr)} `rollout` tasks are "
               f"shorter than 1 s (min/median/max {g6(allr.min())}/{g6(allr.median())}/{g6(allr.max())} s); {int((allg < 1.0).sum())} of {len(allg)} `gradient_updates` tasks are shorter than 1 s "
               f"(min/median/max {g6(allg.min())}/{g6(allg.median())}/{g6(allg.max())} s). Per configuration: Section 5.2a. (The same situation was reported for SAC in Section 4.1.)")
    out.append("- *Host-side work counted as FLOPs:* none. `FlopCounterMode` counts matmul-type operators only (`aten.mm/addmm/bmm`…). **Not counted (0 FLOPs):** replay-buffer sampling and the "
               "host→device copies (`buffer_sample`: 0 by definition, `ZERO_FLOP_SEGMENTS`), exploration noise / `np.clip`, `env.step()` (MuJoCo), `buffer.add`, target-smoothing noise and clamps, "
               "the Q-value `min`, MSE loss, Adam updates, ReLU/tanh, and the Polyak averaging (counted separately as elementwise operations). **Counted:** `rollout` = one actor forward at batch 1 per env step "
               "(env step and noise excluded); `critic_update` = target-actor fwd + 2 target-Q fwd + 2 × (Q fwd + bwd) at the configured batch; `actor_update` = actor fwd + Q1 fwd + Q1 input-grad bwd + actor bwd; "
               "`target_update` = Polyak operation count; `gradient_updates` (derived) = critic + actor.")
    out.append("- *Segments whose timers or FLOPs are produced differently from SAC:* (i) `actor_update` and `target_update` are timed and counted only on executed delayed iterations (every second `update()` call), "
               "`target_update` covers three networks including the target actor; (ii) the critic FLOPs include a target-**actor** forward (deterministic, no log-prob) instead of SAC's stochastic next-action sampling; "
               "(iii) the actor loss uses Q1 only (SAC: min of both critics); (iv) no α step. (v) warmup is 10000 steps (SAC 5000), i.e. the replay buffer holds 10000 more transitions throughout training.")
    return "\n".join(out)


# ----------------------------------------------------------------------------- assemble
def main():
    spec = DS.spec
    MD = []
    P = MD.append
    recon_txt, recon_stats = recon(DS)
    sweeps_txt = {}
    # section 1
    prov = provenance(DS, ["build_section_4_2.py", "s4_data.py", "s4_blocks.py", "s4_audit.py"],
                      ["algorithms/td3.py", "algorithms/replay_buffer.py", "algorithms/tracker_utils.py"])
    inv_keys = ["hidden_sizes", "batch_size", "updates_per_env_step", "policy_update_delay"]
    aud = audit(DS, inv_keys)
    exec_txt, _ = exec_facts(DS, ["results/_utd_sweep_status.json", "results/_width_sweep_status.json", "results/_batch_size_sweep_status.json",
                                   "results/_ant_overnight_sweep_status.json", "results/td3/_env_sweep_status.json"],
                             ["`scripts/run_td3_env_sweep.sh` (baselines, 10 runs, `--warmup-steps 10000` in the script)",
                              "`scripts/run_utd_sweep.sh` (as of commit `aa9186b`: UTD ∈ {2,4}; the file has since been repointed to MBPO)",
                              "`scripts/run_width_sweep.sh`", "`scripts/run_batch_size_sweep.sh` (HC)", "`scripts/run_ant_overnight_sweep.sh` (Ant batch sizes)"])
    n_checks_before = len(CHECKS)
    sec2 = section2()
    sec6 = section6()
    # sweeps
    blocks = {}
    blocks["baseline"] = common_block(DS, ["base"], with_response=False, with_env=True)
    for name, text, tags, xd in spec.sweeps:
        blocks[name] = common_block(DS, tags, with_response=True, with_env=True, base_tag="base")
    ols = ols_block(DS, OUT_DIR, PREFIX)
    dlt = delta_block(DS, OUT_DIR, PREFIX, [("hidden_sizes", "width"), ("batch_size", "batch size"), ("updates_per_env_step", "UTD")])
    qual = quality_block(DS, pe, IDLE, OUT_DIR, PREFIX)
    learn = learning_block(DS, OUT_DIR, PREFIX)

    common_section7_items(DS, IDLE)
    disc("README.md ('Hyperparameter sensitivity sweeps') and CLAUDE.md say the original SAC-on-Ant/TD3 UTD sweep's top-level status/log bookkeeping was overwritten when "
         "`scripts/run_utd_sweep.sh` was repointed to MBPO. For TD3: `results/_utd_sweep_status.json` indeed holds only the MBPO entries (0 TD3 entries, Section 1.4), "
         "but `results/_utd_sweep.log` (opened with `tee -a`) still contains the complete launch blocks (stdout+stderr) of all 20 TD3 UTD runs "
         "(`utd2`/`utd4`, both envs), so every TD3 UTD run has a log block; only the status file is lost.")
    question("TD3 baseline = batch 100 (the TD3 paper default) is the thesis baseline per the layout instructions; the batch-matched comparison to SAC/MBPO is `b256`. "
             "Whether the SAC-vs-TD3 comparison of the thesis should use `base` or `b256` for TD3 is a decision for the author (both are reported in full).")
    n_pass = sum(1 for _, ok, _ in CHECKS if ok)
    n_fail = sum(1 for _, ok, _ in CHECKS if not ok)
    P(f"# Section 4.2 TD3 — context\n")
    P(f"Data context for thesis Section 4.2 (TD3): {len(DS.runs)} canonical runs (8 configurations × 2 environments × 5 seeds {CANON}). "
      "Facts, numbers and provenance only — no interpretation, no LaTeX. All energies are **gross** (idle floor included, nothing subtracted); "
      "mean ± sample sd (ddof = 1) over the five seeds; 6 significant digits; every table cell shows `HalfCheetah-v5 ‖ Ant-v5` (left ‖ right) unless a "
      "table says otherwise. Conventions of Section 4.1 apply (`measured` vs `allocated` segments; `target_update` in nJ per Polyak operation, never J/FLOP; `buffer_sample` has no FLOPs; "
      "J/FLOP = ratio of means with the sd of the five per-seed ratios; relative changes computed per seed then averaged; environment comparison paired by seed, Ant relative to HC). "
      "Training total = `TOTAL_MEASURED_TRAINING` (rollout + gradient_updates); warmup and idle windows are excluded.\n")
    P(f"Checks run by the generator: {len(CHECKS)}; passed {n_pass}; failed {n_fail}. Independent recomputation vs the pipeline CSVs: "
      f"{recon_stats['n_compared']} values compared, {recon_stats['n_fail']} deviating by more than 1e-9, maximum relative deviation {recon_stats['max_dev']:.3e}.\n")
    P("**Configuration tags:** `base` = hidden (1024,1024), batch 100, UTD 1, policy delay 2, warmup 10000 steps (TD3 paper hyperparameters except width); `utd2`/`utd4`, `w256`/`w512`, "
      "`b256`/`b512`/`b1024` change exactly that one factor of the baseline. Abbreviations: HC = HalfCheetah-v5, Ant = Ant-v5, GU = `gradient_updates`, TOTAL = `TOTAL_MEASURED_TRAINING`, "
      "buf_sample = `buffer_sample`, critic = `critic_update`, actor = `actor_update`, target = `target_update`.\n")
    P("## 0. Provenance\n")
    P(prov)
    P("\n## 1. Data basis and audit\n")
    P(aud)
    P("\n" + exec_txt)
    P("\n" + recon_txt)
    P("\n## 2. Structural facts from code and FLOP files\n")
    P(sec2)
    P("\n## 3. Baseline decomposition and energy per FLOP (4.2.1)\n")
    P(f"Configuration `base` (TD3 paper hyperparameters, hidden (1024,1024), batch 100, UTD 1, warmup 10000). Per-environment values, side by side.\n")
    P(blocks["baseline"])
    P("\nExcluded phases and per-seed totals of all 80 runs: `contexts/Section_4.2_data/td3_segments_long.csv` (all segments incl. warmup and idle, per run) and `td3_cross_seed_summary.csv`.\n")
    for i, (name, text, tags, xd) in enumerate(spec.sweeps, 1):
        P(f"\n## 4.{i} Sweep: {name} — {text}\n")
        P(f"Configurations: {', '.join(f'`{t}` ({spec.tag_desc[t]})' for t in tags)}. Baseline for the response tables: `base`.\n")
        P(blocks[name])
    P("\n## 5. Cross-configuration quantities\n")
    P(ols)
    P("\n" + dlt)
    P("\n" + qual)
    P("\n" + learn)
    P("\n## 6. Algorithm-specific items (TD3)\n")
    P(sec6)
    P("\n## 7. Discrepancies, NOT AVAILABLE items, questions\n")
    P("**(a) Where data or code contradict CLAUDE.md, README.md or `thesis-layout.md`:**\n")
    P("\n".join(f"- {d}" for d in DISCREPANCIES) if DISCREPANCIES else "- none found by the generator's checks")
    P("\n**(b) Quantities that could not be obtained (`NOT AVAILABLE`):**\n")
    P("\n".join(f"- NOT AVAILABLE: {d}" for d in NOT_AVAILABLE) if NOT_AVAILABLE else "- none")
    P("\n**(c) Questions for the author that the repository cannot answer:**\n")
    P("\n".join(f"- {d}" for d in QUESTIONS) if QUESTIONS else "- none")
    P("\n**Index of produced files** (`contexts/Section_4.2_data/`): `td3_segments_long.csv`, `td3_cross_seed_summary.csv`, `td3_run_inventory.csv`, `td3_per_epoch.csv`, "
      "`td3_idle_floor.csv`, `td3_ols_fits.csv`, `td3_delta_pairs.csv`, `td3_mad_flags.csv`, `td3_learning_performance.csv`; scripts `build_section_4_2.py`, `s4_data.py`, `s4_blocks.py`, `s4_audit.py`.")
    OUT_MD.write_text("\n".join(MD) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD} ({len(MD)} blocks); checks {len(CHECKS)} pass {n_pass} fail {n_fail}")
    for n_, ok, d in CHECKS:
        if not ok:
            print("FAIL:", n_, d)


if __name__ == "__main__":
    main()
