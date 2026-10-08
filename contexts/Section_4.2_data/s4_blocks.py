"""
Table blocks shared by the three section generators: the per-configuration "common block" (Section 3/4 of every output
file, spec Part 4.C) and the cross-configuration quantities (Section 5, spec Part 4.D).
Every table has HalfCheetah-v5 on the left and Ant-v5 on the right of the separator '‖' inside each cell.
"""
from __future__ import annotations

from s4_data import *  # noqa: F401,F403
from s4_data import _clean  # noqa: F401


def srt(rs):
    return sorted(rs, key=lambda r: r["seed"])


def envcell(ds, tag, fn):
    parts = []
    for es, _ in ENVS:
        rs = srt(ds.runs_of(es, tag))
        parts.append(fn(rs) if rs else "n/a")
    return SEP.join(parts)


def env_table(ds, tags, columns, first="config"):
    """columns: list of (header, fn(rs)->str).  One row per tag."""
    rows = []
    for t in tags:
        rows.append([t] + [envcell(ds, t, fn) for _, fn in columns])
    return table([first] + [h for h, _ in columns], rows)


def seg_h(s):
    st = SEG_STATUS[s]
    return f"{SEG_SHORT[s]} ({st})" if st in ("measured", "allocated") else SEG_SHORT[s]


def vals(rs, group, key):
    return [r[group][key] for r in rs]


def paired_rel(rs, base, f):
    """paired per-seed relative change in % (config seed / baseline seed - 1)"""
    bm = {r["seed"]: r for r in base}
    return np.array([100 * (f(r) / f(bm[r["seed"]]) - 1) for r in rs])


def pm(a):
    a = np.asarray(a, float)
    return f"{g6(a.mean())} ± {g6(a.std(ddof=1)) if len(a) > 1 else 'NaN'}"


def jpf_cell(rs, s, unit=1.0):
    Em = np.mean([r["E"][s] for r in rs])
    Fm = np.mean([r["F"][s] for r in rs])
    per = np.array([r["E"][s] / r["F"][s] for r in rs])
    return f"{g6(Em / Fm * unit)} ± {g6(per.std(ddof=1) * unit)}"


def jpf_perseed_mean_cell(rs, s, unit=1.0):
    per = np.array([r["E"][s] / r["F"][s] for r in rs])
    return f"{g6(per.mean() * unit)} ± {g6(per.std(ddof=1) * unit)}"


def jpf_segments(ds):
    """segments that carry matmul FLOPs, in table order (GU derived)."""
    out = []
    for t in ds.spec.measured_order:
        if t == "gradient_updates":
            out += ["critic_update", "actor_update", "gradient_updates"]
        else:
            out.append(t)
    return out + ["TOTAL_MEASURED_TRAINING"]


# =============================================================================== the common block
def common_block(ds: Dataset, tags, tag_label_fn=None, with_response=True, with_env=True, base_tag="base", prefix=""):
    spec = ds.spec
    out = []
    note = ("Cell format `HalfCheetah-v5 ‖ Ant-v5`; n = 5 seeds per cell; mean ± sample sd (ddof = 1); gross energy "
            "(idle floor included). `measured` = CodeCarbon task, `allocated` = share of the measured `gradient_updates` "
            "(GU) task by the wall-clock (`perf_counter`) share of the four sub-phases.")
    segs = spec.seg_order
    out.append(f"{prefix}**Energy per segment [J]** — source: `segment_energy.json` (kWh × 3.6e6) per run; GU = Σ of the four "
               f"allocated sub-segments (= measured `gradient_updates` task energy); TOTAL = Σ of all training segments "
               f"(= `TOTAL_MEASURED_TRAINING`; excludes warmup and idle windows). " + note)
    out.append(env_table(ds, tags, [(seg_h(s), lambda rs, s=s: msd(vals(rs, "E", s))) for s in segs]))
    out.append("TOTAL energy in kWh (mean ± sd) and the excluded phases [J] (warmup, idle head, idle tail; not in any total):")
    out.append(env_table(ds, tags,
                         [("TOTAL [kWh]", lambda rs: msd(vals(rs, "E_kwh", "TOTAL_MEASURED_TRAINING"))),
                          ("warmup [J]", lambda rs: msd(vals(rs, "E", "warmup"))),
                          ("idle head [J]", lambda rs: msd(vals(rs, "E", "idle_baseline_head"))),
                          ("idle tail [J]", lambda rs: msd(vals(rs, "E", "idle_baseline_tail")))]))
    # shares
    shares = [s for s in spec.train_segments] + ["gradient_updates"]
    ordered = [s for s in segs if s in shares]
    out.append(f"{prefix}**Shares [%]** — share of each segment in the TOTAL training energy (per seed, then mean ± sd); "
               "`GU` = sum of the four allocated sub-segments:")
    out.append(env_table(ds, tags, [(seg_h(s), lambda rs, s=s: msd(vals(rs, "share", s))) for s in ordered]))
    out.append("Share of each allocated sub-segment **within GU** [%] (= its wall-clock time share T_k / ΣT_j):")
    out.append(env_table(ds, tags, [(seg_h(s), lambda rs, s=s: msd(vals(rs, "share_gu", s))) for s in SUBS]))
    # J/FLOP
    js = jpf_segments(ds)
    cols = []
    for s in js:
        h = {"gradient_updates": "GU (derived: E_GU/(F_critic+F_actor))"}.get(s, SEG_SHORT[s])
        st = "" if s in ("gradient_updates", "TOTAL_MEASURED_TRAINING") else f" ({SEG_STATUS[s]})"
        cols.append((f"{h}{st} [J/FLOP]", lambda rs, s=s: jpf_cell(rs, s)))
    cols.append(("target_update (allocated) [nJ per Polyak op, NOT J/FLOP]", lambda rs: jpf_cell(rs, "target_update", 1e9)))
    out.append(f"{prefix}**Energy per FLOP** — J/FLOP = (mean energy over seeds) / (mean FLOPs over seeds) [ratio of means] "
               "± sd of the five per-seed ratios; matmul FLOPs from `flops_per_call.json` × call counts "
               "(Section 2); `target_update` is energy per elementwise Polyak operation; `buffer_sample` has no FLOPs. "
               "`TOTAL` = TOTAL energy / matmul FLOPs of all training segments except target_update:")
    out.append(env_table(ds, tags, cols))
    if spec.algo == "mbpo":
        cols2 = [(f"{SEG_SHORT[s]} [J/FLOP]", lambda rs, s=s: jpf_perseed_mean_cell(rs, s)) for s in js]
        out.append("MBPO FLOPs differ between seeds. **Mean of the per-seed ratios** (E_i/F_i averaged over seeds) ± sd, "
                   "for comparison with the ratio-of-means table above:")
        out.append(env_table(ds, tags, cols2))
    if with_env:
        both = [t for t in tags if ds.has("HC", t) and ds.has("Ant", t)]
        rows = []
        for t in both:
            hc, ant = srt(ds.runs_of("HC", t)), srt(ds.runs_of("Ant", t))
            cells = [t]
            for s in js:
                ja = np.mean(vals(ant, "E", s)) / np.mean(vals(ant, "F", s))
                jh = np.mean(vals(hc, "E", s)) / np.mean(vals(hc, "F", s))
                cells.append(g6(ja / jh))
            ja = np.mean(vals(ant, "E", "target_update")) / np.mean(vals(ant, "F", "target_update"))
            jh = np.mean(vals(hc, "E", "target_update")) / np.mean(vals(hc, "F", "target_update"))
            cells.append(g6(ja / jh))
            rows.append(cells)
        out.append("**J/FLOP ratio Ant / HC** (ratio of the two ratio-of-means J/FLOP values; `target` is a J/op ratio):")
        out.append(table(["config"] + [SEG_SHORT[s] for s in js] + ["target (J/op)"], rows))
    # time / power / throughput
    meas = spec.measured_order
    tcols = [(f"{SEG_SHORT[m]} (measured) duration [s]", lambda rs, m=m: msd(vals(rs, "D", m))) for m in meas]
    tcols += [(f"T {SEG_SHORT[s]} (perf_counter) [s]", lambda rs, s=s: msd([r["T"][s] for r in rs])) for s in SUBS]
    tcols += [("TOTAL duration (Σ measured tasks) [s]", lambda rs: msd(vals(rs, "D", "TOTAL_MEASURED_TRAINING")))]
    out.append(f"{prefix}**Duration [s]** — measured tasks: Σ over the task's CodeCarbon `duration`s (per-task CSV); "
               "allocated sub-segments: summed `perf_counter` times (`_sub_segment_wall_time_seconds`); TOTAL = Σ of "
               "the measured training tasks:")
    out.append(env_table(ds, tags, tcols))
    pcols = [(f"P {SEG_SHORT[m]} [W]", lambda rs, m=m: msd(vals(rs, "P", m))) for m in meas]
    pcols += [("P TOTAL [W]", lambda rs: msd(vals(rs, "P", "TOTAL_MEASURED_TRAINING"))),
              ("P whole run (Σ energy / Σ duration of all per-task rows: idle head + warmup + training tasks + idle tail) [W]", lambda rs: msd([r["P_run"] for r in rs])),
              ("P_rollout / P_GU", lambda rs: msd([r["P"]["rollout"] / r["P"]["gradient_updates"] for r in rs]))]
    out.append("**Mean power [W]** = per-seed energy / duration (measured tasks; TOTAL = Σ energy / Σ duration of the "
               "measured training tasks), then mean ± sd:")
    out.append(env_table(ds, tags, pcols))
    tp_cols = [(f"{SEG_SHORT[s]} [GFLOP/s]", lambda rs, s=s: msd([r["tp"][s] for r in rs]))
               for s in [m for m in meas] + ["TOTAL_MEASURED_TRAINING"] if s in ds.runs[0]["tp"]]
    tp_cols = [(f"{SEG_SHORT[s]} [GFLOP/s]", lambda rs, s=s: msd([r["tp"][s] for r in rs]))
               for s in meas + ["TOTAL_MEASURED_TRAINING"] if s in ds.runs[0]["tp"]]
    tp_cols += [("critic (FLOPs / T_perf) [GFLOP/s]", lambda rs: msd([r["F"]["critic_update"] / r["T"]["critic_update"] / 1e9 for r in rs])),
                ("actor (FLOPs / T_perf) [GFLOP/s]", lambda rs: msd([r["F"]["actor_update"] / r["T"]["actor_update"] / 1e9 for r in rs])),
                ("sub-timer coverage ΣT / D_GU [%] mean ± sd", lambda rs: msd([100 * r["coverage"] for r in rs])),
                ("coverage min–max [%]", lambda rs: f"{g6(100 * min(r['coverage'] for r in rs))}–{g6(100 * max(r['coverage'] for r in rs))}")]
    out.append("**Achieved throughput** = matmul FLOPs / duration [GFLOP/s] per seed (host wall time for the sub-segments; "
               "the GU row uses the measured GU task duration), and **sub-segment timer coverage** = Σ(four perf_counter "
               "times) / summed measured GU task duration:")
    out.append(env_table(ds, tags, tp_cols))
    # hardware composition
    for m in meas:
        cols = []
        for c, nm in (("cpu", "CPU"), ("gpu", "GPU"), ("ram", "RAM")):
            cols.append((f"{nm} share of {SEG_SHORT[m]} energy [%]",
                         lambda rs, c=c, m=m: msd([100 * r["hw"][m][c] / r["hw"][m]["E"] for r in rs])))
        cols.append((f"{SEG_SHORT[m]} energy in tasks shorter than 1 s polling interval: n tasks < 1 s of n tasks",
                     lambda rs, m=m: f"{sum(int((task_rows(r['tasks'], m)['duration'] < 1.0).sum()) for r in rs)}/"
                                     f"{sum(len(task_rows(r['tasks'], m)) for r in rs)}"))
        out.append(f"{prefix}**Hardware composition of the measured `{m}` task** (per-task CSV `cpu_energy`, `gpu_energy`, "
                   "`ram_energy` summed over the task's rows; per-run percentage, then mean ± sd; the last column counts "
                   "tasks (over all 5 seeds) whose CodeCarbon `duration` < `measure_power_secs` = 1.0 s):")
        out.append(env_table(ds, tags, cols))
    cols = [(f"{SEG_SHORT[m]} [%]", lambda rs, m=m: msd([r["idle_frac"][m] for r in rs])) for m in meas]
    cols.append(("TOTAL [%]", lambda rs: msd([r["idle_frac"]["TOTAL_MEASURED_TRAINING"] for r in rs])))
    cols.append(("P_idle (mean of head/tail, as recorded) [W]", lambda rs: msd([r["P_idle"] for r in rs])))
    out.append("**Idle-floor fraction** (definition of Section 4.1): P̄_idle × duration / energy, P̄_idle = mean of the run's "
               "as-recorded idle-head and idle-tail mean power; diagnostic only, nothing is subtracted:")
    out.append(env_table(ds, tags, cols))
    # response
    if with_response:
        rtags = [t for t in tags if t != base_tag]
        out.append(response_block(ds, rtags, base_tag, prefix))
    if with_env:
        out.append(env_comparison_block(ds, tags, prefix))
    return "\n".join(out)


# =============================================================================== response to the sweep
def response_block(ds: Dataset, rtags, base_tag, prefix=""):
    spec = ds.spec
    out = []
    if not rtags:
        return ""
    meas = spec.measured_order

    def base_of(es):
        return srt(ds.runs_of(es, base_tag))

    def rc(f):
        def fn_es(rs, es):
            return pm(paired_rel(rs, base_of(es), f))
        return fn_es

    def tbl(cols, rtags_=rtags):
        rows = []
        for t in rtags_:
            cells = [t]
            for _, f in cols:
                parts = []
                for es, _e in ENVS:
                    rs = srt(ds.runs_of(es, t))
                    if not rs or not base_of(es):
                        parts.append("n/a")
                    else:
                        parts.append(pm(paired_rel(rs, base_of(es), f)))
                cells.append(SEP.join(parts))
            rows.append(cells)
        return table(["config"] + [h for h, _ in cols], rows)

    out.append(f"{prefix}**Response to the sweep — paired relative change vs the same environment's baseline `{base_tag}` [%]** "
               "(per seed: config / baseline − 1, then mean ± sd over the 5 seeds; baseline = same seed):")
    out.append("Energy:")
    out.append(tbl([(f"E {SEG_SHORT[s]}", lambda r, s=s: r["E"][s]) for s in spec.seg_order]))
    out.append("Duration:")
    out.append(tbl([(f"D {SEG_SHORT[m]}", lambda r, m=m: r["D"][m]) for m in meas]
                   + [("D TOTAL", lambda r: r["D"]["TOTAL_MEASURED_TRAINING"])]
                   + [(f"T {SEG_SHORT[s]} (perf_counter)", lambda r, s=s: r["T"][s]) for s in SUBS]))
    out.append("Mean power:")
    out.append(tbl([(f"P {SEG_SHORT[m]}", lambda r, m=m: r["P"][m]) for m in meas]
                   + [("P TOTAL", lambda r: r["P"]["TOTAL_MEASURED_TRAINING"]),
                      ("P whole run", lambda r: r["P_run"])]))
    out.append("Achieved throughput:")
    out.append(tbl([(f"{SEG_SHORT[m]} GFLOP/s", lambda r, m=m: r["tp"][m]) for m in meas if m in ds.runs[0]["tp"]]
                   + [("TOTAL GFLOP/s", lambda r: r["tp"]["TOTAL_MEASURED_TRAINING"])]))
    out.append("Energy per FLOP (J/FLOP, per-seed ratio):")
    out.append(tbl([(f"{SEG_SHORT[s]}", lambda r, s=s: r["E"][s] / r["F"][s]) for s in jpf_segments(ds)]
                   + [("target (J/op)", lambda r: r["E"]["target_update"] / r["F"]["target_update"])]))
    # FLOP factors (x of baseline, paired)
    rows = []
    fsegs = [s for s in jpf_segments(ds)] + ["target_update"]
    for t in rtags:
        cells = [t]
        for s in fsegs:
            parts = []
            for es, _e in ENVS:
                rs = srt(ds.runs_of(es, t))
                if not rs or not base_of(es):
                    parts.append("n/a")
                    continue
                bm = {r["seed"]: r for r in base_of(es)}
                fac = np.array([r["F"][s] / bm[r["seed"]]["F"][s] for r in rs])
                parts.append(g6(fac.mean()) + ("" if fac.std(ddof=1) < 1e-12 * abs(fac.mean()) else f" ± {g6(fac.std(ddof=1))}"))
            cells.append(SEP.join(parts))
        rows.append(cells)
    out.append("FLOP factors (config FLOPs / baseline FLOPs, per seed, mean; `± sd` only where FLOPs differ between seeds; "
               "`target` = factor of the Polyak operation count):")
    out.append(table(["config"] + [SEG_SHORT[s] for s in fsegs], rows))
    # share change in pp
    rows = []
    shs = [s for s in spec.seg_order if s in spec.train_segments or s == "gradient_updates"]
    for t in rtags:
        cells = [t]
        for s in shs:
            parts = []
            for es, _e in ENVS:
                rs = srt(ds.runs_of(es, t))
                if not rs or not base_of(es):
                    parts.append("n/a")
                    continue
                bm = {r["seed"]: r for r in base_of(es)}
                d = np.array([r["share"][s] - bm[r["seed"]]["share"][s] for r in rs])
                parts.append(pm(d))
            cells.append(SEP.join(parts))
        rows.append(cells)
    out.append("Change of the share of TOTAL energy vs baseline [percentage points, paired mean ± sd]:")
    out.append(table(["config"] + [SEG_SHORT[s] for s in shs], rows))
    return "\n".join(out)


# =============================================================================== environment comparison
def env_comparison_block(ds: Dataset, tags, prefix=""):
    spec = ds.spec
    both = [t for t in tags if ds.has("HC", t) and ds.has("Ant", t)]
    if not both:
        return ""
    out = []
    shs = [s for s in spec.train_segments]
    rows = []
    for t in both:
        hc, ant = srt(ds.runs_of("HC", t)), srt(ds.runs_of("Ant", t))
        eT = np.mean(vals(ant, "E", "TOTAL_MEASURED_TRAINING")) / np.mean(vals(hc, "E", "TOTAL_MEASURED_TRAINING"))
        eR = np.mean(vals(ant, "E", "rollout")) / np.mean(vals(hc, "E", "rollout"))
        diffs = {s: np.mean(vals(ant, "share", s)) - np.mean(vals(hc, "share", s)) for s in shs}
        smax = max(diffs, key=lambda s: abs(diffs[s]))
        pT = np.mean(vals(ant, "P", "TOTAL_MEASURED_TRAINING")) / np.mean(vals(hc, "P", "TOTAL_MEASURED_TRAINING"))
        rr_a = np.mean(vals(ant, "P", "rollout")) / np.mean(vals(ant, "P", "gradient_updates"))
        rr_h = np.mean(vals(hc, "P", "rollout")) / np.mean(vals(hc, "P", "gradient_updates"))
        jf = (np.mean(vals(ant, "E", "TOTAL_MEASURED_TRAINING")) / np.mean(vals(ant, "F", "TOTAL_MEASURED_TRAINING"))) / \
             (np.mean(vals(hc, "E", "TOTAL_MEASURED_TRAINING")) / np.mean(vals(hc, "F", "TOTAL_MEASURED_TRAINING")))
        rows.append([t, g6(eT), g6(eR), g6(abs(diffs[smax])), SEG_SHORT[smax], g6(pT), g6(rr_a / rr_h), g6(jf)])
    out.append(f"{prefix}**Environment comparison, Ant-v5 relative to HalfCheetah-v5** (ratios of the means over the 5 seeds; "
               "shares of TOTAL energy; max |share difference| = largest |mean share Ant − mean share HC| over all training "
               "segments incl. algorithm-specific ones, GU excluded because it is the sum of its sub-segments; "
               "P_rollout/P_GU double ratio = (P_rollout/P_GU)_Ant / (P_rollout/P_GU)_HC from the environment means):")
    out.append(table(["config", "E_TOTAL Ant/HC", "E_rollout Ant/HC", "max |Δshare| [pp]", "attained by", "P_TOTAL Ant/HC",
                      "(P_roll/P_GU) Ant/HC", "J/FLOP TOTAL Ant/HC"], rows))
    # paired per-seed ratios (sd)
    rows = []
    for t in both:
        hc, ant = srt(ds.runs_of("HC", t)), srt(ds.runs_of("Ant", t))
        cells = [t]
        for s in ["rollout"] + [x for x in spec.train_segments if x != "rollout"] + ["gradient_updates", "TOTAL_MEASURED_TRAINING"]:
            em = np.mean(vals(ant, "E", s)) / np.mean(vals(hc, "E", s))
            ep = np.array([a["E"][s] / h["E"][s] for h, a in zip(hc, ant)])
            cells.append(f"{g6(em)} (paired sd {g6(ep.std(ddof=1))})")
        rows.append(cells)
    out.append("Energy ratio Ant/HC per segment (ratio of means; sd of the five paired per-seed ratios):")
    out.append(table(["config"] + [SEG_SHORT[s] for s in ["rollout"] + [x for x in spec.train_segments if x != "rollout"] + ["gradient_updates", "TOTAL_MEASURED_TRAINING"]], rows))
    # share differences with sign agreement
    rows = []
    for t in both:
        hc, ant = srt(ds.runs_of("HC", t)), srt(ds.runs_of("Ant", t))
        cells = [t]
        for s in shs:
            d = np.array([a["share"][s] - h["share"][s] for h, a in zip(hc, ant)])
            agree = int((np.sign(d) == np.sign(d.mean())).sum())
            cells.append(f"{g6(d.mean())} ± {g6(d.std(ddof=1))} ({agree}/5 same sign)")
        rows.append(cells)
    out.append("Per-segment **share difference Ant − HC [pp]**, paired by seed: mean ± sd of the five per-seed differences, "
               "and the number of seeds (of 5) whose difference has the same sign as the mean:")
    out.append(table(["config"] + [SEG_SHORT[s] for s in shs], rows))
    rows = []
    for t in both:
        hc, ant = srt(ds.runs_of("HC", t)), srt(ds.runs_of("Ant", t))
        cells = [t]
        for name, f in (("rollout share [pp]", lambda r: r["share"]["rollout"]),
                        ("TOTAL energy [J]", lambda r: r["E"]["TOTAL_MEASURED_TRAINING"])):
            d = np.array([f(a) - f(h) for h, a in zip(hc, ant)])
            p, cnt = signflip_p(d)
            cells.append(f"mean Δ {g6(d.mean())}; {cnt}/32 patterns; p = {g6(p)}")
        rows.append(cells)
    out.append("Exact paired two-sided sign-flip permutation test (all 2⁵ = 32 sign patterns; statistic |mean of the paired "
               "differences Ant − HC|; p = #{patterns with |mean| ≥ observed}/32; smallest attainable p = 2/32 = 0.0625):")
    out.append(table(["config", "rollout share", "TOTAL energy"], rows))
    return "\n".join(out)


# =============================================================================== CSV exports
def export_csvs(ds: Dataset, out_dir: Path, pe: pd.DataFrame, prefix: str):
    spec = ds.spec
    rows = []
    for r in ds.runs:
        for s in spec.seg_order + ["warmup", "idle_baseline_head", "idle_baseline_tail"]:
            ftype = ("mixed_total" if s == "TOTAL_MEASURED_TRAINING" else "elementwise" if s == "target_update"
                     else "none" if s in ("buffer_sample", "warmup", "idle_baseline_head", "idle_baseline_tail") else "matmul")
            fl = r["F"].get(s)
            rows.append(dict(
                config=r["tag"], env=r["env"], seed=r["seed"], segment=s, status=SEG_STATUS[s],
                energy_kwh=r["E_kwh"][s], energy_j=r["E"][s], duration_s=r["D"].get(s), mean_power_w=r["P"].get(s),
                flops=fl, call_count=r["calls"].get(s), flop_type=ftype,
                j_per_flop=(r["E"][s] / fl if fl else None),
                share_of_total_pct=r["share"].get(s) if s in r["share"] else (100.0 if s == "TOTAL_MEASURED_TRAINING" else None),
                share_of_gu_pct=r["share_gu"].get(s) if s in SUBS else (100.0 if s == "gradient_updates" else None),
                idle_floor_fraction_pct=r["idle_frac"].get(s)))
    pd.DataFrame(rows).to_csv(out_dir / f"{prefix}_segments_long.csv", index=False, float_format="%.17g")
    inv = []
    for r in ds.runs:
        md = r["meta"]
        ec, ac = md["experiment_config"], md["algo_config"]
        inv.append(dict(algo=ds.algo, config=r["tag"], env=r["env"], seed=r["seed"], run_dir=r["run_dir"],
                        timestamp=r["timestamp"], git_commit=md["git_commit"], torch_version=md["torch_version"],
                        codecarbon_version=md.get("codecarbon_version"), gpu_name=md.get("cuda_device_name"),
                        gpu_min_clock_mhz=ec.get("gpu_min_clock_mhz"), gpu_max_clock_mhz=ec.get("gpu_max_clock_mhz"),
                        warmup_steps=ec["warmup_steps"], train_steps=ec["train_steps"], steps_per_epoch=ec["steps_per_epoch"],
                        start_time_utc=md["start_time_utc"], end_time_utc=md["end_time_utc"],
                        algo_config=json.dumps(ac, sort_keys=True)))
    pd.DataFrame(inv).to_csv(out_dir / f"{prefix}_run_inventory.csv", index=False)
    pe.to_csv(out_dir / f"{prefix}_per_epoch.csv", index=False, float_format="%.17g")
    summ = []
    for es, t in ds.cfgs():
        rs = srt(ds.runs_of(es, t))
        for s in spec.seg_order + ["warmup", "idle_baseline_head", "idle_baseline_tail"]:
            E = np.array(vals(rs, "E", s))
            D = np.array([r["D"][s] for r in rs])
            P = np.array([r["P"][s] for r in rs])
            d = dict(config=t, env=rs[0]["env"], segment=s, status=SEG_STATUS[s], n=len(rs),
                     energy_j_mean=E.mean(), energy_j_sd=E.std(ddof=1), energy_j_min=E.min(), energy_j_max=E.max(),
                     energy_kwh_mean=E.mean() / KWH_TO_J, duration_s_mean=D.mean(), duration_s_sd=D.std(ddof=1),
                     mean_power_w_mean=P.mean(), mean_power_w_sd=P.std(ddof=1))
            if s in rs[0]["share"]:
                sv = np.array(vals(rs, "share", s))
                d.update(share_of_total_pct_mean=sv.mean(), share_of_total_pct_sd=sv.std(ddof=1))
            if s in SUBS:
                sv = np.array(vals(rs, "share_gu", s))
                d.update(share_of_gu_pct_mean=sv.mean(), share_of_gu_pct_sd=sv.std(ddof=1))
            if s in rs[0]["F"] and rs[0]["F"][s]:
                Fv = np.array([r["F"][s] for r in rs])
                per = E / Fv
                d.update(flops_mean=Fv.mean(), flops_min=Fv.min(), flops_max=Fv.max(), flops_sd=Fv.std(ddof=1),
                         j_per_flop_ratio_of_means=E.mean() / Fv.mean(), j_per_flop_per_seed_mean=per.mean(),
                         j_per_flop_per_seed_sd=per.std(ddof=1))
            summ.append(d)
    pd.DataFrame(summ).to_csv(out_dir / f"{prefix}_cross_seed_summary.csv", index=False, float_format="%.17g")
    idle = []
    for r in ds.runs:
        th, tt = task_rows(r["tasks"], "idle_baseline_head").iloc[0], task_rows(r["tasks"], "idle_baseline_tail").iloc[0]
        wh = th["ram_energy"] * KWH_TO_J / th["ram_power"]
        wt = tt["ram_energy"] * KWH_TO_J / tt["ram_power"]
        ph, pt = r["P"]["idle_baseline_head"], r["P"]["idle_baseline_tail"]
        ph_c, pt_c = r["E"]["idle_baseline_head"] / wh, r["E"]["idle_baseline_tail"] / wt
        idle.append(dict(config=r["tag"], env=r["env"], seed=r["seed"], run_dir=r["run_dir"], P_head_w=ph, P_tail_w=pt,
                         signed_tail_minus_head_w=pt - ph, rel_head_tail_diff=abs(pt - ph) / ph,
                         head_duration_s=th["duration"], head_integration_window_s=wh, tail_duration_s=tt["duration"],
                         tail_integration_window_s=wt, P_head_window_corrected_w=ph_c, P_tail_window_corrected_w=pt_c,
                         rel_diff_window_corrected=abs(pt_c - ph_c) / ph_c))
    IDLE = pd.DataFrame(idle)
    IDLE.to_csv(out_dir / f"{prefix}_idle_floor.csv", index=False, float_format="%.17g")
    return IDLE


# =============================================================================== Section 5.1 marginal cost
def sweep_points(ds, es, tags, seg):
    return [(r["F"][seg], r["E"][seg], t, r["seed"]) for t in tags for r in srt(ds.runs_of(es, t))]


def ols_block(ds: Dataset, out_dir: Path, prefix: str):
    spec = ds.spec
    out = []
    rows, csv_rows, skipped = [], [], []
    for name, text, tags, xdim in spec.sweeps:
        for es, env_id in ENVS:
            if not all(ds.has(es, t) for t in tags):
                skipped.append(f"{name} / {env_id}: not all configurations of the sweep exist in this environment "
                               f"({[t for t in tags if not ds.has(es, t)]} missing) — no fit")
                continue
            for s in ("gradient_updates", "TOTAL_MEASURED_TRAINING"):
                cfg_means = {t: np.mean([r["F"][s] for r in ds.runs_of(es, t)]) for t in tags}
                distinct = len({round(v / max(cfg_means.values()), 9) for v in cfg_means.values()})
                if distinct < 3:
                    skipped.append(f"{name} / {env_id} / {SEG_SHORT[s]}: only {distinct} distinct FLOP value(s) among the "
                                   f"{len(tags)} configurations ({', '.join(f'{t}={g6(v)}' for t, v in cfg_means.items())}) — no fit")
                    continue
                pts = sweep_points(ds, es, tags, s)
                F = np.array([p_[0] for p_ in pts], float)
                E = np.array([p_[1] for p_ in pts], float)
                b, a = np.polyfit(F, E, 1)
                pred = a + b * F
                ssr = ((E - pred) ** 2).sum()
                r2 = 1 - ssr / ((E - E.mean()) ** 2).sum()
                rse = math.sqrt(ssr / (len(E) - 2))
                rows.append([name, env_id, ", ".join(tags), SEG_SHORT[s], len(E), distinct, g6(a), g6(b * 1e12), g6(r2), g6(rse)])
                csv_rows.append(dict(sweep=name, env=env_id, configs="+".join(tags), segment=s, n=len(E),
                                     distinct_flop_values=distinct, a_J=a, b_J_per_FLOP=b, b_pJ_per_FLOP=b * 1e12,
                                     r2=r2, rse_J=rse))
    out.append("**5.1a Fixed-versus-marginal fit E = a + b·F** — ordinary least squares over per-run points (one point per run: "
               "x = the run's FLOPs of the segment, y = its gross energy of the same segment; n = 5 × number of "
               "configurations of the sweep), exactly as in Section 4.1 (`ols_section`). a = intercept [J], b = slope "
               "[pJ/FLOP], residual SE = sqrt(SSR/(n−2)) [J]. GU = `gradient_updates` (FLOPs = critic + actor); "
               "TOTAL = `TOTAL_MEASURED_TRAINING`. A sweep with fewer than 3 distinct per-configuration FLOP values is skipped:")
    out.append(table(["sweep", "env", "configurations", "segment", "n", "distinct FLOP values", "a [J]", "b [pJ/FLOP]", "R²",
                      "residual SE [J]"], rows))
    if skipped:
        out.append("Skipped fits:\n" + "\n".join("- " + s for s in skipped))
    pd.DataFrame(csv_rows).to_csv(out_dir / f"{prefix}_ols_fits.csv", index=False, float_format="%.17g")
    return "\n".join(out)


def delta_block(ds: Dataset, out_dir: Path, prefix: str, x_dims):
    import flop_dashboard as fd
    out = []
    pr = fd.load_per_run(str(REPO / "flop_analysis" / "output"))
    pr = pr[(pr.algo == ds.algo) & (pr.included_in_cross_seed_avg)]
    cs = fd.load_cross_seed(str(REPO / "flop_analysis" / "output"))
    cs = cs[cs.algo == ds.algo]
    out.append(f"**5.1b Pairwise ΔEnergy/ΔFLOPs** — `compute_delta_pairs()` imported from `flop_dashboard.py` and called "
               f"unmodified (inputs: `per_run_energy_per_flop.csv` loaded with `load_per_run`, rows of `{ds.algo}` with "
               f"`included_in_cross_seed_avg`, `PER_RUN_DIMENSIONS` so the seed is held fixed; and `cross_seed_energy_per_flop.csv` "
               f"with `DIMENSIONS`). Rule: pairs differ only in the X dimension; matmul/mixed_total rows only; pairs with |ΔF| < "
               f"{fd.MIN_REL_DELTA_FLOPS:.0%} of the reference FLOPs are dropped; pooled value = ΣΔE/ΣΔF over the used pairs. "
               "Reference = smallest X value (`first`) or the next-smaller one (`previous`). Pair list: "
               f"`contexts/Section_{ds.spec.section}_data/{prefix}_delta_pairs.csv`.")
    all_pairs, pooled = [], {}
    for x_dim, name in x_dims:
        for mode in ("first", "previous"):
            for level, df, dims, ecol in (("per-run", pr, fd.PER_RUN_DIMENSIONS, "total_energy_joules"),
                                          ("cross-seed", cs, fd.DIMENSIONS, "mean_energy_joules")):
                pairs, n_ex = fd.compute_delta_pairs(df, dims, x_dim, ecol, "total_flops", mode)
                if pairs.empty:
                    continue
                pairs = pairs.copy()
                pairs["x_dim"], pairs["ref_mode"], pairs["level"] = x_dim, mode, level
                all_pairs.append(pairs)
                xcol = dims[x_dim]["col"]
                for (env, seg), g in pairs.groupby(["env_id", "segment"]):
                    ok = g[g.status == "ok"]
                    val = ok.delta_energy_j.sum() / ok.delta_flops.sum() if len(ok) else float("nan")
                    xs = ", ".join(f"{a}←{b}" for a, b in sorted(set(zip(ok[xcol].astype(str), ok["reference_value"].astype(str)))))
                    pooled[(name, mode, level, ENV_SHORT[env], seg)] = (val, len(ok), len(g) - len(ok), xs)
    if all_pairs:
        pd.concat(all_pairs, ignore_index=True).to_csv(out_dir / f"{prefix}_delta_pairs.csv", index=False, float_format="%.17g")
    segs = ["rollout", "critic_update", "actor_update"] + [s for s in ds.spec.extra_measured] + ["TOTAL_MEASURED_TRAINING"]
    rows = []
    for _, name in x_dims:
        for mode in ("first", "previous"):
            for seg in segs:
                h_ = pooled.get((name, mode, "per-run", "HC", seg))
                a_ = pooled.get((name, mode, "per-run", "Ant", seg))
                if not h_ and not a_:
                    continue
                fmt = lambda v: "n/a" if v is None else ("dropped (ΔF≈0)" if math.isnan(v[0]) else g6(v[0] * 1e12))  # noqa: E731
                rows.append([name, mode, SEG_SHORT[seg], (h_ or a_)[3] or "—", fmt(h_), fmt(a_),
                             f"{h_[1] if h_ else 'n/a'} / {a_[1] if a_ else 'n/a'}",
                             f"{h_[2] if h_ else 'n/a'} / {a_[2] if a_ else 'n/a'}"])
    out.append(table(["sweep", "reference", "segment", "X pairs (x←ref)", "HC ΣΔE/ΣΔF [pJ/FLOP]", "Ant ΣΔE/ΣΔF [pJ/FLOP]",
                      "pairs used HC / Ant", "pairs dropped HC / Ant"], rows))
    diffs = [abs(v[0] - pooled[(k[0], k[1], "cross-seed", k[3], k[4])][0]) / abs(v[0]) for k, v in pooled.items()
             if k[2] == "per-run" and not math.isnan(v[0]) and (k[0], k[1], "cross-seed", k[3], k[4]) in pooled
             and not math.isnan(pooled[(k[0], k[1], "cross-seed", k[3], k[4])][0]) and v[0] != 0]
    if diffs:
        out.append(f"Per-run vs cross-seed pooled values: max relative difference {max(diffs):.2e} over {len(diffs)} comparable "
                   "(sweep, reference, env, segment) cells. The cross-seed CSV has no `gradient_updates` row, so the dashboard "
                   "rule yields no GU pooled value; the OLS slope b in 5.1a is the closest equivalent.")
    return "\n".join(out)


# =============================================================================== Section 5.2 measurement quality
def quality_block(ds: Dataset, pe: pd.DataFrame, IDLE: pd.DataFrame, out_dir: Path, prefix: str):
    spec = ds.spec
    out = []
    tasks = spec.per_epoch_tasks
    mps = ds.runs[0]["meta"]["experiment_config"]["measure_power_secs"]
    tags = [t for t in spec.tags]

    def sub(es, t):
        return pe[(pe.env == dict(ENVS)[es]) & (pe.tag == t)]

    def ec(fn):
        rows = []
        return rows

    def build(cols):
        rows = []
        for t in tags:
            cells = [t]
            for _, fn in cols:
                parts = []
                for es, _e in ENVS:
                    if not ds.has(es, t):
                        parts.append("n/a")
                    else:
                        parts.append(fn(es, t))
                cells.append(SEP.join(parts))
            rows.append(cells)
        return table(["config"] + [h for h, _ in cols], rows)

    out.append(f"**5.2a Task durations and the 1 s polling interval** — all per-epoch measured tasks of all 5 seeds "
               f"(100 epochs × 5 seeds = 500 tasks per task type; `measure_power_secs` = {mps} s). Cell: median [min–max] "
               "of the single-task CodeCarbon `duration` [s]; and number of tasks shorter than the polling interval / total:")
    cols = []
    for k in tasks:
        cols.append((f"{SEG_SHORT[k]} duration median [min–max] s",
                     lambda es, t, k=k: (lambda d: f"{g6(d.median())} [{g6(d.min())}–{g6(d.max())}]")(sub(es, t)[sub(es, t).task == k].duration_s)))
        cols.append((f"{SEG_SHORT[k]} tasks < 1 s",
                     lambda es, t, k=k: (lambda d: f"{int((d < mps).sum())}/{len(d)}")(sub(es, t)[sub(es, t).task == k].duration_s)))
    out.append(build(cols))
    cnt_rows = []
    for k in tasks:
        d = pe[pe.task == k].duration_s
        cnt_rows.append([k, int((d < mps).sum()), len(d), g6(d.min()), g6(d.median()), g6(d.max())])
    singles = [m for m in spec.measured_order if m not in tasks]
    out.append(f"Across all {len(ds.runs)} runs: " + "; ".join(f"`{r_[0]}`: {r_[1]} of {r_[2]} tasks < {mps} s (min/median/max "
                                                           f"{r_[3]}/{r_[4]}/{r_[5]} s)" for r_ in cnt_rows) +
               (". Single (non-per-epoch) measured tasks: " + ", ".join(
                   f"`{m}` duration {msd([r['hw'][m]['duration'] for r in ds.runs])} s" for m in singles) if singles else "") + ".")
    # span deviation
    cols = []
    for k in tasks:
        cols.append((f"{SEG_SHORT[k]} integration span / duration: median [min–max]",
                     lambda es, t, k=k: (lambda w: f"{g6((w.ram_integration_window_s / w.duration_s).median())} "
                                                   f"[{g6((w.ram_integration_window_s / w.duration_s).min())}–"
                                                   f"{g6((w.ram_integration_window_s / w.duration_s).max())}]")(sub(es, t)[sub(es, t).task == k])))
    out.append("**5.2b Integration span versus task duration** — span = `ram_energy / ram_power` of the task row (CodeCarbon "
               "models RAM at constant power, so this is the time over which the task's energy was integrated); the cell is the "
               "ratio span/duration (1 = integrated over exactly the recorded duration):")
    out.append(build(cols))
    # coverage + CV + epoch stats
    stats = {}
    for r in ds.runs:
        for k in tasks:
            v = pe[(pe.env == r["env"]) & (pe.tag == r["tag"]) & (pe.seed == r["seed"]) & (pe.task == k)].sort_values("epoch")
            y, yd = v.energy_j.to_numpy(), v.duration_s.to_numpy()
            stats[(r["es"], r["tag"], r["seed"], k)] = dict(
                cv=100 * y.std(ddof=1) / y.mean(), ep0_ratio=y[0] / y[1:].mean(),
                dur_trend=100 * (yd[:10].mean() - yd[10:].mean()) / yd[10:].mean(),
                en_trend=100 * (y[:10].mean() - y[10:].mean()) / y[10:].mean())

    def stat_cell(key, fmt="msd"):
        def fn(es, t):
            vv = [stats[(es, t, s, key[0])][key[1]] for s in CANON]
            return msd(vv)
        return fn
    cols = []
    for k in tasks:
        cols.append((f"{SEG_SHORT[k]} per-epoch energy CV within run [%]", stat_cell((k, "cv"))))
    for k in tasks:
        cols.append((f"{SEG_SHORT[k]} energy epoch 0 / mean(epochs 1–99)", stat_cell((k, "ep0_ratio"))))
    for k in tasks:
        cols.append((f"{SEG_SHORT[k]} duration (mean ep 0–9 − mean ep 10–99)/mean ep 10–99 [%]", stat_cell((k, "dur_trend"))))
    out.append("**5.2c Per-epoch stationarity** (per run: CV of the 100 per-epoch energies; ratio of the epoch-0 energy to the mean "
               "of epochs 1–99; relative difference of the mean duration of epochs 0–9 versus epochs 10–99; then mean ± sd over the "
               "5 seeds; per-epoch values in the per-epoch CSV):")
    out.append(build(cols))
    cols = [("CV of TOTAL energy over seeds [%]", lambda es, t: g6(100 * np.std(vals(srt(ds.runs_of(es, t)), "E", "TOTAL_MEASURED_TRAINING"), ddof=1)
                                                               / np.mean(vals(srt(ds.runs_of(es, t)), "E", "TOTAL_MEASURED_TRAINING")))),
            ("allocation coverage min–max [%]", lambda es, t: f"{g6(100 * min(r['coverage'] for r in ds.runs_of(es, t)))}–{g6(100 * max(r['coverage'] for r in ds.runs_of(es, t)))}")]
    for m in spec.measured_order:
        cols.append((f"CV of {SEG_SHORT[m]} energy over seeds [%]", lambda es, t, m=m: g6(100 * np.std(vals(srt(ds.runs_of(es, t)), "E", m), ddof=1) / np.mean(vals(srt(ds.runs_of(es, t)), "E", m))) if m != "gradient_updates" else g6(100 * np.std(vals(srt(ds.runs_of(es, t)), "E", "gradient_updates"), ddof=1) / np.mean(vals(srt(ds.runs_of(es, t)), "E", "gradient_updates")))))
    out.append("**5.2d Between-seed variability and allocation coverage:**")
    out.append(build(cols))
    # idle drift
    cols = [("P_head [W]", lambda es, t: msd(IDLE[(IDLE.env == dict(ENVS)[es]) & (IDLE.config == t)].P_head_w)),
            ("P_tail [W]", lambda es, t: msd(IDLE[(IDLE.env == dict(ENVS)[es]) & (IDLE.config == t)].P_tail_w)),
            ("P_tail − P_head [W]", lambda es, t: msd(IDLE[(IDLE.env == dict(ENVS)[es]) & (IDLE.config == t)].signed_tail_minus_head_w)),
            ("|P_tail − P_head| / P_head [%]", lambda es, t: msd(100 * IDLE[(IDLE.env == dict(ENVS)[es]) & (IDLE.config == t)].rel_head_tail_diff))]
    out.append("**5.2e Idle head versus tail** (as-recorded mean power = idle-task energy / its CodeCarbon `duration`):")
    out.append(build(cols))
    a = IDLE.rel_head_tail_diff * 100
    worst = IDLE.loc[IDLE.rel_head_tail_diff.idxmax()]
    ac_ = IDLE.rel_diff_window_corrected * 100
    worst_c = IDLE.loc[IDLE.rel_diff_window_corrected.idxmax()]
    out.append(f"Over all {len(IDLE)} runs of {spec.name}: |P_tail − P_head|/P_head (as recorded) median {g6(a.median())} %, mean "
               f"{g6(a.mean())} %, maximum {g6(a.max())} % (run `{worst.run_dir}`, config {worst.config}, {worst.env}, seed "
               f"{worst.seed}); tail > head in {int((IDLE.signed_tail_minus_head_w > 0).sum())}/{len(IDLE)} runs. "
               f"Window-corrected (energy ÷ integrated span, see `idle_power_analysis.py`): median {g6(ac_.median())} %, maximum "
               f"{g6(ac_.max())} % (run `{worst_c.run_dir}`). Median head integration span "
               f"{g6(IDLE.head_integration_window_s.median())} s vs recorded duration {g6(IDLE.head_duration_s.median())} s; tail "
               f"{g6(IDLE.tail_integration_window_s.median())} s vs {g6(IDLE.tail_duration_s.median())} s.")
    # MAD flags
    flags = []
    n_cells = 0
    for es, t in ds.cfgs():
        rs = srt(ds.runs_of(es, t))
        for s in spec.seg_order:
            v = np.array(vals(rs, "E", s))
            med = np.median(v)
            mad = np.median(np.abs(v - med))
            n_cells += len(rs)
            for r, x in zip(rs, v):
                if mad > 0 and abs(x - med) > 3 * mad:
                    flags.append(dict(env=r["env"], config=t, seed=r["seed"], segment=s, value_j=x, median_j=med, mad_j=mad,
                                      abs_dev_over_mad=abs(x - med) / mad, dev_pct_of_median=100 * abs(x - med) / med))
    FL = pd.DataFrame(flags)
    FL.to_csv(out_dir / f"{prefix}_mad_flags.csv", index=False, float_format="%.17g")
    big = FL.loc[FL.dev_pct_of_median.idxmax()] if len(FL) else None
    out.append(f"**5.2f Outlier flags** — rule: |x − median| > 3 × MAD (unscaled MAD, n = 5 per configuration), applied to the energy "
               f"of every segment of the energy table (incl. GU and TOTAL). Flagged, **not excluded**: {len(FL)} of {n_cells} "
               f"(run, segment) cells" + (f"; largest deviation {g6(big.dev_pct_of_median)} % of the median ({big.env} {big.config} "
                                         f"seed {int(big.seed)} {SEG_SHORT[big.segment]}: {g6(big.value_j)} J vs median {g6(big.median_j)} J)"
                                         if big is not None else "") + f". Full list: `contexts/Section_{spec.section}_data/{prefix}_mad_flags.csv`.")
    if len(FL):
        cnt = FL.groupby(["env", "config"]).size()
        out.append("Flag count per configuration: " + ", ".join(f"{e}/{c}: {n}" for (e, c), n in cnt.items()) + ".")
    return "\n".join(out)


# =============================================================================== Section 5.3 episodes / returns
def learning_block(ds: Dataset, out_dir: Path, prefix: str):
    perf = []
    for r in ds.runs:
        eps = r["tm"]["episodes"]
        ret = [e["return"] for e in eps if e["phase"] == "train"]
        k = max(1, len(ret) // 10)
        perf.append(dict(config=r["tag"], env=r["env"], seed=r["seed"], n_train_episodes=len(ret), window_episodes=k,
                         final_10pct_mean_return=sum(ret[-k:]) / k, max_return=max(ret)))
    PF = pd.DataFrame(perf)
    PF.to_csv(out_dir / f"{prefix}_learning_performance.csv", index=False, float_format="%.17g")
    rows = []
    for t in ds.spec.tags:
        cells = [t]
        for es, env_id in ENVS:
            sub = PF[(PF.env == env_id) & (PF.config == t)]
            if sub.empty:
                cells.append("n/a")
                continue
            cells.append(f"episodes {sub.n_train_episodes.min()}–{sub.n_train_episodes.max()}; window {sub.window_episodes.min()}–"
                         f"{sub.window_episodes.max()}; return {msd(sub.final_10pct_mean_return)}")
        rows.append(cells)
    return ("**5.3 Episodes and returns (sanity information only)** — definitions of `aggregate_results.py`: episodes = "
            "`training_metrics.json[\"episodes\"]` with `phase == \"train\"` (warmup and the trailing partial `train_incomplete` "
            "episode excluded); final-10 %-mean return = mean of the last max(1, n_episodes // 10) of them; returns are those of "
            "the stochastic training policy (no evaluation episodes). Cell: completed training episodes per run (min–max over "
            "the 5 seeds); number of episodes in the 10 % window (min–max); mean ± sd over seeds of the window mean return:\n"
            + table(["config", "HalfCheetah-v5", "Ant-v5"], rows))
