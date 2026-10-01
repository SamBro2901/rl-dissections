"""
Interactive Plotly Dash dashboard for browsing the energy-per-FLOP analysis
output (`flop_analysis/output/*.csv`), produced by
`flop_analysis/compute_energy_per_flop.py`.

Two tabs:
  - "Cross-seed comparison" -- cross_seed_energy_per_flop.csv, one row per
    (algo, env, architecture, UTD, segment) averaged over the canonical
    5-seed sweep.
  - "Per-run detail" -- per_run_energy_per_flop.csv, one row per
    (run, segment); includes dev/exploratory runs excluded from the
    cross-seed average (flagged via included_in_cross_seed_avg) and a
    per-seed breakdown.

Both tabs share the same pattern: a set of crossfiltering dropdowns (each
dropdown's options narrow based on every other dropdown's current selection)
to pick which rows are in play, then freely-assignable X-axis / Color /
Facet dropdowns to pivot the filtered rows into a grouped bar chart (or, on
the per-run tab, a box plot of the per-seed spread), plus a sortable/
filterable data table of the exact underlying rows for precise lookups --
together these cover arbitrary permutations/combinations of the CSV columns
without hardcoding a fixed set of charts.

Both tabs also offer a derived "ΔEnergy / ΔFLOPs" metric: the marginal energy
per extra FLOP between configurations that differ only in the X-axis
dimension (see DELTA_METRIC / compute_delta_pairs below).

Usage:
    python flop_dashboard.py [--flop-dir flop_analysis/output] [--port 8051]
"""
import argparse
import os
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, dash_table
from dash.dash_table.Format import Format, Scheme

FLOP_DIR = os.path.join("flop_analysis", "output")

SEGMENT_ORDER = [
    "idle_baseline_head", "warmup", "rollout", "buffer_sample",
    "critic_update", "actor_update", "target_update",
    "dynamics_model_update", "synthetic_rollout_generation",
    "world_model_pretrain", "idle_baseline_tail", "TOTAL_MEASURED_TRAINING",
]
ALGO_ORDER = ["sac", "td3", "mbpo", "tdmpc2"]
ALGO_COLORS = {"sac": "#1f77b4", "td3": "#2ca02c", "mbpo": "#d62728", "tdmpc2": "#9467bd"}

# Segments whose energy-per-FLOP is not a matmul-FLOP-normalized quantity
# (CPU-side sampling has 0 FLOPs; Polyak averaging is elementwise, not
# matmul) -- flagged in the UI rather than silently dropped, per the
# methodology's "measured vs. allocated / caveats" reporting standard.
NON_FLOP_SEGMENTS = {"buffer_sample", "target_update", "warmup", "idle_baseline_head", "idle_baseline_tail"}

# hidden_sizes ("256x256" / "512x512" / "1024x1024") is written as its own
# column by compute_energy_per_flop.py (flop_keys.hidden_sizes) so the width
# sweep can be compared independent of the other architecture details (batch
# size, MBPO's ensemble/model config, ...). For TD-MPC2 it's derived from
# mlp_dim, the closest analogue (it has no hidden_sizes field).

# The "bs{batch_size}" prefix is common to all four algos' signatures (see
# flop_keys.py's sig_sac_td3/sig_mbpo/sig_tdmpc2), so this
# extracts for every algo -- pulled into its own facet for the batch-size sweep.
BATCH_SIZE_RE = re.compile(r"^bs(\d+)(?:_|$)")

# mbpo_rollout_regime values look like "rl1-15_re20-100" (see
# flop_keys.mbpo_rollout_regime) -- plain alphabetical sort orders them
# rl1-15 < rl1-25 < rl1-1 (string comparison sees "_" > "5"/"2"), so pivoting
# on this dimension needs its own numeric sort key to read left-to-right as
# increasing rollout length.
ROLLOUT_REGIME_RE = re.compile(r"^rl(\d+)-(\d+)_re(\d+)-(\d+)$")


def rollout_regime_sort_key(value):
    m = ROLLOUT_REGIME_RE.match(str(value))
    return tuple(int(x) for x in m.groups()) if m else (0, 0, 0, 0)


# Numeric-aware ordering for string-valued dimensions ("256x256" < "512x512" <
# "1024x1024", "bs100" < "bs256", "UTD 1" < "UTD 4") -- plain string sort gets
# these wrong, which matters for the delta metric below (its reference value
# is the first category in this order), not just for chart readability.
_NUM_CHUNK_RE = re.compile(r"(\d+)")


def natural_sort_key(value):
    return tuple((0, int(t)) if t.isdigit() else (1, t) for t in _NUM_CHUNK_RE.split(str(value)) if t)


# TD-MPC2's num_q (Q-ensemble size) and horizon (MPPI planning horizon) are
# already part of sig_tdmpc2 (see flop_keys.py) -- unlike mbpo_rollout_regime
# they don't need any extra grouping-key plumbing in compute_energy_per_flop.py,
# since two runs with different num_q/horizon already get different
# architecture_signature strings and are never averaged together. They're
# just pulled back out of the signature string here for the dashboard, the
# same way batch_size is pulled out below. The "h" component of
# sig_tdmpc2 is this horizon, not a network width.
TDMPC2_HORIZON_RE = re.compile(r"^bs\d+_h(\d+)(?:_|$)")
TDMPC2_NUM_Q_RE = re.compile(r"_nq(\d+)(?:_|$)")
TDMPC2_ALGOS = {"tdmpc2"}


def extract_tdmpc2_horizon(algo, architecture_signature):
    if algo not in TDMPC2_ALGOS or not isinstance(architecture_signature, str):
        return None
    m = TDMPC2_HORIZON_RE.match(architecture_signature)
    return m.group(1) if m else None


def extract_tdmpc2_num_q(algo, architecture_signature):
    if algo not in TDMPC2_ALGOS or not isinstance(architecture_signature, str):
        return None
    m = TDMPC2_NUM_Q_RE.search(architecture_signature)
    return m.group(1) if m else None


def extract_batch_size(architecture_signature):
    if not isinstance(architecture_signature, str):
        return None
    m = BATCH_SIZE_RE.match(architecture_signature)
    return m.group(1) if m else None


# Dimensions pivotable onto X-axis / Color / Facet, keyed by the option value
# used in the dropdowns. "col" is the dataframe column (post category-cast).
DIMENSIONS = {
    "segment": {"label": "Segment", "col": "segment", "order": SEGMENT_ORDER},
    "algo": {"label": "Algorithm", "col": "algo", "order": ALGO_ORDER},
    "env_id": {"label": "Environment", "col": "env_id", "order": None},
    "updates_per_env_step": {"label": "UTD (updates/env step)", "col": "utd_str", "order": None,
                              "sort_key": natural_sort_key},
    "architecture_signature": {"label": "Architecture", "col": "architecture_signature", "order": None,
                                "sort_key": natural_sort_key},
    "hidden_sizes": {"label": "Hidden sizes", "col": "hidden_sizes", "order": None, "sort_key": natural_sort_key},
    "batch_size": {"label": "Batch size", "col": "batch_size", "order": None, "sort_key": natural_sort_key},
    "mbpo_rollout_regime": {"label": "MBPO Rollout Regime", "col": "mbpo_rollout_regime", "order": None,
                             "sort_key": rollout_regime_sort_key},
    "tdmpc2_horizon": {"label": "TD-MPC2 Horizon", "col": "tdmpc2_horizon", "order": None,
                        "sort_key": natural_sort_key},
    "tdmpc2_num_q": {"label": "TD-MPC2 Num Q", "col": "tdmpc2_num_q", "order": None, "sort_key": natural_sort_key},
}
# Only meaningful for algo == "mbpo" / "tdmpc2" respectively (NaN elsewhere)
# -- excluded from the X-axis/Color/Facet pivot dropdowns unless that algo is
# among the selected algos (see the *-pivot-options callbacks, mirroring
# each field's own FILTER dropdown visibility toggle).
MBPO_ROLLOUT_DIM = "mbpo_rollout_regime"
TDMPC2_EXTRA_DIMS = {"tdmpc2_horizon", "tdmpc2_num_q"}
GATED_DIMS = {MBPO_ROLLOUT_DIM} | TDMPC2_EXTRA_DIMS
PER_RUN_DIMENSIONS = dict(DIMENSIONS, seed=dict(label="Seed", col="seed_str", order=None, sort_key=natural_sort_key))

CROSS_SEED_METRICS = [
    ("mean_energy_per_flop_j_per_flop", "Energy per FLOP (J/FLOP)"),
    ("mean_energy_kwh", "Mean Energy (kWh)"),
    ("mean_energy_joules", "Mean Energy (J)"),
    ("mean_duration_s", "Mean Duration (s)"),
    ("mean_power_w", "Mean Power (W)"),
    ("mean_cpu_power_w", "Mean CPU Power (W)"),
    ("mean_gpu_power_w", "Mean GPU Power (W)"),
    ("mean_ram_power_w", "Mean RAM Power (W)"),
    ("total_flops", "Mean Total FLOPs"),
    ("n_seeds", "# Seeds included"),
]
PER_RUN_METRICS = [
    ("energy_per_flop_j_per_flop", "Energy per FLOP (J/FLOP)"),
    ("total_energy_kwh", "Energy (kWh)"),
    ("total_energy_joules", "Energy (J)"),
    ("duration_s", "Duration (s)"),
    ("mean_power_w", "Mean Power (W)"),
    ("mean_cpu_power_w", "Mean CPU Power (W)"),
    ("mean_gpu_power_w", "Mean GPU Power (W)"),
    ("mean_ram_power_w", "Mean RAM Power (W)"),
    ("total_flops", "Total FLOPs"),
    ("call_count", "Call count"),
]

# ---- Marginal metric: ΔEnergy / ΔFLOPs ----
# Not a CSV column -- computed on the fly from matched pairs of rows that
# differ ONLY in the chosen X-axis dimension (every other config dimension,
# plus flop_type, and seed on the per-run tab, held fixed). For each pair,
# ΔE = E(x) - E(reference x) and ΔF = F(x) - F(reference x); the reference is
# either the smallest X value present in that matched group (in the
# dimension's ascending sort order -- e.g. the smallest batch size/width/UTD)
# or the next-smaller one (consecutive slopes). Bars pool pairs as ΣΔE / ΣΔF; the per-run box plot
# shows each pair's own ΔE/ΔF. Only matmul / mixed_total rows are used --
# target_update's elementwise op counts and buffer_sample's zero FLOPs
# can't be differenced against matmul FLOPs.
DELTA_METRIC = "__delta_energy_per_delta_flop__"
DELTA_METRIC_LABEL = "ΔEnergy / ΔFLOPs (J per extra FLOP)"
DELTA_FLOP_TYPES = {"matmul", "mixed_total"}
TOTAL_SEGMENT = "TOTAL_MEASURED_TRAINING"
# X-axes where a "pair" isn't the same workload at a different configuration:
# seeds are replicates (ΔF ~0, exactly 0 outside MBPO; ΔE is pure noise), and
# segments are different kernels entirely (rollout vs critic_update isn't
# "the same work with more FLOPs").
DELTA_BLOCKED_X_DIMS = {"seed", "segment"}
DELTA_MATCH_DIMS = ["algo", "env_id", "hidden_sizes", "batch_size", "updates_per_env_step",
                    "mbpo_rollout_regime", "tdmpc2_horizon", "tdmpc2_num_q", "segment"]
# architecture_signature is a composite of these (plus per-algo constants,
# see flop_keys.py), so pivoting on it must release all of them from the
# match key -- otherwise no two rows with different signatures could pair.
SIGNATURE_COMPONENT_DIMS = {"hidden_sizes", "batch_size", "tdmpc2_horizon", "tdmpc2_num_q"}
# Pairs whose FLOP difference is below this fraction of the reference's
# FLOPs are dropped from the chart (kept in the table, flagged): ΔE there is
# dominated by run-to-run measurement noise and the ratio blows up. MBPO's
# dynamics_model_update/synthetic_rollout FLOPs vary a few % across seeds
# (early-stopped model fits, Ant terminations), so tiny ΔF is common there.
MIN_REL_DELTA_FLOPS = 0.01
DELTA_REF_OPTIONS = [
    {"label": "Smallest X value", "value": "first"},
    {"label": "Next-smaller X value (consecutive)", "value": "previous"},
]
DELTA_PAIR_NUMERIC_COLS = ["energy_j", "ref_energy_j", "delta_energy_j", "flops", "ref_flops",
                           "delta_flops", "delta_energy_per_delta_flop"]
DELTA_CONTEXT_COLS = ["algo", "env_id", "architecture_signature", "hidden_sizes", "batch_size", "utd_str",
                      "mbpo_rollout_regime", "tdmpc2_horizon", "tdmpc2_num_q", "seed_str", "segment", "flop_type"]

NONE_VALUE = "__none__"


def dim_rank_fn(dims, dim_key):
    order = dims[dim_key]["order"]
    if order:
        idx = {v: i for i, v in enumerate(order)}
        return lambda v: (0, idx[v], ()) if v in idx else (1, 0, natural_sort_key(v))
    return dims[dim_key].get("sort_key") or natural_sort_key


def compute_delta_pairs(df, dims, x_dim, energy_col, flop_col, ref_mode):
    """Return (pairs, n_excluded_rows). One row per matched pair; see the
    DELTA_METRIC comment above for the definition."""
    x_col = dims[x_dim]["col"]
    usable = df[df["flop_type"].isin(DELTA_FLOP_TYPES) & df[x_col].notna()].dropna(subset=[energy_col, flop_col])
    n_excluded = len(df) - len(usable)

    match_dims = [k for k in DELTA_MATCH_DIMS + (["seed"] if "seed" in dims else []) if k != x_dim]
    if x_dim == "architecture_signature":
        match_dims = [k for k in match_dims if k not in SIGNATURE_COMPONENT_DIMS]
    key_cols = [dims[k]["col"] for k in match_dims] + ["flop_type"]
    carry_cols = [v["col"] for v in dims.values() if v["col"] not in key_cols and v["col"] != x_col]
    carry_cols += [c for c in ("seed",) if c in usable.columns and c not in carry_cols]
    if usable.empty:
        return pd.DataFrame(), n_excluded

    # Duplicate (config, x) rows -- e.g. repeated dev runs at the same seed on
    # the per-run tab's "All runs" mode -- are averaged before differencing.
    agg = (
        usable.groupby(key_cols + [x_col], dropna=False)
        .agg(energy_j=(energy_col, "mean"), flops=(flop_col, "mean"),
             **{c: (c, "first") for c in carry_cols})
        .reset_index()
    )
    rank = dim_rank_fn(dims, x_dim)
    pos = {v: i for i, v in enumerate(sorted(agg[x_col].unique(), key=rank))}
    agg["_pos"] = agg[x_col].map(pos)

    records = []
    for _, grp in agg.groupby(key_cols, dropna=False, sort=False):
        rows = grp.sort_values("_pos").to_dict("records")
        for i in range(1, len(rows)):
            tgt = rows[i]
            ref = rows[0] if ref_mode == "first" else rows[i - 1]
            d_e = tgt["energy_j"] - ref["energy_j"]
            d_f = tgt["flops"] - ref["flops"]
            ok = ref["flops"] > 0 and abs(d_f) >= MIN_REL_DELTA_FLOPS * ref["flops"]
            rec = {c: tgt[c] for c in key_cols + carry_cols + [x_col]}
            rec.update(
                reference_value=ref[x_col],
                energy_j=tgt["energy_j"], ref_energy_j=ref["energy_j"], delta_energy_j=d_e,
                flops=tgt["flops"], ref_flops=ref["flops"], delta_flops=d_f,
                delta_energy_per_delta_flop=d_e / d_f if ok else float("nan"),
                status="ok" if ok else f"dropped: |ΔF| < {MIN_REL_DELTA_FLOPS:.0%} of reference FLOPs",
            )
            records.append(rec)
    return pd.DataFrame(records), n_excluded


def delta_note(pairs, n_excluded, dims, x_dim, ref_mode, per_run):
    n_ok = int((pairs["status"] == "ok").sum()) if not pairs.empty else 0
    n_dropped = len(pairs) - n_ok
    ref_desc = ("the smallest " if ref_mode == "first" else "the next-smaller ") + dims[x_dim]["label"] + " value"
    held = "every other config dimension, flop_type" + (" and seed" if per_run else "")
    return (
        f"ΔEnergy / ΔFLOPs: each pair differs only in {dims[x_dim]['label']} ({held} held fixed), "
        f"measured against {ref_desc} present in its matched group, so the reference value itself has no bar. "
        f"Bars pool pairs as ΣΔE / ΣΔF" + ("; the box plot shows each pair's own ratio" if per_run else "") + ". "
        f"{n_ok} pairs used, {n_dropped} dropped (|ΔF| < {MIN_REL_DELTA_FLOPS:.0%} of reference FLOPs), "
        f"{n_excluded} rows excluded (non-matmul flop_type or no {dims[x_dim]['label']} value). "
        "With only two X values the reference choice doesn't matter (the ratio is symmetric)."
    )


def delta_table(pairs, x_col):
    context = [c for c in DELTA_CONTEXT_COLS if c in pairs.columns and c != x_col]
    cols = context + [x_col, "reference_value"] + DELTA_PAIR_NUMERIC_COLS + ["status"]
    return dash_table.DataTable(
        columns=numeric_table_columns(pairs[cols], set(DELTA_PAIR_NUMERIC_COLS)),
        data=pairs[cols].to_dict("records"),
        filter_action="native", sort_action="native", page_size=15,
        style_table={"overflowX": "auto"}, style_cell={"fontSize": 12, "fontFamily": "monospace", "padding": "4px"},
        style_header={"fontWeight": "bold"},
    )


def disable_log_if_nonpositive(fig, values, log_y):
    """Log axes silently drop values <= 0 (common for ΔE/ΔF, where ΔE can be
    negative within noise) -- fall back to linear and say so on the chart."""
    if log_y and (pd.Series(values).dropna() <= 0).any():
        fig.add_annotation(text="Log Y disabled: some values ≤ 0", xref="paper", yref="paper",
                           x=1, y=1.12, xanchor="right", showarrow=False, font=dict(size=11, color="#b71c1c"))
        return False
    return log_y


def load_cross_seed(flop_dir):
    path = os.path.join(flop_dir, "cross_seed_energy_per_flop.csv")
    df = pd.read_csv(path)
    df["utd_str"] = "UTD " + df["updates_per_env_step"].astype(str)
    df["hidden_sizes"] = df["hidden_sizes"].astype(str)
    df["batch_size"] = [extract_batch_size(s) for s in df["architecture_signature"]]
    df["tdmpc2_horizon"] = [extract_tdmpc2_horizon(a, s) for a, s in zip(df["algo"], df["architecture_signature"])]
    df["tdmpc2_num_q"] = [extract_tdmpc2_num_q(a, s) for a, s in zip(df["algo"], df["architecture_signature"])]
    df["is_flop_normalized"] = ~df["segment"].isin(NON_FLOP_SEGMENTS)
    return df


def load_per_run(flop_dir):
    path = os.path.join(flop_dir, "per_run_energy_per_flop.csv")
    df = pd.read_csv(path)
    df["utd_str"] = "UTD " + df["updates_per_env_step"].astype(str)
    df["seed_str"] = df["seed"].astype(str)
    df["hidden_sizes"] = df["hidden_sizes"].astype(str)
    df["batch_size"] = [extract_batch_size(s) for s in df["architecture_signature"]]
    df["tdmpc2_horizon"] = [extract_tdmpc2_horizon(a, s) for a, s in zip(df["algo"], df["architecture_signature"])]
    df["tdmpc2_num_q"] = [extract_tdmpc2_num_q(a, s) for a, s in zip(df["algo"], df["architecture_signature"])]
    df["is_flop_normalized"] = ~df["segment"].isin(NON_FLOP_SEGMENTS)
    if df["included_in_cross_seed_avg"].dtype == object:
        df["included_in_cross_seed_avg"] = df["included_in_cross_seed_avg"].astype(str).str.strip().eq("True")
    return df


# mbpo_rollout_regime / tdmpc2_horizon / tdmpc2_num_q (see flop_analysis/flop_keys.py
# and the extract_tdmpc2_* functions above) are only populated for their one
# relevant algo -- NaN everywhere else. They're included in the generic
# crossfilter fields so they interact normally with every other filter, but
# their dropdowns are only shown in the UI while the relevant algo is among
# the selected algos (see the *-extra-visibility callbacks below).
#
# NOTE: FILTER_FIELDS_WITH_EXTRAS's order must exactly match the Dash
# callback Output()/dropdown order (rollout/horizon/num_q slotted in between
# UTD and Segment) -- crossfilter_options()/apply_filters() zip these field
# lists positionally against the callback's Output tuple, so a mismatch here
# silently wires the wrong dropdown's options to the wrong field.
FILTER_FIELDS = ["algo", "env_id", "architecture_signature", "hidden_sizes", "batch_size", "updates_per_env_step", "segment"]
PER_RUN_FILTER_FIELDS = FILTER_FIELDS + ["seed"]
MBPO_ROLLOUT_FIELD = "mbpo_rollout_regime"
TDMPC2_HORIZON_FIELD = "tdmpc2_horizon"
TDMPC2_NUM_Q_FIELD = "tdmpc2_num_q"
FILTER_FIELDS_WITH_EXTRAS = ["algo", "env_id", "architecture_signature", "hidden_sizes", "batch_size",
                             "updates_per_env_step", MBPO_ROLLOUT_FIELD, TDMPC2_HORIZON_FIELD, TDMPC2_NUM_Q_FIELD,
                             "segment"]
PER_RUN_FILTER_FIELDS_WITH_EXTRAS = FILTER_FIELDS_WITH_EXTRAS + ["seed"]


def crossfilter_options(df, fields, current):
    """For each field in `fields`, compute sorted unique values available
    once every OTHER field's current selection has been applied -- a true
    mutual crossfilter (not just upstream-cascading), all resolved inside
    one callback so Dash doesn't see a dependency cycle."""
    options = {}
    for field in fields:
        mask = pd.Series(True, index=df.index)
        for other in fields:
            if other == field:
                continue
            vals = current.get(other)
            if vals:
                mask &= df[other].isin(vals)
        options[field] = sorted(df.loc[mask, field].dropna().unique().tolist())
    return options


def apply_filters(df, fields, current):
    mask = pd.Series(True, index=df.index)
    for field in fields:
        vals = current.get(field)
        if vals:
            mask &= df[field].isin(vals)
    return df[mask]


def empty_figure(message):
    fig = go.Figure()
    fig.update_layout(
        annotations=[dict(text=message, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14))],
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        height=260, margin=dict(l=40, r=20, t=30, b=20),
    )
    return fig


def dim_label(dim_key, dims):
    return dims[dim_key]["label"] if dim_key else "(none)"


def build_category_orders(df, dims, dim_keys):
    """Dimensions with a fixed conceptual "order" (segment, algo) keep it;
    every other dimension reads ascending (e.g. 256x256 -> 512x512 ->
    1024x1024), on the X-axis, Color and Facet alike."""
    category_orders = {}
    for k in dim_keys:
        col = dims[k]["col"]
        order = dims[k]["order"]
        present = df[col].dropna().unique().tolist()
        if order:
            category_orders[col] = [v for v in order if v in present] + sorted(set(present) - set(order))
        else:
            category_orders[col] = sorted(present, key=dims[k].get("sort_key") or natural_sort_key)
    return category_orders


def build_grouped_bar(df, dims, x_dim, color_dim, facet_dim, metric_col, metric_label, log_y, pool_deltas=False):
    """pool_deltas: df is a compute_delta_pairs() frame; each bar is
    ΣΔE / ΣΔF over its pairs rather than the mean of metric_col."""
    if df.empty:
        return empty_figure("No rows match the current filters.")
    if not x_dim:
        return empty_figure("Pick an X-axis dimension.")

    group_keys = [d for d in [x_dim, color_dim, facet_dim] if d]
    seen = set()
    group_keys = [k for k in group_keys if not (k in seen or seen.add(k))]
    group_cols = [dims[k]["col"] for k in group_keys]

    if pool_deltas:
        g = (
            df.groupby(group_cols, dropna=False)
            .agg(n=("delta_energy_j", "count"), sum_de=("delta_energy_j", "sum"), sum_df=("delta_flops", "sum"))
            .reset_index()
        )
        g["value"] = g["sum_de"] / g["sum_df"]
        custom_data = ["n", "sum_de", "sum_df"]
        hovertemplate = (metric_label + ": %{y:.4g}<br>ΣΔEnergy: %{customdata[1]:.4g} J"
                         "<br>ΣΔFLOPs: %{customdata[2]:.4g}<br>Pairs pooled: %{customdata[0]}<extra></extra>")
    else:
        g = (
            df.groupby(group_cols, dropna=False)[metric_col]
            .agg(value="mean", n="count")
            .reset_index()
        )
        custom_data = ["n"]
        hovertemplate = metric_label + ": %{y:.4g}<br>Rows averaged: %{customdata[0]}<extra></extra>"

    category_orders = build_category_orders(g, dims, group_keys)

    color_col = dims[color_dim]["col"] if color_dim else None
    facet_col = dims[facet_dim]["col"] if facet_dim else None
    color_map = None
    if color_dim == "algo":
        color_map = ALGO_COLORS

    fig = px.bar(
        g, x=dims[x_dim]["col"], y="value", color=color_col, facet_col=facet_col,
        barmode="group", category_orders=category_orders, color_discrete_map=color_map,
        custom_data=custom_data,
        labels={"value": metric_label, dims[x_dim]["col"]: dims[x_dim]["label"],
                **({color_col: dims[color_dim]["label"]} if color_col else {}),
                **({facet_col: dims[facet_dim]["label"]} if facet_col else {})},
    )
    fig.update_traces(hovertemplate=hovertemplate)
    fig.update_layout(
        height=520,
        margin=dict(l=60, r=20, t=60, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    log_y = disable_log_if_nonpositive(fig, g["value"], log_y)
    fig.update_xaxes(matches=None, showticklabels=True)
    # Set type via update_yaxes so it applies to every facet's y-axis, not just the first
    fig.update_yaxes(matches=None, showticklabels=True, type="log" if log_y else "linear")
    return fig


def build_box(df, dims, x_dim, color_dim, facet_dim, metric_col, metric_label, log_y):
    if df.empty:
        return empty_figure("No rows match the current filters.")
    if not x_dim:
        return empty_figure("Pick an X-axis dimension.")
    color_col = dims[color_dim]["col"] if color_dim else None
    facet_col = dims[facet_dim]["col"] if facet_dim else None
    color_map = ALGO_COLORS if color_dim == "algo" else None

    category_orders = build_category_orders(df, dims, [k for k in (x_dim, color_dim, facet_dim) if k])

    fig = px.box(
        df, x=dims[x_dim]["col"], y=metric_col, color=color_col, facet_col=facet_col,
        points="all", category_orders=category_orders, color_discrete_map=color_map,
        labels={metric_col: metric_label, dims[x_dim]["col"]: dims[x_dim]["label"],
                **({color_col: dims[color_dim]["label"]} if color_col else {}),
                **({facet_col: dims[facet_dim]["label"]} if facet_col else {})},
        hover_data=[c for c in ("seed", "run_dir", "reference_value") if c in df.columns] or None,
    )
    fig.update_layout(
        height=520,
        margin=dict(l=60, r=20, t=40, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    log_y = disable_log_if_nonpositive(fig, df[metric_col], log_y)
    fig.update_xaxes(matches=None, showticklabels=True)
    # Set type via update_yaxes so it applies to every facet's y-axis, not just the first
    fig.update_yaxes(matches=None, showticklabels=True, type="log" if log_y else "linear")
    return fig


def numeric_table_columns(df, precise_cols):
    cols = []
    for c in df.columns:
        if c in precise_cols:
            cols.append({"name": c, "id": c, "type": "numeric",
                         "format": Format(precision=4, scheme=Scheme.decimal_or_exponent)})
        else:
            cols.append({"name": c, "id": c})
    return cols


def filter_dropdown(id_, label):
    return html.Div(
        [html.Label(label), dcc.Dropdown(id=id_, options=[], value=[], multi=True, placeholder="All")],
        className="filter-field",
    )


def pivot_dropdown(id_, label, options, value):
    return html.Div(
        [html.Label(label), dcc.Dropdown(id=id_, options=options, value=value, clearable=False)],
        className="pivot-field",
    )


def delta_ref_dropdown(prefix):
    """Only shown while the ΔE/ΔF metric is selected (see *-delta-visibility)."""
    return html.Div(
        [html.Label("Δ reference"),
         dcc.Dropdown(id=f"{prefix}-deltaref", options=DELTA_REF_OPTIONS, value="first", clearable=False)],
        id=f"{prefix}-deltaref-wrap", className="pivot-field", style={"display": "none"},
    )


def make_app(flop_dir):
    app = Dash(__name__)
    app.title = "FLOP / Energy-per-FLOP Dashboard"

    cross_df = load_cross_seed(flop_dir)
    per_run_df = load_per_run(flop_dir)

    dim_opts = [{"label": v["label"], "value": k} for k, v in DIMENSIONS.items()]
    dim_opts_with_none = [{"label": "(none)", "value": NONE_VALUE}] + dim_opts
    pr_dim_opts = [{"label": v["label"], "value": k} for k, v in PER_RUN_DIMENSIONS.items()]
    pr_dim_opts_with_none = [{"label": "(none)", "value": NONE_VALUE}] + pr_dim_opts

    def filters_block(prefix):
        return html.Div(
            [
                filter_dropdown(f"{prefix}-algo", "Algorithm"),
                filter_dropdown(f"{prefix}-env", "Environment"),
                filter_dropdown(f"{prefix}-arch", "Architecture"),
                filter_dropdown(f"{prefix}-hidden", "Hidden sizes"),
                filter_dropdown(f"{prefix}-batch", "Batch size"),
                filter_dropdown(f"{prefix}-utd", "UTD"),
                html.Div(
                    [html.Label("MBPO Rollout Regime"), dcc.Dropdown(id=f"{prefix}-rollout", options=[], value=[], multi=True, placeholder="All")],
                    id=f"{prefix}-rollout-wrap", className="filter-field", style={"display": "none"},
                ),
                html.Div(
                    [html.Label("TD-MPC2 Horizon"), dcc.Dropdown(id=f"{prefix}-horizon", options=[], value=[], multi=True, placeholder="All")],
                    id=f"{prefix}-horizon-wrap", className="filter-field", style={"display": "none"},
                ),
                html.Div(
                    [html.Label("TD-MPC2 Num Q"), dcc.Dropdown(id=f"{prefix}-numq", options=[], value=[], multi=True, placeholder="All")],
                    id=f"{prefix}-numq-wrap", className="filter-field", style={"display": "none"},
                ),
                filter_dropdown(f"{prefix}-segment", "Segment"),
            ] + ([filter_dropdown(f"{prefix}-seed", "Seed")] if prefix == "pr" else []),
            className="filters-row",
        )

    cross_tab = html.Div(
        [
            html.P(
                "Averaged across the canonical 5-seed sweep (per architecture/UTD/segment). "
                "Segments in " + ", ".join(sorted(NON_FLOP_SEGMENTS)) + " are not matmul-FLOP-normalized "
                "(CPU-side or elementwise ops; see the flop_type column -- target_update's value is J per "
                "elementwise op, not J/FLOP). TOTAL_MEASURED_TRAINING = all training energy (incl. buffer_sample "
                "and target_update) / matmul FLOPs only.",
                className="note",
            ),
            html.H4("Filter rows"),
            filters_block("cs"),
            html.H4("Chart"),
            html.Div(
                [
                    pivot_dropdown("cs-xaxis", "X-axis", dim_opts, "segment"),
                    pivot_dropdown("cs-color", "Color", dim_opts_with_none, "algo"),
                    pivot_dropdown("cs-facet", "Facet", dim_opts_with_none, "env_id"),
                    pivot_dropdown("cs-metric", "Metric",
                                    [{"label": l, "value": v} for v, l in CROSS_SEED_METRICS]
                                    + [{"label": DELTA_METRIC_LABEL, "value": DELTA_METRIC}],
                                    "mean_energy_per_flop_j_per_flop"),
                    delta_ref_dropdown("cs"),
                    html.Div([html.Label("Log Y"), dcc.Checklist(id="cs-logy", options=[{"label": "", "value": "log"}], value=["log"])],
                             className="pivot-field"),
                ],
                className="pivot-row",
            ),
            html.P(id="cs-delta-note", className="note", style={"display": "none"}),
            dcc.Loading(dcc.Graph(id="cs-graph")),
            html.H4("Underlying rows"),
            dcc.Loading(html.Div(id="cs-table-wrap")),
        ],
        className="tab-content",
    )

    per_run_tab = html.Div(
        [
            html.P(
                "One row per (run, segment), including dev/exploratory runs outside the canonical "
                "5-seed set (flagged below). Use this tab to inspect per-seed spread or a specific run.",
                className="note",
            ),
            html.H4("Filter rows"),
            filters_block("pr"),
            html.Div(
                [
                    html.Label("Seed set: "),
                    dcc.RadioItems(
                        id="pr-canonical",
                        options=[
                            {"label": "Canonical 5-seed sweep only", "value": "canonical"},
                            {"label": "All runs (incl. dev/exploratory)", "value": "all"},
                        ],
                        value="canonical", inline=True,
                    ),
                ],
                className="controls",
            ),
            html.H4("Chart"),
            html.Div(
                [
                    pivot_dropdown("pr-charttype", "Chart type",
                                    [{"label": "Grouped bar (mean)", "value": "bar"},
                                     {"label": "Box plot (per-seed spread)", "value": "box"}],
                                    "box"),
                    pivot_dropdown("pr-xaxis", "X-axis", pr_dim_opts, "segment"),
                    pivot_dropdown("pr-color", "Color", pr_dim_opts_with_none, "algo"),
                    pivot_dropdown("pr-facet", "Facet", pr_dim_opts_with_none, "env_id"),
                    pivot_dropdown("pr-metric", "Metric",
                                    [{"label": l, "value": v} for v, l in PER_RUN_METRICS]
                                    + [{"label": DELTA_METRIC_LABEL, "value": DELTA_METRIC}],
                                    "energy_per_flop_j_per_flop"),
                    delta_ref_dropdown("pr"),
                    html.Div([html.Label("Log Y"), dcc.Checklist(id="pr-logy", options=[{"label": "", "value": "log"}], value=["log"])],
                             className="pivot-field"),
                ],
                className="pivot-row",
            ),
            html.P(id="pr-delta-note", className="note", style={"display": "none"}),
            dcc.Loading(dcc.Graph(id="pr-graph")),
            html.H4("Underlying rows"),
            dcc.Loading(html.Div(id="pr-table-wrap")),
        ],
        className="tab-content",
    )

    app.layout = html.Div(
        [
            html.H2("FLOP / Energy-per-FLOP Dashboard"),
            dcc.Tabs(
                id="tabs",
                value="cross",
                children=[
                    dcc.Tab(label="Cross-seed comparison", value="cross", children=[cross_tab]),
                    dcc.Tab(label="Per-run detail", value="perrun", children=[per_run_tab]),
                ],
            ),
        ],
        className="app-container",
    )

    # ---- Cross-seed tab: algo-gated filter visibility (rollout only while
    # "mbpo" is selected; horizon/num_q only while "tdmpc2" is selected) ----
    @app.callback(
        Output("cs-rollout-wrap", "style"),
        Output("cs-horizon-wrap", "style"), Output("cs-numq-wrap", "style"),
        Input("cs-algo", "value"),
    )
    def _cs_extra_visibility(algo):
        algo = algo or []
        rollout_style = {"display": "block"} if "mbpo" in algo else {"display": "none"}
        tdmpc2_style = {"display": "block"} if "tdmpc2" in algo else {"display": "none"}
        return rollout_style, tdmpc2_style, tdmpc2_style

    # ---- Cross-seed tab: X-axis/Color/Facet pivot options -- only offer the
    # algo-gated dimensions while their algo is selected, same gating as the
    # filter dropdowns above.
    @app.callback(
        Output("cs-xaxis", "options"), Output("cs-color", "options"), Output("cs-facet", "options"),
        Input("cs-algo", "value"),
    )
    def _cs_pivot_options(algo):
        algo = algo or []
        hidden = set() if "mbpo" in algo else {MBPO_ROLLOUT_DIM}
        if "tdmpc2" not in algo:
            hidden |= TDMPC2_EXTRA_DIMS
        xaxis_opts = [o for o in dim_opts if o["value"] not in hidden]
        other_opts = [o for o in dim_opts_with_none if o["value"] not in hidden]
        return xaxis_opts, other_opts, other_opts

    # ---- Cross-seed tab: crossfilter dropdown options ----
    @app.callback(
        Output("cs-algo", "options"), Output("cs-env", "options"), Output("cs-arch", "options"),
        Output("cs-hidden", "options"), Output("cs-batch", "options"), Output("cs-utd", "options"),
        Output("cs-rollout", "options"), Output("cs-horizon", "options"), Output("cs-numq", "options"),
        Output("cs-segment", "options"),
        Input("cs-algo", "value"), Input("cs-env", "value"), Input("cs-arch", "value"),
        Input("cs-hidden", "value"), Input("cs-batch", "value"), Input("cs-utd", "value"),
        Input("cs-rollout", "value"), Input("cs-horizon", "value"), Input("cs-numq", "value"),
        Input("cs-segment", "value"),
    )
    def _cs_options(algo, env, arch, hidden, batch, utd, rollout, horizon, numq, segment):
        current = {"algo": algo, "env_id": env, "architecture_signature": arch,
                   "hidden_sizes": hidden, "batch_size": batch, "updates_per_env_step": utd,
                   MBPO_ROLLOUT_FIELD: rollout, TDMPC2_HORIZON_FIELD: horizon, TDMPC2_NUM_Q_FIELD: numq,
                   "segment": segment}
        opts = crossfilter_options(cross_df, FILTER_FIELDS_WITH_EXTRAS, current)
        return tuple([{"label": str(v), "value": v} for v in opts[f]] for f in FILTER_FIELDS_WITH_EXTRAS)

    # ---- Both tabs: ΔE/ΔF reference selector only visible for that metric ----
    for prefix in ("cs", "pr"):
        @app.callback(Output(f"{prefix}-deltaref-wrap", "style"), Input(f"{prefix}-metric", "value"))
        def _delta_visibility(metric):
            return {"display": "block"} if metric == DELTA_METRIC else {"display": "none"}

    hidden_note = ("", {"display": "none"})

    def delta_outputs(filtered, dims, x_dim, color_dim, facet_dim, energy_col, ref_mode, logy, charttype):
        """(figure, table, note text, note style) for the ΔE/ΔF metric."""
        per_run = "seed" in dims
        if not x_dim:
            return empty_figure("Pick an X-axis dimension."), None, *hidden_note
        if x_dim in DELTA_BLOCKED_X_DIMS:
            return empty_figure(f"ΔEnergy / ΔFLOPs across {dims[x_dim]['label']} values is not meaningful -- pick a "
                                "configuration dimension (batch size, hidden sizes, UTD, ...) as X-axis."), None, *hidden_note
        pairs, n_excluded = compute_delta_pairs(filtered, dims, x_dim, energy_col, "total_flops", ref_mode)
        note_text = delta_note(pairs, n_excluded, dims, x_dim, ref_mode, per_run)
        if pairs.empty:
            return empty_figure(f"No matched pairs differing only in {dims[x_dim]['label']}."), None, note_text, {"display": "block"}
        ok = pairs[pairs["status"] == "ok"]
        # TOTAL_MEASURED_TRAINING already sums its component segments' energy,
        # so pooling it with them in one bar would count that energy twice.
        if ("segment" not in (x_dim, color_dim, facet_dim) and ok["segment"].nunique() > 1
                and (ok["segment"] == TOTAL_SEGMENT).any()):
            ok = ok[ok["segment"] == TOTAL_SEGMENT]
            note_text += (f" Segment isn't on X/Color/Facet, so only {TOTAL_SEGMENT} pairs are charted (pooling it "
                          "with its own component segments would double-count their energy) -- filter Segment or "
                          "pivot on it to see individual segments.")
        note = (note_text, {"display": "block"})
        if ok.empty:
            fig = empty_figure("Every matched pair was dropped (ΔFLOPs ≈ 0) -- see the table.")
        elif charttype == "box":
            fig = build_box(ok, dims, x_dim, color_dim, facet_dim, "delta_energy_per_delta_flop",
                            DELTA_METRIC_LABEL, bool(logy))
        else:
            fig = build_grouped_bar(ok, dims, x_dim, color_dim, facet_dim, "delta_energy_per_delta_flop",
                                    DELTA_METRIC_LABEL, bool(logy), pool_deltas=True)
        return fig, delta_table(pairs, dims[x_dim]["col"]), *note

    @app.callback(
        Output("cs-graph", "figure"), Output("cs-table-wrap", "children"),
        Output("cs-delta-note", "children"), Output("cs-delta-note", "style"),
        Input("cs-algo", "value"), Input("cs-env", "value"), Input("cs-arch", "value"),
        Input("cs-hidden", "value"), Input("cs-batch", "value"), Input("cs-utd", "value"),
        Input("cs-rollout", "value"), Input("cs-horizon", "value"), Input("cs-numq", "value"),
        Input("cs-segment", "value"),
        Input("cs-xaxis", "value"), Input("cs-color", "value"), Input("cs-facet", "value"),
        Input("cs-metric", "value"), Input("cs-logy", "value"), Input("cs-deltaref", "value"),
    )
    def _cs_update(algo, env, arch, hidden, batch, utd, rollout, horizon, numq, segment, x_dim, color_dim, facet_dim,
                   metric, logy, delta_ref):
        current = {"algo": algo, "env_id": env, "architecture_signature": arch,
                   "hidden_sizes": hidden, "batch_size": batch, "updates_per_env_step": utd,
                   MBPO_ROLLOUT_FIELD: rollout, TDMPC2_HORIZON_FIELD: horizon, TDMPC2_NUM_Q_FIELD: numq,
                   "segment": segment}
        filtered = apply_filters(cross_df, FILTER_FIELDS_WITH_EXTRAS, current)
        color_dim = None if color_dim == NONE_VALUE else color_dim
        facet_dim = None if facet_dim == NONE_VALUE else facet_dim
        if metric == DELTA_METRIC:
            return delta_outputs(filtered, DIMENSIONS, x_dim, color_dim, facet_dim, "mean_energy_joules",
                                 delta_ref, logy, "bar")
        metric_label = dict(CROSS_SEED_METRICS)[metric]
        fig = build_grouped_bar(filtered, DIMENSIONS, x_dim, color_dim, facet_dim, metric, metric_label, bool(logy))
        display_cols = ["algo", "env_id", "architecture_signature", "hidden_sizes", "batch_size", "updates_per_env_step",
                         "mbpo_rollout_regime", "tdmpc2_horizon", "tdmpc2_num_q", "segment", "flop_type",
                         "n_seeds", "mean_energy_kwh", "mean_energy_joules",
                         "mean_duration_s", "mean_power_w", "mean_cpu_power_w", "mean_gpu_power_w", "mean_ram_power_w",
                         "total_flops", "mean_energy_per_flop_j_per_flop"]
        table = dash_table.DataTable(
            columns=numeric_table_columns(filtered[display_cols], {
                "mean_energy_kwh", "mean_energy_joules", "mean_duration_s", "mean_power_w",
                "mean_cpu_power_w", "mean_gpu_power_w", "mean_ram_power_w",
                "total_flops", "mean_energy_per_flop_j_per_flop",
            }),
            data=filtered[display_cols].to_dict("records"),
            filter_action="native", sort_action="native", page_size=15,
            style_table={"overflowX": "auto"}, style_cell={"fontSize": 12, "fontFamily": "monospace", "padding": "4px"},
            style_header={"fontWeight": "bold"},
        )
        return fig, table, *hidden_note

    # ---- Per-run tab: algo-gated filter visibility (same gating as cross-seed tab) ----
    @app.callback(
        Output("pr-rollout-wrap", "style"),
        Output("pr-horizon-wrap", "style"), Output("pr-numq-wrap", "style"),
        Input("pr-algo", "value"),
    )
    def _pr_extra_visibility(algo):
        algo = algo or []
        rollout_style = {"display": "block"} if "mbpo" in algo else {"display": "none"}
        tdmpc2_style = {"display": "block"} if "tdmpc2" in algo else {"display": "none"}
        return rollout_style, tdmpc2_style, tdmpc2_style

    # ---- Per-run tab: X-axis/Color/Facet pivot options (same gating as cross-seed tab) ----
    @app.callback(
        Output("pr-xaxis", "options"), Output("pr-color", "options"), Output("pr-facet", "options"),
        Input("pr-algo", "value"),
    )
    def _pr_pivot_options(algo):
        algo = algo or []
        hidden = set() if "mbpo" in algo else {MBPO_ROLLOUT_DIM}
        if "tdmpc2" not in algo:
            hidden |= TDMPC2_EXTRA_DIMS
        xaxis_opts = [o for o in pr_dim_opts if o["value"] not in hidden]
        other_opts = [o for o in pr_dim_opts_with_none if o["value"] not in hidden]
        return xaxis_opts, other_opts, other_opts

    # ---- Per-run tab: crossfilter dropdown options ----
    @app.callback(
        Output("pr-algo", "options"), Output("pr-env", "options"), Output("pr-arch", "options"),
        Output("pr-hidden", "options"), Output("pr-batch", "options"), Output("pr-utd", "options"),
        Output("pr-rollout", "options"), Output("pr-horizon", "options"), Output("pr-numq", "options"),
        Output("pr-segment", "options"),
        Output("pr-seed", "options"),
        Input("pr-algo", "value"), Input("pr-env", "value"), Input("pr-arch", "value"),
        Input("pr-hidden", "value"), Input("pr-batch", "value"), Input("pr-utd", "value"),
        Input("pr-rollout", "value"), Input("pr-horizon", "value"), Input("pr-numq", "value"),
        Input("pr-segment", "value"),
        Input("pr-seed", "value"), Input("pr-canonical", "value"),
    )
    def _pr_options(algo, env, arch, hidden, batch, utd, rollout, horizon, numq, segment, seed, canonical):
        base = per_run_df[per_run_df["included_in_cross_seed_avg"]] if canonical == "canonical" else per_run_df
        current = {"algo": algo, "env_id": env, "architecture_signature": arch,
                   "hidden_sizes": hidden, "batch_size": batch, "updates_per_env_step": utd,
                   MBPO_ROLLOUT_FIELD: rollout, TDMPC2_HORIZON_FIELD: horizon, TDMPC2_NUM_Q_FIELD: numq,
                   "segment": segment, "seed": seed}
        opts = crossfilter_options(base, PER_RUN_FILTER_FIELDS_WITH_EXTRAS, current)
        return tuple([{"label": str(v), "value": v} for v in opts[f]] for f in PER_RUN_FILTER_FIELDS_WITH_EXTRAS)

    @app.callback(
        Output("pr-graph", "figure"), Output("pr-table-wrap", "children"),
        Output("pr-delta-note", "children"), Output("pr-delta-note", "style"),
        Input("pr-algo", "value"), Input("pr-env", "value"), Input("pr-arch", "value"),
        Input("pr-hidden", "value"), Input("pr-batch", "value"), Input("pr-utd", "value"),
        Input("pr-rollout", "value"), Input("pr-horizon", "value"), Input("pr-numq", "value"),
        Input("pr-segment", "value"),
        Input("pr-seed", "value"), Input("pr-canonical", "value"),
        Input("pr-charttype", "value"),
        Input("pr-xaxis", "value"), Input("pr-color", "value"), Input("pr-facet", "value"),
        Input("pr-metric", "value"), Input("pr-logy", "value"), Input("pr-deltaref", "value"),
    )
    def _pr_update(algo, env, arch, hidden, batch, utd, rollout, horizon, numq, segment, seed, canonical, charttype,
                    x_dim, color_dim, facet_dim, metric, logy, delta_ref):
        base = per_run_df[per_run_df["included_in_cross_seed_avg"]] if canonical == "canonical" else per_run_df
        current = {"algo": algo, "env_id": env, "architecture_signature": arch,
                   "hidden_sizes": hidden, "batch_size": batch, "updates_per_env_step": utd,
                   MBPO_ROLLOUT_FIELD: rollout, TDMPC2_HORIZON_FIELD: horizon, TDMPC2_NUM_Q_FIELD: numq,
                   "segment": segment, "seed": seed}
        filtered = apply_filters(base, PER_RUN_FILTER_FIELDS_WITH_EXTRAS, current)
        color_dim = None if color_dim == NONE_VALUE else color_dim
        facet_dim = None if facet_dim == NONE_VALUE else facet_dim
        if metric == DELTA_METRIC:
            return delta_outputs(filtered, PER_RUN_DIMENSIONS, x_dim, color_dim, facet_dim, "total_energy_joules",
                                 delta_ref, logy, charttype)
        metric_label = dict(PER_RUN_METRICS)[metric]
        if charttype == "box":
            fig = build_box(filtered, PER_RUN_DIMENSIONS, x_dim, color_dim, facet_dim, metric, metric_label, bool(logy))
        else:
            fig = build_grouped_bar(filtered, PER_RUN_DIMENSIONS, x_dim, color_dim, facet_dim, metric, metric_label, bool(logy))
        display_cols = ["algo", "env_id", "architecture_signature", "hidden_sizes", "batch_size", "updates_per_env_step",
                         "mbpo_rollout_regime", "tdmpc2_horizon", "tdmpc2_num_q", "seed",
                         "included_in_cross_seed_avg", "segment", "flop_type", "call_count", "total_flops",
                         "total_energy_kwh", "total_energy_joules",
                         "duration_s", "mean_power_w", "mean_cpu_power_w", "mean_gpu_power_w", "mean_ram_power_w",
                         "energy_per_flop_j_per_flop",
                         "run_dir", "note"]
        table = dash_table.DataTable(
            columns=numeric_table_columns(filtered[display_cols], {
                "total_flops", "total_energy_kwh", "total_energy_joules",
                "duration_s", "mean_power_w", "mean_cpu_power_w", "mean_gpu_power_w", "mean_ram_power_w",
                "energy_per_flop_j_per_flop",
            }),
            data=filtered[display_cols].to_dict("records"),
            filter_action="native", sort_action="native", page_size=15,
            style_table={"overflowX": "auto"}, style_cell={"fontSize": 12, "fontFamily": "monospace", "padding": "4px"},
            style_header={"fontWeight": "bold"},
        )
        return fig, table, *hidden_note

    return app


APP_CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; background: #fafafa; color: #1a1a1a; }
.app-container { max-width: 1300px; margin: 0 auto; padding: 24px; }
.tab-content { padding: 16px 0; }
.note { background: #fff8e1; border: 1px solid #ffe082; border-radius: 6px; padding: 10px 14px; font-size: 13px; }
.filters-row { display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 12px; }
.filter-field { flex: 1 1 190px; min-width: 170px; }
.pivot-row { display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-end; margin-bottom: 12px; padding: 12px; background: #f0f0f0; border-radius: 6px; }
.pivot-field { flex: 1 1 170px; min-width: 150px; }
.controls { margin: 12px 0; }
h4 { margin-top: 18px; margin-bottom: 6px; }
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flop-dir", default=FLOP_DIR)
    parser.add_argument("--port", type=int, default=8051)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    app = make_app(args.flop_dir)
    app.index_string = app.index_string.replace("</head>", f"<style>{APP_CSS}</style></head>")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
