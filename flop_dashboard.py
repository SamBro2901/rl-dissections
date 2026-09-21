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

# architecture_signature is "bs{batch_size}_h{hidden_sizes}..." for sac/td3/mbpo
# (see flop_analysis/flop_keys.py) -- pull the hidden_sizes component out into
# its own facet so the width sweep (256x256 / 512x512 / 1024x1024) can be
# compared independent of the other architecture details (batch size, MBPO's
# ensemble/model config, ...). TD-MPC2's signature also has an "h" component
# but it's the MPPI planning horizon, not a network width -- excluded below.
HIDDEN_SIZES_RE = re.compile(r"^bs\d+_h([\dx]+)(?:_|$)")
HIDDEN_SIZES_ALGOS = {"sac", "td3", "mbpo"}

# The "bs{batch_size}" prefix is common to all four algos' signatures (see
# flop_keys.py's sig_sac_td3/sig_mbpo/sig_tdmpc2), so unlike hidden_sizes this
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


# TD-MPC2's num_q (Q-ensemble size) and horizon (MPPI planning horizon) are
# already part of sig_tdmpc2 (see flop_keys.py) -- unlike mbpo_rollout_regime
# they don't need any extra grouping-key plumbing in compute_energy_per_flop.py,
# since two runs with different num_q/horizon already get different
# architecture_signature strings and are never averaged together. They're
# just pulled back out of the signature string here for the dashboard, the
# same way hidden_sizes/batch_size are pulled out below. The "h" component of
# sig_tdmpc2 is this horizon (not a network width, see HIDDEN_SIZES_ALGOS above).
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


def extract_hidden_sizes(algo, architecture_signature):
    if algo not in HIDDEN_SIZES_ALGOS or not isinstance(architecture_signature, str):
        return None
    m = HIDDEN_SIZES_RE.match(architecture_signature)
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
    "updates_per_env_step": {"label": "UTD (updates/env step)", "col": "utd_str", "order": None},
    "architecture_signature": {"label": "Architecture", "col": "architecture_signature", "order": None},
    "hidden_sizes": {"label": "Hidden sizes", "col": "hidden_sizes", "order": None},
    "batch_size": {"label": "Batch size", "col": "batch_size", "order": None},
    "mbpo_rollout_regime": {"label": "MBPO Rollout Regime", "col": "mbpo_rollout_regime", "order": None,
                             "sort_key": rollout_regime_sort_key},
    "tdmpc2_horizon": {"label": "TD-MPC2 Horizon", "col": "tdmpc2_horizon", "order": None},
    "tdmpc2_num_q": {"label": "TD-MPC2 Num Q", "col": "tdmpc2_num_q", "order": None},
}
# Only meaningful for algo == "mbpo" / "tdmpc2" respectively (NaN elsewhere)
# -- excluded from the X-axis/Color/Facet pivot dropdowns unless that algo is
# among the selected algos (see the *-pivot-options callbacks, mirroring
# each field's own FILTER dropdown visibility toggle).
MBPO_ROLLOUT_DIM = "mbpo_rollout_regime"
TDMPC2_EXTRA_DIMS = {"tdmpc2_horizon", "tdmpc2_num_q"}
GATED_DIMS = {MBPO_ROLLOUT_DIM} | TDMPC2_EXTRA_DIMS
PER_RUN_DIMENSIONS = dict(DIMENSIONS, seed=dict(label="Seed", col="seed_str", order=None))

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

NONE_VALUE = "__none__"


def load_cross_seed(flop_dir):
    path = os.path.join(flop_dir, "cross_seed_energy_per_flop.csv")
    df = pd.read_csv(path)
    df["utd_str"] = "UTD " + df["updates_per_env_step"].astype(str)
    df["hidden_sizes"] = [extract_hidden_sizes(a, s) for a, s in zip(df["algo"], df["architecture_signature"])]
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
    df["hidden_sizes"] = [extract_hidden_sizes(a, s) for a, s in zip(df["algo"], df["architecture_signature"])]
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


def build_grouped_bar(df, dims, x_dim, color_dim, facet_dim, metric_col, metric_label, log_y):
    if df.empty:
        return empty_figure("No rows match the current filters.")
    if not x_dim:
        return empty_figure("Pick an X-axis dimension.")

    group_keys = [d for d in [x_dim, color_dim, facet_dim] if d]
    seen = set()
    group_keys = [k for k in group_keys if not (k in seen or seen.add(k))]
    group_cols = [dims[k]["col"] for k in group_keys]

    g = (
        df.groupby(group_cols, dropna=False)[metric_col]
        .agg(value="mean", n="count")
        .reset_index()
    )

    category_orders = {}
    for k in group_keys:
        col = dims[k]["col"]
        order = dims[k]["order"]
        sort_key = dims[k].get("sort_key")
        if order:
            present = g[col].unique().tolist()
            category_orders[col] = [v for v in order if v in present] + sorted(set(present) - set(order))
        elif sort_key:
            present = g[col].dropna().unique().tolist()
            category_orders[col] = sorted(present, key=sort_key)

    color_col = dims[color_dim]["col"] if color_dim else None
    facet_col = dims[facet_dim]["col"] if facet_dim else None
    color_map = None
    if color_dim == "algo":
        color_map = ALGO_COLORS

    fig = px.bar(
        g, x=dims[x_dim]["col"], y="value", color=color_col, facet_col=facet_col,
        barmode="group", category_orders=category_orders, color_discrete_map=color_map,
        custom_data=["n"],
        labels={"value": metric_label, dims[x_dim]["col"]: dims[x_dim]["label"],
                **({color_col: dims[color_dim]["label"]} if color_col else {}),
                **({facet_col: dims[facet_dim]["label"]} if facet_col else {})},
    )
    fig.update_traces(
        hovertemplate=metric_label + ": %{y:.4g}<br>Rows averaged: %{customdata[0]}<extra></extra>"
    )
    fig.update_layout(
        height=520,
        margin=dict(l=60, r=20, t=60, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis_type="log" if log_y else "linear",
    )
    fig.update_xaxes(matches=None, showticklabels=True)
    fig.update_yaxes(matches=None, showticklabels=True)
    return fig


def build_box(df, dims, x_dim, color_dim, facet_dim, metric_col, metric_label, log_y):
    if df.empty:
        return empty_figure("No rows match the current filters.")
    if not x_dim:
        return empty_figure("Pick an X-axis dimension.")
    color_col = dims[color_dim]["col"] if color_dim else None
    facet_col = dims[facet_dim]["col"] if facet_dim else None
    color_map = ALGO_COLORS if color_dim == "algo" else None

    category_orders = {}
    for k in [x_dim, color_dim, facet_dim]:
        if not k:
            continue
        col = dims[k]["col"]
        if dims[k]["order"]:
            present = df[col].unique().tolist()
            category_orders[col] = [v for v in dims[k]["order"] if v in present] + sorted(set(present) - set(dims[k]["order"]))
        elif dims[k].get("sort_key"):
            present = df[col].dropna().unique().tolist()
            category_orders[col] = sorted(present, key=dims[k]["sort_key"])

    fig = px.box(
        df, x=dims[x_dim]["col"], y=metric_col, color=color_col, facet_col=facet_col,
        points="all", category_orders=category_orders, color_discrete_map=color_map,
        labels={metric_col: metric_label, dims[x_dim]["col"]: dims[x_dim]["label"],
                **({color_col: dims[color_dim]["label"]} if color_col else {}),
                **({facet_col: dims[facet_dim]["label"]} if facet_col else {})},
        hover_data=["seed", "run_dir"] if "run_dir" in df.columns else None,
    )
    fig.update_layout(
        height=520,
        margin=dict(l=60, r=20, t=40, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis_type="log" if log_y else "linear",
    )
    fig.update_xaxes(matches=None, showticklabels=True)
    fig.update_yaxes(matches=None, showticklabels=True)
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
                "(CPU-side or elementwise ops) -- their Energy/FLOP is excluded from TOTAL_MEASURED_TRAINING.",
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
                                    [{"label": l, "value": v} for v, l in CROSS_SEED_METRICS],
                                    "mean_energy_per_flop_j_per_flop"),
                    html.Div([html.Label("Log Y"), dcc.Checklist(id="cs-logy", options=[{"label": "", "value": "log"}], value=["log"])],
                             className="pivot-field"),
                ],
                className="pivot-row",
            ),
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
                                    [{"label": l, "value": v} for v, l in PER_RUN_METRICS],
                                    "energy_per_flop_j_per_flop"),
                    html.Div([html.Label("Log Y"), dcc.Checklist(id="pr-logy", options=[{"label": "", "value": "log"}], value=["log"])],
                             className="pivot-field"),
                ],
                className="pivot-row",
            ),
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

    @app.callback(
        Output("cs-graph", "figure"), Output("cs-table-wrap", "children"),
        Input("cs-algo", "value"), Input("cs-env", "value"), Input("cs-arch", "value"),
        Input("cs-hidden", "value"), Input("cs-batch", "value"), Input("cs-utd", "value"),
        Input("cs-rollout", "value"), Input("cs-horizon", "value"), Input("cs-numq", "value"),
        Input("cs-segment", "value"),
        Input("cs-xaxis", "value"), Input("cs-color", "value"), Input("cs-facet", "value"),
        Input("cs-metric", "value"), Input("cs-logy", "value"),
    )
    def _cs_update(algo, env, arch, hidden, batch, utd, rollout, horizon, numq, segment, x_dim, color_dim, facet_dim, metric, logy):
        current = {"algo": algo, "env_id": env, "architecture_signature": arch,
                   "hidden_sizes": hidden, "batch_size": batch, "updates_per_env_step": utd,
                   MBPO_ROLLOUT_FIELD: rollout, TDMPC2_HORIZON_FIELD: horizon, TDMPC2_NUM_Q_FIELD: numq,
                   "segment": segment}
        filtered = apply_filters(cross_df, FILTER_FIELDS_WITH_EXTRAS, current)
        color_dim = None if color_dim == NONE_VALUE else color_dim
        facet_dim = None if facet_dim == NONE_VALUE else facet_dim
        metric_label = dict(CROSS_SEED_METRICS)[metric]
        fig = build_grouped_bar(filtered, DIMENSIONS, x_dim, color_dim, facet_dim, metric, metric_label, bool(logy))
        display_cols = ["algo", "env_id", "architecture_signature", "hidden_sizes", "batch_size", "updates_per_env_step",
                         "mbpo_rollout_regime", "tdmpc2_horizon", "tdmpc2_num_q", "segment",
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
        return fig, table

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
        Input("pr-algo", "value"), Input("pr-env", "value"), Input("pr-arch", "value"),
        Input("pr-hidden", "value"), Input("pr-batch", "value"), Input("pr-utd", "value"),
        Input("pr-rollout", "value"), Input("pr-horizon", "value"), Input("pr-numq", "value"),
        Input("pr-segment", "value"),
        Input("pr-seed", "value"), Input("pr-canonical", "value"),
        Input("pr-charttype", "value"),
        Input("pr-xaxis", "value"), Input("pr-color", "value"), Input("pr-facet", "value"),
        Input("pr-metric", "value"), Input("pr-logy", "value"),
    )
    def _pr_update(algo, env, arch, hidden, batch, utd, rollout, horizon, numq, segment, seed, canonical, charttype,
                    x_dim, color_dim, facet_dim, metric, logy):
        base = per_run_df[per_run_df["included_in_cross_seed_avg"]] if canonical == "canonical" else per_run_df
        current = {"algo": algo, "env_id": env, "architecture_signature": arch,
                   "hidden_sizes": hidden, "batch_size": batch, "updates_per_env_step": utd,
                   MBPO_ROLLOUT_FIELD: rollout, TDMPC2_HORIZON_FIELD: horizon, TDMPC2_NUM_Q_FIELD: numq,
                   "segment": segment, "seed": seed}
        filtered = apply_filters(base, PER_RUN_FILTER_FIELDS_WITH_EXTRAS, current)
        color_dim = None if color_dim == NONE_VALUE else color_dim
        facet_dim = None if facet_dim == NONE_VALUE else facet_dim
        metric_label = dict(PER_RUN_METRICS)[metric]
        if charttype == "box":
            fig = build_box(filtered, PER_RUN_DIMENSIONS, x_dim, color_dim, facet_dim, metric, metric_label, bool(logy))
        else:
            fig = build_grouped_bar(filtered, PER_RUN_DIMENSIONS, x_dim, color_dim, facet_dim, metric, metric_label, bool(logy))
        display_cols = ["algo", "env_id", "architecture_signature", "hidden_sizes", "batch_size", "updates_per_env_step",
                         "mbpo_rollout_regime", "tdmpc2_horizon", "tdmpc2_num_q", "seed",
                         "included_in_cross_seed_avg", "segment", "call_count", "total_flops",
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
        return fig, table

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
