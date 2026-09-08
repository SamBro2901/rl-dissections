"""
Interactive Plotly Dash dashboard for browsing RL energy experiment results.

Scans results/<algo>/<env_id>/<seed>/<timestamp>/ for run folders (any folder
containing a metadata.json is treated as one run) and lets you pick a run from
a dropdown. For the selected run it renders:

  - CodeCarbon per-task metrics from emissions_base_*.csv
  - Training episode metrics from training_metrics.json ("episodes")
  - Training epoch metrics from training_metrics.json ("epochs")

Every numeric column in those files can be toggled on/off with a checklist;
each checked column gets its own stacked subplot sharing the x-axis, since
the columns span very different units (watts, kWh, %, seconds, ...).

Usage:
    python dashboard.py [--results-dir results] [--port 8050]
"""
import argparse
import glob
import json
import os
from datetime import datetime
from functools import lru_cache

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output

RESULTS_DIR = "results"

PHASE_COLORS = {
    "idle_baseline_head": "#9e9e9e",
    "idle_baseline_tail": "#616161",
    "warmup": "#f2a900",
    "rollout": "#1f77b4",
    "gradient_updates": "#d62728",
    "train": "#1f77b4",
}

# Task phases that are calibration/exploration overhead rather than actual
# training work — excluded from the "average power / total energy during
# training" summary stats.
NON_TRAINING_PHASES = {"idle_baseline_head", "idle_baseline_tail", "warmup"}

# Preferred display order for phase/task-type bars and legends.
PHASE_ORDER = ["idle_baseline_head", "warmup", "rollout", "gradient_updates", "idle_baseline_tail"]

# Columns that are numeric but not useful as plottable metrics.
NON_METRIC_COLUMNS = {
    "step", "episode_idx", "epoch", "seed", "elapsed_seconds",
    "longitude", "latitude", "cpu_count", "gpu_count", "ram_total_size",
}

# Units shown alongside axis labels wherever a column has a well-defined unit.
UNITS = {
    "duration": "s",
    "elapsed_seconds": "s",
    "total_duration": "s",
    "emissions": "kg CO2",
    "emissions_rate": "kg CO2/s",
    "cpu_power": "W",
    "gpu_power": "W",
    "ram_power": "W",
    "cpu_energy": "kWh",
    "gpu_energy": "kWh",
    "ram_energy": "kWh",
    "energy_consumed": "kWh",
    "water_consumed": "L",
    "cpu_utilization_percent": "%",
    "gpu_utilization_percent": "%",
    "ram_utilization_percent": "%",
    "ram_used_gb": "GB",
    "return": "reward",
    "length": "steps",
    "env_step": "steps",
    "cumulative_reward": "reward",
    "epoch_reward_sum": "reward",
    "epoch_reward_mean_per_step": "reward/step",
    "mean_episode_return": "reward",
    "critic_loss_mean": "loss",
    "actor_loss_mean": "loss",
    "buffer_size": "samples",
    "num_episodes_completed": "episodes",
}


def axis_label(col):
    unit = UNITS.get(col)
    return f"{col} ({unit})" if unit else col


def discover_runs(results_dir):
    """Find every folder under results_dir that has a metadata.json (one per run)."""
    runs = []
    for meta_path in glob.glob(os.path.join(results_dir, "**", "metadata.json"), recursive=True):
        run_dir = os.path.dirname(meta_path)
        rel = os.path.relpath(run_dir, results_dir)
        parts = rel.replace("\\", "/").split("/")
        label = " / ".join(parts) if len(parts) > 1 else rel
        runs.append({"label": label, "value": run_dir})
    runs.sort(key=lambda r: r["value"])
    return runs


@lru_cache(maxsize=16)
def load_emissions_base(run_dir):
    matches = glob.glob(os.path.join(run_dir, "emissions_base_*.csv"))
    if not matches:
        return None
    df = pd.read_csv(matches[0])
    df = df.reset_index(drop=True)
    df.insert(0, "step", range(len(df)))
    if "task_name" in df.columns:
        df["task_type"] = df["task_name"].astype(str).str.replace(r"_\d+$", "", regex=True)
    if "duration" in df.columns:
        df["elapsed_seconds"] = df["duration"].cumsum()
    return df


def training_only(df):
    """Rows for actual training work (rollout + gradient updates), excluding
    the idle-baseline calibration windows and the random-action warmup phase."""
    if df is None or "task_type" not in df.columns:
        return None
    return df[~df["task_type"].isin(NON_TRAINING_PHASES)]


@lru_cache(maxsize=16)
def load_training_metrics(run_dir):
    path = os.path.join(run_dir, "training_metrics.json")
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        data = json.load(f)
    episodes = pd.DataFrame(data.get("episodes", []))
    epochs = pd.DataFrame(data.get("epochs", []))
    return episodes, epochs


@lru_cache(maxsize=16)
def load_metadata(run_dir):
    path = os.path.join(run_dir, "metadata.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def numeric_columns(df):
    if df is None or df.empty:
        return []
    cols = df.select_dtypes(include="number").columns.tolist()
    return [c for c in cols if c not in NON_METRIC_COLUMNS and df[c].notna().any()]


def default_selection(available, preferred):
    picked = [c for c in preferred if c in available]
    return picked if picked else available[:4]


def empty_figure(message):
    fig = go.Figure()
    fig.update_layout(
        annotations=[dict(text=message, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14))],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=200,
        margin=dict(l=40, r=20, t=30, b=20),
    )
    return fig


def stacked_metric_figure(df, x_col, columns, phase_col=None, title=""):
    if df is None:
        return empty_figure("No data found for this run.")
    if not columns:
        return empty_figure("Select at least one column to plot.")

    n = len(columns)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=min(0.25 / n, 0.08),
                         subplot_titles=columns)

    phases_seen = set()
    has_phase = bool(phase_col) and phase_col in df.columns
    for i, col in enumerate(columns, start=1):
        line_trace = go.Scatter(
            x=df[x_col], y=df[col], mode="lines", line=dict(color="#c8c8c8", width=1.5),
            name=col, showlegend=False,
        )
        if has_phase:
            line_trace.hoverinfo = "skip"
        else:
            line_trace.hovertemplate = f"{col}=%{{y}}<br>{x_col}=%{{x}}<extra></extra>"
        fig.add_trace(line_trace, row=i, col=1)

        if has_phase:
            for phase, group in df.groupby(phase_col, sort=False):
                show_legend = i == 1 and phase not in phases_seen
                phases_seen.add(phase)
                fig.add_trace(
                    go.Scatter(
                        x=group[x_col], y=group[col], mode="markers",
                        marker=dict(size=5, color=PHASE_COLORS.get(phase, "#333333")),
                        name=str(phase), legendgroup=str(phase), showlegend=show_legend,
                        text=group["task_name"] if "task_name" in group.columns else None,
                        hovertemplate=f"{col}=%{{y}}<br>{x_col}=%{{x}}<br>%{{text}}<extra></extra>",
                    ),
                    row=i, col=1,
                )

    fig.update_layout(
        height=max(260, 230 * n),
        title=title,
        margin=dict(l=60, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
        hovermode="x unified",
    )
    for i, col in enumerate(columns, start=1):
        fig.update_yaxes(title_text=UNITS.get(col, ""), row=i, col=1)
    fig.update_xaxes(title_text=axis_label(x_col), row=n, col=1)
    return fig


def duration_bar_figure(run_dir):
    df = load_emissions_base(run_dir)
    if df is None or "task_type" not in df.columns or "duration" not in df.columns:
        return empty_figure("No task-duration data found for this run.")

    grouped = df.groupby("task_type", sort=False)["duration"].agg(["sum", "size", "mean"]).reset_index()
    grouped.columns = ["task_type", "total_duration", "count", "avg_duration"]
    grouped["order"] = grouped["task_type"].apply(lambda t: PHASE_ORDER.index(t) if t in PHASE_ORDER else len(PHASE_ORDER))
    grouped = grouped.sort_values("order")

    fig = go.Figure(
        go.Bar(
            x=grouped["task_type"],
            y=grouped["total_duration"],
            marker_color=[PHASE_COLORS.get(t, "#333333") for t in grouped["task_type"]],
            customdata=grouped[["count", "avg_duration"]].values,
            hovertemplate="%{x}<br>Total: %{y:.2f} s<br>%{customdata[0]} tasks, avg %{customdata[1]:.2f} s/task<extra></extra>",
        )
    )
    fig.update_layout(
        title="Phase duration breakdown",
        height=380,
        margin=dict(l=60, r=20, t=60, b=40),
    )
    fig.update_xaxes(title_text="Phase")
    fig.update_yaxes(title_text=axis_label("total_duration"))
    return fig


def _phase_grouped(df):
    grouped = df.copy()
    grouped["order"] = grouped["task_type"].apply(lambda t: PHASE_ORDER.index(t) if t in PHASE_ORDER else len(PHASE_ORDER))
    return grouped.sort_values("order")


def component_bar_figure(run_dir, components, agg, colors, title, y_title, hover_unit):
    """Stacked bar of one or more emissions_base_* columns, grouped by phase."""
    df = load_emissions_base(run_dir)
    present = [c for c in components if df is not None and c in df.columns]
    if df is None or "task_type" not in df.columns or not present:
        return empty_figure("No data found for this run.")

    grouped = df.groupby("task_type", sort=False)[present].agg(agg).reset_index()
    grouped = _phase_grouped(grouped)

    fig = go.Figure()
    for col in present:
        fig.add_trace(
            go.Bar(
                x=grouped["task_type"],
                y=grouped[col],
                name=col,
                marker_color=colors.get(col, "#333333"),
                hovertemplate=f"%{{x}}<br>{col}: %{{y:.4f}} {hover_unit}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        height=380,
        barmode="stack",
        margin=dict(l=60, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
    )
    fig.update_xaxes(title_text="Phase")
    fig.update_yaxes(title_text=y_title)
    return fig


def power_bar_figure(run_dir):
    return component_bar_figure(
        run_dir,
        components=["cpu_power", "gpu_power", "ram_power"],
        agg="mean",
        colors={"cpu_power": "#1f77b4", "gpu_power": "#d62728", "ram_power": "#2ca02c"},
        title="Phase power breakdown (avg)",
        y_title="Avg power (W)",
        hover_unit="W",
    )


def energy_bar_figure(run_dir):
    return component_bar_figure(
        run_dir,
        components=["cpu_energy", "gpu_energy", "ram_energy"],
        agg="sum",
        colors={"cpu_energy": "#1f77b4", "gpu_energy": "#d62728", "ram_energy": "#2ca02c"},
        title="Phase energy breakdown (total)",
        y_title="Total energy (kWh)",
        hover_unit="kWh",
    )


def info_card(label, value):
    return html.Div(
        [html.Div(label, className="info-label"), html.Div(value, className="info-value")],
        className="info-card",
    )


def session_duration_seconds(meta):
    try:
        start = datetime.fromisoformat(meta["start_time_utc"])
        end = datetime.fromisoformat(meta["end_time_utc"])
        return (end - start).total_seconds()
    except (KeyError, ValueError, TypeError):
        return None


def build_run_info(run_dir):
    meta = load_metadata(run_dir)
    cfg = meta.get("experiment_config", {})
    train_df = training_only(load_emissions_base(run_dir))
    duration_s = session_duration_seconds(meta)

    cards = [
        info_card("Algorithm", meta.get("algo_name", "-")),
        info_card("Environment", meta.get("env_id", "-")),
        info_card("Seed", meta.get("seed", "-")),
        info_card("Train steps", cfg.get("train_steps", "-")),
        info_card("Start (UTC)", meta.get("start_time_utc", "-")),
        info_card("End (UTC)", meta.get("end_time_utc", "-")),
        info_card("Session duration (min)", f"{duration_s / 60:.1f}" if duration_s is not None else "-"),
    ]
    if train_df is not None and not train_df.empty:
        cards += [
            info_card("Training energy (kWh)", f"{train_df['energy_consumed'].sum():.6f}"),
            info_card("Training emissions (kg CO2)", f"{train_df['emissions'].sum():.6f}"),
            info_card("Avg CPU power — training (W)", f"{train_df['cpu_power'].mean():.1f}"),
            info_card("Avg GPU power — training (W)", f"{train_df['gpu_power'].mean():.1f}"),
        ]
    return html.Div(cards, className="info-grid")


def format_param_value(value):
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    if value is None:
        return "null"
    return str(value)


def build_algo_params(run_dir):
    meta = load_metadata(run_dir)
    algo_cfg = meta.get("algo_config", {})
    if not algo_cfg:
        return html.Div("No algorithm parameters found in metadata.json.")
    cards = [info_card(k, format_param_value(v)) for k, v in algo_cfg.items()]
    return html.Div(cards, className="info-grid")


def make_app(results_dir):
    app = Dash(__name__)
    app.title = "RL Energy Experiments Dashboard"

    runs = discover_runs(results_dir)
    initial_run = runs[0]["value"] if runs else None

    app.layout = html.Div(
        [
            html.H2("RL Energy Experiments Dashboard"),
            html.Div(
                [
                    html.Label("Run:"),
                    dcc.Dropdown(id="run-dropdown", options=runs, value=initial_run, clearable=False),
                ],
                className="run-picker",
            ),
            html.Div(id="run-info"),
            html.Hr(),

            html.H3("Algorithm parameters"),
            html.Div(id="algo-params"),
            html.Hr(),

            html.H3("Phase duration breakdown"),
            dcc.Loading(dcc.Graph(id="duration-bar-graph")),

            html.H3("Phase power breakdown"),
            dcc.Loading(dcc.Graph(id="power-bar-graph")),

            html.H3("Phase energy breakdown"),
            dcc.Loading(dcc.Graph(id="energy-bar-graph")),
            html.Hr(),

            html.H3("CodeCarbon task metrics (emissions_base_*.csv)"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("X-axis: "),
                            dcc.RadioItems(
                                id="emissions-xaxis",
                                options=[
                                    {"label": "Task step", "value": "step"},
                                    {"label": "Elapsed time (s)", "value": "elapsed_seconds"},
                                ],
                                value="step", inline=True,
                            ),
                        ]
                    ),
                    html.Label("Columns:"),
                    dcc.Checklist(id="emissions-columns", options=[], value=[], inline=True, className="col-checklist"),
                ],
                className="controls",
            ),
            dcc.Loading(dcc.Graph(id="emissions-graph")),
            html.Hr(),

            html.H3("Training — episodes"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("X-axis: "),
                            dcc.RadioItems(
                                id="episodes-xaxis",
                                options=[
                                    {"label": "Episode index", "value": "episode_idx"},
                                    {"label": "Env step", "value": "env_step"},
                                ],
                                value="episode_idx", inline=True,
                            ),
                        ]
                    ),
                    html.Label("Columns:"),
                    dcc.Checklist(id="episodes-columns", options=[], value=[], inline=True, className="col-checklist"),
                ],
                className="controls",
            ),
            dcc.Loading(dcc.Graph(id="episodes-graph")),
            html.Hr(),

            html.H3("Training — epochs"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("X-axis: "),
                            dcc.RadioItems(
                                id="epochs-xaxis",
                                options=[
                                    {"label": "Epoch", "value": "epoch"},
                                    {"label": "Env step", "value": "env_step"},
                                ],
                                value="epoch", inline=True,
                            ),
                        ]
                    ),
                    html.Label("Columns:"),
                    dcc.Checklist(id="epochs-columns", options=[], value=[], inline=True, className="col-checklist"),
                ],
                className="controls",
            ),
            dcc.Loading(dcc.Graph(id="epochs-graph")),
        ],
        className="app-container",
    )

    @app.callback(Output("run-info", "children"), Input("run-dropdown", "value"))
    def _update_run_info(run_dir):
        if not run_dir:
            return html.Div("No runs found under results/.")
        return build_run_info(run_dir)

    @app.callback(Output("algo-params", "children"), Input("run-dropdown", "value"))
    def _update_algo_params(run_dir):
        if not run_dir:
            return html.Div()
        return build_algo_params(run_dir)

    @app.callback(Output("duration-bar-graph", "figure"), Input("run-dropdown", "value"))
    def _update_duration_bar(run_dir):
        if not run_dir:
            return empty_figure("No run selected.")
        return duration_bar_figure(run_dir)

    @app.callback(Output("power-bar-graph", "figure"), Input("run-dropdown", "value"))
    def _update_power_bar(run_dir):
        if not run_dir:
            return empty_figure("No run selected.")
        return power_bar_figure(run_dir)

    @app.callback(Output("energy-bar-graph", "figure"), Input("run-dropdown", "value"))
    def _update_energy_bar(run_dir):
        if not run_dir:
            return empty_figure("No run selected.")
        return energy_bar_figure(run_dir)

    @app.callback(
        Output("emissions-columns", "options"),
        Output("emissions-columns", "value"),
        Input("run-dropdown", "value"),
    )
    def _update_emissions_columns(run_dir):
        df = load_emissions_base(run_dir) if run_dir else None
        cols = numeric_columns(df)
        preferred = [
            "cpu_power", "gpu_power", "ram_power", "cpu_energy", "gpu_energy",
            "ram_energy", "energy_consumed", "emissions", "cpu_utilization_percent",
            "gpu_utilization_percent",
        ]
        options = [{"label": c, "value": c} for c in cols]
        return options, default_selection(cols, preferred)

    @app.callback(
        Output("emissions-graph", "figure"),
        Input("run-dropdown", "value"),
        Input("emissions-xaxis", "value"),
        Input("emissions-columns", "value"),
    )
    def _update_emissions_graph(run_dir, x_col, columns):
        df = load_emissions_base(run_dir) if run_dir else None
        return stacked_metric_figure(df, x_col or "step", columns, phase_col="task_type",
                                      title="CodeCarbon per-task metrics")

    @app.callback(
        Output("episodes-columns", "options"),
        Output("episodes-columns", "value"),
        Input("run-dropdown", "value"),
    )
    def _update_episodes_columns(run_dir):
        episodes, _ = load_training_metrics(run_dir) if run_dir else (None, None)
        cols = numeric_columns(episodes)
        options = [{"label": c, "value": c} for c in cols]
        return options, default_selection(cols, ["return", "length"])

    @app.callback(
        Output("episodes-graph", "figure"),
        Input("run-dropdown", "value"),
        Input("episodes-xaxis", "value"),
        Input("episodes-columns", "value"),
    )
    def _update_episodes_graph(run_dir, x_col, columns):
        episodes, _ = load_training_metrics(run_dir) if run_dir else (None, None)
        if episodes is None or episodes.empty:
            return empty_figure("No training_metrics.json found for this run.")
        return stacked_metric_figure(episodes, x_col or "episode_idx", columns, phase_col="phase",
                                      title="Training episodes")

    @app.callback(
        Output("epochs-columns", "options"),
        Output("epochs-columns", "value"),
        Input("run-dropdown", "value"),
    )
    def _update_epochs_columns(run_dir):
        _, epochs = load_training_metrics(run_dir) if run_dir else (None, None)
        cols = numeric_columns(epochs)
        options = [{"label": c, "value": c} for c in cols]
        return options, default_selection(cols, ["mean_episode_return", "critic_loss_mean", "actor_loss_mean", "alpha_end"])

    @app.callback(
        Output("epochs-graph", "figure"),
        Input("run-dropdown", "value"),
        Input("epochs-xaxis", "value"),
        Input("epochs-columns", "value"),
    )
    def _update_epochs_graph(run_dir, x_col, columns):
        _, epochs = load_training_metrics(run_dir) if run_dir else (None, None)
        if epochs is None or epochs.empty:
            return empty_figure("No training_metrics.json found for this run.")
        return stacked_metric_figure(epochs, x_col or "epoch", columns, phase_col=None,
                                      title="Training epochs")

    return app


APP_CSS = """
body { font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; background: #fafafa; color: #1a1a1a; }
.app-container { max-width: 1200px; margin: 0 auto; padding: 24px; }
.run-picker { max-width: 640px; margin-bottom: 16px; }
.controls { margin: 12px 0; padding: 12px; background: #f0f0f0; border-radius: 6px; }
.col-checklist { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 6px; }
.col-checklist label { margin-right: 4px; }
.info-grid { display: flex; flex-wrap: wrap; gap: 12px; margin: 12px 0; }
.info-card { background: #fff; border: 1px solid #e0e0e0; border-radius: 6px; padding: 10px 14px; min-width: 140px; }
.info-label { font-size: 11px; text-transform: uppercase; color: #777; letter-spacing: 0.03em; }
.info-value { font-size: 16px; font-weight: 600; margin-top: 2px; }
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    app = make_app(args.results_dir)
    app.index_string = app.index_string.replace("</head>", f"<style>{APP_CSS}</style></head>")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
