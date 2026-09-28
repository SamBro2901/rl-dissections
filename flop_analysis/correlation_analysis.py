"""
Correlation of return and energy with training compute (FLOPs), per env.

One point per (env, config) -- FLOPs are identical across the 5 canonical
seeds of a config, so seeds are averaged, never used as separate points.

  1. Return vs. FLOPs  (main question): Spearman rho + p-value + 95% bootstrap CI.
  2. Energy vs. FLOPs  (control):       same, plus OLS log10(energy) ~ log10(FLOPs).

Inputs:
  - flop_analysis/output/per_run_energy_per_flop.csv, TOTAL_MEASURED_TRAINING
    rows (total_flops, total_energy_joules), joined to run folders on run_dir.
  - Each run's training_metrics.json: return = mean of the last 10% of
    train-phase episode returns (same definition as aggregate_results.py's
    reward__last_10pct_mean_episode_return).
  - Each run's metadata.json: algo_config, for the config ID.

Bootstrap: 5000 resamples, seeds resampled with replacement *within* each
config, config means recomputed, rho recomputed. Fixed RNG seed.

Outputs (flop_analysis/output/correlation/): config_means.csv,
correlation_summary.csv, correlation_plots.html. Energy is in Joules.

Usage: python flop_analysis/correlation_analysis.py [--results-dir results] [--out-dir ...]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flop_keys import mbpo_rollout_regime, signature  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
PER_RUN_CSV = REPO / "flop_analysis" / "output" / "per_run_energy_per_flop.csv"
CANONICAL_SEEDS = {331, 958, 14577, 43611, 85062}
ENVS = ["HalfCheetah-v5", "Ant-v5"]
N_BOOT = 5000
RNG_SEED = 0
ALGO_COLORS = {"sac": "#1f77b4", "mbpo": "#ff7f0e", "td3": "#2ca02c", "tdmpc2": "#d62728"}


def last_10pct_return(run_dir: Path) -> float:
    episodes = json.loads((run_dir / "training_metrics.json").read_text())["episodes"]
    rets = [e["return"] for e in episodes if e["phase"] == "train"]
    tail = rets[-max(1, len(rets) // 10):]
    return float(np.mean(tail))


def config_id(algo: str, algo_cfg: dict) -> str:
    cid = f"{algo}|{signature(algo, algo_cfg)}|utd{algo_cfg['updates_per_env_step']}"
    if algo == "mbpo":
        cid += f"|{mbpo_rollout_regime(algo_cfg)}"
    return cid


def load_runs(results_dir: Path) -> pd.DataFrame:
    if not PER_RUN_CSV.exists():
        subprocess.run([sys.executable, str(REPO / "flop_analysis" / "compute_energy_per_flop.py")], check=True)
    df = pd.read_csv(PER_RUN_CSV)
    df = df[(df.segment == "TOTAL_MEASURED_TRAINING")
            & df.env_id.isin(ENVS) & df.seed.isin(CANONICAL_SEEDS)]

    rows = []
    for r in df.itertuples():
        # run_dir in the CSV is relative to the repo root, e.g. results/sac/Ant-v5/seed_331/<ts>
        run_dir = results_dir / Path(r.run_dir).relative_to("results")
        meta = json.loads((run_dir / "metadata.json").read_text())
        rows.append({
            "env": r.env_id, "algo": r.algo, "seed": r.seed, "run_dir": r.run_dir,
            "config_id": config_id(r.algo, meta["algo_config"]),
            "flops": r.total_flops, "energy": r.total_energy_joules,
            "return": last_10pct_return(run_dir),
        })
    return pd.DataFrame(rows)


def sanity_check(runs: pd.DataFrame) -> None:
    """Expect 5 runs and one FLOP value per (env, config). Known exception:
    MBPO's FLOPs vary a little per seed, because the dynamics-model fit
    early-stops on holdout error (model_train_epochs differs per seed) and
    synthetic rollouts terminate at different rates. For those groups the
    config point uses the seed-mean FLOPs."""
    g = runs.groupby(["env", "config_id"])
    check = pd.DataFrame({"n_runs": g.size(), "n_flop_values": g.flops.nunique(),
                          "flop_spread_pct": (g.flops.max() - g.flops.min()) / g.flops.mean() * 100})
    bad = check[(check.n_runs != 5) | (check.n_flop_values != 1)]
    if bad.empty:
        print("Sanity check passed: every (env, config) has 5 runs and one unique FLOP value.")
    else:
        print("Sanity check exceptions (n_runs != 5 or more than one FLOP value):")
        print(bad.round(2).to_string())
    print("Configs per env:")
    print(check.groupby(level="env").size().to_string())


def boot_rho(env_runs: pd.DataFrame, col: str, rng: np.random.Generator) -> np.ndarray:
    """Resample seeds within each config, recompute config means, recompute rho.
    FLOPs are resampled paired with the seed's value (only matters for MBPO,
    whose FLOPs vary slightly per seed -- see sanity_check)."""
    flop_means, val_means = [], []
    for _, grp in env_runs.groupby("config_id"):
        idx = rng.integers(0, len(grp), size=(N_BOOT, len(grp)))
        flop_means.append(grp.flops.to_numpy()[idx].mean(axis=1))
        val_means.append(grp[col].to_numpy()[idx].mean(axis=1))
    flop_means, val_means = np.column_stack(flop_means), np.column_stack(val_means)  # (N_BOOT, n_configs)
    return np.array([stats.spearmanr(f, v).statistic for f, v in zip(flop_means, val_means)])


def analyse(runs: pd.DataFrame, cm: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for env in ENVS:
        env_cm, env_runs = cm[cm.env == env], runs[runs.env == env]
        for analysis, col in [("return_vs_flops", "return"), ("energy_vs_flops", "energy")]:
            res = stats.spearmanr(env_cm.flops, env_cm[f"{col}_mean"])
            ci = np.nanpercentile(boot_rho(env_runs, col, rng), [2.5, 97.5])
            row = {"env": env, "analysis": analysis, "n_configs": len(env_cm),
                   "spearman_rho": res.statistic, "ci_low": ci[0], "ci_high": ci[1],
                   "p_value": res.pvalue, "loglog_slope": np.nan, "slope_ci_low": np.nan,
                   "slope_ci_high": np.nan, "r2": np.nan, "intercept": np.nan}
            if col == "energy":
                fit = stats.linregress(np.log10(env_cm.flops), np.log10(env_cm.energy_mean))
                t = stats.t.ppf(0.975, len(env_cm) - 2)
                row.update(loglog_slope=fit.slope, slope_ci_low=fit.slope - t * fit.stderr,
                           slope_ci_high=fit.slope + t * fit.stderr, r2=fit.rvalue ** 2,
                           intercept=fit.intercept)
            rows.append(row)
    return pd.DataFrame(rows)


def plot(runs: pd.DataFrame, cm: pd.DataFrame, summary: pd.DataFrame, out: Path) -> None:
    def title(env, analysis):
        s = summary[(summary.env == env) & (summary.analysis == analysis)].iloc[0]
        t = f"{env}: ρ={s.spearman_rho:.2f} [{s.ci_low:.2f}, {s.ci_high:.2f}], n={s.n_configs}"
        if analysis == "energy_vs_flops":
            t += f", slope={s.loglog_slope:.2f}"
        return t

    panels = [("return", "return_vs_flops"), ("energy", "energy_vs_flops")]
    fig = make_subplots(rows=2, cols=2, horizontal_spacing=0.08, vertical_spacing=0.12,
                        subplot_titles=[title(env, a) for _, a in panels for env in ENVS])
    for i, (col, analysis) in enumerate(panels, start=1):
        for j, env in enumerate(ENVS, start=1):
            for algo, color in ALGO_COLORS.items():
                r = runs[(runs.env == env) & (runs.algo == algo)]
                c = cm[(cm.env == env) & (cm.algo == algo)]
                if c.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=r.flops, y=r[col], mode="markers", showlegend=False, hoverinfo="skip",
                    marker=dict(color=color, size=4, opacity=0.25)), row=i, col=j)
                fig.add_trace(go.Scatter(
                    x=c.flops, y=c[f"{col}_mean"], mode="markers", name=algo,
                    legendgroup=algo, showlegend=(i == 1 and j == 1),
                    marker=dict(color=color, size=9, line=dict(width=1, color="white")),
                    error_y=dict(type="data", array=c[f"{col}_std"], thickness=1),
                    text=c.config_id,
                    hovertemplate="%{text}<br>FLOPs=%{x:.3e}<br>" + col + "=%{y:.4g}<extra></extra>"),
                    row=i, col=j)
            if col == "energy":
                s = summary[(summary.env == env) & (summary.analysis == analysis)].iloc[0]
                xs = np.logspace(np.log10(cm[cm.env == env].flops.min()),
                                 np.log10(cm[cm.env == env].flops.max()), 50)
                fig.add_trace(go.Scatter(
                    x=xs, y=10 ** (s.intercept + s.loglog_slope * np.log10(xs)), mode="lines",
                    line=dict(color="black", dash="dash", width=1), name="log-log OLS fit",
                    legendgroup="fit", showlegend=(j == 1), hoverinfo="skip"), row=i, col=j)
            fig.update_xaxes(type="log", title_text="Total training FLOPs", row=i, col=j)
            if col == "energy":
                fig.update_yaxes(type="log", title_text="Total measured training energy (J)", row=i, col=j)
            else:
                fig.update_yaxes(title_text="Return (last 10% of train episodes)", row=i, col=j)
    fig.update_layout(height=1000, width=1400, template="plotly_white",
                      title="Return and energy vs. training compute (one marker per config, mean ± std over 5 seeds)")
    fig.write_html(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=REPO / "results")
    ap.add_argument("--out-dir", type=Path, default=REPO / "flop_analysis" / "output" / "correlation")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.results_dir)
    sanity_check(runs)

    cm = (runs.groupby(["env", "algo", "config_id"])
          .agg(n_seeds=("seed", "size"), flops=("flops", "mean"), flops_std=("flops", "std"),
               return_mean=("return", "mean"), return_std=("return", "std"),
               energy_mean=("energy", "mean"), energy_std=("energy", "std"))
          .reset_index())
    cm.to_csv(args.out_dir / "config_means.csv", index=False)

    summary = analyse(runs, cm)
    plot(runs, cm, summary, args.out_dir / "correlation_plots.html")
    summary = summary.drop(columns="intercept")
    summary.to_csv(args.out_dir / "correlation_summary.csv", index=False)

    pd.set_option("display.width", 250)
    print("\ncorrelation_summary.csv:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
