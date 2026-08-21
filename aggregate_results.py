"""
Walks results/**/segment_energy.json and metadata.json, and produces one
tidy CSV row per run for downstream analysis (pandas / R / plotting).

Usage:
    python aggregate_results.py --results-dir results --out summary.csv
"""
import argparse
import glob
import json
import os

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out", default="summary.csv")
    args = p.parse_args()

    rows = []
    pattern = os.path.join(args.results_dir, "**", "segment_energy.json")
    for energy_path in glob.glob(pattern, recursive=True):
        run_dir = os.path.dirname(energy_path)
        meta_path = os.path.join(run_dir, "metadata.json")
        if not os.path.exists(meta_path):
            continue

        with open(energy_path) as f:
            energy = json.load(f)
        with open(meta_path) as f:
            meta = json.load(f)

        row = {
            "run_dir": run_dir,
            "algo": meta.get("algo_name"),
            "env": meta.get("env_id"),
            "seed": meta.get("seed"),
            "device": meta.get("device"),
            "cuda_device_name": meta.get("cuda_device_name"),
            "git_commit": meta.get("git_commit"),
            "start_time_utc": meta.get("start_time_utc"),
            "end_time_utc": meta.get("end_time_utc"),
        }
        for k, v in energy.items():
            if k.startswith("_") and not k.startswith("_total"):
                continue  # skip internal debug fields like _sub_segment_wall_time_seconds
            row[f"energy_kwh__{k}".replace("_total_kg_co2eq", "total_kg_co2eq")] = v

        metrics_path = os.path.join(run_dir, "training_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            epochs = metrics.get("epochs") or []
            episodes = metrics.get("episodes") or []
            episode_returns = [e["return"] for e in episodes if e["phase"] == "train"]
            row["reward__final_cumulative_reward"] = epochs[-1]["cumulative_reward"] if epochs else None
            row["reward__num_episodes_completed"] = len(episode_returns)
            row["reward__final_epoch_mean_episode_return"] = epochs[-1]["mean_episode_return"] if epochs else None
            row["reward__max_episode_return"] = max(episode_returns) if episode_returns else None
            row["reward__last_10pct_mean_episode_return"] = (
                sum(episode_returns[-max(1, len(episode_returns) // 10):])
                / len(episode_returns[-max(1, len(episode_returns) // 10):])
                if episode_returns else None
            )

        rows.append(row)

    if not rows:
        print(f"No runs found under {args.results_dir}")
        return

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")
    print(df.head())


if __name__ == "__main__":
    main()
