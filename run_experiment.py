"""
CLI entry point.

Example (quick smoke test on a cheap env, short run):
    python run_experiment.py --algo sac --env Pendulum-v1 --seed 0 \
        --train-steps 5000 --warmup-steps 1000 \
        --idle-baseline-seconds 10 --idle-tail-seconds 10 --settle-seconds 5 \
        --no-gpu-lock --no-thermal-gate

Example (real MuJoCo run, full protocol):
    python run_experiment.py --algo sac --env HalfCheetah-v5 --seed 0

Algorithm hyperparameters are NOT set via CLI flags here -- they live in each
algorithm's config dataclass in configs/config.py (see ALGO_CONFIGS), keyed by
--algo. To deviate from an algorithm's defaults for a single run, pass a JSON
file of overrides:

    python run_experiment.py --algo sac --env HalfCheetah-v5 \
        --algo-config-overrides configs/overrides/sac_lowlr.json

where sac_lowlr.json contains e.g. {"actor_lr": 1e-4, "critic_lr": 1e-4}.
"""
import argparse
import json

from configs.config import ALGO_CONFIGS, ExperimentConfig
from experiment_runner import run_experiment


def parse_args():
    p = argparse.ArgumentParser(description="Run one energy-profiled RL training experiment.")
    p.add_argument("--algo", default="sac", choices=sorted(ALGO_CONFIGS), help="Algorithm to run.")
    p.add_argument("--env", default="HalfCheetah-v5", help="Gymnasium environment id.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")

    p.add_argument("--train-steps", type=int, default=100_000)
    p.add_argument("--warmup-steps", type=int, default=5_000)
    p.add_argument("--steps-per-epoch", type=int, default=1000,
                    help="Env steps per CodeCarbon-tagged rollout/gradient_updates task pair.")

    p.add_argument("--settle-seconds", type=float, default=30.0)
    p.add_argument("--idle-baseline-seconds", type=float, default=90.0)
    p.add_argument("--idle-tail-seconds", type=float, default=90.0)

    p.add_argument("--no-gpu-lock", action="store_true", help="Disable GPU clock locking.")
    p.add_argument("--no-thermal-gate", action="store_true", help="Disable thermal gating between runs.")
    p.add_argument("--tracking-mode", default="machine", choices=["machine", "process"])
    p.add_argument("--output-dir", default="results")
    p.add_argument("--country-iso-code", default="DEU")

    p.add_argument("--algo-config-overrides", default=None,
                    help="Optional path to a JSON file of field overrides for the "
                         "selected algorithm's config dataclass (see ALGO_CONFIGS "
                         "in configs/config.py). Unset fields keep their dataclass defaults.")

    return p.parse_args()


def build_algo_config(algo_name: str, overrides_path: str | None):
    """Look up the algorithm's config dataclass from the registry and apply
    any JSON overrides for this run. This is how each algorithm's config
    file (its dataclass in configs/config.py) gets picked, instead of the
    runner hardcoding one algorithm's hyperparameters as CLI flags."""
    config_cls = ALGO_CONFIGS[algo_name]

    overrides = {}
    if overrides_path:
        with open(overrides_path) as f:
            overrides = json.load(f)
        valid_fields = {f.name for f in __import__("dataclasses").fields(config_cls)}
        unknown = set(overrides) - valid_fields
        if unknown:
            raise ValueError(
                f"Unknown field(s) {sorted(unknown)} in {overrides_path} "
                f"for {config_cls.__name__}; valid fields: {sorted(valid_fields)}"
            )

    return config_cls(**overrides)


def main():
    args = parse_args()

    exp_cfg = ExperimentConfig(
        algo_name=args.algo,
        env_id=args.env,
        seed=args.seed,
        device=args.device,
        train_steps=args.train_steps,
        warmup_steps=args.warmup_steps,
        settle_seconds=args.settle_seconds,
        idle_baseline_seconds=args.idle_baseline_seconds,
        idle_tail_seconds=args.idle_tail_seconds,
        steps_per_epoch=args.steps_per_epoch,
        lock_gpu_clocks=not args.no_gpu_lock,
        thermal_gate_enabled=not args.no_thermal_gate,
        tracking_mode=args.tracking_mode,
        output_dir=args.output_dir,
        country_iso_code=args.country_iso_code,
    )

    algo_cfg = build_algo_config(args.algo, args.algo_config_overrides)

    run_dir, energy_log = run_experiment(exp_cfg, algo_cfg)
    print(f"\nRun artifacts written to: {run_dir}")


if __name__ == "__main__":
    main()
