"""
CLI entry point.

Example (quick smoke test on a cheap env, short run):
    python run_experiment.py --algo sac --env Pendulum-v1 --seed 0 \
        --train-steps 5000 --warmup-steps 1000 \
        --idle-baseline-seconds 10 --idle-tail-seconds 10 --settle-seconds 5 \
        --no-gpu-lock --no-thermal-gate

Example (real MuJoCo run, full protocol):
    python run_experiment.py --algo sac --env HalfCheetah-v5 --seed 0
"""
import argparse

from configs.config import ExperimentConfig, SACConfig
from experiment_runner import run_experiment


def parse_args():
    p = argparse.ArgumentParser(description="Run one energy-profiled RL training experiment.")
    p.add_argument("--algo", default="sac", choices=["sac"], help="Algorithm to run.")
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

    # SAC hyperparameters (defaults match standard SAC continuous-control settings)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--updates-per-env-step", type=int, default=1)

    return p.parse_args()


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

    algo_cfg = SACConfig(
        batch_size=args.batch_size,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        tau=args.tau,
        updates_per_env_step=args.updates_per_env_step,
    )

    run_dir, energy_log = run_experiment(exp_cfg, algo_cfg)
    print(f"\nRun artifacts written to: {run_dir}")


if __name__ == "__main__":
    main()
