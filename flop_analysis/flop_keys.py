"""
Shared architecture-signature keys used by both measure_flops.py and
compute_energy_per_flop.py, so a run is always joined against the FLOP
constants measured for its *actual* architecture -- not just its algorithm
and environment.

This repo's results/ directory contains more than the UTD sweep: e.g.
results/sac/HalfCheetah-v5 has runs at hidden_sizes (256,256), (512,512), and
(1024,1024) from the project's evolution, alongside the final UTD sweep at
(1024,1024). FLOP counts depend on architecture, so measure_flops.py must
measure every distinct combination that appears (per the methodology doc's
"any hyperparameter sweep that changes batch size, network width, ensemble
size... needs its own Step 1 measurement" caveat), and compute_energy_per_flop.py
must look up each run under its own signature, not just its (algo, env_id).
"""
from __future__ import annotations


def sig_sac_td3(algo_config: dict) -> str:
    h = "x".join(str(x) for x in algo_config["hidden_sizes"])
    return f"bs{algo_config['batch_size']}_h{h}"


def sig_mbpo(algo_config: dict) -> str:
    base = sig_sac_td3(algo_config)
    mh = "x".join(str(x) for x in algo_config["model_hidden_sizes"])
    return f"{base}_ens{algo_config['ensemble_size']}_mh{mh}_mb{algo_config['model_train_batch_size']}"


def sig_tdmpc2(algo_config: dict) -> str:
    return (
        f"bs{algo_config['batch_size']}"
        f"_h{algo_config['horizon']}"
        f"_ns{algo_config['num_samples']}"
        f"_it{algo_config['iterations']}"
        f"_pit{algo_config['num_pi_trajs']}"
        f"_ne{algo_config['num_elites']}"
        f"_nq{algo_config['num_q']}"
        f"_enc{algo_config['num_enc_layers']}x{algo_config['enc_dim']}"
        f"_mlp{algo_config['mlp_dim']}"
        f"_lat{algo_config['latent_dim']}"
        f"_ep{algo_config['episodic']}"
    )


SIGNATURE_FNS = {
    "sac": sig_sac_td3,
    "td3": sig_sac_td3,
    "mbpo": sig_mbpo,
    "tdmpc2": sig_tdmpc2,
}


def signature(algo: str, algo_config: dict) -> str:
    return SIGNATURE_FNS[algo](algo_config)
