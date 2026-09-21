#!/usr/bin/env bash
#
# Network-width sweep: hidden_sizes in {(256,256), (512,512)}, across seeds
# {331, 958, 14577, 43611, 85062}, envs {HalfCheetah-v5, Ant-v5}, for:
#   - sac
#   - mbpo
#   - td3
# -- 3 algos x 2 envs x 2 widths x 5 seeds = 60 runs total.
#
# TD-MPC2 is intentionally excluded: it has no `hidden_sizes` field (unlike
# SACConfig/MBPOConfig/TD3Config) -- its architecture (enc_dim/mlp_dim/
# latent_dim) is the paper's own fixed spec, reused unmodified per
# TDMPC2Config's docstring in configs/config.py, so there's no equivalent
# "width" knob to sweep without deviating from that paper-fidelity choice.
#
# This revisits the (256,256)/(512,512) architectures mentioned in the repo
# history (dev exploration before settling on (1024,1024), see CLAUDE.md) --
# but this time at the canonical 5-seed set, both envs, and the current
# settled protocol/hyperparameters, so the results are directly comparable
# to the existing (1024,1024) canonical runs and go through
# flop_analysis/ the same way (flop_keys.sig_sac_td3/sig_mbpo already key
# on hidden_sizes, so measure_flops.py just needs to pick up these new
# architectures the next time it's run).
#
# MBPO on Ant-v5 needs its rollout-schedule override (configs/overrides/
# mbpo_ant.json) *combined* with the width override, since only one
# --algo-config-overrides file can be passed per run -- see
# configs/overrides/mbpo_ant_width{256,512}.json. MBPO on HalfCheetah-v5 and
# SAC/TD3 on both envs only need the width override on its own.
#
# Each run needs sudo (for GPU clock locking + CPU governor pinning, see
# cli_commands.txt), so this script primes a sudo credential cache once up
# front and refreshes it in the background for the life of the sweep --
# nothing is written to /etc/sudoers, so you must run this interactively
# and enter your password once before walking away.
#
# Because this can run for hours and you may not be at your desk, launch it
# inside `screen` (or `nohup ... &`) so it survives an SSH disconnect:
#
#   screen -S width_sweep
#   ./run_width_sweep.sh
#   # Ctrl-A D to detach; reattach later with: screen -r width_sweep
#
# SHUTDOWN: once the sweep finishes (all 60 runs attempted, pass or fail),
# this script powers the machine off automatically -- see the bottom of the
# file. Don't launch this unless you actually want the desktop to shut down
# unattended when it's done.
#
set -uo pipefail
cd "$(dirname "$0")"

# algo:env combos to sweep -- all three width-capable algos, both canonical envs.
COMBOS=("sac:HalfCheetah-v5" "sac:Ant-v5" "mbpo:HalfCheetah-v5" "mbpo:Ant-v5" "td3:HalfCheetah-v5" "td3:Ant-v5")
SEEDS=(331 958 14577 43611 85062)
WIDTHS=(256 512)
PYTHON_BIN="rl-exp/bin/python"

# Per-algo --warmup-steps, same convention as run_utd_sweep.sh: TD3's paper
# hyperparameters use 10,000 steps for HalfCheetah-v1/Ant-v1 ("stable length
# environments"); SAC and MBPO use the CLI default for both envs.
declare -A ALGO_WARMUP_STEPS=(
    ["sac"]=""
    ["mbpo"]=""
    ["td3"]="10000"
)

# Resolve the --algo-config-overrides path for a given algo/env/width.
# MBPO on Ant-v5 needs its rollout-schedule override folded in alongside the
# width override (see comment above); everything else just needs the width.
override_file() {
    local algo="$1" env="$2" width="$3"
    if [[ "$algo" == "mbpo" && "$env" == "Ant-v5" ]]; then
        echo "configs/overrides/mbpo_ant_width${width}.json"
    else
        echo "configs/overrides/${algo}_width${width}.json"
    fi
}

STATUS_FILE="results/_width_sweep_status.json"
LOG_FILE="results/_width_sweep.log"
mkdir -p "$(dirname "$STATUS_FILE")"

TOTAL=$(( ${#COMBOS[@]} * ${#WIDTHS[@]} * ${#SEEDS[@]} ))

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# ---------------- sudo credential priming + keepalive ----------------
log "Priming sudo credentials (you may be prompted for your password now)..."
if ! sudo -v; then
    log "ERROR: sudo authentication failed. Aborting before starting any runs."
    exit 1
fi

SUDO_KEEPALIVE_PID=""
cleanup() {
    if [[ -n "$SUDO_KEEPALIVE_PID" ]]; then
        kill "$SUDO_KEEPALIVE_PID" 2>/dev/null
    fi
}
trap cleanup EXIT

(
    while true; do
        sudo -n true 2>/dev/null
        sleep 60
        kill -0 "$$" 2>/dev/null || exit
    done
) &
SUDO_KEEPALIVE_PID=$!
log "sudo credential keepalive started (pid $SUDO_KEEPALIVE_PID, refreshes every 60s)."

# ---------------- build the initial status file (all runs 'pending') ----------------
python3 - "$STATUS_FILE" "$TOTAL" "${COMBOS[*]}" "${WIDTHS[*]}" "${SEEDS[*]}" <<'PYEOF'
import json, sys, datetime

status_file, total, combos_str, widths_str, seeds_str = sys.argv[1:6]
combos = [tuple(c.split(":")) for c in combos_str.split()]
widths = [int(w) for w in widths_str.split()]
seeds = [int(s) for s in seeds_str.split()]

runs = []
idx = 1
for algo, env in combos:
    for width in widths:
        for seed in seeds:
            runs.append({
                "index": idx,
                "algo": algo,
                "env": env,
                "seed": seed,
                "hidden_sizes": [width, width],
                "status": "pending",
                "run_dir": None,
                "started_utc": None,
                "finished_utc": None,
                "exit_code": None,
            })
            idx += 1

now = datetime.datetime.now(datetime.timezone.utc).isoformat()
data = {
    "total_runs": int(total),
    "started_utc": now,
    "updated_utc": now,
    "runs": runs,
}
with open(status_file, "w") as f:
    json.dump(data, f, indent=2)
PYEOF

update_run_status() {
    # args: index field1=value1 [field2=value2 ...] (values are JSON-encoded already)
    local idx="$1"; shift
    local jq_filter=".updated_utc = \$now"
    local jq_args=(--argjson now "\"$(date -u +%Y-%m-%dT%H:%M:%S)\"" --argjson idx "$idx")
    for kv in "$@"; do
        local key="${kv%%=*}"
        local val="${kv#*=}"
        jq_filter+=" | (.runs[] | select(.index == \$idx) | .${key}) = ${val}"
    done
    tmp="$(mktemp)"
    jq "$jq_filter" "${jq_args[@]}" "$STATUS_FILE" > "$tmp" && mv "$tmp" "$STATUS_FILE"
}

# ---------------- run the sweep ----------------
idx=0
fail_count=0
for combo in "${COMBOS[@]}"; do
    algo="${combo%%:*}"
    env="${combo#*:}"
    warmup_steps="${ALGO_WARMUP_STEPS[$algo]}"
    for width in "${WIDTHS[@]}"; do
        overrides="$(override_file "$algo" "$env" "$width")"
        for seed in "${SEEDS[@]}"; do
            idx=$((idx + 1))

            log "[$idx/$TOTAL] Starting: algo=$algo env=$env seed=$seed width=${width}x${width}"
            update_run_status "$idx" \
                "status=\"running\"" \
                "started_utc=\"$(date -u +%Y-%m-%dT%H:%M:%S)\""

            run_output="$(mktemp)"
            cmd=(sudo "$PYTHON_BIN" run_experiment.py \
                --algo "$algo" --env "$env" --seed "$seed" \
                --algo-config-overrides "$overrides")
            if [[ -n "$warmup_steps" ]]; then
                cmd+=(--warmup-steps "$warmup_steps")
            fi
            "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE" | tee "$run_output"
            exit_code="${PIPESTATUS[0]}"

            run_dir="$(grep -o 'Run artifacts written to: .*' "$run_output" | sed 's/Run artifacts written to: //' | tail -1)"
            rm -f "$run_output"

            if [[ "$exit_code" -eq 0 ]]; then
                log "[$idx/$TOTAL] Finished OK: algo=$algo env=$env seed=$seed width=${width}x${width} -> ${run_dir:-unknown}"
                update_run_status "$idx" \
                    "status=\"done\"" \
                    "run_dir=\"${run_dir:-null}\"" \
                    "finished_utc=\"$(date -u +%Y-%m-%dT%H:%M:%S)\"" \
                    "exit_code=$exit_code"
            else
                fail_count=$((fail_count + 1))
                log "[$idx/$TOTAL] FAILED (exit $exit_code): algo=$algo env=$env seed=$seed width=${width}x${width} -- see $LOG_FILE"
                update_run_status "$idx" \
                    "status=\"failed\"" \
                    "finished_utc=\"$(date -u +%Y-%m-%dT%H:%M:%S)\"" \
                    "exit_code=$exit_code"
            fi
        done
    done
done

log "Sweep complete: $((TOTAL - fail_count))/$TOTAL runs succeeded, $fail_count failed."

# ---------------- automatic shutdown ----------------
# Requested so the desktop powers itself down unattended once the sweep is
# done (user will be away from the desk for the multi-hour duration of this
# sweep). Runs regardless of fail_count -- the point is not to leave the
# machine on unattended, not to gate it on every run succeeding. Uses the
# sudo credential primed/kept alive above, so no further password prompt.
# 1-minute delay (rather than "now") only as a small buffer in case someone
# is in fact still at the desk and wants to cancel with `sudo shutdown -c`.
log "Powering down in 1 minute (sudo shutdown -h +1) -- run 'sudo shutdown -c' now to cancel."
sudo shutdown -h +1 "rl-dissection width sweep finished ($((TOTAL - fail_count))/$TOTAL runs OK) -- powering down"

exit $(( fail_count > 0 ? 1 : 0 ))
