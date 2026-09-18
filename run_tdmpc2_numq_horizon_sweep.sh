#!/usr/bin/env bash
#
# TD-MPC2 num_q / horizon sweep: two independent single-parameter ablations
# (not a cross-product grid), each across {HalfCheetah-v5, Ant-v5}, canonical
# 5 seeds {331, 958, 14577, 43611, 85062} -- tdmpc2 only.
#
#   - num_q  in {3, 7}   (TDMPC2Config's default, 5, is already covered by
#                         the existing canonical TD-MPC2 seed sweep --
#                         see run_tdmpc2_seed_sweep.sh -- so it's skipped here)
#   - horizon in {1, 5}  (default, 3, likewise already covered and skipped)
#
# -- (2 + 2) values x 2 envs x 5 seeds = 40 runs total.
#
# Ant-v5 needs configs/overrides/tdmpc2_ant.json's episodic=true layered
# under each num_q/horizon override (run_experiment.py only accepts one
# --algo-config-overrides file per run), hence the separate
# configs/overrides/tdmpc2_ant_{numq,horizon}{value}.json files -- vs plain
# configs/overrides/tdmpc2_{numq,horizon}{value}.json for HalfCheetah-v5,
# which otherwise uses TDMPC2Config defaults. Same convention as
# run_mbpo_rollout_length_sweep.sh / run_utd_sweep.sh's mbpo_ant_utd*.json.
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
#   screen -S tdmpc2_numq_horizon_sweep
#   ./run_tdmpc2_numq_horizon_sweep.sh
#   # Ctrl-A D to detach; reattach later with: screen -r tdmpc2_numq_horizon_sweep
#
# SHUTDOWN: once the sweep finishes (all 40 runs attempted, pass or fail),
# this script powers the machine off automatically -- see the bottom of the
# file. Don't launch this unless you actually want the desktop to shut down
# unattended when it's done.
#
set -uo pipefail
cd "$(dirname "$0")"

ENVS=("HalfCheetah-v5" "Ant-v5")
SEEDS=(331 958 14577 43611 85062)
PYTHON_BIN="rl-exp/bin/python"

# param:value pairs to sweep, in order. Defaults (num_q=5, horizon=3) are
# intentionally omitted -- see header comment.
PARAM_VALUES=("num_q:3" "num_q:7" "horizon:1" "horizon:5")

override_file() {
    local param="$1" value="$2" env="$3"
    if [[ "$env" == "Ant-v5" ]]; then
        echo "configs/overrides/tdmpc2_ant_${param}${value}.json"
    else
        echo "configs/overrides/tdmpc2_${param}${value}.json"
    fi
}

STATUS_FILE="results/tdmpc2/_numq_horizon_sweep_status.json"
LOG_FILE="results/tdmpc2/_numq_horizon_sweep.log"
mkdir -p "$(dirname "$STATUS_FILE")"

TOTAL=$(( ${#PARAM_VALUES[@]} * ${#ENVS[@]} * ${#SEEDS[@]} ))

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
python3 - "$STATUS_FILE" "$TOTAL" "${PARAM_VALUES[*]}" "${ENVS[*]}" "${SEEDS[*]}" <<'PYEOF'
import json, sys, datetime

status_file, total, param_values_str, envs_str, seeds_str = sys.argv[1:6]
param_values = [tuple(pv.split(":")) for pv in param_values_str.split()]
envs = envs_str.split()
seeds = [int(s) for s in seeds_str.split()]

runs = []
idx = 1
for param, value in param_values:
    for env in envs:
        for seed in seeds:
            runs.append({
                "index": idx,
                "algo": "tdmpc2",
                "env": env,
                "seed": seed,
                "param": param,
                "value": int(value),
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
for pv in "${PARAM_VALUES[@]}"; do
    param="${pv%%:*}"
    value="${pv#*:}"
    for env in "${ENVS[@]}"; do
        overrides="$(override_file "$param" "$value" "$env")"
        for seed in "${SEEDS[@]}"; do
            idx=$((idx + 1))

            log "[$idx/$TOTAL] Starting: algo=tdmpc2 env=$env seed=$seed $param=$value"
            update_run_status "$idx" \
                "status=\"running\"" \
                "started_utc=\"$(date -u +%Y-%m-%dT%H:%M:%S)\""

            run_output="$(mktemp)"
            sudo "$PYTHON_BIN" run_experiment.py \
                --algo tdmpc2 --env "$env" --seed "$seed" \
                --algo-config-overrides "$overrides" \
                2>&1 | tee -a "$LOG_FILE" | tee "$run_output"
            exit_code="${PIPESTATUS[0]}"

            run_dir="$(grep -o 'Run artifacts written to: .*' "$run_output" | sed 's/Run artifacts written to: //' | tail -1)"
            rm -f "$run_output"

            if [[ "$exit_code" -eq 0 ]]; then
                log "[$idx/$TOTAL] Finished OK: algo=tdmpc2 env=$env seed=$seed $param=$value -> ${run_dir:-unknown}"
                update_run_status "$idx" \
                    "status=\"done\"" \
                    "run_dir=\"${run_dir:-null}\"" \
                    "finished_utc=\"$(date -u +%Y-%m-%dT%H:%M:%S)\"" \
                    "exit_code=$exit_code"
            else
                fail_count=$((fail_count + 1))
                log "[$idx/$TOTAL] FAILED (exit $exit_code): algo=tdmpc2 env=$env seed=$seed $param=$value -- see $LOG_FILE"
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
# done. Runs regardless of fail_count -- the point is not to leave the
# machine on unattended, not to gate it on every run succeeding. Uses the
# sudo credential primed/kept alive above, so no further password prompt.
# 1-minute delay (rather than "now") only as a small buffer in case someone
# is in fact still at the desk and wants to cancel with `sudo shutdown -c`.
log "Powering down in 1 minute (sudo shutdown -h +1) -- run 'sudo shutdown -c' now to cancel."
sudo shutdown -h +1 "rl-dissection TD-MPC2 num_q/horizon sweep finished ($((TOTAL - fail_count))/$TOTAL runs OK) -- powering down"

exit $(( fail_count > 0 ? 1 : 0 ))
