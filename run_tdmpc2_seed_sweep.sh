#!/usr/bin/env bash
#
# TD-MPC2 seed sweep: tdmpc2 across envs {HalfCheetah-v5, Ant-v5}, seeds
# {331, 958, 14577, 43611, 85062} -- 10 runs total.
#
# Env-specific overrides: TDMPC2Config's defaults (configs/config.py) are the
# paper's own values, used unmodified for HalfCheetah-v5. Ant-v5 needs
# configs/overrides/tdmpc2_ant.json (episodic=True), since Gymnasium's
# default terminate_when_unhealthy=True means Ant can end an episode early
# and the MPPI planner should account for that (see TDMPC2Config's
# docstring). No --warmup-steps override is needed for either env: the
# paper's seed-steps heuristic (5 * episode_length = 5000 for these
# 1000-step envs) already matches the CLI's default warmup_steps.
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
#   screen -S tdmpc2_seed_sweep
#   ./run_tdmpc2_seed_sweep.sh
#   # Ctrl-A D to detach; reattach later with: screen -r tdmpc2_seed_sweep
#
# Check progress at any time (from another terminal / after reattaching) by
# inspecting results/tdmpc2/_seed_sweep_status.json (or tailing
# results/tdmpc2/_seed_sweep.log).
#
set -uo pipefail
cd "$(dirname "$0")"

ALGO="tdmpc2"
ENVS=("HalfCheetah-v5" "Ant-v5")
SEEDS=(331 958 14577 43611 85062)
PYTHON_BIN="rl-exp/bin/python"

# Per-env --algo-config-overrides path; empty string means "no override,
# use TDMPC2Config defaults".
declare -A ENV_OVERRIDES=(
    ["HalfCheetah-v5"]=""
    ["Ant-v5"]="configs/overrides/tdmpc2_ant.json"
)

STATUS_FILE="results/tdmpc2/_seed_sweep_status.json"
LOG_FILE="results/tdmpc2/_seed_sweep.log"
mkdir -p "$(dirname "$STATUS_FILE")"

TOTAL=$(( ${#SEEDS[@]} * ${#ENVS[@]} ))

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
python3 - "$STATUS_FILE" "$TOTAL" "$ALGO" "${SEEDS[*]}" "${ENVS[*]}" <<'PYEOF'
import json, sys, datetime

status_file, total, algo, seeds_str, envs_str = sys.argv[1:6]
seeds = [int(s) for s in seeds_str.split()]
envs = envs_str.split()

runs = []
idx = 1
for env in envs:
    for seed in seeds:
        runs.append({
            "index": idx,
            "algo": algo,
            "env": env,
            "seed": seed,
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
for env in "${ENVS[@]}"; do
    overrides="${ENV_OVERRIDES[$env]}"
    for seed in "${SEEDS[@]}"; do
        idx=$((idx + 1))

        log "[$idx/$TOTAL] Starting: algo=$ALGO env=$env seed=$seed overrides=${overrides:-none}"
        update_run_status "$idx" \
            "status=\"running\"" \
            "started_utc=\"$(date -u +%Y-%m-%dT%H:%M:%S)\""

        run_output="$(mktemp)"
        cmd=(sudo "$PYTHON_BIN" run_experiment.py --algo "$ALGO" --env "$env" --seed "$seed")
        if [[ -n "$overrides" ]]; then
            cmd+=(--algo-config-overrides "$overrides")
        fi
        "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE" | tee "$run_output"
        exit_code="${PIPESTATUS[0]}"

        run_dir="$(grep -o 'Run artifacts written to: .*' "$run_output" | sed 's/Run artifacts written to: //' | tail -1)"
        rm -f "$run_output"

        if [[ "$exit_code" -eq 0 ]]; then
            log "[$idx/$TOTAL] Finished OK: env=$env seed=$seed -> ${run_dir:-unknown}"
            update_run_status "$idx" \
                "status=\"done\"" \
                "run_dir=\"${run_dir:-null}\"" \
                "finished_utc=\"$(date -u +%Y-%m-%dT%H:%M:%S)\"" \
                "exit_code=$exit_code"
        else
            fail_count=$((fail_count + 1))
            log "[$idx/$TOTAL] FAILED (exit $exit_code): env=$env seed=$seed -- see $LOG_FILE"
            update_run_status "$idx" \
                "status=\"failed\"" \
                "finished_utc=\"$(date -u +%Y-%m-%dT%H:%M:%S)\"" \
                "exit_code=$exit_code"
        fi
    done
done

log "Sweep complete: $((TOTAL - fail_count))/$TOTAL runs succeeded, $fail_count failed."
exit $(( fail_count > 0 ? 1 : 0 ))
