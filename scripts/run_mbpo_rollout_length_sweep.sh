#!/usr/bin/env bash
#
# MBPO model-rollout-length sweep: rollout_max_length in {1, 15}, Ant-v5
# only, canonical 5 seeds {331, 958, 14577, 43611, 85062} -- MBPO only.
# -- 2 rollout lengths x 5 seeds = 10 runs total.
#
# rollout_min_length is held at 1 in both cases (so length=1 is a fixed,
# unscheduled 1-step rollout -- the same regime as MBPOConfig's own
# HalfCheetah-v5 defaults -- while length=15 linearly schedules 1->15 between
# epochs 20-100, same schedule window as the established Ant-v5 override,
# configs/overrides/mbpo_ant.json, just capped at 15 instead of 25). See
# configs/overrides/mbpo_ant_rollout{1,15}.json.
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
#   screen -S mbpo_rollout_length_sweep
#   ./run_mbpo_rollout_length_sweep.sh
#   # Ctrl-A D to detach; reattach later with: screen -r mbpo_rollout_length_sweep
#
# SHUTDOWN: once the sweep finishes (all 10 runs attempted, pass or fail),
# this script powers the machine off automatically -- see the bottom of the
# file. Don't launch this unless you actually want the desktop to shut down
# unattended when it's done.
#
set -uo pipefail
cd "$(dirname "$0")"

ENV="Ant-v5"
SEEDS=(331 958 14577 43611 85062)
ROLLOUT_LENGTHS=(1 15)
PYTHON_BIN="rl-exp/bin/python"

override_file() {
    local length="$1"
    echo "configs/overrides/mbpo_ant_rollout${length}.json"
}

STATUS_FILE="results/_mbpo_rollout_length_sweep_status.json"
LOG_FILE="results/_mbpo_rollout_length_sweep.log"
mkdir -p "$(dirname "$STATUS_FILE")"

TOTAL=$(( ${#ROLLOUT_LENGTHS[@]} * ${#SEEDS[@]} ))

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
python3 - "$STATUS_FILE" "$TOTAL" "$ENV" "${SEEDS[*]}" "${ROLLOUT_LENGTHS[*]}" <<'PYEOF'
import json, sys, datetime

status_file, total, env, seeds_str, lengths_str = sys.argv[1:6]
seeds = [int(s) for s in seeds_str.split()]
lengths = [int(l) for l in lengths_str.split()]

runs = []
idx = 1
for length in lengths:
    for seed in seeds:
        runs.append({
            "index": idx,
            "algo": "mbpo",
            "env": env,
            "seed": seed,
            "rollout_max_length": length,
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
for length in "${ROLLOUT_LENGTHS[@]}"; do
    overrides="$(override_file "$length")"
    for seed in "${SEEDS[@]}"; do
        idx=$((idx + 1))

        log "[$idx/$TOTAL] Starting: algo=mbpo env=$ENV seed=$seed rollout_max_length=$length"
        update_run_status "$idx" \
            "status=\"running\"" \
            "started_utc=\"$(date -u +%Y-%m-%dT%H:%M:%S)\""

        run_output="$(mktemp)"
        sudo "$PYTHON_BIN" run_experiment.py \
            --algo mbpo --env "$ENV" --seed "$seed" \
            --algo-config-overrides "$overrides" \
            2>&1 | tee -a "$LOG_FILE" | tee "$run_output"
        exit_code="${PIPESTATUS[0]}"

        run_dir="$(grep -o 'Run artifacts written to: .*' "$run_output" | sed 's/Run artifacts written to: //' | tail -1)"
        rm -f "$run_output"

        if [[ "$exit_code" -eq 0 ]]; then
            log "[$idx/$TOTAL] Finished OK: algo=mbpo env=$ENV seed=$seed rollout_max_length=$length -> ${run_dir:-unknown}"
            update_run_status "$idx" \
                "status=\"done\"" \
                "run_dir=\"${run_dir:-null}\"" \
                "finished_utc=\"$(date -u +%Y-%m-%dT%H:%M:%S)\"" \
                "exit_code=$exit_code"
        else
            fail_count=$((fail_count + 1))
            log "[$idx/$TOTAL] FAILED (exit $exit_code): algo=mbpo env=$ENV seed=$seed rollout_max_length=$length -- see $LOG_FILE"
            update_run_status "$idx" \
                "status=\"failed\"" \
                "finished_utc=\"$(date -u +%Y-%m-%dT%H:%M:%S)\"" \
                "exit_code=$exit_code"
        fi
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
sudo shutdown -h +1 "rl-dissection MBPO rollout-length sweep finished ($((TOTAL - fail_count))/$TOTAL runs OK) -- powering down"

exit $(( fail_count > 0 ? 1 : 0 ))
