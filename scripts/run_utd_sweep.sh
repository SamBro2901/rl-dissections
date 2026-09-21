#!/usr/bin/env bash
#
# UTD (update-to-data ratio) sweep: updates_per_env_step in {2, 4}, across
# seeds {331, 958, 14577, 43611, 85062}, for:
#   - mbpo on HalfCheetah-v5 and Ant-v5
# -- 20 runs total.
#
# Shuts the machine down when the sweep is done (see bottom of script) --
# this is meant to be launched unattended (e.g. overnight).
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
#   screen -S utd_sweep
#   ./run_utd_sweep.sh
#   # Ctrl-A D to detach; reattach later with: screen -r utd_sweep
#
set -uo pipefail
cd "$(dirname "$0")"

# algo:env combos to sweep.
COMBOS=("mbpo:HalfCheetah-v5" "mbpo:Ant-v5")
SEEDS=(331 958 14577 43611 85062)
UTDS=(2 4)
PYTHON_BIN="rl-exp/bin/python"

# Per-algo --warmup-steps. MBPO uses the CLI default for both envs, so no
# override is needed here.
declare -A ALGO_WARMUP_STEPS=(
    ["mbpo"]=""
)

# Per-(algo,env) --algo-config-overrides file prefix, combined below with
# "_utd${utd}.json". MBPO needs Ant-v5's own rollout-schedule overrides
# (mbpo_ant.json) layered under the UTD override, since run_experiment.py
# only accepts one overrides file per run -- hence the separate
# mbpo_ant_utd{2,4}.json files (vs. plain mbpo_utd{2,4}.json for
# HalfCheetah-v5, which uses MBPOConfig defaults otherwise).
declare -A OVERRIDE_PREFIX=(
    ["mbpo:HalfCheetah-v5"]="mbpo"
    ["mbpo:Ant-v5"]="mbpo_ant"
)

STATUS_FILE="results/_utd_sweep_status.json"
LOG_FILE="results/_utd_sweep.log"
mkdir -p "$(dirname "$STATUS_FILE")"

TOTAL=$(( ${#COMBOS[@]} * ${#UTDS[@]} * ${#SEEDS[@]} ))

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
python3 - "$STATUS_FILE" "$TOTAL" "${COMBOS[*]}" "${UTDS[*]}" "${SEEDS[*]}" <<'PYEOF'
import json, sys, datetime

status_file, total, combos_str, utds_str, seeds_str = sys.argv[1:6]
combos = [tuple(c.split(":")) for c in combos_str.split()]
utds = [int(u) for u in utds_str.split()]
seeds = [int(s) for s in seeds_str.split()]

runs = []
idx = 1
for algo, env in combos:
    for utd in utds:
        for seed in seeds:
            runs.append({
                "index": idx,
                "algo": algo,
                "env": env,
                "seed": seed,
                "updates_per_env_step": utd,
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
    override_prefix="${OVERRIDE_PREFIX[$combo]}"
    for utd in "${UTDS[@]}"; do
        overrides="configs/overrides/${override_prefix}_utd${utd}.json"
        for seed in "${SEEDS[@]}"; do
            idx=$((idx + 1))

            log "[$idx/$TOTAL] Starting: algo=$algo env=$env seed=$seed utd=$utd"
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
                log "[$idx/$TOTAL] Finished OK: algo=$algo env=$env seed=$seed utd=$utd -> ${run_dir:-unknown}"
                update_run_status "$idx" \
                    "status=\"done\"" \
                    "run_dir=\"${run_dir:-null}\"" \
                    "finished_utc=\"$(date -u +%Y-%m-%dT%H:%M:%S)\"" \
                    "exit_code=$exit_code"
            else
                fail_count=$((fail_count + 1))
                log "[$idx/$TOTAL] FAILED (exit $exit_code): algo=$algo env=$env seed=$seed utd=$utd -- see $LOG_FILE"
                update_run_status "$idx" \
                    "status=\"failed\"" \
                    "finished_utc=\"$(date -u +%Y-%m-%dT%H:%M:%S)\"" \
                    "exit_code=$exit_code"
            fi
        done
    done
done

log "Sweep complete: $((TOTAL - fail_count))/$TOTAL runs succeeded, $fail_count failed."

# ---------------- shutdown ----------------
# Always powers the machine off once the sweep is done (success or not) --
# this is meant to run unattended overnight. 60s grace period so you can
# still Ctrl-C it if you're at the console when it finishes.
log "Sweep finished. Shutting down in 60s -- press Ctrl-C now to cancel."
sleep 60
sudo shutdown -h now
exit $(( fail_count > 0 ? 1 : 0 ))
