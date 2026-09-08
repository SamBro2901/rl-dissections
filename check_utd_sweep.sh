#!/usr/bin/env bash
#
# Progress checker for run_utd_sweep.sh. Safe to run any time, from any
# terminal (including a fresh SSH session) -- it only reads status/log
# files, it doesn't touch the running sweep.
#
set -uo pipefail
cd "$(dirname "$0")"

STATUS_FILE="results/sac/HalfCheetah-v5/_utd_sweep_status.json"
LOG_FILE="results/sac/HalfCheetah-v5/_utd_sweep.log"

if [[ ! -f "$STATUS_FILE" ]]; then
    echo "No sweep status file found at $STATUS_FILE -- has run_utd_sweep.sh been started?"
    exit 1
fi

bar() {
    # args: done total width
    local done="$1" total="$2" width="${3:-30}"
    local filled=0
    if [[ "$total" -gt 0 ]]; then
        filled=$(( done * width / total ))
    fi
    local empty=$((width - filled))
    printf '['
    [[ "$filled" -gt 0 ]] && printf '%0.s#' $(seq 1 "$filled")
    [[ "$empty" -gt 0 ]] && printf '%0.s.' $(seq 1 "$empty")
    printf '] %d/%d' "$done" "$total"
}

total="$(jq -r '.total_runs' "$STATUS_FILE")"
done_count="$(jq -r '[.runs[] | select(.status == "done")] | length' "$STATUS_FILE")"
failed_count="$(jq -r '[.runs[] | select(.status == "failed")] | length' "$STATUS_FILE")"
started="$(jq -r '.started_utc' "$STATUS_FILE")"
updated="$(jq -r '.updated_utc' "$STATUS_FILE")"

echo "UTD sweep progress (started $started UTC, last update $updated UTC)"
echo "Overall: $(bar "$done_count" "$total")  ($failed_count failed)"
echo

if [[ "$failed_count" -gt 0 ]]; then
    echo "Failed runs:"
    jq -r --argjson total "$total" '.runs[] | select(.status == "failed") | "  [\(.index)/\($total)] seed=\(.seed) utd=\(.updates_per_env_step) exit_code=\(.exit_code)"' "$STATUS_FILE"
    echo
fi

current="$(jq -c '.runs[] | select(.status == "running")' "$STATUS_FILE" | head -1)"
if [[ -z "$current" ]]; then
    if [[ "$done_count" -eq "$total" ]]; then
        echo "Sweep finished: $done_count/$total runs succeeded, $failed_count failed."
    else
        echo "No run currently marked 'running' -- sweep may be between runs, not started yet, or stalled."
        echo "Check $LOG_FILE (tail below) for what's happening:"
        tail -n 15 "$LOG_FILE" 2>/dev/null
    fi
    exit 0
fi

cur_idx="$(echo "$current" | jq -r '.index')"
cur_seed="$(echo "$current" | jq -r '.seed')"
cur_utd="$(echo "$current" | jq -r '.updates_per_env_step')"
cur_started="$(echo "$current" | jq -r '.started_utc')"

echo "Current run: [$cur_idx/$total] seed=$cur_seed utd=$cur_utd (started $cur_started UTC)"

run_dir="$(ls -td "results/${ALGO:-sac}/HalfCheetah-v5/seed_${cur_seed}"/*/ 2>/dev/null | head -1)"
if [[ -n "$run_dir" && -f "${run_dir}run.log" ]]; then
    epoch_line="$(grep -o 'Epoch [0-9]*/[0-9]* done.*' "${run_dir}run.log" | tail -1)"
    if [[ -n "$epoch_line" ]]; then
        cur_epoch="$(echo "$epoch_line" | grep -o '^Epoch [0-9]*' | grep -o '[0-9]*')"
        max_epoch="$(echo "$epoch_line" | grep -o '/[0-9]*' | head -1 | tr -d '/')"
        echo "  Training: $(bar "$cur_epoch" "$max_epoch" 20)  ($epoch_line)"
    else
        phase_line="$(grep -E 'Settling|Recording idle|Starting warmup|Starting measured training|Waiting for thermal' "${run_dir}run.log" | tail -1)"
        echo "  Phase: ${phase_line:-starting up...}"
    fi
    echo "  Run dir: $run_dir"
else
    echo "  (run directory/log not found yet -- probably still in the thermal-gate/settle phase)"
fi
