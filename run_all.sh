#!/usr/bin/env bash
# run_all.sh -- run the analysis pipeline.
#
# Usage  : ./run_all.sh --from-eeg | --from-sequences  [--sensitivity]
#
# Templates: one set, the group templates of the October 2025 clustering run
# (K = 4..9, deposited; K = 5 = microstates_K5_templates_reference.npz). They
# are inputs; 02_microstates/fit_K5.py and fit_all_K.py document the
# clustering procedure and are not run here.
#
# --from-eeg        (needs MICROSTATE_DATA_DIR, MICROSTATE_REFERENCE_TEMPLATES,
#                   MICROSTATE_TEMPLATES_DIR, MICROSTATE_OUT_DIR)
#                   verify_inputs, reshape, backfit (reference templates),
#                   gev_by_k (GEV of the deposited K = 4..9 templates ->
#                   all_K_gev.csv), then the common steps below.
# --from-sequences  (needs MICROSTATE_SEQUENCES_DIR, MICROSTATE_TEMPLATES_DIR,
#                   MICROSTATE_OUT_DIR; MICROSTATE_REFERENCE_TEMPLATES only for
#                   canonical_similarity)
#                   verify_inputs, import_sequences (the deposited label
#                   sequences replace reshape and backfit), then the common
#                   steps below.
# Common steps      k_template_correlations (|r| within each K = 4..9 solution,
#                   T4 values of K = 6; no EEG needed), metrics, trial_table,
#                   consensus, motif_analysis (exact
#                   windows), split_half, Needleman-Wunsch, gee_sex_condition,
#                   canonical_similarity (only if MICROSTATE_REFERENCE_MAPS_DIR
#                   and MICROSTATE_REFERENCE_TEMPLATES are set), and the last
#                   step assemble_supplementary (Supplementary Tables S1-S5).
# --sensitivity     also runs full_prepost and motif_analysis with proportional
#                   windows (files with the suffix _proportional). Exact
#                   windows are the primary analysis.
#
# Each step writes $MICROSTATE_OUT_DIR/logs/<step>.log with its command (path
# arguments by file name only), start time, end time and duration. The script stops at the first failing step.
# Python interpreter: $PYTHON if set, otherwise "python" (activate the
# environment first, see README.md). Python 3.11.14.

set -euo pipefail
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"

usage() {
    echo "usage: $0 --from-eeg | --from-sequences  [--sensitivity]" >&2
    exit 2
}

need() {
    for var in "$@"; do
        if [ -z "${!var:-}" ]; then
            echo "error: environment variable $var is not set" >&2
            exit 2
        fi
    done
}

MODE=""
SENSITIVITY=0
for arg in "$@"; do
    case "$arg" in
        --from-eeg|--from-sequences) [ -z "$MODE" ] || usage; MODE="$arg" ;;
        --sensitivity) SENSITIVITY=1 ;;
        *) usage ;;
    esac
done
[ -n "$MODE" ] || usage

case "$MODE" in
    --from-eeg)       need MICROSTATE_DATA_DIR MICROSTATE_REFERENCE_TEMPLATES MICROSTATE_TEMPLATES_DIR MICROSTATE_OUT_DIR ;;
    --from-sequences) need MICROSTATE_SEQUENCES_DIR MICROSTATE_TEMPLATES_DIR MICROSTATE_OUT_DIR ;;
esac

LOG_DIR="$MICROSTATE_OUT_DIR/logs"
mkdir -p "$LOG_DIR"

# step <name> <script relative to this folder> [arguments...]
step() {
    local name="$1" script="$2"
    shift 2
    local log="$LOG_DIR/$name.log"
    local t0
    t0=$SECONDS
    {
        echo "step: $name"
        # path arguments are logged by file name only (no absolute paths)
        printf 'command: %s' "$script"
        for arg in "$@"; do
            if [ -e "$arg" ]; then printf ' %s' "$(basename "$arg")"; else printf ' %s' "$arg"; fi
        done
        echo
        echo "start: $(date '+%Y-%m-%d %H:%M:%S')"
    } > "$log"
    echo "[$name] start"
    local status=0
    "$PY" "$HERE/$script" "$@" >> "$log" 2>&1 || status=$?
    {
        echo "end: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "duration_seconds: $((SECONDS - t0))"
        echo "exit_status: $status"
    } >> "$log"
    if [ "$status" -ne 0 ]; then
        echo "[$name] FAILED (exit $status); see $log" >&2
        exit "$status"
    fi
    echo "[$name] done in $((SECONDS - t0)) s"
}

skip() {
    local name="$1" reason="$2"
    {
        echo "step: $name"
        echo "skipped: $reason"
    } > "$LOG_DIR/$name.log"
    echo "[$name] skipped: $reason"
}

K_TEMPLATES=()
for K in 4 5 6 7 8 9; do
    K_TEMPLATES+=("$MICROSTATE_TEMPLATES_DIR/microstates_K${K}_templates.npz")
done

if [ "$MODE" = "--from-eeg" ]; then
    step verify_inputs          tools/verify_inputs.py \
        "$MICROSTATE_REFERENCE_TEMPLATES" "${K_TEMPLATES[@]}"
    step reshape                01_reshape/reshape.py
    step backfit_reference      03_sequences/backfit.py
    step gev_by_k               02_microstates/gev_by_k.py \
        --templates "${K_TEMPLATES[@]}" --out all_K_gev.csv
else
    step verify_inputs          tools/verify_inputs.py \
        "$MICROSTATE_SEQUENCES_DIR/microstate_sequences_K5_reference.csv" \
        "${K_TEMPLATES[@]}"
    step import_sequences       03_sequences/import_sequences.py
fi

step k_template_correlations    02_microstates/k_template_correlations.py
step metrics                    03_sequences/metrics.py
step trial_table                03_sequences/trial_table.py
step consensus                  05_alignment/consensus.py
step motif_analysis_exact       04_statistics/motif_analysis.py --windows exact
step split_half                 04_statistics/split_half.py
step within_condition_nw        05_alignment/within_condition_nw_analysis.py --source eeg
step gee_sex_condition          04_statistics/gee_sex_condition.py --source eeg

if [ -n "${MICROSTATE_REFERENCE_MAPS_DIR:-}" ] && [ -n "${MICROSTATE_REFERENCE_TEMPLATES:-}" ]; then
    step canonical_similarity   02_microstates/canonical_similarity.py
else
    skip canonical_similarity "MICROSTATE_REFERENCE_MAPS_DIR or MICROSTATE_REFERENCE_TEMPLATES not set"
fi

if [ "$SENSITIVITY" -eq 1 ]; then
    step full_prepost                   03_sequences/full_prepost.py
    step motif_analysis_proportional    04_statistics/motif_analysis.py --windows proportional --source eeg
fi

step assemble_supplementary     06_tables/assemble_supplementary.py
