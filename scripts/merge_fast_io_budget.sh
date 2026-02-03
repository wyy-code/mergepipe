#!/usr/bin/env bash
# Merge (use cached ANALYZE); write logs to logs/ and parse into results/metrics.csv.
#
# Usage:
### ==================================================================
# bash scripts/merge_fast_io_budget.sh \
#     --db mergepipe.sqlite \
#     --base model_base \
#     --experts experts_1_path experts_2_path experts_3_path \
#     --out outputs_path \
#     --model-id merged_demo
# Notes:
# - This script parses required args itself (so it can validate early).
# - You can still pass any extra args supported by example_usage_explain_lineage.py.

set -euo pipefail
# set -x

# -------- Defaults (can override via env vars) --------
BACKEND="${BACKEND:-pt}"
DEVICE="${DEVICE:-$([[ "$BACKEND" == "pt" ]] && echo cpu || echo '')}"
BLOCK_SIZE="${BLOCK_SIZE:-640000}"
TOP_P="${TOP_P:-0.8}"
TIES_THR="${TIES_THR:-0.5}"
DARE_SCALE="${DARE_SCALE:-0.8}"
IO_BUDGET_MB="${IO_BUDGET_MB:-906}"
SCORING="${SCORING:-l2cos}"
EXPERTS_KIND="${EXPERTS_KIND:-weights}"
DISABLE_BUDGET="${DISABLE_BUDGET:-0}"

# Strategy switch (avg|ties|dare|auto)
STRATEGY="${STRATEGY:-ties}"

case "$STRATEGY" in
  avg)
    TOP_P=1.0
    TIES_THR=0
    DARE_SCALE=0
    ;;
  ties)
    TOP_P=0.35
    TIES_THR=0.6
    DARE_SCALE=0
    ;;
  dare)
    TOP_P=0.25
    TIES_THR=0
    DARE_SCALE="${DARE_SCALE:-0.8}"
    ;;
  auto)
    ;;
  *)
    echo "Unknown STRATEGY='$STRATEGY' (expect: avg|ties|dare|auto)" >&2
    exit 2
    ;;
esac

# -------- Parse required args (and keep passthrough args) --------
DB=""
BASE=""
OUT=""
MODEL_ID=""
EXPERTS=()
PASSTHRU=()

if [[ $# -eq 0 ]]; then
  echo "[ERR] No arguments provided." >&2
  echo "Example:" >&2
  echo "  bash $0 --db mergedb.sqlite --base /path/base --experts e1 e2 --out outputs --model-id demo" >&2
  exit 2
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --db)
      DB="${2:-}"; shift 2
      ;;
    --base)
      BASE="${2:-}"; shift 2
      ;;
    --out)
      OUT="${2:-}"; shift 2
      ;;
    --model-id)
      MODEL_ID="${2:-}"; shift 2
      ;;
    --experts)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        EXPERTS+=("$1"); shift
      done
      ;;
    *)
      PASSTHRU+=("$1"); shift
      ;;
  esac
done

# -------- Validate required args --------
if [[ -z "$DB" ]]; then
  echo "[ERR] Missing required arg: --db" >&2; exit 2
fi
if [[ -z "$BASE" ]]; then
  echo "[ERR] Missing required arg: --base" >&2; exit 2
fi
if [[ ${#EXPERTS[@]} -eq 0 ]]; then
  echo "[ERR] Missing required arg: --experts (need at least 1 expert path)" >&2; exit 2
fi
if [[ -z "$OUT" ]]; then
  echo "[ERR] Missing required arg: --out" >&2; exit 2
fi
if [[ -z "$MODEL_ID" ]]; then
  echo "[ERR] Missing required arg: --model-id" >&2; exit 2
fi

mkdir -p logs results
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/merge_${STAMP}.log"

echo "[P3] Merge with cached ANALYZE → log: $LOG"

# Build command array safely
CMD=(python example_usage_explain_lineage.py
  --db "$DB"
  --base "$BASE"
  --experts "${EXPERTS[@]}"
  --out "$OUT"
  --model-id "$MODEL_ID"
  --experts-kind "$EXPERTS_KIND"
  --backend "$BACKEND"
  --block-size "$BLOCK_SIZE"
  --top-p "$TOP_P"
  --ties-thr "$TIES_THR"
  --dare-scale "$DARE_SCALE"
  --io-budget-mb "$IO_BUDGET_MB"
  --scoring "$SCORING"
)

if [[ "$DISABLE_BUDGET" == "1" ]]; then
  CMD+=(--disable-budget)
fi
# Only add --device when non-empty (for pt backend by default)
if [[ -n "${DEVICE:-}" ]]; then
  CMD+=(--device "$DEVICE")
fi

# Append passthrough args last, so user-specified flags override defaults
if [[ ${#PASSTHRU[@]} -gt 0 ]]; then
  CMD+=("${PASSTHRU[@]}")
fi

"${CMD[@]}" 2>&1 | tee "$LOG"

OUTCSV="results/exec_logs_${STAMP}.csv"
python scripts/export_exec_logs_to_csv.py \
  --db "$DB" \
  --out "$OUTCSV" \
  --stage MERGE

cp -f "$OUTCSV" results/exec_logs_latest.csv
echo "[P3] Exported → $OUTCSV (and results/exec_logs_latest.csv)"
