#!/usr/bin/env bash
set -euo pipefail

COUNT="${COUNT:-100}"
LAYERS="${LAYERS:-all}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1}"
PATCH_SIZE="${PATCH_SIZE:-}"

for corner in topleft topright bottomleft bottomright; do
  case "$corner" in
    topleft)
      prompt="Is there anything in the top left corner?"
      ;;
    topright)
      prompt="Is there anything in the top right corner?"
      ;;
    bottomleft)
      prompt="Is there anything in the bottom left corner?"
      ;;
    bottomright)
      prompt="Is there anything in the bottom right corner?"
      ;;
  esac

  dataset_dir="data/dataset_${corner}"
  stats_dir="head_diff_stats_${corner}"

  python head_attention_stats_qwen2.py \
    --dataset "${dataset_dir}" \
    --prompt "${prompt}" \
    --layers "${LAYERS}" \
    --max-pairs "${COUNT}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --out-dir "${stats_dir}" \
    --save-diffs \
    ${PATCH_SIZE:+--corner ${corner} --patch-size ${PATCH_SIZE}}
done

python analyze_count_qwen2.py \
  head_diff_stats_topleft \
  head_diff_stats_topright \
  head_diff_stats_bottomleft \
  head_diff_stats_bottomright \
  --out-dir count_shared_unique \
  --reduce-level max \
  --use-scale count \
  --diff-brightness raw
