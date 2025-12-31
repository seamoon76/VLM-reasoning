#!/usr/bin/env bash
set -euo pipefail

COUNT="${COUNT:-100}"
LAYERS="${LAYERS:-all}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1}"
PATCH_SIZE="${PATCH_SIZE:-}"

for corner in topleft topright bottomleft bottomright; do
  case "$corner" in
    topleft)
      relation="upper_left_of"
      prompt="Is there anything in the top left corner?"
      ;;
    topright)
      relation="upper_right_of"
      prompt="Is there anything in the top right corner?"
      ;;
    bottomleft)
      relation="lower_left_of"
      prompt="Is there anything in the bottom left corner?"
      ;;
    bottomright)
      relation="lower_right_of"
      prompt="Is there anything in the bottom right corner?"
      ;;
  esac

  dataset_dir="dataset_${corner}"
  stats_dir="head_diff_stats_${corner}"

  python - <<PY
from pathlib import Path
import generate_dataset as gd

gd.RELATIONS = ("${relation}",)
gd.generate_dataset(
    out_dir=Path("${dataset_dir}"),
    count=${COUNT},
    img_size=64,
    min_shape=12,
    max_shape=22,
    min_gap=6,
    seed=7,
)
PY

  python head_attention_stats.py \
    --dataset "${dataset_dir}" \
    --prompt "${prompt}" \
    --layers "${LAYERS}" \
    --max-pairs "${COUNT}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --out-dir "${stats_dir}" \
    --save-diffs \
    ${PATCH_SIZE:+--corner ${corner} --patch-size ${PATCH_SIZE}}
done

python analyze_diff_abs.py head_diff_stats_topleft head_diff_stats_topright head_diff_stats_bottomleft head_diff_stats_bottomright --all-layers