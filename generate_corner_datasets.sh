#!/usr/bin/env bash
set -euo pipefail

COUNT="${COUNT:-100}"

mkdir -p data

for corner in topleft topright bottomleft bottomright; do
  case "$corner" in
    topleft)
      relation="upper_left_of"
      ;;
    topright)
      relation="upper_right_of"
      ;;
    bottomleft)
      relation="lower_left_of"
      ;;
    bottomright)
      relation="lower_right_of"
      ;;
  esac

  dataset_dir="data/dataset_${corner}"

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
done
