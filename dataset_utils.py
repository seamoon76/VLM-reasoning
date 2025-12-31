from pathlib import Path
from typing import List, Dict, Any

import json
from PIL import Image


def load_dataset(dataset_dir: str | Path) -> List[Dict[str, Any]]:
    """
    Load paired images and captions from a generated dataset folder.

    Returns a list of dicts with keys:
    - id, caption, relation, subject_shape, object_shape
    - bounding_boxes, note
    - full_image (PIL.Image), control_image (PIL.Image)
    """
    dataset_dir = Path(dataset_dir)
    meta_path = dataset_dir / "metadata.jsonl"
    entries: List[Dict[str, Any]] = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            full_name = Path(record["full_image"]).name
            ctrl_name = Path(record["control_image"]).name
            full_img = Image.open(dataset_dir / "images" / full_name).convert("RGB")
            ctrl_img = Image.open(dataset_dir / "images" / ctrl_name).convert("RGB")
            record["full_image"] = full_img
            record["control_image"] = ctrl_img
            entries.append(record)
    return entries
