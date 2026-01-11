import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, Optional
import random

from PIL import Image, ImageDraw


SHAPES = ("triangle", "square", "circle", "pentagon", "star")
RELATIONS = (
    # "left_of",
    # "right_of",
    # "above",
    # "below",
    # "upper_left_of",
    # "upper_right_of",
    # "lower_left_of",
    "lower_right_of",
)

# Palette sampled per-pair so colors vary across samples.
COLOR_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (52, 101, 164),   # blue-ish
    (192, 57, 43),    # red
    (39, 174, 96),    # green
    (142, 68, 173),   # purple
    (243, 156, 18),   # orange
    (41, 128, 185),   # light blue
    (22, 160, 133),   # teal
    (127, 140, 141),  # gray
)


@dataclass
class SampleMetadata:
    id: str
    caption: str
    relation: str
    subject_shape: str
    object_shape: str
    full_image: str
    control_image: str
    bounding_boxes: Dict[str, Tuple[int, int, int, int]]
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paired synthetic images of simple shapes with shared captions."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("dataset"), help="Output directory.")
    parser.add_argument("--count", type=int, default=20, help="Number of pairs to generate.")
    parser.add_argument("--image-size", type=int, default=64, help="Square image size in pixels.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument("--min-shape", type=int, default=12, help="Minimum shape side/diameter.")
    parser.add_argument("--max-shape", type=int, default=22, help="Maximum shape side/diameter.")
    parser.add_argument("--min-gap", type=int, default=6, help="Minimum pixel gap between shapes.")
    return parser.parse_args()


def draw_shape(draw: ImageDraw.ImageDraw, shape: str, bbox: Tuple[int, int, int, int], color: Tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = bbox
    if shape == "square":
        draw.rectangle([x1, y1, x2, y2], fill=color)
    elif shape == "circle":
        draw.ellipse([x1, y1, x2, y2], fill=color)
    elif shape == "triangle":
        mid_x = (x1 + x2) // 2
        draw.polygon([(mid_x, y1), (x2, y2), (x1, y2)], fill=color)
    elif shape == "pentagon":
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        # Five points around the center.
        points = []
        for i in range(5):
            angle = (-90 + i * 72) * math.pi / 180.0
            px = cx + (w / 2) * 0.95 * math.cos(angle)
            py = cy + (h / 2) * 0.95 * math.sin(angle)
            points.append((px, py))
        draw.polygon(points, fill=color)
    elif shape == "star":
        w = x2 - x1
        h = y2 - y1
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        points = []
        for i in range(10):
            angle = (-90 + i * 36) * math.pi / 180.0
            radius = 0.48 if i % 2 == 0 else 0.22
            px = cx + (w / 2) * radius * math.cos(angle)
            py = cy + (h / 2) * radius * math.sin(angle)
            points.append((px, py))
        draw.polygon(points, fill=color)
    else:
        raise ValueError(f"Unknown shape: {shape}")


def relation_caption(subject: str, relation: str, obj: str) -> str:
    phrasing = {
        "left_of": f"Is the {subject} to the left of the {obj}?",
        "right_of": f"Is the {subject} to the right of the {obj}?",
        "above": f"Is the {subject} above the {obj}?",
        "below": f"Is the {subject} below the {obj}?",
        "upper_left_of": f"Is the {subject} above and to the left of the {obj}?",
        "upper_right_of": f"Is the {subject} above and to the right of the {obj}?",
        "lower_left_of": f"Is the {subject} below and to the left of the {obj}?",
        "lower_right_of": f"Is the {subject} below and to the right of the {obj}?",
    }
    return phrasing[relation]


def sample_positions(
    img_size: int,
    relation: str,
    min_shape: int,
    max_shape: int,
    min_gap: int,
    rng: random.Random,
) -> Dict[str, Tuple[int, int, int, int]]:
    """Sample bounding boxes (x1, y1, x2, y2) for subject and object respecting the relation."""
    grid_map = {
        "left_of": ((None, 0), (None, 2)),
        "right_of": ((None, 2), (None, 0)),
        "above": ((0, None), (2, None)),
        "below": ((2, None), (0, None)),
        "upper_left_of": ((0, 0), (2, 2)),
        "upper_right_of": ((0, 2), (2, 0)),
        "lower_left_of": ((2, 0), (0, 2)),
        "lower_right_of": ((2, 2), (0, 0)),
    }
    margin = max(4, img_size // 16)
    cell_pad = max(2, img_size // 32)
    grid_size = (img_size - 2 * margin) / 3.0

    def cell_bounds(row: int, col: int) -> Tuple[int, int, int, int]:
        x0 = margin + col * grid_size
        y0 = margin + row * grid_size
        x1 = margin + (col + 1) * grid_size
        y1 = margin + (row + 1) * grid_size
        return (math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1))

    def choose_cell(slot: Tuple[Optional[int], Optional[int]]) -> Tuple[int, int]:
        row, col = slot
        row = rng.randint(0, 2) if row is None else row
        col = rng.randint(0, 2) if col is None else col
        return row, col

    def sample_in_axis(min_v: int, max_v: int, size: int) -> Optional[int]:
        min_v = min_v + cell_pad
        max_v = max_v - cell_pad
        upper = max_v - size
        if upper < min_v:
            return None
        return rng.randint(min_v, upper)

    def has_min_gap(bbox_sub: Tuple[int, int, int, int], bbox_obj: Tuple[int, int, int, int]) -> bool:
        sx1, sy1, sx2, sy2 = bbox_sub
        ox1, oy1, ox2, oy2 = bbox_obj
        if relation in ("left_of", "upper_left_of", "lower_left_of"):
            if sx2 + min_gap > ox1:
                return False
        if relation in ("right_of", "upper_right_of", "lower_right_of"):
            if ox2 + min_gap > sx1:
                return False
        if relation in ("above", "upper_left_of", "upper_right_of"):
            if sy2 + min_gap > oy1:
                return False
        if relation in ("below", "lower_left_of", "lower_right_of"):
            if oy2 + min_gap > sy1:
                return False
        return True

    for _ in range(800):
        sub_cell = choose_cell(grid_map[relation][0])
        obj_cell = choose_cell(grid_map[relation][1])
        sub_bounds = cell_bounds(sub_cell[0], sub_cell[1])
        obj_bounds = cell_bounds(obj_cell[0], obj_cell[1])

        max_sub = min(max_shape, sub_bounds[2] - sub_bounds[0] - 2 * cell_pad)
        max_obj = min(max_shape, obj_bounds[2] - obj_bounds[0] - 2 * cell_pad)
        if max_sub < min_shape or max_obj < min_shape:
            continue

        w_sub = rng.randint(min_shape, max_sub)
        h_sub = w_sub
        w_obj = rng.randint(min_shape, max_obj)
        h_obj = w_obj

        x_sub = sample_in_axis(sub_bounds[0], sub_bounds[2], w_sub)
        y_sub = sample_in_axis(sub_bounds[1], sub_bounds[3], h_sub)
        x_obj = sample_in_axis(obj_bounds[0], obj_bounds[2], w_obj)
        y_obj = sample_in_axis(obj_bounds[1], obj_bounds[3], h_obj)
        if None in (x_sub, y_sub, x_obj, y_obj):
            continue

        bbox_sub = (x_sub, y_sub, x_sub + w_sub, y_sub + h_sub)
        bbox_obj = (x_obj, y_obj, x_obj + w_obj, y_obj + h_obj)
        if not has_min_gap(bbox_sub, bbox_obj):
            continue
        return {"subject": bbox_sub, "object": bbox_obj}

    raise RuntimeError("Failed to sample positions that satisfy constraints; try a larger canvas.")


def create_image(
    size: int,
    subject_shape: str,
    object_shape: str,
    boxes: Dict[str, Tuple[int, int, int, int]],
    colors: Dict[str, Tuple[int, int, int]],
    include_subject: bool,
    include_object: bool,
) -> Image.Image:
    img = Image.new("RGB", (size, size), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)
    if include_subject:
        draw_shape(draw, subject_shape, boxes["subject"], colors[subject_shape])
    if include_object:
        draw_shape(draw, object_shape, boxes["object"], colors[object_shape])
    return img


def generate_dataset(
    out_dir: Path,
    count: int,
    img_size: int,
    min_shape: int,
    max_shape: int,
    min_gap: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for idx in range(count):
        pair_id = f"pair_{idx:05d}"
        subject_shape, object_shape = rng.sample(SHAPES, 2)
        relation = rng.choice(RELATIONS)
        boxes = sample_positions(img_size, relation, min_shape, max_shape, min_gap, rng)
        caption = relation_caption(subject_shape, relation, object_shape)
        # Sample distinct colors for subject and object for this pair.
        color_subject, color_object = rng.sample(COLOR_PALETTE, 2)
        colors = {subject_shape: color_subject, object_shape: color_object}

        img_full = create_image(
            img_size,
            subject_shape,
            object_shape,
            boxes,
            colors,
            include_subject=True,
            include_object=True,
        )
        img_control = create_image(
            img_size,
            subject_shape,
            object_shape,
            boxes,
            colors,
            include_subject=False,
            include_object=True,
        )

        full_path = image_dir / f"{pair_id}_full.png"
        control_path = image_dir / f"{pair_id}_control.png"
        img_full.save(full_path)
        img_control.save(control_path)

        entries.append(
            SampleMetadata(
                id=pair_id,
                caption=caption,
                relation=relation,
                subject_shape=subject_shape,
                object_shape=object_shape,
                full_image=str(full_path),
                control_image=str(control_path),
                bounding_boxes={"subject": boxes["subject"], "object": boxes["object"]},
                note="control image removes the subject_shape; object stays in the same position",
            )
        )

    meta_path = out_dir / "metadata.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(asdict(entry)) + "\n")


def main() -> None:
    args = parse_args()
    generate_dataset(
        out_dir=args.out_dir,
        count=args.count,
        img_size=args.image_size,
        min_shape=args.min_shape,
        max_shape=args.max_shape,
        min_gap=args.min_gap,
        seed=args.seed,
    )
    print(f"Generated {args.count} pairs in {args.out_dir}")


if __name__ == "__main__":
    main()
